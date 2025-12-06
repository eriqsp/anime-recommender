import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from keras import layers, Model, losses, optimizers, callbacks
from dotenv import load_dotenv
import os
from data_handling.cleaning import DataCleaner
import joblib


class ModelBuild:
    def __init__(self, data: pd.DataFrame, filepath: str):
        self.df = data
        self.filepath = filepath

        self.genres = None
        self.tags = None
        self.studios = None
        self.episodes = None
        self.popularity = None

        self.concat_features = []
        self.inputs = []
        self.outputs = None

        self.model = None

        self.loss_dict = {
            "genres_out": losses.BinaryCrossentropy(),
            "tags_out": losses.BinaryCrossentropy(),
            "studios_out": losses.BinaryCrossentropy(),
            "episodes_out": losses.MeanSquaredError(),
            "pop_out": losses.MeanSquaredError()
        }
        self.loss_weights = {
            "genres_out": 1.0,
            "tags_out": 0.9,
            "studios_out": 0.8,
            "episodes_out": 0.5,
            "pop_out": 0.5
        }

    def build_and_save(self):
        texts, titles = self._init_features()

        # create inputs for the nn
        self._text_branch(texts)
        self._multi_hot()
        self._num_inputs()

        # create outputs for the nn
        self._output_heads()

        self._model_init()

        inputs = self._training(texts)
        encoder = Model(inputs=self.model.inputs, outputs=self.model.get_layer("l2_embedding").output)
        all_embeddings = encoder.predict(inputs, batch_size=256)  # shape (N, 128)

        nbrs = NearestNeighbors(n_neighbors=11, metric='cosine').fit(all_embeddings)

        # saving
        self.model.save(os.path.join(self.filepath, "full_multitask_model_tf"))
        encoder.save(os.path.join(self.filepath, "anime_encoder_tf"))
        joblib.dump(nbrs, os.path.join(self.filepath, "nbrs_cosine.joblib"))
        np.save(os.path.join(self.filepath, "all_embeddings.npy"), all_embeddings)
        self.df[['item_id', 'title', 'image_url', 'average_score', 'synopsis']].to_csv(os.path.join(self.filepath, "titles.csv"), index=False)

    def _parse_array_col(self, col):
        return np.vstack(self.df[col].values)

    def _init_features(self):
        self.genres = self._parse_array_col("genres_multi")
        self.tags = self._parse_array_col("tags_multi")
        self.studios = self._parse_array_col("studios_multi")

        # make sure equal rows
        assert len(self.df) == len(self.genres) == len(self.tags) == len(self.studios)

        texts = self.df["text"].fillna("").astype(str).values
        titles = self.df["title"].values
        self.episodes = self.df["episodes"].fillna(0).astype(np.float32).values.reshape(-1, 1)
        self.popularity = self.df["popularity"].fillna(0).astype(np.float32).values.reshape(-1, 1)
        return texts, titles

    def _text_branch(self, texts):
        max_vocab = 30000
        max_len = 300  # max tokens per text (adjust as needed)
        vectorizer = layers.TextVectorization(max_tokens=max_vocab, output_sequence_length=max_len)
        vectorizer.adapt(texts)

        text_inputs = layers.Input(shape=(1,), dtype=tf.string, name="text")
        x = vectorizer(text_inputs)  # turns the string into a sequence of integers (token ids)
        x = layers.Embedding(input_dim=max_vocab, output_dim=128, name="text_emb")(x)  # converts the tokens ids into vectors
        x = layers.Conv1D(128, kernel_size=3, activation="relu")(x)  # helps to extract relevant local features (convolutional)
        x = layers.GlobalMaxPooling1D()(x)  # takes the maximum value across the sequence dimension for each filter (filter -> results from the conv1d)
        text_feat = layers.Dense(128, activation="relu")(x)
        self.concat_features.append(text_feat)
        self.inputs.append(text_inputs)

    def _multi_hot(self):
        genres_in = layers.Input(shape=(self.genres.shape[1],), name="genres")
        tags_in = layers.Input(shape=(self.tags.shape[1],), name="tags")
        studios_in = layers.Input(shape=(self.studios.shape[1],), name="studios")

        g = layers.Dense(128, activation="relu")(genres_in)
        t = layers.Dense(128, activation="relu")(tags_in)
        s = layers.Dense(64, activation="relu")(studios_in)
        self.concat_features.append(g)
        self.concat_features.append(t)
        self.concat_features.append(s)

        self.inputs.append(genres_in)
        self.inputs.append(tags_in)
        self.inputs.append(studios_in)

    def _num_inputs(self):
        episodes_in = layers.Input(shape=(1,), name="episodes")
        pop_in = layers.Input(shape=(1,), name="popularity")
        num = layers.Concatenate()([episodes_in, pop_in])
        num = layers.Dense(32, activation="relu")(num)
        self.concat_features.append(num)
        self.inputs.append(episodes_in)
        self.inputs.append(pop_in)

    def _output_heads(self):
        concat = layers.Concatenate()(self.concat_features)
        x = layers.Dense(256, activation="relu")(concat)
        x = layers.Dropout(0.2)(x)
        embedding = layers.Dense(128, activation=None, name="embedding")(x)  # raw embedding
        embedding_norm = layers.Lambda(lambda z: tf.nn.l2_normalize(z, axis=1), name="l2_embedding")(embedding)

        self.outputs = [
            # multi-label heads -> sigmoid
            layers.Dense(self.genres.shape[1], activation="sigmoid", name="genres_out")(embedding),
            layers.Dense(self.tags.shape[1], activation="sigmoid", name="tags_out")(embedding),
            layers.Dense(self.studios.shape[1], activation="sigmoid", name="studios_out")(embedding),
            # numeric heads -> regression
            layers.Dense(1, activation="sigmoid", name="episodes_out")(embedding),
            layers.Dense(1, activation="sigmoid", name="pop_out")(embedding),
            embedding_norm
        ]

    def _model_init(self):
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()

    """ 
    - train a model that reconstructs its own inputs (autoencoder)
    - in each training step, the model compares the input with the output (reconstructed by the decoder)
    - the parameters are updated to minimize reconstruction error
    - the middle layer becomes the embedding (latent vector)
    - takes the embedded vector as output for each anime
    - input > encoder > embedding > decoder > output
    """
    def _training(self, texts):
        self.model.compile(optimizer=optimizers.Adam(1e-3), loss=self.loss_dict, loss_weights=self.loss_weights)

        # prepare inputs
        batch_text = texts.reshape(-1, 1)  # TextInput expects shape (batch,1) of strings
        inputs = {
            "text": batch_text,
            "genres": self.genres.astype(np.float32),
            "tags": self.tags.astype(np.float32),
            "studios": self.studios.astype(np.float32),
            "episodes": self.episodes.astype(np.float32),
            "popularity": self.popularity.astype(np.float32)
        }
        targets = {
            "genres_out": self.genres.astype(np.float32),
            "tags_out": self.tags.astype(np.float32),
            "studios_out": self.studios.astype(np.float32),
            "episodes_out": self.episodes.astype(np.float32),
            "pop_out": self.popularity.astype(np.float32)
        }

        early = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        split = int(len(self.df) * 0.9)
        self.model.fit(
            {k: v[:split] for k, v in inputs.items()},
            {k: v[:split] for k, v in targets.items()},

            validation_data=(
                {k: v[split:] for k, v in inputs.items()},
                {k: v[split:] for k, v in targets.items()}
            ),
            epochs=20, batch_size=64, callbacks=[early]
                       )
        return inputs


if __name__ == '__main__':
    load_dotenv()

    dc = DataCleaner(os.getenv('PARQUET_FILENAME'), add_numerical=True)
    df = dc.final_stage_df()

    mb = ModelBuild(df, os.getenv('FILEPATH'))
    mb.build_and_save()
