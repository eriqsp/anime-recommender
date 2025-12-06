import pandas as pd
import requests
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
import os


load_dotenv()
filepath = os.getenv('FILEPATH')
input_file = os.path.join(filepath, 'titles.csv')
output_file = os.path.join(filepath, 'streaming_data.csv')
temp_checkpoint_file = os.path.join(filepath, 'streaming_data_checkpoint.csv')

# API endpoints
anilist_api_url = 'https://graphql.anilist.co'
jikan_base_url = "https://api.jikan.moe/v4"

tracked_services = ["Crunchyroll", "Netflix", "Hulu", "Funimation", "Prime Video", "Disney+", "HIDIVE"]

retry_exceptions = (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),  # wait 4s, 8s, 16s, etc
    retry=retry_if_exception_type(retry_exceptions)
)
def fetch_anilist_id_mal_id(anilist_id: int):
    query = """
    query ($id: Int) {
      Media(id: $id, type: ANIME) {
        idMal
      }
    }
    """
    variables = {'id': anilist_id}

    response = requests.post(
        anilist_api_url,
        json={'query': query, 'variables': variables},
        timeout=15
    )

    # AniList API uses standard HTTP error codes, so we rely on retry_if_exception_type
    response.raise_for_status()
    data = response.json()

    # AniList returns idMal (int or null) under data > Media
    mal_id = data.get('data', {}).get('Media', {}).get('idMal')
    return mal_id


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(retry_exceptions)
)
def fetch_streaming_data_jikan(mal_id: int):
    url = f"{jikan_base_url}/anime/{mal_id}/streaming"

    response = requests.get(url, timeout=15)

    if response.status_code == 404:
        raise requests.exceptions.HTTPError(f"MAL ID {mal_id} not found on Jikan (404)")

    response.raise_for_status()
    data = response.json()

    streaming_links = data.get('data', [])

    found_services = []
    for stream in streaming_links:
        stream_name = stream['name']

        if stream_name in tracked_services:
            found_services.append(f"{stream_name}|{stream['url']}")

    return ",".join(found_services)


def enrich_data_with_streaming():
    if os.path.exists(temp_checkpoint_file):
        print(f"Checkpoint file found. Resuming from '{temp_checkpoint_file}'")
        df = pd.read_csv(temp_checkpoint_file)
        #df['streaming'] = df['streaming'].fillna("")
    else:
        print(f"Loading initial data from '{input_file}'")
        df = pd.read_csv(input_file)

        if 'streaming' not in df.columns:
            df['streaming'] = "-"

        df['id'] = df['item_id'].astype(str).str.extract(r'(\d+)').astype('Int64')

    total_rows = len(df)
    start_index = df[df['streaming'] == "-"].index.min()

    if pd.isna(start_index):
        print("All rows are already processed. Exiting.")
        if os.path.exists(temp_checkpoint_file):
            os.remove(temp_checkpoint_file)
        df.to_csv(output_file, index=False)
        return

    print(f"Starting processing from index {start_index} of {total_rows}...")

    for index in range(start_index, total_rows):
        row = df.loc[index]

        anilist_id = row['id']
        title = row['title']
        streaming_data = None

        try:
            mal_id = fetch_anilist_id_mal_id(int(anilist_id))

            if mal_id is None:
                print(f"[{index + 1}/{total_rows}] no MAL ID found for AniList ID {anilist_id} ({title}). Skipping.")
            else:
                streaming_data = fetch_streaming_data_jikan(mal_id)

                status_msg = f"Found {len(streaming_data.split(','))} service(s)" if streaming_data else "No services found"
                print(f"[{index + 1}/{total_rows}] Processed: {title} (MAL ID: {mal_id}). {status_msg}")

        except requests.exceptions.HTTPError as e:
            print(f"[{index + 1}/{total_rows}] HTTP Error processing {title}: {e}")
        except tenacity.RetryError as e:
            print(f"[{index + 1}/{total_rows}] RETRY FAILED on {title}: {e}")
        except Exception as e:
            print(f"[{index + 1}/{total_rows}] UNEXPECTED ERROR on {title}: {e}. Stopping to check")
            break

        df.loc[index, 'streaming'] = streaming_data

        # checkpoint
        if (index + 1) % 100 == 0:
            df.to_csv(temp_checkpoint_file, index=False)
            print(f"\n--- CHECKPOINT SAVED. Total processed: {index + 1} ---\n")

    if (total_rows - 1) == index:
        print(f"All {total_rows} titles processed successfully!")
        df.to_csv(output_file, index=False)  # save final df
        if os.path.exists(temp_checkpoint_file):
            os.remove(temp_checkpoint_file)
    else:
        df.to_csv(temp_checkpoint_file, index=False)
        print(f"\nProcess stopped at index {index}. Data saved to '{temp_checkpoint_file}'. Rerun script to resume")


if __name__ == "__main__":
    enrich_data_with_streaming()
