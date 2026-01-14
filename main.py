import asyncio
import pandas as pd
from autobatcher import BatchOpenAI
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# Configurations
CSV_PATH = "unsplash-research-dataset-lite-latest/photos.csv000"
PHOTO_URL_COLUMN = "photo_image_url"
DESCRIPTION_COLUMN = "photo_description"
PHOTOGRAPHER_COLUMN = "photographer_username"
OUTPUT_CSV = "unsplash_summaries_1000.csv"
NUM_IMAGES = 1000

async def summarize_photos():
    # Load CSV (Unsplash Lite photos dataset is tab-separated)
    df = pd.read_csv(
        CSV_PATH,
        sep="\t",
        dtype=str,
        on_bad_lines='skip'  # skip rows that don't match the columns
    )

    # Debug: print column names
    print(df.columns.tolist())

    # Keep only rows with valid URLs
    df = df[df[PHOTO_URL_COLUMN].notna()]

    # Take only the first NUM_IMAGES rows
    df = df.head(NUM_IMAGES)

    # Create Autobatcher client
    client = BatchOpenAI(
        api_key=API_KEY,
        base_url="https://api.doubleword.ai/v1",
        batch_size=1000,
    )

    # Async summarization function
    async def summarize(url: str, description: str, photographer: str) -> str:
        prompt = (
            "Summarize this image for a social media-style post.\n\n"
            f"Image: {url}\n"
            f"Caption: {description}\n\n"
            f"Photographer: {photographer}\n\n"
            "Write a concise summary ignoring any irrelevant metadata."
        )
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # Prepare async tasks
    tasks = [summarize(row[PHOTO_URL_COLUMN], row[DESCRIPTION_COLUMN], row[PHOTOGRAPHER_COLUMN]) for _, row in df.iterrows()]

    # Run tasks concurrently (batched under the hood)
    summaries = await asyncio.gather(*tasks)

    # Attach summaries back to DataFrame
    df["summary"] = summaries

    # Save results
    df[[PHOTO_URL_COLUMN, DESCRIPTION_COLUMN, PHOTOGRAPHER_COLUMN, "summary"]].to_csv("unsplash_summaries.csv", index=False)

    # Close Autobatcher client
    await client.close()

# Entry point
if __name__ == "__main__":
    asyncio.run(summarize_photos())