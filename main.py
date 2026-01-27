import asyncio
import pandas as pd
from autobatcher import BatchOpenAI
import os
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import hashlib
import re
from pathlib import Path
import ssl
import certifi
import aiohttp

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("DW_API_KEY")
if not API_KEY:
    raise RuntimeError("DW_API_KEY not set in .env")

# Configurations
CSV_PATH = "unsplash-research-dataset-lite-latest/photos.csv000"
PHOTO_URL_COLUMN = "photo_image_url"
DESCRIPTION_COLUMN = "photo_description"
PHOTOGRAPHER_COLUMN = "photographer_username"
OUTPUT_CSV = "unsplash_summaries_1000.csv"

NUM_IMAGES = 1000
TARGET_WIDTH = 1280  # Set your desired width
TARGET_HEIGHT = 720  # Set your desired height

async def summarize_photos():

    # Set up cache directory for images
    CACHE_DIR = Path("image_cache")
    CACHE_DIR.mkdir(exist_ok=True)

    def sanitize_url(url):
        # Use a hash for uniqueness and avoid filesystem issues
        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        # Try to preserve file extension if present
        ext = url.split('.')[-1].split('?')[0].lower()
        if ext not in {"jpg", "jpeg", "png", "gif", "bmp", "webp"}:
            ext = "jpg"
        return f"{url_hash}.{ext}"

    # Load CSV (Unsplash Lite photos dataset is tab-separated)
    df = pd.read_csv(
        CSV_PATH,
        sep="\t",
        dtype=str,
        on_bad_lines='skip'  # skip rows that don't match the columns
    )

    # Keep only rows with valid URLs
    df = df[df[PHOTO_URL_COLUMN].notna()]

    # Take only the first NUM_IMAGES rows
    df = df.head(NUM_IMAGES)

    # Create Autobatcher client
    client = BatchOpenAI(
        api_key=API_KEY,
        base_url="https://api.doubleword.ai/v1",
        batch_size=NUM_IMAGES,
    )

    async def fetch_and_resize_image(session, url, target_width, target_height):
        try:
            cache_filename = CACHE_DIR / sanitize_url(url)
            if cache_filename.exists():
                with open(cache_filename, "rb") as f:
                    img_bytes = f.read()
            else:
                import aiohttp
                timeout = aiohttp.ClientTimeout(total=200)
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status != 200:
                        print(f"Failed to fetch image: {url} (HTTP {resp.status})")
                        return None
                    img_bytes = await resp.read()
                with open(cache_filename, "wb") as f:
                    f.write(img_bytes)
                    
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as img_exc:
                print(f"Error opening image {url}: {img_exc}")
                return None

            # Crop to target aspect ratio (center crop)
            orig_width, orig_height = img.size
            target_aspect = target_width / target_height
            orig_aspect = orig_width / orig_height
            if orig_aspect > target_aspect:
                # Image is wider than target: crop width
                new_width = int(orig_height * target_aspect)
                left = (orig_width - new_width) // 2
                right = left + new_width
                top = 0
                bottom = orig_height
            else:
                # Image is taller than target: crop height
                new_height = int(orig_width / target_aspect)
                top = (orig_height - new_height) // 2
                bottom = top + new_height
                left = 0
                right = orig_width
            try:
                img = img.crop((left, top, right, bottom))
                img = img.resize((target_width, target_height), Image.LANCZOS)
            except Exception as crop_exc:
                print(f"Error cropping/resizing image {url}: {crop_exc}")
                return None
            buf = io.BytesIO()
            try:
                img.save(buf, format="JPEG")
            except Exception as save_exc:
                print(f"Error saving image {url}: {save_exc}")
                return None
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return b64
        except Exception as e:
            print(f"Error processing image {url}: {type(e).__name__}: {e}")
            return None


    # Async summarization function
    async def summarize(image: str, description: str, photographer: str, max_retries: int = 3) -> str:
        prompt = (
            "Summarize this image for a social media-style post.\n\n"
            f"Caption: {description}\n\n"
            f"Photographer: {photographer}\n\n"
            "Write a concise summary ignoring any irrelevant metadata."
        )
        try:
            # If the input is a URL, use it directly; otherwise, treat as base64
            if isinstance(image, str) and (image.startswith('http://') or image.startswith('https://')):
                image_url_payload = {"url": image}
            else:
                image_url_payload = {"url": f"data:image/jpeg;base64,{image}"}

            response = await client.chat.completions.create(
                model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": image_url_payload
                            }
                        ]
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as api_exc:
            print(f"API error (attempt {attempt}) for image: {type(api_exc).__name__}: {api_exc}")
            last_exc = api_exc
            await asyncio.sleep(2 * attempt)  # Exponential backoff

    # Fetch, resize, and encode images in parallel
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        fetch_tasks = [fetch_and_resize_image(session, row[PHOTO_URL_COLUMN], TARGET_WIDTH, TARGET_HEIGHT) for _, row in df.iterrows()]
        b64_images = await asyncio.gather(*fetch_tasks)

    # Prepare async summarization tasks (skip images that failed to process)
    tasks = [
        summarize(b64, row[DESCRIPTION_COLUMN], row[PHOTOGRAPHER_COLUMN])
        if b64 is not None else asyncio.sleep(0, result="[IMAGE ERROR]")
        for b64, (_, row) in zip(b64_images, df.iterrows())
    ]

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