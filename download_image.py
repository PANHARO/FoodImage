"""Build an approximately 500-image Laphet Thoke class and replace Mohinga.

Images are searched only with Laphet Thoke / Burmese tea-leaf-salad terms.
Every accepted image is converted to RGB, center-cropped to 224x224, saved as
an optimized JPEG, and deduplicated before the 80/10/10 split.

Run from the project root:
    python download_laphet_thoke.py
"""

import argparse
import csv
import hashlib
import html
import io
import json
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


CLASS_NAME = "laphet_thoke"
DISPLAY_NAME = "Laphet Thoke"
COUNTRY = "Myanmar"
DATASET_ROOT = Path("dataset")
CSV_ROOT = Path("csv")
STAGING_ROOT = Path(".laphet_thoke_staging")
USER_AGENT = (
    "FoodDatasetBuilder/1.0 "
    "(educational image-classification dataset; contact: local-user)"
)
SEARCH_QUERIES = [
    # Multiple spellings improve coverage without introducing another dish.
    '"laphet thoke"',
    '"lahpet thoke"',
    "laphet thoke Myanmar",
    "lahpet thoke Myanmar",
    "laphet thoke food",
    "lahpet thoke salad",
    '"Burmese tea leaf salad"',
    '"Myanmar tea leaf salad"',
    "Burmese fermented tea leaf salad",
    "Myanmar fermented tea leaf salad",
    "Burmese pickled tea leaf salad",
    "lahpet salad Burma",
    "laphet thoke recipe",
    "lahpet thoke recipe",
    "laphet thoke traditional dish",
    "lahpet thoke traditional dish",
    "tea leaf salad Burma",
    "tea leaf salad Yangon",
    "Myanmar tea salad dish",
    "Burmese tea salad dish",
    "traditional Burmese tea leaf salad",
    "fermented tea leaf salad Burma",
    "Myanmar lahpet salad",
    "Burmese lahpet salad",
    "laphet thoke bowl",
    "lahpet thoke bowl",
    "laphet thoke plate",
    "lahpet thoke plate",
    "laphet thoke restaurant",
    "lahpet thoke restaurant",
    '"lephet thoke"',
    '"laphat thoke"',
    '"lahpet thohk"',
    '"laphet thohk"',
    "Burmese tea leaf salad recipe",
    "Myanmar tea leaf salad recipe",
    "Burmese tea leaf salad restaurant",
    "Myanmar tea leaf salad restaurant",
    "fermented tea leaves salad Myanmar food",
    "pickled tea leaves salad Burmese food",
    "Yangon tea leaf salad",
    "Mandalay tea leaf salad",
    "Rangoon tea leaf salad",
    "Burma Superstar tea leaf salad",
    "Burma Love tea leaf salad",
    "tea leaf salad peanuts sesame tomatoes",
    "site:flickr.com lahpet thoke",
    "site:blogspot.com Burmese tea leaf salad",
]
CSV_FIELDS = [
    "file_path", "abs_path", "label", "display_name", "country", "split",
    "width", "height", "orig_width", "orig_height", "source", "source_url",
    "sha256", "created_at",
]
SOURCE_FIELDS = [
    "filename", "split", "query", "source", "image_url", "landing_page",
    "title", "creator", "license", "sha256", "orig_width", "orig_height",
    "final_bytes",
]
BLOCKED_DOMAINS = {
    "img.freepik.com",
    "static.vecteezy.com",
    "i.etsystatic.com",
    "m.media-amazon.com",
    "myanmarfoodusa.com",
}
POSITIVE_TERMS = (
    "laphet", "lahpet", "lephet", "laphat", "tea leaf salad",
    "tea-leaf-salad", "tea_leaf_salad", "fermented tea leaf salad",
)
BLOCKED_TERMS = (
    "premium vector", "premium-vector", "/vector/", "ai-generated",
    "ai generated", "clipart", " tea leaf salad kit", " salad kit",
    "burger", "chicken", "curry", "pudding", "tikki", "lamb-mandi",
    "lamb mandi", "palm-sugar", "palm sugar", "nan gyi", "noodle salad",
)


def request_bytes(url, timeout=30, max_bytes=15_000_000):
    url = urllib.parse.quote(
        url, safe=":/?&=%#[]@!$'()*+,;~"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("image exceeds maximum download size")
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("image exceeds maximum download size")
        return data


def request_text(url, timeout=30):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "ignore")


def openverse_candidates(pages_per_query):
    endpoint = "https://api.openverse.org/v1/images/"
    seen_urls = set()
    for query in SEARCH_QUERIES:
        for page in range(1, min(pages_per_query, 10) + 1):
            params = urllib.parse.urlencode(
                {"q": query, "page": page, "page_size": 50, "mature": "false"}
            )
            try:
                payload = json.loads(request_text(f"{endpoint}?{params}"))
            except urllib.error.HTTPError as error:
                print(f"Openverse warning ({query}, page {page}): {error}")
                if error.code in {401, 429}:
                    print("Openverse unavailable; continuing with Bing only.")
                    return
                break
            except Exception as error:
                print(f"Openverse warning ({query}, page {page}): {error}")
                break
            results = payload.get("results", [])
            if not results:
                break
            for item in results:
                image_url = item.get("url")
                if not image_url or image_url in seen_urls:
                    continue
                seen_urls.add(image_url)
                yield {
                    "query": query,
                    "source": f"openverse:{item.get('source', 'unknown')}",
                    "image_url": image_url,
                    "landing_page": item.get("foreign_landing_url", ""),
                    "title": item.get("title", ""),
                    "creator": item.get("creator", ""),
                    "license": " ".join(
                        value for value in [
                            item.get("license", ""),
                            item.get("license_version", ""),
                        ] if value
                    ),
                }
            if page >= payload.get("page_count", page):
                break
            time.sleep(0.25)


def bing_candidates(pages_per_query):
    endpoint = "https://www.bing.com/images/async"
    seen_urls = set()
    for query in SEARCH_QUERIES:
        search_query = query
        for page in range(pages_per_query):
            params = urllib.parse.urlencode({
                "q": search_query,
                "first": page * 35,
                "count": 35,
                "adlt": "strict",
                "safeSearch": "Strict",
            })
            try:
                body = request_text(f"{endpoint}?{params}")
            except Exception as error:
                print(f"Bing warning ({query}, page {page + 1}): {error}")
                break

            found = 0
            for encoded in re.findall(r'\bm="([^"]+)"', body):
                try:
                    item = json.loads(html.unescape(encoded))
                except (json.JSONDecodeError, TypeError):
                    continue
                image_url = item.get("murl")
                if not image_url or image_url in seen_urls:
                    continue
                if not image_url.startswith(("http://", "https://")):
                    continue
                seen_urls.add(image_url)
                found += 1
                yield {
                    "query": query,
                    "source": "bing",
                    "image_url": image_url,
                    "landing_page": item.get("purl", ""),
                    "title": item.get("t", ""),
                    "creator": "",
                    "license": "verify at source",
                }
            if found == 0:
                break
            time.sleep(0.2)


def duckduckgo_candidates(pages_per_query):
    seen_urls = set()
    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/138.0 Safari/537.36"
        ),
        "Referer": "https://duckduckgo.com/",
    }
    for query in SEARCH_QUERIES:
        search_query = query
        search_url = "https://duckduckgo.com/?" + urllib.parse.urlencode({
            "q": search_query, "iax": "images", "ia": "images"
        })
        try:
            request = urllib.request.Request(search_url, headers=browser_headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8", "ignore")
            token_match = (
                re.search(r'vqd=["\']([^"\']+)', body)
                or re.search(r"vqd=([0-9-]+)", body)
            )
            if not token_match:
                continue
            token = token_match.group(1)
            next_url = "https://duckduckgo.com/i.js?" + urllib.parse.urlencode({
                "l": "us-en",
                "o": "json",
                "q": search_query,
                "vqd": token,
                "f": ",,,",
                "p": "1",
            })
            for _ in range(min(pages_per_query, 2)):
                request = urllib.request.Request(next_url, headers=browser_headers)
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                for item in payload.get("results", []):
                    image_url = item.get("image")
                    if not image_url or image_url in seen_urls:
                        continue
                    if not image_url.startswith(("http://", "https://")):
                        continue
                    seen_urls.add(image_url)
                    yield {
                        "query": query,
                        "source": "duckduckgo",
                        "image_url": image_url,
                        "landing_page": item.get("url", ""),
                        "title": item.get("title", ""),
                        "creator": "",
                        "license": "verify at source",
                    }
                next_path = payload.get("next")
                if not next_path:
                    break
                next_url = urllib.parse.urljoin(
                    "https://duckduckgo.com/", next_path
                )
                time.sleep(0.4)
        except urllib.error.HTTPError as error:
            print(f"DuckDuckGo warning ({query}): {error}")
            if error.code in {401, 403, 429}:
                print("DuckDuckGo unavailable; continuing with Bing only.")
                return
        except Exception as error:
            print(f"DuckDuckGo warning ({query}): {error}")
        time.sleep(0.4)


def candidate_is_relevant(candidate):
    parsed = urllib.parse.urlsplit(candidate["image_url"])
    domain = parsed.netloc.lower().removeprefix("www.")
    if domain in BLOCKED_DOMAINS:
        return False
    metadata = " ".join([
        candidate.get("title", ""),
        candidate.get("image_url", ""),
        candidate.get("landing_page", ""),
    ]).lower()
    if not any(term in metadata for term in POSITIVE_TERMS):
        return False
    return not any(term in metadata for term in BLOCKED_TERMS)


def difference_hash(image):
    gray = image.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
    pixels = list(gray.get_flattened_data())
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value


def normalize_candidate(candidate, quality):
    try:
        raw = request_bytes(candidate["image_url"])
        with Image.open(io.BytesIO(raw)) as opened:
            image = ImageOps.exif_transpose(opened)
            orig_width, orig_height = image.size
            if min(orig_width, orig_height) < 224:
                raise ValueError("image is smaller than 224 pixels")
            ratio = orig_width / orig_height
            if not 0.45 <= ratio <= 2.2:
                raise ValueError("extreme aspect ratio")
            image = image.convert("RGB")
            image = ImageOps.fit(
                image,
                (224, 224),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            perceptual_hash = difference_hash(image)
            output = io.BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
                subsampling=2,
            )
        normalized = output.getvalue()
        digest = hashlib.sha256(normalized).hexdigest()
        return {
            **candidate,
            "bytes": normalized,
            "sha256": digest,
            "dhash": perceptual_hash,
            "orig_width": orig_width,
            "orig_height": orig_height,
            "final_bytes": len(normalized),
        }
    except (
        OSError,
        ValueError,
        TimeoutError,
        UnidentifiedImageError,
        urllib.error.URLError,
    ):
        return None


def is_near_duplicate(candidate_hash, accepted_hashes, max_distance=3):
    return any(
        (candidate_hash ^ existing_hash).bit_count() <= max_distance
        for existing_hash in accepted_hashes
    )


def collect_images(target, pages_per_query, workers, quality):
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    image_stage = STAGING_ROOT / "images"
    image_stage.mkdir(parents=True)

    candidates = []
    candidate_urls = set()
    for candidate in openverse_candidates(pages_per_query):
        if (
            candidate_is_relevant(candidate)
            and candidate["image_url"] not in candidate_urls
        ):
            candidate_urls.add(candidate["image_url"])
            candidates.append(candidate)
    for candidate in bing_candidates(pages_per_query):
        if (
            candidate_is_relevant(candidate)
            and candidate["image_url"] not in candidate_urls
        ):
            candidate_urls.add(candidate["image_url"])
            candidates.append(candidate)
    for candidate in duckduckgo_candidates(pages_per_query):
        if (
            candidate_is_relevant(candidate)
            and candidate["image_url"] not in candidate_urls
        ):
            candidate_urls.add(candidate["image_url"])
            candidates.append(candidate)
    print(f"Found {len(candidates)} unique candidate URLs.")

    accepted = []
    exact_hashes = set()
    perceptual_hashes = []
    batch_size = max(100, workers * 8)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            futures = [
                executor.submit(normalize_candidate, candidate, quality)
                for candidate in batch
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    result = None
                if result is None or result["sha256"] in exact_hashes:
                    continue
                if is_near_duplicate(result["dhash"], perceptual_hashes):
                    continue
                filename = f"{CLASS_NAME}_{result['sha256'][:12]}.jpg"
                stage_path = image_stage / filename
                stage_path.write_bytes(result.pop("bytes"))
                result["filename"] = filename
                result["stage_path"] = stage_path
                exact_hashes.add(result["sha256"])
                perceptual_hashes.append(result["dhash"])
                accepted.append(result)
                if len(accepted) % 25 == 0:
                    print(f"Accepted {len(accepted)}/{target} images.")
                if len(accepted) >= target:
                    return accepted
    return accepted


def replace_dataset(records, seed):
    random.Random(seed).shuffle(records)
    total = len(records)
    train_count = round(total * 0.8)
    val_count = round(total * 0.1)
    assignments = (
        ["train"] * train_count
        + ["val"] * val_count
        + ["test"] * (total - train_count - val_count)
    )

    for split in ("train", "val", "test"):
        new_dir = DATASET_ROOT / split / CLASS_NAME
        old_dir = DATASET_ROOT / split / "mohinga"
        if new_dir.exists():
            shutil.rmtree(new_dir)
        if old_dir.exists():
            shutil.rmtree(old_dir)
        new_dir.mkdir(parents=True)

    created_at = datetime.now(timezone.utc).isoformat()
    dataset_rows = []
    source_rows = []
    for record, split in zip(records, assignments):
        destination = DATASET_ROOT / split / CLASS_NAME / record["filename"]
        shutil.move(str(record["stage_path"]), destination)
        relative_path = Path(split) / CLASS_NAME / record["filename"]
        dataset_rows.append({
            "file_path": str(relative_path),
            "abs_path": str(destination.resolve()),
            "label": CLASS_NAME,
            "display_name": DISPLAY_NAME,
            "country": COUNTRY,
            "split": split,
            "width": 224,
            "height": 224,
            "orig_width": record["orig_width"],
            "orig_height": record["orig_height"],
            "source": record["source"],
            "source_url": record["image_url"],
            "sha256": record["sha256"],
            "created_at": created_at,
        })
        source_rows.append({
            "filename": record["filename"],
            "split": split,
            "query": record["query"],
            "source": record["source"],
            "image_url": record["image_url"],
            "landing_page": record["landing_page"],
            "title": record["title"],
            "creator": record["creator"],
            "license": record["license"],
            "sha256": record["sha256"],
            "orig_width": record["orig_width"],
            "orig_height": record["orig_height"],
            "final_bytes": record["final_bytes"],
        })

    update_csv_files(dataset_rows, source_rows)
    shutil.rmtree(STAGING_ROOT, ignore_errors=True)
    return {split: assignments.count(split) for split in ("train", "val", "test")}


def update_csv_files(dataset_rows, source_rows):
    CSV_ROOT.mkdir(exist_ok=True)
    by_split = {
        split: [row for row in dataset_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    for split, new_rows in by_split.items():
        path = CSV_ROOT / f"{split}.csv"
        preserved = []
        if path.exists():
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                preserved = [
                    row for row in csv.DictReader(handle)
                    if row.get("label") not in {"mohinga", CLASS_NAME}
                ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(preserved + new_rows)

    per_class = CSV_ROOT / "per_class"
    per_class.mkdir(exist_ok=True)
    old_csv = per_class / "mohinga.csv"
    if old_csv.exists():
        old_csv.unlink()
    with (per_class / f"{CLASS_NAME}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(dataset_rows)
    with (per_class / f"{CLASS_NAME}_sources.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(source_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=480)
    parser.add_argument("--pages-per-query", type=int, default=15)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--jpeg-quality", type=int, default=78)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.target < 10:
        parser.error("--target must be at least 10")
    if not 40 <= args.jpeg_quality <= 90:
        parser.error("--jpeg-quality must be between 40 and 90")

    records = collect_images(
        args.target, args.pages_per_query, args.workers, args.jpeg_quality
    )
    if len(records) < args.target:
        raise RuntimeError(
            f"Only {len(records)} valid unique images were found; "
            f"the existing Mohinga dataset was NOT changed. "
            f"Increase --pages-per-query and retry."
        )

    counts = replace_dataset(records[:args.target], args.seed)
    print("\nLaphet Thoke replacement complete:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counts[split]} images")
    print("  size: 224x224 RGB optimized JPEG")
    print(f"  total: {sum(counts.values())} images")


if __name__ == "__main__":
    main()
