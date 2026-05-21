from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


DEFAULT_CATEGORIES = ["cs.CL", "cs.LG", "cs.CV", "cs.AI", "cs.NE", "stat.ML"]
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
API_URL = "https://export.arxiv.org/api/query"
COLUMNS = ["paper_id", "title", "abstract", "category", "published", "updated"]


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def arxiv_id(value: str | None) -> str:
    value = clean(value)
    value = value.removeprefix("https://arxiv.org/abs/").removeprefix("http://arxiv.org/abs/")
    return re.sub(r"v\d+$", "", value)


def load_existing_csv(csv_path: Path, categories: list[str]) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="paper_id", keep="first")
    df = df[df["category"].isin(categories)].copy()
    df.to_csv(csv_path, index=False)
    return df


def load_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {}
    return json.loads(progress_path.read_text(encoding="utf-8"))


def save_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def append_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def fetch_category_page(category: str, start: int, page_size: int, sleep_seconds: float, retries: int = 4) -> tuple[ET.Element, int | None]:
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": page_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "final-nlp-project/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                root = ET.fromstring(response.read())
            total_text = root.findtext(f"{OPENSEARCH}totalResults", "")
            total_results = int(total_text) if total_text.isdigit() else None
            return root, total_results
        except (urllib.error.URLError, TimeoutError, ET.ParseError) as exc:
            wait = max(sleep_seconds, 5 * attempt)
            print(f"    fetch failed attempt {attempt}/{retries}: {exc}; waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {category} page starting at {start}")


def parse_entry(entry: ET.Element) -> dict[str, str]:
    primary = entry.find(f"{ARXIV}primary_category")
    primary_category = primary.attrib.get("term", "") if primary is not None else ""
    return {
        "paper_id": arxiv_id(entry.findtext(f"{ATOM}id", "")),
        "title": clean(entry.findtext(f"{ATOM}title", "")),
        "abstract": clean(entry.findtext(f"{ATOM}summary", "")),
        "category": primary_category,
        "published": entry.findtext(f"{ATOM}published", ""),
        "updated": entry.findtext(f"{ATOM}updated", ""),
    }


def download_dataset(
    csv_path: Path,
    progress_path: Path,
    categories: list[str],
    papers_per_category: int,
    page_size: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = load_existing_csv(csv_path, categories)
    progress = load_progress(progress_path)
    seen_ids = set(existing_df["paper_id"].astype(str)) if len(existing_df) else set()

    print(f"Resume CSV: {csv_path}")
    print(f"Progress file: {progress_path}")
    print(f"Already saved rows: {len(existing_df)}")
    if len(existing_df):
        print("Existing category counts:")
        print(existing_df["category"].value_counts().sort_index())
    print()

    for category in categories:
        current_count = int((existing_df["category"] == category).sum()) if len(existing_df) else 0
        start = int(progress.get(category, {}).get("next_start", 0))

        print("=" * 80)
        print(f"Category {category}: have {current_count}/{papers_per_category}; resume start={start}")

        if current_count >= papers_per_category:
            print(f"Category {category}: complete, skipping")
            continue

        while current_count < papers_per_category:
            root, total_results = fetch_category_page(category, start, page_size, sleep_seconds)
            entries = root.findall(f"{ATOM}entry")
            if not entries:
                print(f"Category {category}: no entries returned at start={start}; stopping")
                break

            new_rows = []
            primary_matches = 0
            duplicates = 0

            for entry in entries:
                row = parse_entry(entry)
                if row["category"] == category:
                    primary_matches += 1
                else:
                    continue

                if row["paper_id"] in seen_ids:
                    duplicates += 1
                    continue

                new_rows.append(row)
                seen_ids.add(row["paper_id"])
                current_count += 1
                if current_count >= papers_per_category:
                    break

            append_rows(csv_path, new_rows)
            start += page_size
            progress[category] = {
                "next_start": start,
                "saved_count": current_count,
                "target_count": papers_per_category,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_progress(progress_path, progress)

            print(
                f"Category {category} page start={start - page_size}: "
                f"entries={len(entries)}, primary_matches={primary_matches}, "
                f"new_saved={len(new_rows)}, duplicates={duplicates}, "
                f"total_saved={current_count}/{papers_per_category}, "
                f"api_totalResults={total_results}"
            )
            print(f"    CSV saved after this page: {csv_path}")

            if total_results is not None and start >= total_results:
                print(f"Category {category}: reached API totalResults={total_results}; stopping")
                break

            if current_count < papers_per_category:
                time.sleep(sleep_seconds)

    df = load_existing_csv(csv_path, categories)
    print("=" * 80)
    print(f"Final saved rows: {len(df)}")
    print("Final category counts:")
    print(df["category"].value_counts().sort_index())
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a resumable ArXiv category dataset using the official API.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/arxiv_1000_per_category.csv"))
    parser.add_argument("--progress", type=Path, default=Path("data/raw/arxiv_1000_per_category_progress.json"))
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--papers-per-category", type=int, default=1000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=3.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dataset(
        csv_path=args.output,
        progress_path=args.progress,
        categories=args.categories,
        papers_per_category=args.papers_per_category,
        page_size=args.page_size,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
