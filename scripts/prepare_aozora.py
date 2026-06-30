from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


DEFAULT_MANIFEST = Path("manifests/aozora_works.json")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


@dataclass(frozen=True)
class Work:
    author_id: str
    author: str
    author_japanese: str
    title: str
    title_japanese: str
    card_url: str

    @property
    def work_id(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.title.lower()).strip("_")


class ZipLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.zip_links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = {name: value for name, value in attrs}
        href = attr_map.get("href")
        if href and href.endswith(".zip"):
            self.zip_links.append(href)


def load_manifest(path: Path) -> list[Work]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return [Work(**item) for item in payload["works"]]


def read_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mini-transformer-data-prep/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def find_text_zip_url(card_url: str) -> str:
    html = read_url(card_url).decode("utf-8")
    parser = ZipLinkParser()
    parser.feed(html)
    if not parser.zip_links:
        raise RuntimeError(f"No zip link found on Aozora card: {card_url}")
    return urllib.parse.urljoin(card_url, parser.zip_links[0])


def download_zip(work: Work, raw_dir: Path) -> Path:
    zip_url = find_text_zip_url(work.card_url)
    author_dir = raw_dir / work.author_id
    author_dir.mkdir(parents=True, exist_ok=True)
    zip_path = author_dir / Path(urllib.parse.urlparse(zip_url).path).name
    if not zip_path.exists():
        zip_path.write_bytes(read_url(zip_url))
    return zip_path


def extract_text(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        text_names = [name for name in archive.namelist() if name.endswith(".txt")]
        if not text_names:
            raise RuntimeError(f"No text file found in {zip_path}")
        raw_bytes = archive.read(text_names[0])
    return raw_bytes.decode("shift_jis", errors="replace")


def strip_header(lines: list[str]) -> list[str]:
    divider_indexes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("-------------------------------------------------------")
    ]
    if len(divider_indexes) >= 2:
        return lines[divider_indexes[1] + 1 :]

    for index, line in enumerate(lines[:20]):
        if not line.strip():
            return lines[index + 1 :]
    return lines


def strip_footer(lines: list[str]) -> list[str]:
    footer_prefixes = (
        "底本：",
        "親本：",
        "入力：",
        "校正：",
        "青空文庫作成ファイル：",
    )
    for index, line in enumerate(lines):
        if line.startswith(footer_prefixes):
            return lines[:index]
    return lines


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    lines = text.split("\n")
    lines = strip_header(lines)
    lines = strip_footer(lines)
    text = "\n".join(lines)

    # Remove Aozora ruby/annotation markup while keeping the visible text.
    text = text.replace("｜", "")
    text = re.sub(r"《[^》]*》", "", text)
    text = re.sub(r"［＃[^］]*］", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def prepare_work(work: Work, raw_dir: Path, processed_dir: Path) -> dict[str, object]:
    zip_path = download_zip(work, raw_dir)
    cleaned = clean_text(extract_text(zip_path))

    author_dir = processed_dir / work.author_id
    author_dir.mkdir(parents=True, exist_ok=True)
    output_path = author_dir / f"{work.work_id}.txt"
    output_path.write_text(cleaned, encoding="utf-8")

    return {
        "author_id": work.author_id,
        "author": work.author,
        "author_japanese": work.author_japanese,
        "title": work.title,
        "title_japanese": work.title_japanese,
        "characters": len(cleaned),
        "lines": cleaned.count("\n"),
        "raw_zip": str(zip_path),
        "processed_text": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and clean Aozora Bunko text for the mini Transformer project."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_PROCESSED_DIR / "summary.json",
        help="Where to write the character-count summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    works = load_manifest(args.manifest)
    summary = [
        prepare_work(work, args.raw_dir, args.processed_dir)
        for work in works
    ]

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps({"works": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("author_id,title,characters,processed_text")
    for item in summary:
        print(
            f"{item['author_id']},{item['title']},"
            f"{item['characters']},{item['processed_text']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
