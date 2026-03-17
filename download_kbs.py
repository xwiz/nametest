from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    url: str
    zip_name: str
    extract_dir: str


MEDQUAD = DownloadSpec(
    name="medquad",
    url="https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip",
    zip_name="medquad_master.zip",
    extract_dir="medquad",
)

FDC_FOUNDATION = DownloadSpec(
    name="fdc_foundation_csv",
    url="https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2025-12-18.zip",
    zip_name="FoodData_Central_foundation_food_csv_2025-12-18.zip",
    extract_dir="fdc_foundation",
)

FDC_FULL = DownloadSpec(
    name="fdc_full_csv",
    url="https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_csv_2025-12-18.zip",
    zip_name="FoodData_Central_csv_2025-12-18.zip",
    extract_dir="fdc_full",
)


def _download(url: str, dest: Path, *, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"[skip] {dest} already exists")
        return

    print(f"[down]  {url}")
    print(f"[to]   {dest}")

    t0 = time.time()
    with urllib.request.urlopen(url) as resp:
        total = resp.headers.get("Content-Length")
        total_n = int(total) if total and total.isdigit() else None

        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            read = 0
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                read += len(chunk)
                if total_n:
                    pct = int(read / total_n * 100)
                    sys.stdout.write(f"\r      {pct:>3}% ({read/1e6:.1f}MB/{total_n/1e6:.1f}MB)")
                    sys.stdout.flush()
            if total_n:
                sys.stdout.write("\n")

    os.replace(tmp, dest)
    dt = time.time() - t0
    print(f"[ok]   downloaded in {dt:.1f}s\n")


def _extract(zip_path: Path, out_dir: Path, *, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"[skip] {out_dir} already extracted")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ext]  {zip_path} -> {out_dir}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    print("[ok]   extracted\n")


def _find_single_subdir(root: Path) -> Path:
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="Download/extract destination folder")
    ap.add_argument("--no-medquad", action="store_true")
    ap.add_argument(
        "--fdc",
        choices=["none", "foundation", "full"],
        default="foundation",
        help="Which FoodData Central dataset to download (default: foundation)",
    )
    ap.add_argument("--force", action="store_true", help="Re-download/re-extract even if present")
    ap.add_argument("--no-extract", action="store_true", help="Download zips only")

    ap.add_argument(
        "--build-kb-out",
        default=None,
        help="If set, build a fragment KB .txt at this path after downloads",
    )
    ap.add_argument("--max-medquad-pairs", type=int, default=None)
    ap.add_argument("--max-fdc-rows", type=int, default=None)
    ap.add_argument(
        "--nutrients",
        default="protein,energy",
        help="Comma-separated nutrient names to include when building KB",
    )

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    specs: list[DownloadSpec] = []

    if not args.no_medquad:
        specs.append(MEDQUAD)

    if args.fdc == "foundation":
        specs.append(FDC_FOUNDATION)
    elif args.fdc == "full":
        specs.append(FDC_FULL)

    downloaded: dict[str, Path] = {}
    extracted: dict[str, Path] = {}

    for spec in specs:
        zip_path = data_dir / spec.zip_name
        _download(spec.url, zip_path, force=args.force)
        downloaded[spec.name] = zip_path

        if not args.no_extract:
            out_dir = data_dir / spec.extract_dir
            _extract(zip_path, out_dir, force=args.force)
            extracted[spec.name] = out_dir

    if args.build_kb_out:
        try:
            from build_health_kb import build_kb_lines
        except Exception as e:
            print(f"[err] Could not import build_health_kb.py: {e}")
            raise

        nutrients = {x.strip().lower() for x in (args.nutrients or "").split(",") if x.strip()}
        medquad_root = None
        if "medquad" in extracted:
            # GitHub zip extracts into a single nested folder.
            medquad_root = str(_find_single_subdir(extracted["medquad"]))

        fdc_root = None
        if args.fdc != "none":
            if args.fdc == "foundation" and "fdc_foundation_csv" in extracted:
                fdc_root = str(_find_single_subdir(extracted["fdc_foundation_csv"]))
            if args.fdc == "full" and "fdc_full_csv" in extracted:
                fdc_root = str(_find_single_subdir(extracted["fdc_full_csv"]))

        lines = build_kb_lines(
            medquad_dir=medquad_root,
            fdc=fdc_root,
            max_medquad_pairs=args.max_medquad_pairs,
            max_fdc_rows=args.max_fdc_rows,
            nutrient_names=nutrients or None,
        )

        out_path = Path(args.build_kb_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            for ln in lines:
                fh.write(ln)
                fh.write("\n")

        print(f"[ok]   built KB: {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
