from __future__ import annotations

import argparse
import csv
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator


_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s


def _split_sentences(s: str) -> list[str]:
    s = _clean_text(s)
    if not s:
        return []

    # Common prefixes in MedQuAD answers.
    for prefix in ("Summary :", "Summary:", "SUMMARY :", "SUMMARY:"):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
            break

    parts = [p.strip() for p in _SENT_SPLIT_RE.split(s) if p.strip()]
    return parts if parts else [s]


def iter_medquad_fragments(
    medquad_dir: str,
    *,
    max_pairs: int | None = None,
    split_sentences: bool = True,
) -> Iterator[str]:
    """Yield short natural-language fragments from a MedQuAD directory."""
    n_pairs = 0

    for root, _dirs, files in os.walk(medquad_dir):
        for fn in files:
            if not fn.lower().endswith(".xml"):
                continue
            path = os.path.join(root, fn)
            try:
                tree = ET.parse(path)
            except ET.ParseError:
                continue

            doc = tree.getroot()
            focus = _clean_text(doc.findtext("Focus") or "")

            for qapair in doc.findall(".//QAPair"):
                if max_pairs is not None and n_pairs >= max_pairs:
                    return

                q = _clean_text(qapair.findtext("Question") or "")
                a = _clean_text(qapair.findtext("Answer") or "")

                # Some subsets have answers removed.
                if not a:
                    continue

                if split_sentences:
                    answer_parts = _split_sentences(a)
                else:
                    answer_parts = [a]

                for part in answer_parts:
                    frag = part
                    if focus:
                        frag = f"{focus}: {frag}"
                    if q and len(q) <= 140:
                        frag = f"{frag}"

                    frag = _clean_text(frag)
                    if frag:
                        yield frag

                n_pairs += 1


def _read_csv_from_zip_or_dir(
    zip_or_dir: str,
    relative_path: str,
) -> Iterable[dict[str, str]]:
    if os.path.isdir(zip_or_dir):
        p = os.path.join(zip_or_dir, relative_path)
        if not os.path.exists(p):
            base = os.path.basename(relative_path)
            found = None
            for root, _dirs, files in os.walk(zip_or_dir):
                for fn in files:
                    if fn == base:
                        found = os.path.join(root, fn)
                        break
                if found:
                    break
            if found is None:
                raise FileNotFoundError(f"Missing {relative_path!r} under {zip_or_dir!r}")
            p = found
        with open(p, "r", encoding="utf-8", newline="") as fh:
            yield from csv.DictReader(fh)
        return

    with zipfile.ZipFile(zip_or_dir, "r") as z:
        # Try exact match first, else search by basename.
        name = None
        if relative_path in z.namelist():
            name = relative_path
        else:
            base = os.path.basename(relative_path)
            for n in z.namelist():
                if os.path.basename(n) == base:
                    name = n
                    break
        if name is None:
            raise FileNotFoundError(f"Missing {relative_path!r} in {zip_or_dir!r}")

        with z.open(name, "r") as raw:
            text = (ln.decode("utf-8", errors="replace") for ln in raw)
            yield from csv.DictReader(text)


def iter_fooddata_central_fragments(
    fdc_zip_or_dir: str,
    *,
    max_rows: int | None = None,
    nutrient_names: set[str] | None = None,
) -> Iterator[str]:
    """Yield nutrition fact fragments from a FoodData Central CSV export."""

    foods: dict[str, str] = {}
    nutrients: dict[str, tuple[str, str]] = {}

    for r in _read_csv_from_zip_or_dir(fdc_zip_or_dir, "food.csv"):
        fid = (r.get("fdc_id") or r.get("id") or "").strip()
        desc = (r.get("description") or r.get("food_description") or "").strip()
        if fid and desc:
            foods[fid] = _clean_text(desc.lower())

    for r in _read_csv_from_zip_or_dir(fdc_zip_or_dir, "nutrient.csv"):
        nid = (r.get("id") or r.get("nutrient_id") or "").strip()
        name = _clean_text((r.get("name") or "").lower())
        unit = _clean_text((r.get("unit_name") or r.get("unit") or "").lower())
        if nid and name:
            nutrients[nid] = (name, unit)

    n = 0
    for r in _read_csv_from_zip_or_dir(fdc_zip_or_dir, "food_nutrient.csv"):
        if max_rows is not None and n >= max_rows:
            return

        fid = (r.get("fdc_id") or r.get("food_id") or "").strip()
        nid = (r.get("nutrient_id") or r.get("id") or "").strip()
        amt = (r.get("amount") or r.get("value") or "").strip()

        if not fid or not nid or not amt:
            continue

        food = foods.get(fid)
        nutrient = nutrients.get(nid)
        if not food or not nutrient:
            continue

        n_name, n_unit = nutrient
        if nutrient_names is not None and n_name not in nutrient_names:
            continue

        frag = f"{food} has {amt} {n_unit} {n_name}."
        frag = _clean_text(frag)
        if frag:
            yield frag
            n += 1


def build_kb_lines(
    *,
    medquad_dir: str | None,
    fdc: str | None,
    max_medquad_pairs: int | None,
    max_fdc_rows: int | None,
    nutrient_names: set[str] | None,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    if medquad_dir:
        for ln in iter_medquad_fragments(medquad_dir, max_pairs=max_medquad_pairs, split_sentences=True):
            if ln not in seen:
                seen.add(ln)
                out.append(ln)

    if fdc:
        for ln in iter_fooddata_central_fragments(fdc, max_rows=max_fdc_rows, nutrient_names=nutrient_names):
            if ln not in seen:
                seen.add(ln)
                out.append(ln)

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--medquad-dir", default=None, help="Path to MedQuAD root directory (XML files)")
    p.add_argument(
        "--fdc",
        default=None,
        help="Path to USDA FoodData Central CSV ZIP or extracted directory (food.csv, nutrient.csv, food_nutrient.csv)",
    )
    p.add_argument("--out", required=True, help="Output .txt file (one fragment per line)")
    p.add_argument("--max-medquad-pairs", type=int, default=None)
    p.add_argument("--max-fdc-rows", type=int, default=None)
    p.add_argument(
        "--nutrients",
        default=None,
        help="Comma-separated nutrient names to include (e.g. protein,energy,total lipid (fat))",
    )

    args = p.parse_args()

    nutrients = None
    if args.nutrients:
        nutrients = {_clean_text(x.lower()) for x in args.nutrients.split(",") if _clean_text(x)}

    lines = build_kb_lines(
        medquad_dir=args.medquad_dir,
        fdc=args.fdc,
        max_medquad_pairs=args.max_medquad_pairs,
        max_fdc_rows=args.max_fdc_rows,
        nutrient_names=nutrients,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for ln in lines:
            fh.write(ln)
            fh.write("\n")

    print(f"Wrote {len(lines)} lines to {args.out}")


if __name__ == "__main__":
    main()
