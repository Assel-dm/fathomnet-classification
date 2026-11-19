#!/usr/bin/env python3
"""FGVC 2025 COCO -> labelled ROI classification

Reuse the old split_manifest.json, its retained annotation_ids and local images.
Original files are read only. No YOLO, torch or FathomNet API dependency.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

SPLITS = ("train", "val", "test")
SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "taxon"


def valid_image(path, size=None):
    try:
        with Image.open(path) as image:
            if size and image.size != size:
                return False
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def source_key(image):
    # Identical original URLs are the same source even when COCO ids differ.
    return image.get("coco_url") or image.get("flickr_url") or str(image["id"])


def group_key(image, mode):
    if mode == "image":
        return source_key(image)
    if mode == "day":
        date = str(image.get("date_captured", ""))[:10]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ValueError(f"No capture day for image {image['id']}; choose --group-by image.")
        return "day:" + date
    match = re.search(r"/framegrabs/([^/]+)/images/([^/]+)/", source_key(image))
    if not match:
        raise ValueError(f"No MBARI dive identifier for image {image['id']}; choose day or image.")
    return "dive:" + ":".join(match.groups())


def make_split(images, annotations, seed, mode):
    """Greedily balance ROI counts per class while keeping whole groups together."""
    groups, counts = defaultdict(list), defaultdict(Counter)
    url_group = {}
    for image in images.values():
        key = group_key(image, mode)
        old = url_group.setdefault(source_key(image), key)
        if old != key:
            raise ValueError("The same source URL has inconsistent grouping metadata.")
        groups[key].append(image["id"])
        counts[key].update(a["category_id"] for a in annotations[image["id"]])
    if len(groups) < 3:
        raise ValueError("At least three independent groups are needed for train/val/test.")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))  # randomized ties, larger groups first
    total = sum(counts.values(), Counter())
    fractions = dict(zip(SPLITS, (0.7, 0.15, 0.15)))
    sizes, assigned_groups = Counter(), Counter()
    class_counts = {s: Counter() for s in SPLITS}
    result = []
    for index, key in enumerate(keys):
        empty = [s for s in SPLITS if assigned_groups[s] == 0]
        candidates = empty if len(keys) - index == len(empty) else SPLITS

        def cost(s):
            n = len(groups[key])
            score = ((sizes[s] + n) ** 2 - sizes[s] ** 2) / (fractions[s] * len(images))
            for c, amount in counts[key].items():
                score += ((class_counts[s][c] + amount) ** 2 - class_counts[s][c] ** 2) / (
                    fractions[s] * total[c]
                )
            return score

        chosen = min(candidates, key=cost)
        sizes[chosen] += len(groups[key])
        assigned_groups[chosen] += 1
        class_counts[chosen].update(counts[key])
        result.extend({"image_id": i, "split": chosen, "group": key} for i in groups[key])
    return result


def plan(args):
    data = load(args.coco)
    images = {x["id"]: x for x in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("Duplicate COCO image ids.")
    categories = {x["id"]: x["name"] for x in data.get("categories", [])}
    if len(categories) != len(data.get("categories", [])) or not categories:
        raise ValueError("Expected unique labelled categories, not the blind official test set.")
    annotations = defaultdict(list)
    seen = set()
    for a in data["annotations"]:
        if a["id"] in seen or a["image_id"] not in images:
            raise ValueError("Duplicate annotation id or unknown image_id.")
        seen.add(a["id"])
        if a.get("category_id") not in categories:
            raise ValueError("Missing/unknown category_id: use labelled dataset_train.json.")
        x, y, w, h = a["bbox"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError(f"Invalid xywh bbox for annotation {a['id']}.")
        annotations[a["image_id"]].append(a)
    if not annotations:
        raise ValueError("No labelled annotations.")
    if args.split_manifest:
        entries = load(args.split_manifest)
        if not isinstance(entries, list):
            raise ValueError("split_manifest must be a list of image_id/split records.")
    else:
        images = {i: v for i, v in images.items() if annotations[i]}
        entries = make_split(images, annotations, args.seed, args.group_by)
    ids, urls, groups = {}, {}, {}
    selected = {}
    for entry in entries:
        i, split = entry["image_id"], entry["split"]
        if i in ids or i not in images or split not in SPLITS:
            raise ValueError(f"Duplicate/unknown image or split: {entry}")
        ids[i] = split
        for mapping, key in ((urls, source_key(images[i])), (groups, str(entry.get("group", source_key(images[i]))))):
            if mapping.setdefault(key, split) != split:
                raise ValueError(f"Data leakage: source/group {key} occurs in different splits.")
        ann = annotations[i]
        if "annotation_ids" in entry:
            retained = entry["annotation_ids"]
            wanted = set(retained)
            if len(wanted) != len(retained) or not wanted <= {a["id"] for a in ann}:
                raise ValueError(f"Invalid retained annotation_ids for image {i}.")
            ann = [a for a in ann if a["id"] in wanted]
        selected[i] = {"image": images[i], "entry": entry, "annotations": ann}
    classes = [{"class_id": index, "source_category_id": c, "name": categories[c],
                "directory": f"{index:03d}_{slug(categories[c])}"}
               for index, c in enumerate(sorted(categories))]
    counts = {s: Counter(a["category_id"] for i, v in selected.items() if ids[i] == s for a in v["annotations"])
              for s in SPLITS}
    report = {"source_images": len(data["images"]), "source_annotations": len(data["annotations"]),
              "images_without_annotations": sum(not annotations[i["id"]] for i in data["images"]),
              "categories": len(classes), "selected_images": len(selected),
              "selected_annotations": sum(len(v["annotations"]) for v in selected.values()),
              "split_origin": "existing manifest" if args.split_manifest else "new grouped approximate 70/15/15",
              "group_by": "preserved manifest groups" if args.split_manifest else args.group_by,
              "splits": {s: {"images": sum(v == s for v in ids.values()), "rois": sum(counts[s].values()),
                             "missing_classes": [categories[c] for c in categories if not counts[s][c]],
                             "class_counts": {str(c): counts[s][c] for c in sorted(categories)}} for s in SPLITS},
              "warning": "Image/group separation checked by ids, source URLs and supplied groups; adjacent video frames require dive/day grouping."}
    if any(not report["splits"][s]["rois"] for s in SPLITS):
        raise ValueError("Each of train, val and local test must contain labelled ROIs.")
    return selected, classes, entries, report


def local_candidates(record, roots):
    image, entry = record["image"], record["entry"]
    for field in ("source", "image"):
        if entry.get(field):
            yield Path(entry[field]).expanduser()
    filename = Path(str(image.get("file_name", "")).replace("\\", "/"))
    if filename.is_absolute() or ".." in filename.parts:
        raise ValueError("Unsafe COCO file_name.")
    url = urlparse(source_key(image))
    for root in roots:
        for folder in (root, root / "images"):
            if filename.name:
                yield folder / filename
                yield folder / filename.name
            yield folder / unquote(Path(url.path).name)
            for suffix in SUFFIXES:
                yield folder / f"{image['id']}{suffix}"


def get_source(record, roots, cache, download):
    image = record["image"]
    size = (image["width"], image["height"])
    for path in local_candidates(record, roots):
        if path.is_file() and valid_image(path, size):
            return path, "local"
    dest = cache / f"{image['id']}.img"
    if dest.exists() and valid_image(dest, size):
        return dest, "cached"
    if not download:
        raise FileNotFoundError(f"No original image found for image_id={image['id']}; supply --images-dir or --download-missing.")
    url = image.get("coco_url") or image.get("flickr_url")
    if not url or urlparse(url).scheme not in ("https", "http"):
        raise ValueError("No HTTP(S) source URL.")
    temp = dest.with_suffix(".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FGVC2025-ROI-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if not valid_image(temp, size):
                raise ValueError("Downloaded image is invalid or dimensions differ from COCO; refusing automatic resize.")
            temp.replace(dest)
            return dest, "downloaded"
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def process(record, class_map, out, args):
    image, entry = record["image"], record["entry"]
    rows, failures = [], []
    original = None
    source_error = None
    try:
        for a in sorted(record["annotations"], key=lambda v: v["id"]):
            c = class_map[a["category_id"]]
            relative = Path(entry["split"]) / c["directory"] / f"{image['id']}_{a['id']}.png"
            target = out / relative
            try:
                x, y, w, h = a["bbox"]
                W, H = image["width"], image["height"]
                box = (max(0, math.floor(x)), max(0, math.floor(y)), min(W, math.ceil(x+w)), min(H, math.ceil(y+h)))
                if box[0] >= box[2] or box[1] >= box[3]:
                    raise ValueError("BBox has no intersection with the original image.")
                size = (box[2] - box[0], box[3] - box[1])
                status = "existing"
                if not (target.exists() and valid_image(target, size)):
                    if original is None:
                        if source_error is not None:
                            raise source_error
                        try:
                            path, _ = get_source(record, args.images_dir, out / "_source_cache", args.download_missing)
                            original = Image.open(path)
                            original.load()
                        except Exception as error:
                            source_error = error
                            raise
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix(".part")
                    original.crop(box).convert("RGB").save(temp, format="PNG")
                    temp.replace(target)
                    status = "created"
                rows.append({"path": relative.as_posix(), "class_id": c["class_id"], "name": c["name"],
                             "category_id": a["category_id"], "image_id": image["id"], "annotation_id": a["id"],
                             "split": entry["split"], "width": size[0], "height": size[1],
                             "bbox_xywh": json.dumps(a["bbox"]), "crop_xyxy": json.dumps(box),
                             "clipped": x < 0 or y < 0 or x+w > W or y+h > H,
                             "status": status})
            except Exception as error:
                failures.append({"image_id": image["id"], "annotation_id": a["id"], "error": str(error)})
    finally:
        if original is not None:
            original.close()
    return rows, failures


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", type=Path, required=True)
    split = p.add_mutually_exclusive_group(required=True)
    split.add_argument("--split-manifest", type=Path)
    split.add_argument("--new-split", action="store_true")
    p.add_argument("--group-by", choices=("image", "day", "dive"), default="dive")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--images-dir", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path)
    p.add_argument("--inspect", action="store_true", help="Read-only audit; no downloads or writes.")
    p.add_argument("--download-missing", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit-images", type=int, help="Smoke run; planned splits remain fixed; not a complete dataset.")
    args = p.parse_args(argv)
    if not 1 <= args.workers <= 16 or (args.limit_images is not None and args.limit_images < 1):
        p.error("workers must be 1..16 and limit-images positive")
    selected, classes, entries, report = plan(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.inspect:
        return 0
    if args.output is None:
        p.error("--output is required except with --inspect")
    out = args.output.expanduser().resolve()
    # A separate destination protects existing source datasets.
    for root in [*args.images_dir, args.coco.parent, *([args.split_manifest.parent] if args.split_manifest else [])]:
        if out == root.expanduser().resolve() or out in root.expanduser().resolve().parents:
            p.error("Output must be separate from and cannot contain a source directory.")
    fingerprint = {"schema": 1, "coco_sha256": digest(args.coco),
                   "manifest_sha256": digest(args.split_manifest) if args.split_manifest else None,
                   "seed": args.seed if args.new_split else None,
                   "group_by": args.group_by if args.new_split else None,
                   "crop_policy": "xywh; floor top-left; ceil bottom-right; clip; RGB PNG; no EXIF transpose"}
    marker = out / "metadata" / "preparation.json"
    if out.exists() and not marker.exists() and any(out.iterdir()):
        p.error("Refusing nonempty output without preparation identity.")
    if marker.exists() and load(marker) != fingerprint:
        p.error("Output belongs to another COCO/split/policy; choose a new output.")
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_json(marker, fingerprint)
    write_json(out / "metadata" / "classes.json", classes)
    write_json(out / "metadata" / "split_manifest.json", entries)
    # Keep all class directories, even for missing validation/test classes.
    for s in SPLITS:
        for c in classes:
            (out / s / c["directory"]).mkdir(parents=True, exist_ok=True)
    (out / "_source_cache").mkdir(exist_ok=True)
    (out / "COMPLETE.json").unlink(missing_ok=True)
    records = [selected[i] for i in sorted(selected)]
    if args.limit_images:
        records = records[:args.limit_images]
    rows, failures = [], []
    mapping = {c["source_category_id"]: c for c in classes}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(process, r, mapping, out, args) for r in records]
        for n, future in enumerate(as_completed(jobs), 1):
            ok, bad = future.result()
            rows.extend(ok)
            failures.extend(bad)
            if n % 100 == 0 or n == len(jobs):
                print(f"Images {n}/{len(jobs)}, successful ROIs {len(rows)}, failures {len(failures)}", flush=True)
    rows.sort(key=lambda v: (v["image_id"], v["annotation_id"]))
    fields = ["path", "class_id", "name", "category_id", "image_id", "annotation_id", "split", "width", "height", "bbox_xywh", "crop_xyxy", "clipped", "status"]
    with (out / "metadata" / "crops.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "metadata" / "failures.json", failures)
    report.update({"processed_images": len(records), "successful_rois": len(rows), "failed_rois": len(failures),
                   "clipped_rois": sum(r["clipped"] for r in rows),
                   "complete": len(rows) == report["selected_annotations"] and not failures,
                   "rgb_png": True})
    write_json(out / "metadata" / "audit.json", report)
    if report["complete"]:
        write_json(out / "COMPLETE.json", {"successful_rois": len(rows), "coco_sha256": fingerprint["coco_sha256"]})
    print(f"Output: {out}; complete={report['complete']}; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        raise SystemExit(2)
