"""Load prepared FGVC 2025 crops and their taxonomies"""

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

SPLIT_KEYS = {"train": "train", "val": "validation", "test": "test"}


def load_fathomnet(root, taxonomy_adapted=None, taxonomy_original=None):
    """Return class metadata and (image_path, class_id) samples per split."""
    root = Path(root).expanduser().resolve()
    metadata = root / "metadata"

    with (metadata / "classes.json").open(encoding="utf-8-sig") as handle:
        classes = sorted(json.load(handle), key=lambda row: row["class_id"])

    if not classes or [row["class_id"] for row in classes] != list(range(len(classes))):
        raise ValueError("classes.json must contain contiguous class IDs starting at 0.")

    class_names = [row["name"] for row in classes]
    if len(set(class_names)) != len(class_names):
        raise ValueError("classes.json contains duplicate class names.")

    samples = {key: [] for key in SPLIT_KEYS.values()}
    seen_annotations = set()
    source_splits = {}
    manifest = metadata / "crops.csv"

    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "path", "split", "class_id", "category_id", "image_id", "annotation_id"
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{manifest}: missing columns {sorted(missing)}.")

        for row in reader:
            location = f"{manifest}:{reader.line_num}"
            split = row["split"]
            if split not in SPLIT_KEYS:
                raise ValueError(f"{location}: unknown split {split!r}.")

            class_id = int(row["class_id"])
            if not 0 <= class_id < len(classes):
                raise ValueError(f"{location}: unknown class_id {class_id}.")

            label = classes[class_id]
            if int(row["category_id"]) != label["source_category_id"]:
                raise ValueError(f"{location}: class_id and COCO category_id disagree.")

            image_id = int(row["image_id"])
            annotation_id = int(row["annotation_id"])
            if source_splits.setdefault(image_id, split) != split:
                raise ValueError(f"{location}: source image {image_id} crosses splits.")
            if annotation_id in seen_annotations:
                raise ValueError(f"{location}: duplicate annotation {annotation_id}.")
            seen_annotations.add(annotation_id)

            relative = PurePosixPath(row["path"])
            expected_parts = (
                split,
                label["directory"],
                f"{image_id}_{annotation_id}.png",
            )
            if relative.parts != expected_parts or ".." in relative.parts:
                raise ValueError(f"{location}: unexpected crop path {row['path']!r}.")

            path = root.joinpath(*relative.parts)
            if not path.is_file():
                raise FileNotFoundError(path)

            samples[SPLIT_KEYS[split]].append((path, class_id))

    for split, entries in samples.items():
        if not entries:
            raise ValueError(f"{manifest}: no crops in {split}.")

    data = {
        "name": "fathomnet",
        "root": root,
        "samples": samples,
        "class_names": class_names,
        "species": class_names,  # Legacy key used by the classifier.
        "taxonomy": pd.DataFrame(
            {
                "species": class_names,
                "taxon": class_names,
                "source_category_id": [
                    row["source_category_id"] for row in classes
                ],
            }
        ),
    }

    if taxonomy_adapted is not None or taxonomy_original is not None:
        if taxonomy_adapted is None or taxonomy_original is None:
            raise ValueError("Supply both FGVC taxonomy files.")
        data.update(
            build_fathomnet_hierarchies(
                classes, taxonomy_adapted, taxonomy_original
            )
        )

    return data


def _read_fgvc_taxonomy(path, crop_classes):
    """Return the tree and ordered root-to-class paths."""
    path = Path(path).expanduser()
    raw = path.read_bytes()
    tree = json.loads(raw.decode("utf-8-sig"))
    entries = sorted(tree["classes"], key=lambda entry: entry["class_id"])

    keys = ("class_id", "name", "source_category_id")
    if [tuple(entry[key] for key in keys) for entry in entries] != [
        tuple(entry[key] for key in keys) for entry in crop_classes
    ]:
        raise ValueError(
            f"{path}: class IDs, names or source category IDs "
            "differ from classes.json."
        )

    nodes = {node["id"]: node for node in tree["nodes"]}
    paths = []
    for entry in entries:
        lineage, seen = [], set()
        node_id = entry["node_id"]

        while node_id is not None:
            if node_id not in nodes or node_id in seen:
                raise ValueError(
                    f"{path}: invalid parent chain for {entry['name']}."
                )
            seen.add(node_id)
            node = nodes[node_id]
            lineage.append(node)
            node_id = node["parent"]

        if not lineage:
            raise ValueError(f"{path}: empty lineage for {entry['name']}.")
        paths.append(lineage[::-1])

    if len({lineage[0]["id"] for lineage in paths}) != 1:
        raise ValueError(f"{path}: class lineages do not share a root.")

    return tree, entries, paths, hashlib.sha256(raw).hexdigest()


def _edge_distances(paths):
    """Count edges between every pair of classes in crop class order."""
    ids = [tuple(node["id"] for node in lineage) for lineage in paths]
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int32)

    for i, left in enumerate(ids):
        for j, right in enumerate(ids[:i]):
            shared = 0
            for a, b in zip(left, right):
                if a != b:
                    break
                shared += 1
            matrix[i, j] = matrix[j, i] = len(left) + len(right) - 2 * shared

    return matrix


def _one_hot_groups(groups):
    """Build one row per class and one column per ancestor."""
    codes, ancestors = pd.factorize(
        np.asarray(groups, dtype=object), sort=False
    )
    return np.eye(len(ancestors), dtype=np.float32)[codes]


def build_fathomnet_hierarchies(
    crop_classes, taxonomy_adapted, taxonomy_original
):
    """Expose the adapted training hierarchy and both evaluation distances."""
    adapted, adapted_classes, adapted_paths, adapted_sha = _read_fgvc_taxonomy(
        taxonomy_adapted, crop_classes
    )
    original, original_classes, original_paths, original_sha = _read_fgvc_taxonomy(
        taxonomy_original, crop_classes
    )

    if "training_tree" in original or "training_tree" not in adapted:
        raise ValueError(
            "FGVC requires an original taxonomy and an adapted training taxonomy."
        )

    levels = adapted["training_tree"]["levels"]
    if not levels or len(set(levels)) != len(levels) or levels[-1] != "species":
        raise ValueError("Expected unique training levels ending in species.")

    leaf_ids = [entry["node_id"] for entry in adapted_classes]
    parent_ids = {node["parent"] for node in adapted["nodes"]}
    if len(set(leaf_ids)) != len(leaf_ids) or any(
        node in parent_ids for node in leaf_ids
    ):
        raise ValueError("Each FGVC output label must have its own terminal leaf.")

    for new, old, new_path, old_path in zip(
        adapted_classes, original_classes, adapted_paths, original_paths
    ):
        if [node["training_level"] for node in new_path] != levels:
            raise ValueError(f"Adapted levels are misaligned for {new['name']}.")

        adapted_real_path = [
            node["id"] for node in new_path if node["kind"] == "taxon"
        ]
        if (
            new["original_node_id"] != old["node_id"]
            or adapted_real_path != [node["id"] for node in old_path]
        ):
            raise ValueError(f"Original lineage differs for {new['name']}.")

    # The losses already treat the root as implicit and species as the output level.
    ranks = [rank for rank in levels if rank != "root"]
    mappings = {
        rank: _one_hot_groups([
            lineage[levels.index(rank)]["id"] for lineage in adapted_paths
        ])
        for rank in ranks[:-1]
    }

    adapted_distances = _edge_distances(adapted_paths)
    original_distances = _edge_distances(original_paths)

    evaluation_mappings, rank_known = {}, {}
    for rank in ("kingdom", "phylum", "class", "order", "family", "genus"):
        ancestors = [
            next(
                (node["id"] for node in reversed(lineage) if node["rank"] == rank),
                "",
            )
            for lineage in original_paths
        ]
        known = np.asarray([bool(node_id) for node_id in ancestors])

        if known.any():
            # Keep predictions with an unknown rank in their own group.
            evaluation_mappings[rank] = _one_hot_groups(ancestors)
            rank_known[rank] = known

    return {
        "ranks": ranks,
        "mappings": mappings,
        "normalized_distances": (
            adapted_distances.astype(np.float32)
            / max(1, int(adapted_distances.max()))
        ),
        "distance_matrix": original_distances,
        "distance_matrix_adapted": adapted_distances,
        "evaluation_mappings": evaluation_mappings,
        "evaluation_rank_known": rank_known,
        "fathomnet_original_taxonomy": original,
        "fathomnet_taxonomy_fingerprints": {
            "adapted_sha256": adapted_sha,
            "original_sha256": original_sha,
        },
    }
