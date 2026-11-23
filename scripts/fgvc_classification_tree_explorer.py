"""Generate an offline HTML explorer for classification of FGVC crops."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .build_fgvc_classification_taxonomy import index_records, read_json, write_text
else:
    from build_fgvc_classification_taxonomy import index_records, read_json, write_text

SPLITS = ("train", "val", "test")

def crop_support(manifest, classes):
    """Count prepared crop rows by class and split and deduplicate source images."""
    reader = csv.DictReader(io.StringIO(manifest, newline=""))
    required = {"class_id", "category_id", "image_id", "annotation_id", "split"}
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"Missing crops.csv columns: {', '.join(sorted(missing))}")

    counts = defaultdict(Counter)
    image_sets = defaultdict(set)
    annotation_ids = set()
    image_splits = {}
    for row in reader:
        location = f"crops.csv line {reader.line_num}"
        try:
            class_id = int(row["class_id"])
            category_id = int(row["category_id"])
            image_id = int(row["image_id"])
            annotation_id = int(row["annotation_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{location}: invalid class, category, image or annotation ID") from error
        split = row["split"]
        entry = classes.get(class_id)
        if entry is None or category_id != entry.get("source_category_id"):
            raise ValueError(f"{location}: class ID and source category do not match the taxonomy")
        if split not in SPLITS:
            raise ValueError(f"{location}: unknown split {split!r}")
        if annotation_id in annotation_ids:
            raise ValueError(f"{location}: duplicate annotation ID {annotation_id}")
        if image_splits.setdefault(image_id, split) != split:
            raise ValueError(f"{location}: image {image_id} occurs in more than one split")
        annotation_ids.add(annotation_id)
        counts[class_id][split] += 1
        image_sets[class_id].add(image_id)
    return counts, image_sets


def prepare_data(taxonomy, crop_manifest=None, title="FathomNet FGVC classification taxonomy"):
    """Attach class metadata and optional counts to the training tree."""
    if taxonomy.get("schema_version") != 1 or taxonomy.get("training_tree", {}).get("version") != 1:
        raise ValueError("Expected the taxonomy produced by build_fgvc_classification_taxonomy.py")
    metadata = taxonomy["training_tree"]
    nodes = index_records(taxonomy.get("nodes"), "id", str)
    classes = index_records(taxonomy.get("classes"), "class_id", int)
    roots = [node_id for node_id, node in nodes.items() if node.get("parent") is None]
    if len(roots) != 1 or nodes[roots[0]].get("depth") != 0:
        raise ValueError("Expected a single root at depth zero")
    children = defaultdict(list)
    for node_id, node in nodes.items():
        parent_id = node.get("parent")
        if parent_id is not None:
            if parent_id not in nodes or node["depth"] != nodes[parent_id]["depth"] + 1:
                raise ValueError(f"Bad parent or depth for {node_id!r}")
            children[parent_id].append(node_id)

    counts, image_sets = crop_support(crop_manifest, classes) if crop_manifest is not None else ({}, {})
    records = {}
    for class_id, entry in classes.items():
        output_id = entry["node_id"]
        source_taxon = nodes.get(entry["original_node_id"])
        if (
            source_taxon is None or source_taxon["kind"] != "taxon" or output_id not in nodes
            or output_id in records or children[output_id]
            or nodes[output_id]["depth"] != metadata["output_depth"]
        ):
            raise ValueError(f"Invalid output leaf or source taxon for class {class_id}")
        records[output_id] = {
            "source_category_id": entry.get("source_category_id"),
            "class_id": class_id,
            "original_name": entry["name"],
            "retained_name": source_taxon["name"],
            "original_rank": source_taxon.get("rank"),
            "original_node_id": source_taxon["id"],
            "renamed": entry["name"] != source_taxon["name"],
        }
    if set(records) != {node_id for node_id in nodes if not children[node_id]}:
        raise ValueError("Training leaves and output classes differ")

    def build(node_id):
        node = nodes[node_id]
        record = records.get(node_id)
        images = set(image_sets.get(record["class_id"], ())) if record else set()
        total = Counter(counts.get(record["class_id"], {})) if record else Counter()
        child_data = []
        for child_id in sorted(children[node_id], key=lambda key: (nodes[key]["name"].casefold(), key)):
            child, child_images = build(child_id)
            images.update(child_images)
            total.update(child["split_counts"] or {})
            child_data.append(child)
        original = nodes.get(node.get("original_node_id") or node.get("next_taxon_id"))
        return {
            "id": node_id,
            "name": node["name"],
            "rank": node.get("rank"),
            "training_level": node["training_level"],
            "kind": node["kind"],
            "aphia_id": node.get("aphia_id"),
            "source_taxon": original["name"] if original else None,
            "originals": [record] if record else [],
            "split_counts": {split: total[split] for split in SPLITS} if crop_manifest is not None else None,
            "subtree_crops": sum(total.values()) if crop_manifest is not None else None,
            "subtree_images": len(images) if crop_manifest is not None else None,
            "children": child_data,
        }, images

    tree, _ = build(roots[0])
    return {
        "title": title,
        "summary": {
            "nodes": len(nodes),
            "original_classes": len(classes),
            "synthetic_nodes": sum(node["kind"] != "taxon" for node in nodes.values()),
            "moved_classes": len(metadata["annotation_branch_class_ids"]),
            "output_depth": metadata["output_depth"],
            "crops": tree["subtree_crops"],
            "images": tree["subtree_images"],
        },
        "tree": tree,
    }


def embedded_json(value):
    """Escape characters that could end the inert HTML data element."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def generate(taxonomy_path, output_path, crops_path=None, title="FathomNet FGVC classification taxonomy"):
    """Embed taxonomic data and D3 in one portable HTML file."""
    assets = Path(__file__).with_name("fgvc_classification_tree_assets")
    template_path = assets / "template.html"
    d3_path = assets / "d3.v7.9.0.min.js"
    inputs = [taxonomy_path, template_path, d3_path, assets / "D3-LICENSE.txt"]
    if crops_path is not None:
        inputs.append(crops_path)
    if Path(output_path).resolve() in {Path(path).resolve() for path in inputs}:
        raise ValueError("Output must differ from all input and asset paths")
    taxonomy, taxonomy_hash = read_json(taxonomy_path)
    crop_bytes = Path(crops_path).read_bytes() if crops_path is not None else None
    manifest = crop_bytes.decode("utf-8-sig") if crop_bytes is not None else None
    data = prepare_data(taxonomy, manifest, title)
    original = taxonomy["training_tree"].get("source", {})
    data["provenance"] = {
        "taxonomy_name": Path(taxonomy_path).name,
        "taxonomy_sha256": taxonomy_hash,
        "original_name": original.get("file"),
        "original_sha256": original.get("sha256"),
        "crops_name": Path(crops_path).name if crops_path is not None else None,
        "crops_sha256": hashlib.sha256(crop_bytes).hexdigest() if crop_bytes is not None else None,
    }
    template = template_path.read_text(encoding="utf-8")
    library = d3_path.read_text(encoding="utf-8")
    if "</script" in library.lower():
        raise ValueError("Unexpected closing script tag in D3 source")
    for token in ("__D3_SOURCE__", "__TREE_DATA__"):
        if template.count(token) != 1:
            raise ValueError(f"Template must contain one {token} placeholder")
    # Keep the third-party license in the standalone output.
    license_text = (assets / "D3-LICENSE.txt").read_text(encoding="utf-8")
    library = library + "\n/*!\n" + license_text + "\n*/\n"
    html = template.replace("__D3_SOURCE__", library).replace("__TREE_DATA__", embedded_json(data))
    write_text(output_path, html)
    return data["summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True, help="Training taxonomy JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML file")
    parser.add_argument("--crops", type=Path, help="Optional metadata/crops.csv for counts per split")
    parser.add_argument("--title", default="FathomNet FGVC classification taxonomy")
    args = parser.parse_args(argv)
    try:
        summary = generate(args.taxonomy, args.output, args.crops, args.title)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Written to {args.output} ({summary['nodes']} nodes, {summary['original_classes']} output classes)")


if __name__ == "__main__":
    main()
