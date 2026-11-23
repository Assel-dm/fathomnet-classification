"""Build the rank-aligned taxonomy for classification of FGVC crops."""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# The two ranks between phylum and class occupy the same slot in FGVC 2025.
# The source taxon's actual rank is retained separately from its training level.
RANK_LEVELS = (
    ("root", ("superdomain", "root")),
    ("kingdom", ("kingdom",)),
    ("phylum", ("phylum",)),
    ("between_phylum_and_class", ("subphylum", "gigaclass")),
    ("class", ("class",)),
    ("order", ("order",)),
    ("infraorder", ("infraorder",)),
    ("family", ("family",)),
    ("genus", ("genus",)),
    ("subgenus", ("subgenus",)),
    ("species", ("species",)),
)
OPTIONAL_LEVELS = {"between_phylum_and_class", "infraorder", "subgenus"}


def read_json(path):
    """Return the JSON object and the SHA-256 of its source bytes."""
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def index_records(records, field, identifier_type):
    """Index records while checking identifier types and uniqueness."""
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records with {field}")
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object with {field}")
        key = record.get(field)
        if type(key) is not identifier_type or (identifier_type is str and not key.strip()):
            raise ValueError(f"Invalid {field}: {key!r}")
        if key in indexed:
            raise ValueError(f"Duplicate {field}: {key!r}")
        indexed[key] = record
    return indexed


def verify_prepared_classes(taxonomy_classes, prepared_classes):
    """Check the class IDs used by the crop dataset without relabelling it."""
    expected = index_records(taxonomy_classes, "class_id", int)
    actual = index_records(prepared_classes, "class_id", int)
    if expected.keys() != actual.keys():
        raise ValueError("Taxonomy and prepared classes.json do not share the same class IDs")
    for class_id, entry in sorted(expected.items()):
        if (
            actual[class_id].get("name") != entry["name"]
            or type(actual[class_id].get("source_category_id")) is not int
            or actual[class_id]["source_category_id"] != entry.get("source_category_id")
        ):
            raise ValueError(f"Taxonomy and prepared classes.json disagree for class {class_id}")


def write_text(path, value):
    """Write a complete output beside its temporary file, then replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_training_taxonomy(taxonomy):
    """Return the transformed tree without changing the source dictionary."""
    if not isinstance(taxonomy, dict) or taxonomy.get("schema_version") != 1:
        raise ValueError("Expected a schema_version=1 taxonomy")
    if "training_tree" in taxonomy:
        raise ValueError("Use the original taxonomy as input")
    source = index_records(taxonomy.get("nodes"), "id", str)
    classes = index_records(taxonomy.get("classes"), "class_id", int)
    if not classes or sorted(classes) != list(range(len(classes))):
        raise ValueError("Class IDs must be contiguous from zero")
    roots = [node_id for node_id, node in source.items() if node.get("parent") is None]
    if len(roots) != 1:
        raise ValueError("Expected exactly one root")
    root = roots[0]
    root_rank = source[root].get("rank")
    if root_rank not in (None, "root", "superdomain", "kingdom"):
        raise ValueError(f"Unsupported root rank: {root_rank!r}")

    learned_nodes = set()
    class_names = set()
    category_ids = set()
    for entry in classes.values():
        node_id, name = entry.get("node_id"), entry.get("name")
        if not isinstance(node_id, str) or node_id not in source or node_id in learned_nodes:
            raise ValueError(f"Unknown or repeated class node: {node_id!r}")
        if not isinstance(name, str) or not name.strip() or name in class_names:
            raise ValueError(f"Invalid or repeated class name: {name!r}")
        category_id = entry.get("source_category_id")
        if category_id is not None:
            if type(category_id) is not int or category_id in category_ids:
                raise ValueError(f"Invalid or repeated COCO category ID: {category_id!r}")
            category_ids.add(category_id)
        learned_nodes.add(node_id)
        class_names.add(name)

    # A training tree only needs nodes on a path to a learned class.
    retained = {root}
    for node_id in learned_nodes:
        path = set()
        current = node_id
        while current not in retained:
            if current in path:
                raise ValueError(f"Cycle in the path from {node_id!r}")
            path.add(current)
            current = source[current].get("parent")
            if not isinstance(current, str) or current not in source:
                raise ValueError(f"Invalid parent in the path from {node_id!r}: {current!r}")
        retained.update(path)

    ranks = {source[node_id].get("rank") for node_id in retained}
    levels = [
        (level, accepted_ranks)
        for level, accepted_ranks in RANK_LEVELS
        if (level not in OPTIONAL_LEVELS or ranks.intersection(accepted_ranks))
        and (level != "root" or root_rank != "kingdom")
    ]
    depth_by_rank = {
        rank: depth for depth, (_, accepted_ranks) in enumerate(levels) for rank in accepted_ranks
    }
    depth = {root: 0}
    original_children = defaultdict(list)
    for node_id in sorted(retained - {root}):
        node = source[node_id]
        rank = node.get("rank")
        if rank not in depth_by_rank or depth_by_rank[rank] == 0:
            raise ValueError(f"Unsupported training rank {rank!r} at {node_id!r}")
        depth[node_id] = depth_by_rank[rank]
        original_children[node["parent"]].append(node_id)

    nodes = {
        node_id: dict(
            source[node_id], depth=depth[node_id], training_level=levels[depth[node_id]][0],
            kind="taxon",
        )
        for node_id in sorted(retained)
    }

    def add_synthetic(identifier, parent, node_depth, kind, name, **details):
        if identifier in source or identifier in nodes:
            raise ValueError(f"Synthetic ID collides with a node: {identifier}")
        nodes[identifier] = {
            "id": identifier,
            "parent": parent,
            "name": name,
            "rank": None,
            "depth": node_depth,
            "training_level": levels[node_depth][0],
            "kind": kind,
            **details,
        }
        return identifier

    # Insert missing levels on each edge independently. The LCA among real
    # biological nodes then stays the same as in the source tree.
    for node_id in sorted(retained - {root}):
        parent = source[node_id]["parent"]
        if depth[parent] >= depth[node_id]:
            raise ValueError(f"Ranks do not progress from {parent!r} to {node_id!r}")
        for node_depth in range(depth[parent] + 1, depth[node_id]):
            level = levels[node_depth][0].replace("_", " ")
            parent = add_synthetic(
                f"fgvc:gap:{node_depth}:{node_id}", parent, node_depth, "rank_gap",
                f"{level.capitalize()} slot for {source[node_id]['name']}",
                next_taxon_id=node_id,
            )
        nodes[node_id]["parent"] = parent

    target_depth = len(levels) - 1
    moved_classes = []
    outputs = []
    for class_id in sorted(classes):
        entry = classes[class_id]
        taxon = entry["node_id"]
        node_id = taxon
        has_learned_descendants = bool(original_children[taxon])
        if has_learned_descendants:
            moved_classes.append(class_id)
        for node_depth in range(depth[taxon] + 1, target_depth + 1):
            level = levels[node_depth][0].replace("_", " ")
            if has_learned_descendants and node_depth == depth[taxon] + 1:
                # Rule 1 creates a sibling branch for the annotation at the parent.
                kind = "annotation_branch"
                name = f"Annotated as {entry['name']}"
            else:
                # Rule 2 extends the annotation path to the species output level.
                kind = "annotation_padding"
                name = f"{level.capitalize()} slot for {entry['name']}"
            if node_depth == target_depth:
                name = f"Annotated as {entry['name']}"
            node_id = add_synthetic(
                f"fgvc:annotation:{class_id}:{node_depth}", node_id, node_depth,
                kind, name, original_node_id=taxon,
            )
        outputs.append(dict(
            entry, node_id=node_id, original_node_id=taxon,
            original_rank=source[taxon].get("rank"),
        ))

    parent_ids = {node["parent"] for node in nodes.values()}
    leaves = set(nodes) - parent_ids
    output_ids = {entry["node_id"] for entry in outputs}
    if leaves != output_ids or {nodes[node_id]["depth"] for node_id in leaves} != {target_depth}:
        raise ValueError("Expected every output to be a distinct leaf at the final depth")

    kinds = Counter(node["kind"] for node in nodes.values())
    return {
        **taxonomy,
        "nodes": sorted(nodes.values(), key=lambda node: (node["depth"], node["id"])),
        "classes": outputs,
        "training_tree": {
            "version": 1,
            "levels": [level for level, _ in levels],
            "output_depth": target_depth,
            "annotation_branch_class_ids": moved_classes,
            "synthetic_nodes_by_kind": {kind: count for kind, count in kinds.items() if kind != "taxon"},
            "unused_source_nodes": len(source) - len(retained),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True, help="Original frozen taxonomy JSON")
    parser.add_argument("--classes", type=Path, help="Optional metadata/classes.json from the prepared crops")
    parser.add_argument("--output", type=Path, required=True, help="New training taxonomy JSON")
    args = parser.parse_args(argv)
    try:
        inputs = [args.taxonomy]
        if args.classes is not None:
            inputs.append(args.classes)
        if args.output.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("Output must differ from the input files")
        source, digest = read_json(args.taxonomy)
        transformed = build_training_taxonomy(source)
        metadata = transformed["training_tree"]
        metadata["source"] = {"file": args.taxonomy.name, "sha256": digest}
        if args.classes is not None:
            prepared, classes_digest = read_json(args.classes)
            verify_prepared_classes(transformed["classes"], prepared)
            metadata["prepared_classes"] = {"file": args.classes.name, "sha256": classes_digest}
        write_text(args.output, json.dumps(transformed, ensure_ascii=False, indent=2) + "\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"{len(transformed['classes'])} output leaves at depth {metadata['output_depth']}")
    print(f"{len(metadata['annotation_branch_class_ids'])} parent classes moved to annotation branches")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
