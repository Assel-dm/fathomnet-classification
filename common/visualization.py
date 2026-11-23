"""Build CSV, PNG and HTML from checkpoints"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ast
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


def save_figure(fig, path):
    """Save a figure"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_label(metadata):
    """Build readable label for figures"""
    names = {"ce": "CE", "soft": "Soft Labels", "hxe": "HXE", "haf": "HAF", "haf_ce": "HAF CE", "haf_shc": "HAF CE + SHC", "haf_shc_margin": "HAF CE + SHC + margin", "haf_shc_gc": "HAF CE + SHC + GC"}
    label = names.get(metadata["variant"], metadata["variant"])
    if metadata["parameter"] is not None:
        symbol = "β" if metadata["variant"] == "soft" else "α"
        label += f" {symbol}={metadata['parameter']:g}"
    return f"{label} - {metadata['stage']} - seed {metadata['seed']}"


def plot_learning_curves(run_dir, label):
    """Plot loss, accuracy and top-5 train/validation"""
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    metrics = (("loss", "Loss"), ("accuracy", "Species accuracy"), ("top5_accuracy", "Top-5 accuracy"), ("ce", "HAF fine CE"), ("shc", "HAF soft hierarchical consistency"), ("margin", "HAF dissimilar-pair margin"), ("gc", "HAF geometric consistency (1 - cosine)"))
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(9, 4))
        if metric in history :
            ax.plot(history.epoch, history[metric], label="Train")
        if f"val_{metric}" in history :
            ax.plot(history.epoch, history[f"val_{metric}"], "--", label="Validation")
        ax.set(title=label, xlabel="Epoch", ylabel=title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend()
        save_figure(fig, run_dir / "figures" / f"learning_{metric}.png")
    
    components = ("ce", "shc", "margin", "gc")
    if any(key in history or f"val_{key}" in history for key in components):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for ax, key in zip(axes.flat, components):
            for column, style, name in ((key, "-", "Train"), (f"val_{key}", "--", "Validation")):
                if column in history:
                    ax.plot(history.epoch, history[column], style, label=name)
            ax.set(title=key.upper(), xlabel="Epoch", ylabel="Loss component")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend()
        fig.suptitle(label)
        fig.tight_layout()
        save_figure(fig, run_dir / "figures" / "learning_components.png")

    if "margin_active_ranks" in history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, key, title in zip(axes, ("margin_active_ranks", "margin_pairs"), ("Ranks with dissimilar pairs per batch", "Selected pairs per batch (sum of ranks)")):
            ax.plot(history.epoch, history[key], label="Train")
            if f"val_{key}" in history:
                ax.plot(history.epoch, history[f"val_{key}"], "--", label="Validation")
            ax.set(title=title, xlabel="Epoch", ylabel="Mean per batch", ylim=(0, None))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle(label)
        fig.tight_layout()
        save_figure(fig, run_dir / "figures" / "learning_margin_diagnostics.png")


def plot_distance_histogram(run_dir, split, label):
    """Show both total-image and error-conditional distance distributions"""
    path = run_dir / f"distance_histogram_{split}.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, selected, metric, title in (
        (axes[0], table, "fraction_of_images", "All images (distance 0 = correct)"),
        (axes[1], table[table.distance > 0], "fraction_of_errors", "Errors only"),
    ):
        ax.bar(selected.distance, selected[metric])
        ax.set(title=title, xlabel="LCA height", ylabel="Fraction", ylim=(0, 1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{label} - {split}")
    fig.tight_layout()
    save_figure(fig, run_dir / "figures" / f"distance_histogram_{split}.png")



def plot_per_class(run_dir, split, label):
    """Print worst F1 and confusion per class"""
    table = pd.read_csv(run_dir / f"per_class_{split}.csv", dtype={"species": str})
    if "species" not in table and "taxon" in table:
        table = table.rename(columns={"taxon": "species"})
    worst = table.sort_values("f1").head(25)
    fig, ax = plt.subplots(figsize=(9, max(4, len(worst) * .26)))
    labels = worst["species"]
    if "species_name" in worst :
        labels = worst["species_name"].fillna(worst["species"]) + " (" + worst["species"] + ")"
    ax.barh(labels, worst["f1"])
    ax.invert_yaxis()
    ax.set(xlabel="F1", xlim=(0, 1), title=f"{label} - {split} - lowest F1")
    ax.grid(axis="x", alpha=.2)
    save_figure(fig, run_dir / "figures" / f"per_class_{split}.png")
    if len(table) <= 40 and (run_dir / f"confusion_{split}.csv").is_file():
        matrix = pd.read_csv(run_dir / f"confusion_{split}.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
        ax.set(xlabel="Predicted species", ylabel="True species", title=f"{label} - {split}")
        fig.colorbar(image, ax=ax, label="Images")
        save_figure(fig, run_dir / "figures" / f"confusion_{split}.png")


def plot_dataset_counts(path, output_dir):
    """Plot repartition of images through species and splits"""
    if not path.exists():
        return
    table = pd.read_csv(path, dtype={"species": str})
    table = table.sort_values("train", ascending=False)
    large = len(table) > 40
    fig, ax = plt.subplots(figsize=(12 if large else max(10, len(table) * .15), 5))
    positions = np.arange(len(table)) if large else table.species
    bottom = np.zeros(len(table))
    for split in ("train", "validation", "test"):
        ax.bar(positions, table[split], bottom=bottom, label=split)
        bottom += table[split].to_numpy()
    if large :
        ax.set_xlabel("Species index, sorted by training frequency")
    else :
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set(ylabel="Images", title="Dataset distribution by species")
    ax.legend()
    save_figure(fig, output_dir / "dataset_distribution.png")
    for rank in ("kingdom", "class", "order"):
        if rank not in table:
            continue

        column = f"{rank}_name" if f"{rank}_name" in table else rank
        grouped = (
            table.groupby(column)[["train", "validation", "test"]]
            .sum()
            .sort_values("train", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(12, max(4, len(grouped) * .25)))
        bottom = np.zeros(len(grouped))

        for split in ("train", "validation", "test"):
            ax.barh(grouped.index, grouped[split], left=bottom, label=split)
            bottom += grouped[split].to_numpy()

        ax.invert_yaxis()
        ax.set(xlabel="Images", title=f"Dataset distribution by {rank}")
        ax.legend()
        save_figure(fig, output_dir / f"dataset_distribution_{rank}.png")


def plot_augmentation_preview(image_path, transforms, settings, output_path):
    """Compare different augmention for an image"""
    from PIL import Image
    with Image.open(image_path) as image:
        original = image.convert("RGB")
    mean = np.array(settings["mean"]).reshape(1, 1, 3)
    std = np.array(settings["std"]).reshape(1, 1, 3)
    fig, axes = plt.subplots(len(transforms), 5, figsize=(14, 3 * len(transforms)), squeeze=False)
    for row, (name, transform) in enumerate(transforms.items()):
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{name}\nOriginal")
        for col in range(1, 5):
            tensor = transform(original.copy())
            visible = np.clip(tensor.permute(1, 2, 0).numpy() * std + mean, 0, 1)
            axes[row, col].imshow(visible)
            axes[row, col].set_title(f"Example {col}")
    for ax in axes.flat:
        ax.axis("off")
    save_figure(fig, output_path)

def draw_tradeoff(ax, table, metric):
    """Plot parameter sweeps with seed means and automatic label placement"""
    from adjustText import adjust_text

    names = {"ce": "CE", "soft": "Soft Labels", "hxe": "HXE", "haf": "HAF"}
    colors = {
        "ce": "#252525", "soft": "#e28b20",
        "hxe": "#007d79", "haf": "#7756a8",
    }
    styles = {"frozen": ":", "partial": "--", "deeper": "-.", "full": "--"}

    groups = [
        key for key in ("model_key", "augmentation", "stage", "variant")
        if key in table
    ]
    varying = [
        key for key in groups
        if key != "variant" and table[key].nunique() > 1
    ]
    texts, target_x, target_y, avoid_x, avoid_y = [], [], [], [], []

    for _, selected in table.groupby(groups, dropna=False):
        selected = selected.dropna(subset=["top1_error", metric])
        if selected.empty:
            continue

        variant = selected.variant.iloc[0]
        label = names.get(variant, variant.upper().replace("_", " "))
        label += "".join(f" · {selected[key].iloc[0]}" for key in varying)
        color = colors.get(variant)

        points = selected.groupby("parameter", dropna=False)[
            ["top1_error", metric]
        ].agg(["mean", "std"]).sort_index()

        x = points[("top1_error", "mean")].to_numpy(dtype=float)
        y = points[(metric, "mean")].to_numpy(dtype=float)
        multiple_seeds = (
            selected.groupby("parameter", dropna=False).seed.nunique().max() > 1
        )

        ax.errorbar(
            x, y,
            xerr=points[("top1_error", "std")].fillna(0)
                 if multiple_seeds else None,
            yerr=points[(metric, "std")].fillna(0)
                 if multiple_seeds else None,
            fmt="*" if variant == "ce" else "o",
            color=color,
            linestyle=styles.get(selected.stage.iloc[0], "--")
                      if variant in ("soft", "hxe") else "none",
            markersize=10 if variant == "ce" else 7,
            capsize=3,
            linewidth=1.3,
            alpha=.9,
            label=label,
        )

        avoid_x.extend(x)
        avoid_y.extend(y)

        # Sample connecting lines so labels also avoid the curves.
        if variant in ("soft", "hxe"):
            for index in range(len(x) - 1):
                avoid_x.extend(np.linspace(x[index], x[index + 1], 25))
                avoid_y.extend(np.linspace(y[index], y[index + 1], 25))

        for parameter, xx, yy in zip(points.index, x, y):
            if variant in ("soft", "hxe") and pd.notna(parameter):
                text = f"{'β' if variant == 'soft' else 'α'}={parameter:g}"
            elif variant == "ce":
                text = "CE"
            else:
                continue

            texts.append(ax.text(
                xx, yy, text,
                fontsize=9, color=color, ha="center", va="center",
            ))
            target_x.append(xx)
            target_y.append(yy)

    ylabel = (
        "Mistake severity" if metric == "mistake_severity"
        else f"Average hierarchical distance @{metric.split('@')[-1]}"
    )
    ax.set(xlabel="Top-1 error", ylabel=ylabel)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.margins(x=.15, y=.18)
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))

    # Adjust labels after setting the final axes limits.
    if texts:
        adjust_text(
            texts,
            x=avoid_x, y=avoid_y,
            target_x=target_x, target_y=target_y,
            ax=ax,
            expand=(1.15, 1.3),
            force_text=(.25, .4),
            force_static=(.2, .3),
            iter_lim=400,
            min_arrow_len=5,
            arrowprops=dict(
                arrowstyle="-", color="#9aa4ad", lw=.6
            ),
        )


def build_html(records, overall, seed_summary, output_path, confusion_splits=()):
    """Build HTML report with Plotly"""
    import base64
    from html import escape

    output_path = Path(output_path)
    reports_dir = output_path.parent
    campaign = reports_dir.parent.name

    def picture(path, caption):
        if not path.exists():
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return (
            f'<figure><figcaption>{escape(caption)}</figcaption>'
            f'<img src="data:image/png;base64,{encoded}" '
            f'alt="{escape(caption)}" loading="lazy"></figure>'
        )

    def table(frame):
        if frame.empty:
            return "<p>No saved data available.</p>"
        return '<div class="table-wrap">' + frame.to_html(
            index=False,
            escape=True,
            na_rep="—",
            float_format=lambda value: f"{value:.4f}",
        ) + "</div>"

    parts = [
        f"<h1>{escape(campaign)}</h1>",
        "<p>Report generated from saved results. "
        '<nav><a href="#comparison">Comparisons</a> · '
        '<a href="#runs">Runs and species</a> · '
        '<a href="#metrics">Overall metrics</a> · '
        '<a href="#seeds">Across seeds</a> · '
        '<a href="#dataset">Dataset distribution</a></nav>',
        "<p>Lower values on both axes are preferable.</p>",
    ]

    if not overall.empty:
        missing = (
            overall["avg_hierarchical_distance@20"].isna().sum()
            if "avg_hierarchical_distance@20" in overall else len(overall)
        )
        if missing:
            parts.append(
                f'<p class="note">Distance @20 is missing for {missing} '
                "evaluated runs. Re-evaluate these runs to add this metric.</p>"
            )

        for split in overall.split.unique():
            parts.append(f"<h3>{escape(str(split).capitalize())}</h3>")
            parts.append(picture(
                reports_dir / f"comparison_{split}.png",
                "Top-1 error versus mistake severity",
            ))
            if split in confusion_splits:
                parts.append(picture(
                    reports_dir / f"confusion_by_distance_{split}.png",
                    "Top-1 confusion by hierarchical distance",
                ))
            for k in (1, 5, 20):
                parts.append(picture(
                    reports_dir / f"avg_hierarchical_distance_{k}_{split}.png",
                    f"Top-1 error versus average hierarchical distance @{k}",
                ))
    else:
        parts.append(
            "<p>No saved evaluation results. "
            "Run an evaluation before comparing methods.</p>"
        )

    parts += [
        '<h2 id="runs">Learning curves and F1 by species</h2>',
        "<p>Expand a run to view its figures. Curves use the original history "
        "values without smoothing. Loss scales depend on the objective and its "
        "parameter; compare them within the same objective. Saved evaluation "
        "metrics refer to the selected checkpoint.</p>",
        "<p>HAF total = fine CE + SHC + margin + GC. GC uses 1 - cosine; "
        "the authors' -cosine convention differs by one constant per adjacent "
        "rank link and has the same gradients. Margin depends on the number of "
        "ranks with dissimilar pairs in each batch. Original validation ordering "
        "and checkpoint selection are preserved.</p>",
    ]

    for record in records:
        run_dir = record["run_dir"]
        parts.append(
            f'<details><summary>{escape(record["label"])}</summary>'
        )
        history = record["history"]
        if "HAF" in record["label"] and history is not None:
            columns = [prefix + key for prefix in ("", "val_") for key in ("ce", "shc", "margin", "gc")]
            if not all(key in history for key in columns):
                parts.append('<p class="note">This history does not contain all train/validation '
                             'HAF components. Missing epoch values cannot be reconstructed from final weights.</p>')
            else:
                for prefix, title in (("", "Train"), ("val_", "Validation")):
                    total = history[[prefix + key for key in ("ce", "shc", "margin", "gc")]].sum(axis=1)
                    if not np.allclose(total, history[prefix + "loss"], atol=1e-5, rtol=1e-5):
                        parts.append(f'<p class="note">{title} total loss does not match the sum of '
                                     'the four saved components. Check the history and loss version.</p>')

        for metric, title in (
            ("loss", "Loss"),
            ("accuracy", "Top-1 accuracy"),
            ("top5_accuracy", "Top-5 accuracy"),
            ("ce", "HAF CE"),
            ("shc", "HAF SHC"),
            ("margin", "HAF margin"),
            ("gc", "HAF GC"),
            ("components", "HAF loss components"),
            ("margin_diagnostics", "Dissimilar-pair availability"),
        ):
            parts.append(picture(
                run_dir / "figures" / f"learning_{metric}.png", title
            ))

        for split, classes in record["classes"].items():
            parts.append(picture(
                run_dir / "figures" / f"per_class_{split}.png",
                f"Lowest species F1 · {split}",
            ))
            parts.append(picture(
                run_dir / "figures" / f"confusion_{split}.png",
                f"Confusion matrix · {split}",
            ))
            columns = [
                key for key in (
                    "species", "species_name", "support",
                    "precision", "recall", "f1"
                )
                if key in classes
            ]
            parts.append(table(
                classes.sort_values("f1").head(25)[columns]
            ))

        parts.append("</details>")

    parts += [
        '<h2 id="metrics">Overall metrics</h2>',
        "<p>Historical rank_accuracy metrics retain probability-sum decoding. "
        "The explicit rank_accuracy_probability_sum columns are aliases; "
        "rank_accuracy_top1_ancestor uses the ancestor of the top-1 species, "
        "as in the paper's hierarchical evaluation.</p>",
        table(overall),
        '<h2 id="seeds">Mean and standard deviation across seeds</h2>',
        table(seed_summary),
        "<p>Single-seed standard deviations appear as “—”. "
        "Complete CSV files remain in runs/ and reports/.</p>",
        '<h2 id="dataset">Dataset distribution</h2>',
    ]

    for path in sorted(reports_dir.glob("dataset_distribution*.png")):
        parts.append(picture(
            path, path.stem.replace("_", " ").capitalize()
        ))
    parts.append(picture(
        reports_dir / "augmentation_preview.png", "Augmentation preview"
    ))

    css = """
    body{font:15px/1.5 system-ui,sans-serif;color:#203340;
         background:#f5f7f9;margin:0}
    main{max-width:1050px;margin:auto;padding:24px}
    h1{overflow-wrap:anywhere}h2{margin-top:32px}
    nav,a{color:#007d79}
    details,figure,.table-wrap{background:white;border:1px solid #dce4e8;
                              border-radius:8px}
    details{padding:16px;margin:12px 0}
    summary{cursor:pointer;font-weight:600}
    figure{margin:16px 0;padding:12px}
    figcaption{font-weight:600;margin-bottom:8px}
    img{display:block;width:100%;height:auto}
    .table-wrap{overflow:auto;max-height:480px}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{padding:8px;white-space:nowrap;border-bottom:1px solid #e2e8ec}
    th{position:sticky;top:0;background:#eef3f5}
    .note{padding:12px;background:#fff2d9}
    @media(max-width:600px){main{padding:12px}details{padding:10px}}
    """

    output_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(campaign)} · results</title><style>{css}</style></head>"
        "<body><main>" + "".join(parts) + "</main></body></html>",
        encoding="utf-8",
    )

def confusion_distance_reference(campaign_dir, fgvc_taxonomy=None):
    """Rebuild the evaluation distances from a campaign's saved taxa."""
    campaign_path = campaign_dir / "campaign.json"
    campaign = (
        json.loads(campaign_path.read_text(encoding="utf-8-sig"))
        if campaign_path.is_file() else {}
    )
    taxonomy = campaign.get("taxonomy", [])
    dataset = str(campaign.get("dataset", "")).lower()

    if "fathom" in dataset or "fgvc" in dataset or (not dataset and fgvc_taxonomy):
        if fgvc_taxonomy:
            source = Path(fgvc_taxonomy).expanduser()
        else:
            runtime_path = campaign_dir / "runtime.json"
            runtime = (
                json.loads(runtime_path.read_text(encoding="utf-8-sig"))
                if runtime_path.is_file() else {}
            )
            roots = [campaign_dir, Path(__file__).resolve().parents[1]]
            if runtime.get("dataset_root"):
                roots.insert(1, Path(runtime["dataset_root"]).expanduser())

            filename = "taxonomy_fathomnet2025.json"
            source = next(
                (
                    path
                    for root in roots
                    for path in (root / filename, root / "metadata" / filename)
                    if path.is_file()
                ),
                None,
            )
            if source is None:
                raise ValueError(
                    "FGVC original taxonomy unavailable. "
                    "Set FATHOMNET_EVALUATION_TAXONOMY to its JSON path."
                )

        original = json.loads(source.read_text(encoding="utf-8-sig"))
        if "training_tree" in original or "nodes" not in original or "classes" not in original:
            raise ValueError("FGVC evaluation requires the original nodes/classes taxonomy JSON.")

        nodes = {node["id"]: node["parent"] for node in original["nodes"]}
        entries = sorted(original["classes"], key=lambda value: value["class_id"])
        if [entry["class_id"] for entry in entries] != list(range(len(entries))):
            raise ValueError("FGVC class IDs must be contiguous and zero-based.")

        lineages, names = [], []
        for entry in entries:
            names.append(entry["name"])
            path, node = [], entry["node_id"]
            while node is not None:
                if node in path or node not in nodes:
                    raise ValueError(f"Invalid FGVC parent chain for {entry['name']}.")
                path.append(node)
                node = nodes[node]
            lineages.append(path[::-1])

        if len(set(names)) != len(names):
            raise ValueError("FGVC class names must be unique.")
        if not lineages or len({path[0] for path in lineages}) != 1:
            raise ValueError("FGVC classes must share a single root.")

        distances = np.zeros((len(names), len(names)), dtype=int)
        for i, left in enumerate(lineages):
            for j, right in enumerate(lineages[:i]):
                shared = next(
                    (k for k, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                    min(len(left), len(right)),
                )
                distances[i, j] = distances[j, i] = len(left) + len(right) - 2 * shared

        return names, distances, "Original FGVC taxonomy (edge count)"

    ranks = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
    lineage_table = pd.DataFrame(taxonomy)
    configured_ranks = campaign.get("taxonomy_ranks") or campaign.get("ranks")
    config_path = campaign_dir / "config_used.py"

    if not configured_ranks and dataset.startswith(("inat21", "inaturalist21")) and config_path.is_file():
        # Read the archived rank list without executing the configuration.
        for statement in ast.parse(config_path.read_text(encoding="utf-8-sig")).body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "INAT21_RANKS"
                for target in statement.targets
            ):
                configured_ranks = ast.literal_eval(statement.value)

    ranks = list(configured_ranks or [rank for rank in ranks if rank in lineage_table])
    if (
        lineage_table.empty
        or len(ranks) < 2
        or ranks[-1] != "species"
        or not set(ranks) <= set(lineage_table)
    ):
        raise ValueError("Saved campaign taxonomy with rank columns is required.")

    if lineage_table[ranks].isna().any().any() or lineage_table.species.duplicated().any():
        raise ValueError("Incomplete or duplicate class lineages in campaign.json.")

    names = lineage_table.species.astype(str).tolist()
    # Same LCA height convention as common.data.build_hierarchy.
    distances = np.full((len(names), len(names)), len(ranks), dtype=np.uint8)
    for depth, rank in enumerate(ranks, start=1):
        ancestors = pd.factorize(lineage_table[rank])[0]
        distances[ancestors[:, None] == ancestors[None, :]] = len(ranks) - depth

    return names, distances, "Saved campaign taxonomy (LCA height)"


def read_confusion_counts(run_dir, split, names):
    """Load top-1 counts or rebuild them from saved per-crop predictions."""
    path = run_dir / f"confusion_{split}.csv"
    if path.is_file():
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        matrix = raw.set_index(raw.columns[0])
        if (
            matrix.index.has_duplicates
            or matrix.columns.has_duplicates
            or set(matrix.index) != set(names)
            or set(matrix.columns) != set(names)
        ):
            raise ValueError(f"Class labels do not match the reference taxonomy in {path.name}.")

        values = matrix.loc[names, names].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError(f"{path.name} must contain raw non-negative confusion counts.")
        return values.astype(np.int64)

    path = run_dir / f"predictions_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing confusion_{split}.csv and {path.name}.")

    predictions = pd.read_csv(path, dtype=str, keep_default_na=False)
    positions = {name: index for index, name in enumerate(names)}

    if {"true_species", "predicted_species"} <= set(predictions):
        indices = [
            predictions[column].map(positions)
            for column in ("true_species", "predicted_species")
        ]
        if any(index.isna().any() for index in indices):
            raise ValueError(f"Unknown class names in {path.name}.")
    elif {"true_id", "predicted_id"} <= set(predictions):
        indices = [
            predictions[column].astype(int)
            for column in ("true_id", "predicted_id")
        ]
        if any(((index < 0) | (index >= len(names))).any() for index in indices):
            raise ValueError(f"Unknown class IDs in {path.name}.")
    else:
        raise ValueError(f"Missing class columns in {path.name}.")

    counts = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(counts, tuple(index.to_numpy(dtype=int) for index in indices), 1)
    return counts


def confusion_by_distance(counts, distances):
    """Compute top-1 error shares and mean directional confusion rates."""
    support = counts.sum(axis=1)
    if not support.sum():
        raise ValueError("Cannot plot a split with no evaluated crops.")

    bins = int(distances.max()) + 1
    errors = np.zeros(bins)
    rates = np.zeros(bins)
    pairs = np.zeros(bins, dtype=int)

    for i in np.flatnonzero(support):
        row = distances[i]
        per_distance = np.bincount(row, weights=counts[i], minlength=bins)
        n_pairs = np.bincount(row, minlength=bins)

        # Remove the diagonal, which contains correct predictions.
        per_distance[row[i]] -= counts[i, i]
        n_pairs[row[i]] -= 1

        errors += per_distance
        rates += per_distance / support[i]
        pairs += n_pairs

    shares = errors / errors.sum() if errors.sum() else np.full(bins, np.nan)
    averages = np.divide(rates, pairs, out=np.full(bins, np.nan), where=pairs > 0)
    return shares, averages, pairs, counts.trace() / support.sum()


def plot_campaign_confusion_distances(
    campaign_dir, records, overall, output_dir, compare_campaigns=(),
):
    """Compare evaluated variants, averaging available seeds within each variant."""
    if overall.empty:
        return set()

    fgvc_override = os.environ.get("FATHOMNET_EVALUATION_TAXONOMY")
    try:
        names, distances, reference = confusion_distance_reference(campaign_dir, fgvc_override)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        warnings.warn(
            f"Skipping confusion by distance for {campaign_dir.name}: {error}",
            stacklevel=2,
        )
        return set()

    candidates = [(campaign_dir, records)]
    for extra in compare_campaigns:
        other = Path(extra).expanduser()
        try:
            other_names, other_distances, other_reference = confusion_distance_reference(
                other, fgvc_override
            )
            if (
                other_names != names
                or other_reference != reference
                or not np.array_equal(other_distances, distances)
            ):
                raise ValueError("Class order or evaluation distances differ.")

            other_records = []
            for path in sorted((other / "runs").glob("*/*/metadata.json")):
                other_records.append({
                    "run_dir": path.parent,
                    "label": run_label(json.loads(path.read_text(encoding="utf-8-sig"))),
                })
            candidates.append((other, other_records))
        except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
            warnings.warn(
                f"Skipping campaign {other} in confusion by distance: {error}",
                stacklevel=2,
            )

    plotted_splits = set()
    for split in overall["split"].unique():
        curves = {}
        reference_support = None

        for folder, record in ((folder, record) for folder, runs in candidates for record in runs):
            run_dir = record["run_dir"]
            if not (run_dir / f"metrics_{split}.json").is_file():
                continue

            try:
                counts = read_confusion_counts(run_dir, split, names)
                support = counts.sum(axis=1)
                if reference_support is not None and not np.array_equal(support, reference_support):
                    raise ValueError("Class supports differ from the other runs on this split.")
                stats = confusion_by_distance(counts, distances)
            except (OSError, ValueError, TypeError) as error:
                warnings.warn(
                    f"Skipping confusion by distance in {run_dir} ({split}): {error}",
                    stacklevel=2,
                )
                continue

            reference_support = support
            label = record["label"].rsplit(" - seed ", 1)[0]
            if len(candidates) > 1:
                label = f"{folder.name} | {label}"
            curves.setdefault(label, []).append(stats)

        if not curves:
            continue

        pairs = next(iter(curves.values()))[0][2]
        x = np.flatnonzero(pairs)
        if not x.size:
            continue

        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
        for label, stats in curves.items():
            shares = [item[0] for item in stats if np.isfinite(item[0]).any()]
            average = np.mean([item[1][x] for item in stats], axis=0)
            accuracy = np.mean([item[3] for item in stats])
            shown_label = f"{label} | top-1 {accuracy:.1%}"
            line = right.plot(x, average, marker="o", markersize=3, label=shown_label)[0]

            if shares:
                left.plot(
                    x, np.mean(shares, axis=0)[x],
                    marker="o", markersize=3, color=line.get_color(),
                )

        left.set(
            title="Distribution of top-1 errors",
            xlabel="Hierarchical distance",
            ylabel="Share of all errors",
        )
        right.set(
            title="Mean confusion by ordered class pair",
            xlabel="Distance (eligible ordered pairs)",
            ylabel="Mean share of crops from the true class",
        )
        left.set_xticks(x)
        right.set_xticks(x, [f"{d}\n({pairs[d]})" for d in x], fontsize=8)

        for axis in (left, right):
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=.25)

        if not left.lines:
            left.text(.5, .5, "No top-1 errors", transform=left.transAxes, ha="center")

        fig.suptitle(f"Confusion by hierarchical distance | {split}\n{reference}")
        handles, labels = right.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(.5, -.025),
            ncol=min(3, len(labels)), frameon=False, fontsize=8,
        )
        fig.tight_layout(rect=(0, 0, 1, .93))
        save_figure(fig, output_dir / f"confusion_by_distance_{split}.png")
        plotted_splits.add(split)

    return plotted_splits

def build_reports(campaign_dir, compare_campaigns=()):
    """Build graphics with available with available plots"""
    campaign_dir = Path(campaign_dir).expanduser()
    if not campaign_dir.is_dir():
        raise FileNotFoundError(campaign_dir)

    reports_dir = campaign_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    records, rows, class_tables = [], [], []
    stage_order = {"frozen": 0, "partial": 1, "deeper": 2, "full": 3}
    paths = sorted(
        (campaign_dir / "runs").glob("*/*/metadata.json"),
        key=lambda path: (
            path.parent.parent.name, stage_order[path.parent.name]
        ),
    )

    for path in paths:
        run_dir = path.parent
        metadata = json.loads(path.read_text(encoding="utf-8"))
        label = run_label(metadata)
        plot_learning_curves(run_dir, label)
        history_path = run_dir / "history.csv"
        record = {
            "run_dir": run_dir,
            "label": label,
            "history": pd.read_csv(history_path)
                       if history_path.exists() else None,
            "classes": {},
        }

        for split in ("validation", "test"):
            metrics_path = run_dir / f"metrics_{split}.json"
            if not metrics_path.exists():
                continue

            scores = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({
                **{key: value for key, value in metadata.items()
                   if key != "training"},
                "label": label, "split": split, **scores,
            })
            per_class_path = run_dir / f"per_class_{split}.csv"
            if not per_class_path.is_file():
                continue
            record["classes"][split] = pd.read_csv(
                per_class_path,
                dtype={"species": str},
            )
            table = record["classes"][split].copy()
            for key, value in (
                ("label", label), ("seed", metadata["seed"]),
                ("stage", metadata["stage"]),
                ("variant", metadata["variant"]),
                ("parameter", metadata["parameter"]), ("split", split),
            ):
                table[key] = (
                    np.nan if key == "parameter" and value is None else value
                )
            class_tables.append(table)
            plot_per_class(run_dir, split, label)
            plot_distance_histogram(run_dir, split, label)

        records.append(record)

    overall = (
        pd.DataFrame(rows) if rows
        else pd.DataFrame(columns=["label", "split"])
    )
    overall.to_csv(reports_dir / "comparison.csv", index=False)
    if class_tables:
        pd.concat(class_tables, ignore_index=True).to_csv(
            reports_dir / "per_class_comparison.csv", index=False
        )

    summary = pd.DataFrame()
    if rows:
        groups = [
            "campaign", "model_key", "augmentation",
            "variant", "parameter", "stage", "split",
        ]
        metrics = [
            column for column in overall.select_dtypes(include="number")
            if column not in {"seed", "parameter"}
        ]
        grouped = overall.groupby(groups, dropna=False)
        summary = grouped[metrics].agg(["mean", "std"])
        summary.columns = [
            f"{metric}_{stat}" for metric, stat in summary.columns
        ]
        summary["n_seeds"] = grouped.seed.nunique()
        summary = summary.reset_index()

        for split in overall.split.unique():
            table = overall[overall.split == split]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            draw_tradeoff(ax, table, "mistake_severity")
            ax.set_title(f"Hierarchical trade-off · {split}")
            save_figure(fig, reports_dir / f"comparison_{split}.png")

            for k in (1, 5, 20):
                metric = f"avg_hierarchical_distance@{k}"
                if metric not in table or table[metric].notna().sum() == 0:
                    print(
                        f"Skipping {metric} · {split}: no saved values. "
                        "Re-evaluate to add this metric."
                    )
                    continue

                fig, ax = plt.subplots(figsize=(9, 5.5))
                draw_tradeoff(ax, table, metric)
                ax.set_title(f"Hierarchical distance @{k} · {split}")
                save_figure(
                    fig,
                    reports_dir / f"avg_hierarchical_distance_{k}_{split}.png",
                )

    summary.to_csv(reports_dir / "seed_summary.csv", index=False)
    confusion_splits = plot_campaign_confusion_distances(
        campaign_dir, records, overall, reports_dir, compare_campaigns
    )
    plot_dataset_counts(reports_dir / "dataset_counts.csv", reports_dir)
    build_html(records, overall, summary, reports_dir / "report.html", confusion_splits)
    print("Report:", reports_dir / "report.html")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path, help="Folder containing runs/ and campaign.json")
    parser.add_argument(
        "--compare", type=Path, nargs="+", default=(),
        help="Additional campaigns for the confusion-by-distance figure",
    )
    args = parser.parse_args()
    build_reports(args.campaign_dir, args.compare)
