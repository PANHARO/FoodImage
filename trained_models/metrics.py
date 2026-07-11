"""Shared Matplotlib reports for the image-classification models."""

import csv
from pathlib import Path


REPORT_ROOT = Path("reports") / "plots"


def save_training_history(history, model_name):
    """Save epoch values as CSV and plot loss and accuracy curves."""
    import matplotlib.pyplot as plt

    output_dir = REPORT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [row["loss"] for row in history], marker="o")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history],
                 marker="o", label="Train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history],
                 marker="o", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(f"{model_name.replace('_', ' ').title()} Training History")
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Training statistics saved to: {output_dir}")


def save_evaluation_plots(labels, predictions, class_names, model_name):
    """Plot a normalized confusion matrix and per-class precision/recall/F1."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    output_dir = REPORT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(class_names)))
    matrix = confusion_matrix(labels, predictions, labels=label_ids)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals,
                           out=np.zeros_like(matrix, dtype=float),
                           where=row_totals != 0)

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, label="Fraction of true class")
    axis.set(title=f"{model_name.replace('_', ' ').title()} Confusion Matrix",
             xlabel="Predicted class", ylabel="True class",
             xticks=label_ids, yticks=label_ids,
             xticklabels=class_names, yticklabels=class_names)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(column, row,
                      f"{matrix[row, column]}\n{normalized[row, column]:.0%}",
                      ha="center", va="center", fontsize=7,
                      color="white" if normalized[row, column] > 0.55 else "black")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=label_ids, zero_division=0
    )
    positions = np.arange(len(class_names))
    width = 0.25
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.bar(positions - width, precision, width, label="Precision")
    axis.bar(positions, recall, width, label="Recall")
    axis.bar(positions + width, f1, width, label="F1 score")
    axis.set(title=f"{model_name.replace('_', ' ').title()} Per-class Statistics",
             xlabel="Class", ylabel="Score", ylim=(0, 1.05),
             xticks=positions, xticklabels=class_names)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "class_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    with (output_dir / "class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        writer.writerows(zip(class_names, precision, recall, f1, support))
    print(f"Evaluation graphs saved to: {output_dir}")
