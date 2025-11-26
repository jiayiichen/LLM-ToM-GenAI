import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from upsetplot import UpSet, from_contents

sns.set(style="whitegrid")

# -------------------------------------------------------------------
# 0. Basic path settings
# -------------------------------------------------------------------
file_paths = [
    "sample_results/gemini/sample_failure_gemini.json",
    "sample_results/Claude/sample_failures.json",
    "sample_results/gpt5.1/sample_failure.json",
    "sample_results/llama_merged/llama_failures.json",
    "sample_results/qwen_merged/qwen_failures.json",
]

# Output directory
viz_dir = "visualization"
os.makedirs(viz_dir, exist_ok=True)

# -------------------------------------------------------------------
# 1. Aggregate errors by Model × Task and plot bar charts + heatmap
# -------------------------------------------------------------------
def extract_model_and_task(file_path: str) -> pd.DataFrame:
    """Extract (Model, Task) level statistics from a single JSON failure file."""
    model_name = os.path.basename(os.path.dirname(file_path))

    with open(file_path, "r") as f:
        data = json.load(f)

    rows = []
    for item in data:
        # SOURCE_FILE is used as the task/subtest name
        task = item.get("SOURCE_FILE", "Unknown Task")
        rows.append(
            {
                "Model": model_name,
                "Task": task,
                "Count": 1,  # Each failure sample counts as 1
            }
        )

    return pd.DataFrame(rows)


def build_aggregated_error_data(paths):
    """Aggregate error data across all models and tasks."""
    dfs = []
    for p in paths:
        try:
            df = extract_model_and_task(p)
            dfs.append(df)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    if not dfs:
        raise ValueError("No data files could be loaded. Please check file paths and content.")

    data = pd.concat(dfs, ignore_index=True)

    # Aggregate to (Model, Task)
    data = data.groupby(["Model", "Task"], as_index=False).sum()

    # Normalize within each model: percentage of errors in this task out of all errors for the model
    data["Total"] = data.groupby("Model")["Count"].transform("sum")
    data["Percentage"] = data["Count"] / data["Total"] * 100.0

    return data


def plot_task_error_bars(data: pd.DataFrame, out_pdf: str):
    """Plot task-level error bar charts, faceted by task."""
    tasks = sorted(data["Task"].unique())
    num_tasks = len(tasks)
    tasks_per_figure = 6  # 2x3 subplots per page

    with PdfPages(out_pdf) as pdf:
        for i in range(0, num_tasks, tasks_per_figure):
            fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
            axes = axes.flatten()

            current_tasks = tasks[i : i + tasks_per_figure]

            for ax, task in zip(axes, current_tasks):
                task_data = data[data["Task"] == task]

                sns.barplot(
                    data=task_data,
                    x="Percentage",
                    y="Model",
                    ax=ax,
                )
                ax.set_title(f"Task: {task}", fontsize=14)
                ax.set_xlabel("Share of errors in this task (%)", fontsize=12)
                ax.set_ylabel("Model", fontsize=12)

            # Hide unused subplots
            for ax in axes[len(current_tasks) :]:
                ax.axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Task-level error barplots saved as PDF at: {out_pdf}")


def plot_error_heatmap(data: pd.DataFrame, out_pdf: str):
    """Plot a heatmap of error distribution by model and task."""
    pivot_table = data.pivot_table(
        index="Task", columns="Model", values="Percentage", aggfunc="sum"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title("Error distribution by task and model", fontsize=16)
    plt.ylabel("Task", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"Heatmap saved as PDF at: {out_pdf}")


# -------------------------------------------------------------------
# 2. Failure overlap (by item) + UpSet plot + CSV export
# -------------------------------------------------------------------
def extract_failure_overlap(paths):
    """
    Construct failure overlap at the item level:
      key   = (story, question, options)
      value = set of models that failed this item
    """
    overlap_data = {}

    for file_path in paths:
        model_name = os.path.basename(os.path.dirname(file_path))

        with open(file_path, "r") as f:
            data = json.load(f)

        for item in data:
            story = item.get("STORY")
            question = item.get("QUESTION")
            options = tuple(
                item.get(option) for option in ["OPTION-A", "OPTION-B", "OPTION-C", "OPTION-D"]
            )

            # Strictly filter out samples with missing fields
            if not story or not question or any(opt is None for opt in options):
                print(f"Warning: Missing required fields in file {file_path}")
                continue

            key = (story, question, options)

            if key not in overlap_data:
                overlap_data[key] = set()
            overlap_data[key].add(model_name)

    return overlap_data


def prepare_upset_series(overlap_data):
    """
    Convert overlap_data to a Series required by upsetplot.from_contents.
    """
    contents = {}  # model_name -> list of failure_ids

    for failure_key, models in overlap_data.items():
        for model in models:
            contents.setdefault(model, []).append(failure_key)

    series = from_contents(contents)
    return series


def plot_upset(overlap_data, out_pdf: str):
    """Plot an UpSet chart for failure overlap across models."""
    series = prepare_upset_series(overlap_data)

    plt.figure(figsize=(12, 8))
    upset = UpSet(series, subset_size="count", show_counts=True)
    upset.plot()
    plt.suptitle("Failure Overlap Across Models (UpSet Plot)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_pdf)
    plt.close()
    print(f"UpSet plot saved as PDF at: {out_pdf}")


def export_failure_table(overlap_data, out_csv=None):
    """
    Export each failed item with the list of models that failed it:
    - story_snippet, question_snippet
    - models (sorted and joined as a string)
    - num_models (number of models that failed)
    """
    if out_csv is None:
        out_csv = os.path.join(viz_dir, "failure_tasks_by_models.csv")

    rows = []
    for failure_key, models in overlap_data.items():
        story, question, options = failure_key

        rows.append(
            {
                "story_snippet": (story or "")[:80],
                "question_snippet": (question or "")[:80],
                "models": ", ".join(sorted(models)),
                "num_models": len(models),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved detailed failure table to: {out_csv}")
    return df


def export_intersection_table(overlap_data, out_csv=None):
    """
    Export (model combination → list of failed items):
    Each row represents a failed item, including:
    - model_combination: e.g., "Claude, gemini, gpt5.1"
    - story_snippet, question_snippet
    - num_models: number of models in the combination
    - intersection_size: total number of items in this combination
    """
    if out_csv is None:
        out_csv = os.path.join(viz_dir, "failure_intersections_detailed.csv")

    groups = {}  # model_combination -> list of failure keys

    for failure_key, models in overlap_data.items():
        comb = tuple(sorted(models))
        groups.setdefault(comb, []).append(failure_key)

    rows = []
    for comb, failures in groups.items():
        comb_str = ", ".join(comb)
        for failure_key in failures:
            story, question, options = failure_key
            rows.append(
                {
                    "model_combination": comb_str,
                    "story_snippet": (story or "")[:80],
                    "question_snippet": (question or "")[:80],
                    "num_models": len(comb),
                    "intersection_size": len(failures),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved detailed intersection table to: {out_csv}")
    return df


# -------------------------------------------------------------------
# 3. Auxiliary visualizations based on CSV
# -------------------------------------------------------------------
def plot_num_models_per_task(tasks_df: pd.DataFrame, out_pdf: str):
    """Plot the distribution of the number of models that failed each task."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="num_models", data=tasks_df)
    plt.xlabel("Number of models that failed this task")
    plt.ylabel("Number of tasks")
    plt.title("Distribution of how many models fail each task")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"Num-models-per-task distribution saved as PDF at: {out_pdf}")


def plot_top_model_combinations(inter_df: pd.DataFrame, out_pdf: str, top_k: int = 10):
    """Plot the top-k model combinations by intersection size."""
    comb_summary = (
        inter_df.groupby("model_combination")
        .agg(intersection_size=("intersection_size", "max"))
        .reset_index()
        .sort_values("intersection_size", ascending=False)
    )

    top_comb = comb_summary.head(top_k)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_comb, x="intersection_size", y="model_combination")
    plt.xlabel("Number of tasks in this intersection")
    plt.ylabel("Model combination")
    plt.title(f"Top {top_k} failure intersections by model combination")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"Top model combinations plot saved as PDF at: {out_pdf}")


def export_hard_cases(tasks_df: pd.DataFrame, inter_df: pd.DataFrame):
    """Export the hardest items and items failed by all models."""
    # 1) Sort by num_models to find the hardest tasks
    hard_tasks = tasks_df.sort_values("num_models", ascending=False).head(20)
    cols_to_show = ["num_models", "models", "story_snippet", "question_snippet"]
    hard_csv = os.path.join(viz_dir, "hard_tasks_top20.csv")
    hard_tasks[cols_to_show].to_csv(hard_csv, index=False)
    print(f"Top-20 hardest tasks saved to: {hard_csv}")

    # 2) Find items failed by all models: construct combination string automatically
    model_names = sorted(
        {os.path.basename(os.path.dirname(p)) for p in file_paths}
    )
    full_comb_str = ", ".join(model_names)

    all_fail_subset = inter_df[inter_df["model_combination"] == full_comb_str]
    if not all_fail_subset.empty:
        cols_to_show2 = [
            "model_combination",
            "num_models",
            "intersection_size",
            "story_snippet",
            "question_snippet",
        ]
        all_fail_csv = os.path.join(viz_dir, "failure_cases_all_models.csv")
        all_fail_subset[cols_to_show2].to_csv(all_fail_csv, index=False)
        print(f"All-model failure cases saved to: {all_fail_csv}")
    else:
        print("No items found where all models fail together.")


# -------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------
def main():
    # 1) Build aggregated error data (Model × Task)
    data = build_aggregated_error_data(file_paths)

    # 2) Plot task-level bar charts (faceted)
    task_bar_pdf = os.path.join(viz_dir, "task_error_analysis.pdf")
    plot_task_error_bars(data, task_bar_pdf)

    # 3) Plot model × Task heatmap
    heatmap_pdf = os.path.join(viz_dir, "heatmap_analysis.pdf")
    plot_error_heatmap(data, heatmap_pdf)

    # 4) Build item-level overlap + UpSet plot
    overlap_data = extract_failure_overlap(file_paths)
    upset_pdf = os.path.join(viz_dir, "overlap_analysis.pdf")
    plot_upset(overlap_data, upset_pdf)

    # 5) Export item-level CSVs
    tasks_df = export_failure_table(overlap_data, os.path.join(viz_dir, "failure_tasks_by_models.csv"))
    inter_df = export_intersection_table(
        overlap_data, os.path.join(viz_dir, "failure_intersections_detailed.csv")
    )

    # 6) Auxiliary visualizations based on CSVs
    num_models_pdf = os.path.join(viz_dir, "viz_num_models_per_task.pdf")
    plot_num_models_per_task(tasks_df, num_models_pdf)

    top_comb_pdf = os.path.join(viz_dir, "viz_top_model_combinations.pdf")
    plot_top_model_combinations(inter_df, top_comb_pdf, top_k=10)

    # 7) Export hardest items / items failed by all models
    export_hard_cases(tasks_df, inter_df)


if __name__ == "__main__":
    main()
