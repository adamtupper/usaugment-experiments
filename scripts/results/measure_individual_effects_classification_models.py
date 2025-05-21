# %%
# Setup
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = "../../results/individual"
CLASSIFICATION_TASKS = {
    "aul_mass_classification": "aul_mass_v5_mass_classification",
    # "butterfly_classification": "butterfly_v4_classification",
    "camus_classification": "camus_v4_classification",
    # "fatty_liver_classification": "fatty_liver_v3_classification",
    # "gbcu_classification": "gbcu_v2_classification",
    "mmotu_classification": "mmotu_v2_classification",
    # "pocus_classification": "pocus_v4_classification",
}

# %%
# Combine the classification results from each class into a a single data frame
classification_results = None
for task in CLASSIFICATION_TASKS:
    for model in ["efficientnetb0_multiclass", "efficientnetb5_multiclass", "mitb2_multiclass"]:
        results = pd.read_csv(
            os.path.join(
                RESULTS_DIR, f"{CLASSIFICATION_TASKS[task]}_{model}_results.csv")
        )
        results["task"] = task
        results["model"] = model
        if classification_results is None:
            classification_results = results
        else:
            classification_results = pd.concat(
                [classification_results, results])

# Calculate the mean and standard error of the mean for each metric (test/precision, test/recall, test/f1,
# test/avg_precision, test/acc) for each task and augmentation
classification_results["augmentation"] = classification_results["augmentation"].astype(
    "category"
)
classification_results["seed"] = classification_results["seed"].astype(
    "category")
classification_results["task"] = classification_results["task"].astype(
    "category")
classification_results["model"] = classification_results["model"].astype(
    "category")

# Group by task and augmentation and calculate the mean and standard error of the mean for each metric
classification_results = classification_results.drop(columns=["seed"])
classification_results = classification_results.groupby(["task", "model", "augmentation"]).agg(
    {
        "test/loss": ["mean", "sem"],
        "test/acc": ["mean", "sem"],
        "test/precision": ["mean", "sem"],
        "test/recall": ["mean", "sem"],
        "test/f1": ["mean", "sem"],
        "test/avg_precision": ["mean", "sem"],
    }
)
classification_results = classification_results.reset_index()
classification_results.columns = classification_results.columns = [
    "_".join(a).strip("_") for a in classification_results.columns.to_flat_index()
]
classification_results.round(3)

# %%
# Separate identity results from the rest of the results
identity_results = classification_results[
    classification_results["augmentation"] == "identity"
]
identity_results = identity_results.drop(columns=["augmentation"])
identity_results.round(3)

# %%
# Calculate the difference between each augmentation and no augmentation (identity) for each metric (test/precision,
# test/recall, test/f1, test/avg_precision) for each task
results_df = classification_results.merge(
    identity_results, on=["task", "model"], suffixes=("", "_identity")
)
results_df["test/loss_diff"] = (
    results_df["test/loss_mean"] - results_df["test/loss_mean_identity"]
)
results_df["test/acc_diff"] = (
    results_df["test/acc_mean"] - results_df["test/acc_mean_identity"]
)
results_df["test/precision_diff"] = (
    results_df["test/precision_mean"] -
    results_df["test/precision_mean_identity"]
)
results_df["test/recall_diff"] = (
    results_df["test/recall_mean"] - results_df["test/recall_mean_identity"]
)
results_df["test/f1_diff"] = (
    results_df["test/f1_mean"] - results_df["test/f1_mean_identity"]
)
results_df["test/avg_precision_diff"] = (
    results_df["test/avg_precision_mean"]
    - results_df["test/avg_precision_mean_identity"]
)
results_df["test/avg_precision_percent_change"] = (
    results_df["test/avg_precision_diff"]
    / results_df["test/avg_precision_mean_identity"]
    * 100
)
results_df.sort_index(axis=1).round(4)

# %%
summary_df = results_df[
    ["task", "augmentation", "model"]
    + [col for col in results_df.columns if col.startswith("test/avg_precision")]
]
summary_df = summary_df.drop(
    ["test/avg_precision_mean_identity", "test/avg_precision_sem_identity"], axis=1
)
summary_df = summary_df.rename(
    columns={
        "test/avg_precision_mean": "AP",
        "test/avg_precision_sem": "AP SE",
        "test/avg_precision_diff": "AP Diff",
        "test/avg_precision_percent_change": "AP % Change",
    }
)

# %%
task = "aul_mass_classification"
model = "efficientnetb5_multiclass"
summary_df[(summary_df["task"] == task) & (summary_df["model"] == model)].sort_values("AP", ascending=False).reset_index(
    drop=True
).round(3)

# %%
summary_df[["augmentation", "AP % Change"]].groupby(
    "augmentation").mean().round(2)

# %%
# For each task, plot the average precision for each augmentation using a bar chart
sns.set_theme(style="whitegrid", context="paper")

models = {
    "efficientnetb0_multiclass": "EfficientNet-B0",
    "efficientnetb5_multiclass": "EfficientNet-B5",
    "mitb2_multiclass": "MiT-B2",
}
titles = {
    "aul_mass_classification": "AUL Mass",
    "butterfly_classification": "Butterfly",
    "camus_classification": "CAMUS",
    "fatty_liver_classification": "Fatty Liver",
    "gbcu_classification": "GBCU",
    "mmotu_classification": "MMOTU",
    "pocus_classification": "POCUS",
}

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey="col")

for i, task in enumerate(CLASSIFICATION_TASKS):
    for j, model in enumerate(["efficientnetb5_multiclass", "mitb2_multiclass"]):
        subset_df = summary_df[(summary_df["task"] == task)
                               & (summary_df["model"] == model)]
        subset_df = subset_df.sort_values("AP")
        identity_results = subset_df[subset_df["augmentation"] == "identity"]

        x_tick_labels = subset_df["augmentation"]
        x_ticks = np.arange(len(x_tick_labels))
        ax = axes[j, i % 3]
        ax.errorbar(
            x=x_ticks,
            y=subset_df["AP"],
            yerr=subset_df["AP SE"],
            fmt="o",
            elinewidth=1,
            capsize=2,
        )

        ax.fill_between(
            (-0.5, len(x_ticks) - 0.5),
            identity_results["AP"].values[0] +
            identity_results["AP SE"].values[0],
            identity_results["AP"].values[0] -
            identity_results["AP SE"].values[0],
            color="tab:grey",
            alpha=0.2,
        )

        ax.set_title(f"{titles[task]} ({models[model]})")
        ax.grid(axis="both", linestyle="--", linewidth=0.5, which="both")

        ax.set_xlabel("")
        ax.set_xticks(x_ticks)
        ax.set_xlim(-0.5, len(x_ticks) - 0.5)
        ax.set_xticklabels(
            [
                x.replace("_", " ")
                .replace("identity", "none")
                .replace("bilateral filter", "speckle reduction")
                .capitalize()
                for x in x_tick_labels
            ],
            rotation=90,
            fontsize=8,
        )

        ax.set_ylabel("mAP")
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
plt.savefig(
    "../../figures/individual_effects_for_different_models_classification.pdf")
plt.show()

# %%
