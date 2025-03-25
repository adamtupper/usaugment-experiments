# %%
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")

RESULTS_DIR = "../../results/"
TASKS = {
    "aul_mass_classification": "aul_mass_v5_mass_classification",
    "camus_classification": "camus_v4_classification",
    "mmotu_classification": "mmotu_v2_classification",
}

# %%
# Combine the results on each task into a a single data frame
for i, task in enumerate(TASKS):
    for j, model in enumerate(
        [
            "efficientnetb0_multiclass",
            "efficientnetb5_multiclass",
            "mitb2_multiclass",
        ]
    ):
        results = pd.read_csv(
            os.path.join(
                RESULTS_DIR,
                f"trivial_augment/{TASKS[task]}_{model}_results.csv",
            )
        )
        results["task"] = task
        results["model"] = model
        results["num_augmentations"] = results["num_augmentations"] - 1
        if i == 0 and j == 0:
            classification_results = results
        else:
            classification_results = pd.concat(
                [classification_results, results]
            )

        # Add the results for the best single augmentation and no augmentation
        individual_results = pd.read_csv(
            os.path.join(
                RESULTS_DIR, f"individual/{TASKS[task]}_{model}_results.csv"
            )
        )
        best = (
            individual_results.groupby("augmentation")["test/avg_precision"]
            .mean()
            .idxmax()
        )
        rows = individual_results[
            individual_results["augmentation"].isin([best, "identity"])
        ].copy()
        rows["task"] = task
        rows["model"] = model
        rows["num_augmentations"] = rows["augmentation"].apply(
            lambda x: 0 if x == "identity" else 1
        )
        rows.drop(columns=["augmentation"], inplace=True)
        classification_results = pd.concat([classification_results, rows])

# Calculate the mean and standard error of the mean for each metric (test/precision, test/recall, test/f1,
# test/avg_precision, test/acc) for each task and number of augmentations
classification_results["num_augmentations"] = classification_results[
    "num_augmentations"
].astype("category")
classification_results["seed"] = classification_results["seed"].astype(
    "category"
)
classification_results["task"] = classification_results["task"].astype(
    "category"
)

# Group by task and number of augmentations and calculate the mean and standard error of the mean for each metric
classification_results = classification_results.drop(columns=["seed"])
classification_results = classification_results.groupby(
    ["task", "model", "num_augmentations"]
).agg(
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
    "_".join(a).strip("_")
    for a in classification_results.columns.to_flat_index()
]
classification_results.round(3)

# %%
# Separate identity results from the rest of the results
identity_results = classification_results[
    classification_results["num_augmentations"] == 0
]
identity_results = identity_results.drop(columns=["num_augmentations"])
identity_results.round(3)

# %%
# Separate the #1 augmentation results from the rest of the results
best_single_results = classification_results[
    classification_results["num_augmentations"] == 1
]
best_single_results = best_single_results.drop(columns=["num_augmentations"])
best_single_results.round(3)

# %%
# Calculate the difference between each number of augmentations and no augmentation
# (identity) for each metric (test/precision, test/recall, test/f1,
# test/avg_precision, test/acc) for each task
results_df = classification_results.merge(
    identity_results, on=["task", "model"], suffixes=("", "_identity")
)
results_df = results_df.merge(
    best_single_results, on=["task", "model"], suffixes=("", "_no1")
)

results_df["test/loss_diff"] = (
    results_df["test/loss_mean"] - results_df["test/loss_mean_identity"]
)
results_df["test/precision_diff"] = (
    results_df["test/precision_mean"]
    - results_df["test/precision_mean_identity"]
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
results_df["test/acc_diff"] = (
    results_df["test/acc_mean"] - results_df["test/acc_mean_identity"]
)
results_df["test/avg_precision_percent_change"] = (
    results_df["test/avg_precision_diff"]
    / results_df["test/avg_precision_mean_identity"]
    * 100
)

# Calculate the differences for the best single augmentation
results_df["test/avg_precision_diff_no1"] = (
    results_df["test/avg_precision_mean"]
    - results_df["test/avg_precision_mean_no1"]
)
results_df["test/avg_precision_percent_change_no1"] = (
    results_df["test/avg_precision_diff_no1"]
    / results_df["test/avg_precision_mean_no1"]
    * 100
)

results_df.sort_index(axis=1).round(4)

# %%
summary_df = results_df[
    ["task", "model", "num_augmentations"]
    + [
        col
        for col in results_df.columns
        if col.startswith("test/avg_precision")
    ]
]
summary_df = summary_df.drop(
    [
        "test/avg_precision_mean_identity",
        "test/avg_precision_sem_identity",
        "test/avg_precision_mean_no1",
        "test/avg_precision_sem_no1",
    ],
    axis=1,
)
summary_df = summary_df.rename(
    columns={
        "test/avg_precision_mean": "AP",
        "test/avg_precision_sem": "AP SE",
        "test/avg_precision_diff": "AP Diff",
        "test/avg_precision_percent_change": "AP % Change",
        "test/avg_precision_diff_no1": "AP Diff (No. 1)",
        "test/avg_precision_percent_change_no1": "AP % Change (No. 1)",
    }
)

# %%
task = "aul_mass_classification"
model = "efficientnetb5_multiclass"
summary_df[
    (summary_df["task"] == task) & (summary_df["model"] == model)
].sort_values("num_augmentations", ascending=True).reset_index(
    drop=True
).round(3)

# %%
summary_df[["num_augmentations", "AP % Change"]].groupby(
    "num_augmentations"
).mean().round(2)

# %%
# For each task, plot the AP for each number of augmentations using a bar chart
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
colour_schemes = {
    "aul_mass_classification": {
        "efficientnetb0_multiclass": {
            0: "tab:blue",
            1: "tab:green",
            2: "tab:green",
            3: "tab:green",
            4: "tab:green",
            5: "tab:green",
            6: "tab:green",
            7: "tab:green",
            8: "tab:green",
            9: "tab:green",
            10: "tab:green",
            11: "tab:green",
            12: "tab:olive",
            13: "tab:olive",
            14: "tab:olive",
        },
        "efficientnetb5_multiclass": {
            0: "tab:blue",
            1: "tab:green",
            2: "tab:green",
            3: "tab:green",
            4: "tab:olive",
            5: "tab:olive",
            6: "tab:olive",
            7: "tab:olive",
            8: "tab:olive",
            9: "tab:olive",
            10: "tab:olive",
            11: "tab:olive",
            12: "tab:red",
            13: "tab:olive",
            14: "tab:red",
        },
        "mitb2_multiclass": {
            0: "tab:blue",
            1: "tab:green",
            2: "tab:green",
            3: "tab:green",
            4: "tab:green",
            5: "tab:green",
            6: "tab:green",
            7: "tab:olive",
            8: "tab:olive",
            9: "tab:olive",
            10: "tab:olive",
            11: "tab:olive",
            12: "tab:olive",
            13: "tab:olive",
            14: "tab:olive",
        },
    },
    "camus_classification": {
        "efficientnetb0_multiclass": {
            0: "tab:blue",
            1: "tab:green",
            2: "tab:green",
            3: "tab:green",
            4: "tab:green",
            5: "tab:green",
            6: "tab:green",
            7: "tab:olive",
            8: "tab:olive",
            9: "tab:olive",
            10: "tab:olive",
            11: "tab:olive",
            12: "tab:red",
            13: "tab:red",
            14: "tab:red",
        },
        "efficientnetb5_multiclass": {
            0: "tab:blue",
            1: "tab:green",
            2: "tab:green",
            3: "tab:olive",
            4: "tab:olive",
            5: "tab:olive",
            6: "tab:olive",
            7: "tab:olive",
            8: "tab:olive",
            9: "tab:olive",
            10: "tab:olive",
            11: "tab:olive",
            12: "tab:olive",
            13: "tab:red",
            14: "tab:olive",
        },
        "mitb2_multiclass": {
            0: "tab:blue",
            1: "tab:green",
            2: "tab:olive",
            3: "tab:olive",
            4: "tab:olive",
            5: "tab:olive",
            6: "tab:olive",
            7: "tab:olive",
            8: "tab:olive",
            9: "tab:olive",
            10: "tab:olive",
            11: "tab:olive",
            12: "tab:olive",
            13: "tab:olive",
            14: "tab:red",
        },
    },
    "mmotu_classification": {
        "efficientnetb0_multiclass": {
            0: "tab:blue",
            1: "tab:green",
            2: "tab:green",
            3: "tab:green",
            4: "tab:green",
            5: "tab:green",
            6: "tab:green",
            7: "tab:green",
            8: "tab:green",
            9: "tab:green",
            10: "tab:green",
            11: "tab:green",
            12: "tab:green",
            13: "tab:green",
            14: "tab:red",
        },
        "efficientnetb5_multiclass": {
            0: "tab:blue",
            1: "tab:green",
            2: "tab:green",
            3: "tab:green",
            4: "tab:green",
            5: "tab:green",
            6: "tab:green",
            7: "tab:green",
            8: "tab:olive",
            9: "tab:olive",
            10: "tab:olive",
            11: "tab:olive",
            12: "tab:olive",
            13: "tab:olive",
            14: "tab:olive",
        },
        "mitb2_multiclass": {
            0: "tab:blue",
            1: "tab:green",
            2: "tab:green",
            3: "tab:green",
            4: "tab:green",
            5: "tab:green",
            6: "tab:green",
            7: "tab:green",
            8: "tab:green",
            9: "tab:green",
            10: "tab:olive",
            11: "tab:olive",
            12: "tab:olive",
            13: "tab:olive",
            14: "tab:olive",
        },
    },
}
colour_to_marker = {
    "tab:blue": "o",
    "tab:green": "s",
    "tab:olive": "D",
    "tab:red": "*",
}

fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey="col")

for i, task in enumerate(TASKS):
    for j, model in enumerate(
        [
            "efficientnetb0_multiclass",
            "efficientnetb5_multiclass",
            "mitb2_multiclass",
        ]
    ):
        subset_df = summary_df[
            (summary_df["task"] == task) & (summary_df["model"] == model)
        ]
        subset_df = subset_df.sort_values("num_augmentations")

        x_tick_labels = subset_df["num_augmentations"]
        x_ticks = np.arange(len(x_tick_labels))
        ax = axes[j, i % 3]

        colour_map = colour_schemes[task][model]
        for ap, se, x in zip(subset_df["AP"], subset_df["AP SE"], x_ticks):
            marker = colour_to_marker[colour_map[x]]
            ax.errorbar(
                x=x,
                y=ap,
                yerr=se,
                elinewidth=1,
                capsize=2,
                color=colour_map[x],
                fmt=marker,
                markersize=7 if marker == "*" else 5
            )

        ax.set_title(f"{titles[task]} ({models[model]})")
        ax.grid(axis="both", linestyle="--", linewidth=0.5, which="both")

        ax.set_xlabel("Number of Augmentations", fontsize=9)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            x_tick_labels,
            fontsize=8,
        )

        ax.set_ylabel("Avg. Precision", fontsize=9)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
plt.savefig(
    "../../figures/trivial_augment_for_different_models_classification.pdf"
)
plt.show()

# %%
