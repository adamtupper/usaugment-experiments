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
    "aul_liver_segmentation": "aul_liver_v5_liver_segmentation",
    "aul_mass_segmentation": "aul_mass_v5_mass_segmentation",
    "camus_segmentation": "camus_v4_segmentation",
    "mmotu_segmentation": "mmotu_v2_segmentation",
    "open_kidney_capsule_segmentation": "open_kidney_v3_capsule_segmentation",
    "psfhs_segmentation": "psfhs_v2_segmentation",
    "stanford_thyroid_segmentation": "stanford_thyroid_v4_segmentation",
}

# %%
# %%
# Combine the results on each task into a a single data frame
for i, task in enumerate(TASKS):
    results = pd.read_csv(
        os.path.join(RESULTS_DIR, f"trivial_augment/{TASKS[task]}_results.csv")
    )
    results["task"] = task
    results["num_augmentations"] = results["num_augmentations"] - 1
    if i == 0:
        segmentation_results = results
    else:
        segmentation_results = pd.concat([segmentation_results, results])

    # Add the results for the best single augmentation and no augmentation
    individual_results = pd.read_csv(
        os.path.join(RESULTS_DIR, f"individual/{TASKS[task]}_results.csv")
    )
    best = individual_results.groupby("augmentation")[
        "test/dice"].mean().idxmax()
    rows = individual_results[
        individual_results["augmentation"].isin([best, "identity"])
    ].copy()
    rows["task"] = task
    rows["num_augmentations"] = rows["augmentation"].apply(
        lambda x: 0 if x == "identity" else 1
    )
    rows.drop(columns=["augmentation"], inplace=True)
    segmentation_results = pd.concat([segmentation_results, rows])

# Calculate the mean and standard error of the mean for each metric (test/dice) for each task and number of augmentations
segmentation_results["num_augmentations"] = segmentation_results[
    "num_augmentations"
].astype("category")
segmentation_results["seed"] = segmentation_results["seed"].astype("category")
segmentation_results["task"] = segmentation_results["task"].astype("category")

# Group by task and number of augmentations and calculate the mean and standard error of the mean for each metric
segmentation_results = segmentation_results.drop(columns=["seed"])
segmentation_results = segmentation_results.groupby(["task", "num_augmentations"]).agg(
    {
        "test/loss": ["mean", "sem"],
        "test/dice": ["mean", "sem"],
    }
)
segmentation_results = segmentation_results.reset_index()
segmentation_results.columns = segmentation_results.columns = [
    "_".join(a).strip("_") for a in segmentation_results.columns.to_flat_index()
]
segmentation_results.round(3)

# %%
# Separate identity results from the rest of the results
identity_results = segmentation_results[segmentation_results["num_augmentations"] == 0]
identity_results = identity_results.drop(columns=["num_augmentations"])
identity_results.round(3)

# %%
# Separate the #1 augmentation results from the rest of the results
best_single_results = segmentation_results[
    segmentation_results["num_augmentations"] == 1
]
best_single_results = best_single_results.drop(columns=["num_augmentations"])
best_single_results.round(3)

# %%
# Calculate the difference between each number of augmentations and no augmentation
# (identity) for each metric (test/dice) for each task
results_df = segmentation_results.merge(
    identity_results, on="task", suffixes=("", "_identity")
)
results_df = results_df.merge(
    best_single_results, on="task", suffixes=("", "_no1"))

results_df["test/loss_diff"] = (
    results_df["test/loss_mean"] - results_df["test/loss_mean_identity"]
)
results_df["test/dice_diff"] = (
    results_df["test/dice_mean"] - results_df["test/dice_mean_identity"]
)
results_df["test/dice_percent_change"] = (
    results_df["test/dice_diff"] / results_df["test/dice_mean_identity"] * 100
)

# Calculate the differences for the best single augmentation
results_df["test/dice_diff_no1"] = (
    results_df["test/dice_mean"] - results_df["test/dice_mean_no1"]
)
results_df["test/dice_percent_change_no1"] = (
    results_df["test/dice_diff_no1"] / results_df["test/dice_mean_no1"] * 100
)

results_df.sort_index(axis=1).round(4)

# %%
summary_df = results_df[
    ["task", "num_augmentations"]
    + [col for col in results_df.columns if col.startswith("test/dice")]
]
summary_df = summary_df.drop(
    [
        "test/dice_mean_identity",
        "test/dice_sem_identity",
        "test/dice_mean_no1",
        "test/dice_sem_no1",
    ],
    axis=1,
)
summary_df = summary_df.rename(
    columns={
        "test/dice_mean": "Dice",
        "test/dice_sem": "Dice SE",
        "test/dice_diff": "Dice Diff",
        "test/dice_percent_change": "Dice % Change",
        "test/dice_diff_no1": "Dice Diff (No. 1)",
        "test/dice_percent_change_no1": "Dice % Change (No. 1)",
    }
)

# %%
task = "stanford_thyroid_segmentation"
summary_df[summary_df["task"] == task].sort_values(
    "num_augmentations", ascending=True
).reset_index(drop=True).round(3)

# %%
summary_df[["num_augmentations", "Dice % Change"]].groupby(
    "num_augmentations"
).mean().round(2)

# %%
# For each task, plot the dice for each number of augmentations using a bar chart
titles = {
    "aul_liver_segmentation": "AUL Liver",
    "aul_mass_segmentation": "AUL Mass",
    "camus_segmentation": "CAMUS",
    "mmotu_segmentation": "MMOTU",
    "open_kidney_capsule_segmentation": "Open Kidney",
    "psfhs_segmentation": "PSFHS",
    "stanford_thyroid_segmentation": "Stanford Thyroid",
}
colour_maps = {
    "aul_liver_segmentation": {
        0: "tab:blue",
        1: "tab:green",
        2: "tab:green",
        3: "tab:green",
        4: "tab:green",
        5: "tab:green",
        6: "tab:green",
        7: "tab:green",
        8: "tab:green",
        9: "tab:olive",
        10: "tab:olive",
        11: "tab:olive",
        12: "tab:red",
        13: "tab:red",
        14: "tab:red",
    },
    "aul_mass_segmentation": {
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
        10: "tab:red",
        11: "tab:red",
        12: "tab:red",
        13: "tab:red",
        14: "tab:red",
    },
    "camus_segmentation": {
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
        11: "tab:olive",
        12: "tab:red",
        13: "tab:red",
        14: "tab:red",
    },
    "mmotu_segmentation": {
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
        14: "tab:olive",
    },
    "open_kidney_capsule_segmentation": {
        0: "tab:blue",
        1: "tab:green",
        2: "tab:olive",
        3: "tab:olive",
        4: "tab:olive",
        5: "tab:olive",
        6: "tab:olive",
        7: "tab:olive",
        8: "tab:olive",
        9: "tab:red",
        10: "tab:red",
        11: "tab:red",
        12: "tab:red",
        13: "tab:red",
        14: "tab:red",
    },
    "psfhs_segmentation": {
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
        14: "tab:red",
    },
    "stanford_thyroid_segmentation": {
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
        11: "tab:olive",
        12: "tab:red",
        13: "tab:red",
        14: "tab:red",
    },
}
colour_to_marker = {
    "tab:blue": "o",
    "tab:green": "s",
    "tab:olive": "D",
    "tab:red": "*",
}

fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))

for i, task in enumerate(TASKS):
    subset_df = summary_df[summary_df["task"] == task]
    subset_df = subset_df.sort_values("num_augmentations")

    x_tick_labels = subset_df["num_augmentations"]
    x_ticks = np.arange(len(x_tick_labels))
    ax = axes[i // 4, i % 4]

    colour_map = colour_maps[task]
    for dice, se, x in zip(subset_df["Dice"], subset_df["Dice SE"], x_ticks):
        marker = colour_to_marker[colour_map[x]]
        ax.errorbar(
            x=x,
            y=dice,
            yerr=se,
            elinewidth=1,
            capsize=2,
            color=colour_map[x],
            fmt=colour_to_marker[colour_map[x]],
            markersize=7 if marker == "*" else 5
        )

    ax.set_title(titles[task])
    ax.grid(axis="both", linestyle="--", linewidth=0.5, which="both")

    ax.set_xlabel("Number of Augmentations")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        x_tick_labels,
        fontsize=8,
    )

    ax.set_ylabel("Dice")
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

axes[1, 3].axis("off")

plt.tight_layout()
plt.savefig("../../figures/trivial_augment_segmentation.pdf")
plt.show()

# %%
