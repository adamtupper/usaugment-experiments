"""Split the Annotated Ultrasound Liver (AUL) dataset into training, validation, and test sets using a 7:1:2 split. The
dataset consists of 735 images from different patients.

Two versions of the dataset are created, one for liver segmentation and one for mass classification/segmentation. This
is because one of the examples (image 374.jpg) is missing a liver segmentation mask and is therefore excluded from the
liver segmentation task. For images 229.jpg and 306.jpg, the scan annotations actually correspond to the liver (they are
missing the scan annotations). See `exploratory_analysis/eda_aul.py` for more information.

Each example (a single image) is represented as an object in one of three JSON array files (`train.json`,
`validation.json`, or `test.json`). Each object has the following key/value pairs:
    - image: The path to the image file.
    - scan_mask:  The path to the scan mask file.
    - liver_mask: The path to the liver mask file.
    - mass_mask: The path to the mass mask file.
    - pathology: The pathology of the mass (normal/no mass, benign or malignant).
    - label: The pathology of the mass encoded as an integer (norma/no mass = 0, benign = 1, malignant = 2).

Usage:
    python prepare_aul.py
        --dataset_dir /path/to/dataset
        --output_dir /path/to/output
        --version N
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import skimage
import skimage.io as io
from sklearn.model_selection import train_test_split

OUTPUT_NAME_MASS = "aul_mass_v{}"
OUTPUT_NAME_LIVER = "aul_liver_v{}"
CLASS_TO_LABEL = {
    "Normal": 0,
    "Benign": 1,
    "Malignant": 2,
}


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="prepare_aul.py")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="The path to the original dataset directory",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory in which to save the processed dataset",
        required=True,
    )
    parser.add_argument(
        "--version",
        type=int,
        help="The version number to assign the processed dataset",
        required=True,
    )

    args = parser.parse_args()
    assert os.path.isdir(args.dataset_dir), "dataset_dir must be an existing directory"
    assert os.path.exists(args.output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(args.output_dir, OUTPUT_NAME_MASS.format(args.version))
    ), (
        "a version of the mass classification/segmentation dataset with this version number already exists"
    )
    assert not os.path.exists(
        os.path.join(args.output_dir, OUTPUT_NAME_LIVER.format(args.version))
    ), (
        "a version of the liver segmentation dataset with this version number already exists"
    )

    return args


def collect_examples(dataset_dir):
    examples = []
    for class_name in CLASS_TO_LABEL.keys():
        images = glob.glob(f"{dataset_dir}/{class_name}/image/*.jpg")
        for image_path in images:
            filename = os.path.basename(image_path).removesuffix(".jpg")
            scan_polygon_path = os.path.join(
                dataset_dir, class_name, "segmentation/outline", f"{filename}.json"
            )
            liver_polygon_path = os.path.join(
                dataset_dir, class_name, "segmentation/liver", f"{filename}.json"
            )
            mass_polygon_path = os.path.join(
                dataset_dir, class_name, "segmentation/mass", f"{filename}.json"
            )

            has_liver_polygon = os.path.exists(liver_polygon_path)
            has_mass_polygon = os.path.exists(mass_polygon_path)

            new_filename = f"{class_name.lower()}_{filename}"

            example = {
                "filename": new_filename,
                "image_path": image_path,
                "scan_polygon_path": scan_polygon_path,
                "liver_polygon_path": liver_polygon_path,
                "mass_polygon_path": mass_polygon_path,
                "pathology": class_name,
                "label": CLASS_TO_LABEL[class_name],
                "image": f"images/{new_filename}.jpg",
                "scan_mask": f"masks/scan/{new_filename}.png",
                "liver_mask": f"masks/liver/{new_filename}.png"
                if has_liver_polygon
                else None,
                "mass_mask": f"masks/mass/{new_filename}.png"
                if (has_mass_polygon or CLASS_TO_LABEL[class_name] == 0)
                else None,
            }

            # Correct the scan/liver annotations for images 229.jpg and 306.jpg
            if filename in ["229", "306"]:
                example["liver_polygon_path"] = example["scan_polygon_path"]
                example["liver_mask"] = f"masks/liver/{new_filename}.png"
                example["scan_polygon_path"] = None

            examples.append(example)

    return examples


def generate_scan_mask(image_path):
    """Generate a scan mask for the given image using morphological operations."""
    image = io.imread(image_path)

    # Threshold the image
    mask = image > 1

    # Morphological operations
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)
    mask = skimage.morphology.remove_small_objects(mask, min_size=1000)

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype("uint8")


def generate_mask(image_shape, polygon_file):
    """Generate a pixel mask for the polygon coordinates in the given file."""
    with open(polygon_file) as f:
        points = json.load(f)

    points = [[y, x] for x, y in points]
    mask = skimage.draw.polygon2mask(image_shape, points).astype("uint8")

    return mask


def copy_images(output_dir, df):
    """Save the images as PNG files in the output directory. 3-channel images are converted to single-channel images by
    using the mean intensity across the three channels.
    """
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for _, example in df.iterrows():
        source_filename = os.path.basename(example["image_path"])
        dest_filename = example["pathology"].lower() + "_" + source_filename

        image = io.imread(example["image_path"])
        if image.ndim == 3:
            image = image.mean(axis=-1).astype("uint8")

        io.imsave(os.path.join(image_dir, dest_filename), image, check_contrast=False)


def generate_masks(output_dir, df):
    """Generate and save the scan, liver, and mass masks for the examples."""
    scan_mask_dir = os.path.join(output_dir, "masks", "scan")
    liver_mask_dir = os.path.join(output_dir, "masks", "liver")
    mass_mask_dir = os.path.join(output_dir, "masks", "mass")
    os.makedirs(scan_mask_dir, exist_ok=True)
    os.makedirs(liver_mask_dir, exist_ok=True)
    os.makedirs(mass_mask_dir, exist_ok=True)

    for _, example in df.iterrows():
        image = io.imread(example["image_path"])

        if example["scan_polygon_path"]:
            scan_mask = generate_mask(image.shape[:2], example["scan_polygon_path"])
            scan_mask_path = os.path.join(scan_mask_dir, f"{example['filename']}.png")
            io.imsave(scan_mask_path, scan_mask, check_contrast=False)
        else:
            scan_mask = generate_scan_mask(example["image_path"])
            scan_mask_path = os.path.join(scan_mask_dir, f"{example['filename']}.png")
            io.imsave(scan_mask_path, scan_mask, check_contrast=False)

        if example["liver_mask"]:
            liver_mask = generate_mask(image.shape[:2], example["liver_polygon_path"])
            liver_mask_path = os.path.join(liver_mask_dir, f"{example['filename']}.png")
            io.imsave(liver_mask_path, liver_mask, check_contrast=False)

        if example["mass_mask"]:
            if os.path.exists(example["mass_polygon_path"]):
                mass_mask = generate_mask(image.shape[:2], example["mass_polygon_path"])
            else:
                # Create an empty mask for images with no mass
                mass_mask = np.zeros(image.shape[:2], dtype="uint8")
            mass_mask_path = os.path.join(mass_mask_dir, f"{example['filename']}.png")
            io.imsave(mass_mask_path, mass_mask, check_contrast=False)


def save_examples(output_dir, df, split):
    """Save the examples to a JSON file."""
    columns_to_drop = [
        "filename",
        "image_path",
        "scan_polygon_path",
        "liver_polygon_path",
        "mass_polygon_path",
    ]
    df = df.drop(columns=columns_to_drop)
    examples = df.to_dict(orient="records")
    with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
        json.dump(examples, f, indent=4)


def save_version_info(output_dir, version):
    """Save a file with the current date and commit hash."""
    with open(os.path.join(output_dir, "version.txt"), "w") as f:
        f.write(f"Version: {version}\n")
        f.write(f"Commit: {os.popen('git rev-parse HEAD').read().strip()}\n")
        f.write(f"Date: {os.popen('date').read().strip()}\n")


def main():
    """Prepare the training, validation, and test sets for the AUL mass classification/segmentation and liver
    segmentation datasets.
    """
    args = parse_args()

    # Collect examples
    examples = collect_examples(args.dataset_dir)
    df = pd.DataFrame.from_records(examples)

    print(f"Total number of examples: {len(df)}")

    # Separate the test set
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=df["label"]
    )

    # Separate the training and validation sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=train_val_df["label"],
    )

    # Create mass classification/segmentation dataset
    output_dir = os.path.join(args.output_dir, OUTPUT_NAME_MASS.format(args.version))
    for split, subset_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        copy_images(output_dir, subset_df)
        generate_masks(output_dir, subset_df)
        save_examples(output_dir, subset_df, split)
        save_version_info(output_dir, args.version)

        print(
            f"Mass classification/segmentation - Number of {split} examples: {len(subset_df)}"
        )

    # Create liver segmentation dataset
    output_dir = os.path.join(args.output_dir, OUTPUT_NAME_LIVER.format(args.version))
    for split, subset_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        subset_df = subset_df.dropna(subset=["liver_mask"])
        copy_images(output_dir, subset_df)
        generate_masks(output_dir, subset_df)
        save_examples(output_dir, subset_df, split)
        save_version_info(output_dir, args.version)

        print(f"Liver segmentation - Number of {split} examples: {len(subset_df)}")


if __name__ == "__main__":
    main()
