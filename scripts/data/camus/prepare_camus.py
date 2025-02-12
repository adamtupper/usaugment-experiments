"""Split the CAMUS dataset into training, validation, and test sets using a 7:1:2 split, ensuring that there is no
patient overlap between the splits.

The images and masks are extracted from the NIfTI files and saved as 8-bit PNG files. The
corresponding metadata for each sequence is the ".cfg" files.

Each example (a single image) is represented as an object in one of three JSON array files (`train.json`,
`validation.json`, or `test.json`). Each object has the following key/value pairs:
    - patient:          The patient ID.
    - view:             The view of the sequence (2CH or 4CH).
    - frame:            The frame number of the image in the sequence.
    - sex:              The sex of the patient.
    - age:              The age of the patient.
    - image_quality:    The image quality of the image.
    - EF:               The ejection fraction.
    - frame_rate:       The frame rate of the sequence.
    - image:            The path to the image file.
    - mask:             The path to the mask file.
    - label:            The quality of the image encoded as an integer (poor = 0, medium = 1, good = 2).

Usage:
    python prepare_camus.py
        --dataset_dir /path/to/dataset
        --output_dir /path/to/output
        --version N
"""

import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
import skimage
import yaml
from sklearn.model_selection import GroupShuffleSplit

OUTPUT_NAME = "camus_v{}"
CLASS_TO_LABEL = {
    "Poor": 0,
    "Medium": 1,
    "Good": 2,
}


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="prepare_camus.py")
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
        os.path.join(args.output_dir, OUTPUT_NAME.format(args.version))
    ), "a version of the dataset with this version number already exists"

    return args


def segment_fan(image: np.ndarray):
    # Threshold the image
    mask = image > 0

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


def extract_images_and_masks(dataset_dir, output_dir):
    """Extract the images and masks from the NIfTI files and save them as 8-bit PNG files."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks", "scan"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks", "structures"), exist_ok=True)

    for nifti_file in glob.glob(
        os.path.join(dataset_dir, "database_nifti", "**", "*_half_sequence*.nii.gz")
    ):
        images = nib.load(nifti_file).get_fdata().astype(np.uint8)
        patient, view = nifti_file.split("/")[-1].split("_")[:2]
        for i in range(images.shape[2]):
            image = np.rot90(images[:, :, i], axes=(1, 0)).astype(
                np.uint8
            )  # Rotate the image 90 degrees clockwise
            output_filename = f"{patient}_view{view}_frame{i + 1}.png"
            if "_gt" in nifti_file:
                skimage.io.imsave(
                    os.path.join(output_dir, "masks", "structures", output_filename),
                    image,
                    check_contrast=False,
                )
            else:
                # Generate the scan mask
                scan_mask = segment_fan(image)
                skimage.io.imsave(
                    os.path.join(output_dir, "masks", "scan", output_filename),
                    scan_mask,
                    check_contrast=False,
                )

                # Save the image
                skimage.io.imsave(
                    os.path.join(output_dir, "images", output_filename),
                    image,
                    check_contrast=False,
                )


def extract_metadata(dataset_dir):
    """Extract the metadata from the ".cfg" files."""
    metadata = []
    for file_path in glob.glob(
        os.path.join(dataset_dir, "database_nifti", "**", "*.cfg")
    ):
        patient_id, filename = file_path.split("/")[-2:]
        patient_id = int(patient_id.removeprefix("patient"))
        view = filename.removesuffix(".cfg").split("_")[-1]

        with open(file_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        for i in range(config["NbFrame"]):
            frame = i + 1
            filename = f"patient{patient_id:04d}_view{view}_frame{frame}.png"
            row = {
                "patient": patient_id,
                "view": view,
                "frame": frame,
                "sex": config["Sex"],
                "age": config["Age"],
                "image_quality": config["ImageQuality"],
                "EF": config["EF"],
                "frame_rate": config["FrameRate"],
                "image": f"images/{filename}",
                "mask": f"masks/structures/{filename}",
                "scan_mask": f"masks/scan/{filename}",
                "label": CLASS_TO_LABEL[config["ImageQuality"]],
            }
            row["ED"] = True if config["ED"] == frame else False
            row["ES"] = True if config["ES"] == frame else False
            metadata.append(row)

    df = pd.DataFrame(metadata)

    return df


def main():
    """Divide the CAMUS dataset into training, validation, and test splits."""
    args = parse_args()

    output_dir = os.path.join(args.output_dir, OUTPUT_NAME.format(args.version))

    # Extract clinical metadata
    metadata = extract_metadata(args.dataset_dir)

    # Extract and save images and masks
    extract_images_and_masks(args.dataset_dir, output_dir)

    # Split the dataset into training, validation, and test sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_indices, test_indices = next(
        splitter.split(X=metadata, groups=metadata["patient"])
    )
    train_val_examples = metadata.iloc[train_val_indices]
    test_examples = metadata.iloc[test_indices]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_indices, val_indices = next(
        splitter.split(X=train_val_examples, groups=train_val_examples["patient"])
    )
    train_examples = train_val_examples.iloc[train_indices]
    val_examples = train_val_examples.iloc[val_indices]

    # Save the training, validation, and test examples as JSON files
    for split, examples in [
        ("train", train_examples),
        ("validation", val_examples),
        ("test", test_examples),
    ]:
        file_path = os.path.join(output_dir, f"{split}.json")
        with open(file_path, "w") as f:
            json.dump(examples.to_dict(orient="records"), f, indent=4)

    # Save a file with the current date and commit hash
    with open(os.path.join(output_dir, "version.txt"), "w") as f:
        f.write(f"Version: {args.version}\n")
        f.write(f"Commit: {os.popen('git rev-parse HEAD').read().strip()}\n")
        f.write(f"Date: {os.popen('date').read().strip()}\n")


if __name__ == "__main__":
    main()
