import os
import glob
import re
import warnings


def get_path_images_masks(path):
    """
    Retrieve file paths for images and masks from a specified directory.

    This function scans a given directory for `.nii` files and categorizes them
    into image files and mask files based on their filenames. Image files are
    identified as those without 'seg' in their names, while mask files contain 'seg'.
    The function ensures that the number of image files matches the number of mask files.

    Parameters
    ----------
    path : str
        Path to the directory containing `.nii` image and mask files.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing two lists:
        - The first list includes paths to image files (excluding 'seg' in their name).
        - The second list includes paths to mask files (containing 'seg' in their name).

    Raises
    ------
    TypeError
        If `path` is not a string.
    ValueError
        If the directory is empty or contains no `.nii` files.
        If the number of image files does not match the number of mask files.
    """

    if not isinstance(path, str):
        raise TypeError("Path must be a string")

    files = glob.glob(os.path.join(path, "*.nii"))

    if not files:
        raise ValueError("The directory is empty or contains no .nii files")

    img = [f for f in files if not f.endswith("seg.nii")]
    mask = [f for f in files if f.endswith("seg.nii")]

    if len(img) != len(mask):
        raise ValueError(
            "The number of image files does not match the number of mask files"
        )

    return img, mask


def extract_id(path):
    """
    Extract the patient ID from the file name.

    This function retrieves the patient ID from a given file path by searching for
    occurrences of the pattern 'PR<number>' in the file name. If multiple IDs are found,
    only the first occurrence is used. If the format is incorrect
    or no ID is found, a warning is issued, and `None` is returned.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    int or None
        The extracted patient ID as an integer if found and correctly formatted;
        otherwise, `None`.

    Raises
    ------
    TypeError
        If `path` is not a string.
        
    Warns
    -----
    UserWarning
        If the filename contains 'PR' with invalid numeric format (e.g., 'PRabc')
        If no 'PR' pattern is found in the filename
    """

    if not isinstance(path, str):
        raise TypeError("Path must be a string")

    filename = os.path.basename(path)
    matches = re.findall(r"PR(\d+)", filename)  # Find all occurrences of "PR<number>"

    if matches:
        return int(matches[0])

    if "PR" in filename:
        warnings.warn(
            f"Invalid patient ID format in file name '{filename}'. Expected 'PR<number>', e.g., 'PR2'. The ID will be automatically assigned.",
            category=UserWarning,
        )
        return None
    else:
        warnings.warn(
            f"No valid patient ID found in file name '{filename}'. Expected format: 'PR<number>', e.g., 'PR2'. The ID will be automatically assigned.",
            category=UserWarning,
        )

    return None


def new_patient_id(patients_id):
    """
    Generate a new unique patient ID.

    This function finds the first available positive integer that is not already
    present in the given set of patient IDs, ensuring uniqueness.

    Parameters
    ----------
    patients_id : set[int]
        Set of existing patient IDs.

    Returns
    -------
    int
        A new unique patient ID.

    Raises
    ------
    TypeError
        If `patients_id` is not a set.
        If any element in `patients_id` is not an integer.
    ValueError
        If any patient ID is negative.
    """
    if not isinstance(patients_id, set):
        raise TypeError("patients_id must be a set")

    if any(not isinstance(i, int) for i in patients_id):
        raise TypeError("All patient IDs must be integers")

    if any(i < 0 for i in patients_id):
        raise ValueError("Patient IDs cannot be negative")

    new_id = 1
    while new_id in patients_id:
        new_id += 1

    return new_id


def assign_patient_ids(images_path):
    """
    Assign patient IDs based on image file paths.

    This function extracts patient IDs from file names using `extract_id()`. If no valid
    ID is found, a new unique ID is generated using `new_patient_id()`. The function
    ensures that each patient is assigned a unique identifier.

    Parameters
    ----------
    images_path : list[str]
        List of file paths to the image files.

    Returns
    -------
    set[int]
        Set of assigned patient IDs.

    Raises
    ------
    TypeError
        If `images_path` is not a list.
    ValueError
        If `images_path` is empty.
    """
    if not isinstance(images_path, list):
        raise TypeError("images_path must be a list")

    if not images_path:
        raise ValueError("The list of image paths cannot be empty")

    patient_ids = set()
    for im_path in images_path:
        patient_id = extract_id(im_path)
        if patient_id is None:
            patient_id = new_patient_id(patient_ids)
            print(
                f"Patient ID not found, automatically assigning new ID, for {im_path} id {patient_id}"
            )
        patient_ids.add(patient_id)

    return patient_ids
