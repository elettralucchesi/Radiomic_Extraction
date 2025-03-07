import os
import glob


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

    files = glob.glob(os.path.join(path, '*.nii'))

    if not files:
        raise ValueError("The directory is empty or contains no .nii files")

    img = [f for f in files if not f.endswith('seg.nii')]
    mask = [f for f in files if f.endswith('seg.nii')]

    if len(img) != len(mask):
        raise ValueError("The number of image files does not match the number of mask files")

    return img, mask