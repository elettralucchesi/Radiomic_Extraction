from scipy.ndimage import label
import SimpleITK as sitk
import os
import numpy as np


def extract_largest_region(mask_slice, label_value):
    """
    Extract the largest connected region of a given label from a binary mask slice.

    This function identifies and extracts the largest connected component 
    corresponding to a specified label in a 2D mask slice. The extracted region 
    retains the original label value, while all other regions are set to zero.

    Parameters
    ----------
    mask_slice : np.ndarray
        A 2D array representing the mask slice.
    label_value : int
        The integer label whose largest connected region should be extracted.

    Returns
    -------
    np.ndarray or None
        A 2D array containing only the largest connected region with the given label,
        or None if no such region is found.

    Raises
    ------
    TypeError
        If `mask_slice` is not a numpy array.
        If `mask_slice` is an integer and `label_value` is a numpy array (possible swap).
        If `label_value` is not an integer.
    ValueError
        If `mask_slice` is not a 2D array.
        If `label_value` is negative.
    """

    if isinstance(mask_slice, int) and isinstance(label_value, np.ndarray):
        raise TypeError(
            "Inputs appear to be swapped. Expected mask_slice as a numpy array and label_value as an integer."
        )
    if not isinstance(mask_slice, np.ndarray):
        raise TypeError("mask_slice must be a numpy array")
    if mask_slice.ndim != 2:
        raise ValueError("mask_slice must be a 2D array")

    if not isinstance(label_value, int):
        raise TypeError("Label value must be an integer")
    if label_value < 0:
        raise ValueError("Label value cannot be negative")

    # Create a binary mask for the specified label
    region_mask = mask_slice == label_value

    # Label the connected components in the binary mask
    labeled_region, num_labels = label(region_mask)

    largest_region = None
    largest_area = 0

    for region_id in range(1, num_labels + 1):
        region = (labeled_region == region_id).astype(mask_slice.dtype) * label_value
        region_area = np.sum(region > 0)

        if region_area > largest_area:
            largest_area = region_area
            largest_region = region

    return largest_region


def process_slice(mask_slice):
    """
    Extract the largest connected region for each label in a mask slice.

    This function iterates through all unique labels in a given 2D mask slice, 
    excluding the background (label 0), and extracts the largest connected region 
    for each label. The first non-empty largest region found is returned along 
    with its corresponding label.

    Parameters
    ----------
    mask_slice : np.ndarray
        A 2D array representing the mask slice.

    Returns
    -------
    tuple[np.ndarray or None, int or None]
        A tuple containing the largest region mask and its corresponding label.
        If no valid region is found, returns (None, None).

    Raises
    ------
    TypeError
        If `mask_slice` is not a numpy array.
    ValueError
        If `mask_slice` is not a 2D array.
    """
    
    if not isinstance(mask_slice, np.ndarray):
        raise TypeError("mask_slice must be a numpy array")
    if mask_slice.ndim != 2:
        raise ValueError("mask_slice must be a 2D array")

    labels = np.unique(mask_slice)
    labels = labels[labels != 0]

    for lbl in labels:
        lbl = int(lbl)  # Convert numpy.int16 to native Python int
        largest_region_mask = extract_largest_region(mask_slice, lbl)
        if largest_region_mask is not None:
            return largest_region_mask, lbl

    # If no region found
    return None, None


def get_slices_2D(image, mask, patient_id):
    """
    Extract 2D slices from a 3D medical image and its corresponding mask.

    This function iterates through all slices of a given 3D image and mask, 
    extracts the largest connected region for each label, and returns relevant 
    metadata for each valid slice.

    Parameters
    ----------
    image : sitk.Image
        The 3D medical image.
    mask : sitk.Image
        The corresponding 3D segmentation mask.
    patient_id : int
        Unique identifier of the patient.

    Returns
    -------
    list[dict]
        A list of dictionaries, where each entry contains:
        - 'PatientID' (str): The patient identifier formatted as 'PR<number>'.
        - 'Label' (int): The extracted region label.
        - 'SliceIndex' (int): The index of the slice in the 3D volume.
        - 'ImageSlice' (sitk.Image): The extracted 2D image slice.
        - 'MaskSlice' (sitk.Image): The extracted 2D mask slice.

    Raises
    ------
    TypeError
        If `image` or `mask` is not a SimpleITK Image.
    ValueError
        If `patient_id` is not an integer.
    """

    if not isinstance(image, sitk.Image):
        raise TypeError(
            f"Expected 'image' to be a SimpleITK Image, but got {type(image)}."
        )

    if not isinstance(mask, sitk.Image):
        raise TypeError(
            f"Expected 'mask' to be a SimpleITK Image, but got {type(mask)}."
        )

    if not isinstance(patient_id, int):
        raise ValueError(
            f"Expected 'patient_id' to be a int, but got {type(patient_id)}."
        )

    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    patient_slices = []

    for slice_idx in range(mask_array.shape[0]):
        mask_slice = mask_array[slice_idx, :, :]
        image_slice = image_array[slice_idx, :, :]

        new_mask_slice, mask_label = process_slice(mask_slice)
        if new_mask_slice is None:
            continue

        new_mask_slice_image = sitk.GetImageFromArray(new_mask_slice)
        image_slice_image = sitk.GetImageFromArray(image_slice)
        patient_slices.append(
            {
                "PatientID": f"PR{patient_id}",
                "Label": mask_label,
                "SliceIndex": slice_idx,
                "ImageSlice": image_slice_image,
                "MaskSlice": new_mask_slice_image,
            }
        )

    return patient_slices


def get_patient_3D_data(image, mask, patient_id):
    """
    Retrieve the full 3D image and segmentation mask for a given patient.

    This function extracts the entire 3D volume of a medical image and its corresponding 
    segmentation mask, returning structured metadata for a specific patient.

    Parameters
    ----------
    image : sitk.Image
        The 3D medical image.
    mask : sitk.Image
        The corresponding 3D segmentation mask.
    patient_id : int
        Unique identifier of the patient.

    Returns
    -------
    list[dict]
        A list containing a single dictionary with:
        - 'PatientID' (str): The patient identifier formatted as 'PR<number>'.
        - 'ImageVolume' (sitk.Image): The full 3D medical image.
        - 'MaskVolume' (sitk.Image): The full 3D segmentation mask.

    Raises
    ------
    TypeError
        If `image` or `mask` is not a SimpleITK Image.
    ValueError
        If `patient_id` is not an integer.
    """
    
    if not isinstance(image, sitk.Image):
        raise TypeError("Expected 'image' to be a SimpleITK Image.")

    if not isinstance(mask, sitk.Image):
        raise TypeError("Expected 'mask' to be a SimpleITK Image.")

    if not isinstance(patient_id, int):
        raise ValueError("Expected 'patient_id' to be a int.")

    return [{"PatientID": f"PR{patient_id}", "ImageVolume": image, "MaskVolume": mask}]


def read_image_and_mask(image_path, mask_path):
    """
    Load a medical image and its corresponding segmentation mask from disk.

    This function reads a medical image and its associated mask using SimpleITK, ensuring 
    that they are in the same directory and have matching dimensions.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    mask_path : str
        Path to the mask file.

    Returns
    -------
    tuple[sitk.Image, sitk.Image]
        A tuple containing:
        - The image as a SimpleITK Image.
        - The corresponding mask as a SimpleITK Image.

    Raises
    ------
    ValueError
        If any input path is empty.
        If the image and mask are not located in the same directory.
        If the image and mask dimensions do not match.
    TypeError
        If the provided paths are not strings.
    """
    
    if not image_path or not mask_path:
        raise ValueError("Image and mask paths cannot be empty.")

    if not isinstance(image_path, str) or not isinstance(mask_path, str):
        raise TypeError("Image and mask paths must be strings.")

    if os.path.dirname(image_path) != os.path.dirname(mask_path):
        raise ValueError("Image and mask must be in the same directory.")

    img = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    if img.GetSize() != mask.GetSize():
        raise ValueError("Image and mask dimensions do not match.")

    return img, mask


def get_patient_image_mask_dict(imgs_path, masks_path, patient_ids, mode):
    """
    Generate a dictionary mapping patient IDs to their corresponding image-mask data.

    This function reads medical images and segmentation masks, associating them 
    with patient IDs and processing them as either 2D slices or full 3D volumes.

    Parameters
    ----------
    imgs_path : list[str]
        List of file paths to image files.
    masks_path : list[str]
        List of file paths to mask files.
    patient_ids : set[int]
        Set of unique patient IDs.
    mode : str
        Processing mode, either '2D' (for extracting slices) or '3D' (for full volumes).

    Returns
    -------
    dict[int, list[dict]]
        A dictionary where each key is a patient ID, and the value is:
        - A list of 2D slice dictionaries (if mode="2D").
        - A list containing a single dictionary with the full 3D volume (if mode="3D").

    Raises
    ------
    ValueError
        If any of `imgs_path`, `masks_path`, or `patient_ids` is empty.
        If the lengths of `imgs_path`, `masks_path`, and `patient_ids` do not match.
        If `mode` is not '2D' or '3D'.
    TypeError
        If `imgs_path` or `masks_path` are not lists of strings.
        If `patient_ids` is not a set of integers.
    """

    if not isinstance(imgs_path, list) or not all(
        isinstance(path, str) for path in imgs_path
    ):
        raise TypeError("imgs_path must be a list of strings.")
    if not isinstance(masks_path, list) or not all(
        isinstance(path, str) for path in masks_path
    ):
        raise TypeError("masks_path must be a list of strings.")
    if not isinstance(patient_ids, set) or not all(
        isinstance(pid, int) for pid in patient_ids
    ):
        raise TypeError("patient_ids must be a set of integers.")

    if len(imgs_path) == 0 or len(masks_path) == 0 or len(patient_ids) == 0:
        raise ValueError("The imgs_path, masks_path, and patient_ids cannot be empty.")

    if len(imgs_path) != len(masks_path) or len(imgs_path) != len(patient_ids):
        raise ValueError(
            "The number of images, masks, and patient_ids must be the same."
        )

    patient_dict = {}

    for pr_id, img_path, mask_path in zip(patient_ids, imgs_path, masks_path):
        img, mask = read_image_and_mask(img_path, mask_path)

        if mode == "2D":
            patient_slices = get_slices_2D(img, mask, pr_id)
            patient_dict[pr_id] = patient_slices
        elif mode == "3D":
            patient_volume = get_volume_3D(img, mask, pr_id)
            patient_dict[pr_id] = patient_volume
        else:
            raise ValueError("Mode should be '2D' or '3D'")

    return patient_dict
