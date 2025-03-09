from scipy.ndimage import label
import SimpleITK as sitk
import os
import numpy as np


def extract_largest_region(mask_slice, label_value):
    """
    Extract the largest connected region of a given label from a binary mask slice.

    GIVEN
    -----
    mask_slice : np.ndarray
        2D array representing the mask slice.
    label_value : int
        Integer label whose largest region should be extracted.

    WHEN
    ----
    The function is called with a valid mask slice and label.

    THEN
    ----
    Returns a 2D array containing only the largest connected region of the given label.

    Raises
    ------
    TypeError
        If `mask_slice` is not a numpy array.
        If `mask_slice` is an integer and `label_value` is a numpy array.
        If `label_value` is not an integer.
    ValueError
        If `label_value` is negative.
    """

    if isinstance(mask_slice, int) and isinstance(label_value, np.ndarray):
        raise TypeError(
            "Inputs appear to be swapped. Expected mask_slice as a numpy array and label_value as an integer.")
    if not isinstance(mask_slice, np.ndarray):
        raise TypeError("mask_slice must be a numpy array")
    if not isinstance(label_value, int):
        raise TypeError("Label value must be an integer")
    if label_value < 0:
        raise ValueError("Label value cannot be negative")

    # Create a binary mask for the specified label
    region_mask = (mask_slice == label_value)

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
    Process a mask slice to extract the largest connected region for each label.

    GIVEN
    -----
    mask_slice : np.ndarray
        2D array representing the mask slice.

    WHEN
    ----
    The function iterates through all labels in the mask.

    THEN
    ----
    Returns a tuple (largest_region_mask, label) containing the largest region found.
    
        Raises
    ------
    TypeError: If mask_slice is not a numpy array.
    ValueError: If mask_slice is not 2D or contains only invalid values.
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
    Extract 2D slices of an image and its corresponding mask.

    GIVEN
    -----
    image : sitk.Image
        The 3D medical image.
    mask : sitk.Image
        The corresponding 3D segmentation mask.
    patient_id : int
        Unique identifier of the patient.

    WHEN
    ----
    The function processes each slice of the mask.

    THEN
    ----
    Returns a list of dictionaries, each containing:
        - 'PatientID': Patient identifier.
        - 'Label': Extracted region label.
        - 'SliceIndex': Slice index in the volume.
        - 'ImageSlice': Image slice in SimpleITK format.
        - 'MaskSlice': Mask slice in SimpleITK format.

    Raises
    ------
    TypeError
        If `image` or `mask` is not a SimpleITK Image.
    ValueError
        If `patient_id` is not an integer.
    """

    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected 'image' to be a SimpleITK Image, but got {type(image)}.")

    if not isinstance(mask, sitk.Image):
        raise TypeError(f"Expected 'mask' to be a SimpleITK Image, but got {type(mask)}.")

    if not isinstance(patient_id, int):
        raise ValueError(f"Expected 'patient_id' to be a int, but got {type(patient_id)}.")

    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    patient_slices = []

    for slice_idx in range(mask_array.shape[0]):
        mask_slice = mask_array[slice_idx, :, :]
        image_slice = image_array[slice_idx, :, :]

        region_mask, region_label = process_slice(mask_slice)
        if region_mask is None:
            continue

        largest_region_mask_image  = sitk.GetImageFromArray(region_mask)
        image_slice_image = sitk.GetImageFromArray(image_slice)
        patient_slices.append({
            'PatientID': f"PR{patient_id}",
            'Label': region_label,
            'SliceIndex': slice_idx,
            'ImageSlice':  image_slice_image,
            'MaskSlice': largest_region_mask_image
        })

    return patient_slices


def get_volume_3D(image, mask, patient_id):
    """
    Extract a 3D volume of an image and its corresponding mask.

    GIVEN
    -----
    image : sitk.Image
        The 3D medical image.
    mask : sitk.Image
        The corresponding 3D segmentation mask.
    patient_id : int
        Unique identifier of the patient.

    WHEN
    ----
    The function processes the full 3D volume.

    THEN
    ----
    Returns a list containing a dictionary with:
        - 'PatientID': Patient identifier.
        - 'ImageVolume': The full 3D image.
        - 'MaskVolume': The full 3D mask.

    Raises
    ------
    TypeError
        If `image` or `mask` is not a SimpleITK Image.
    ValueError
        If `patient_id` is not an integer.
    """
    
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected 'image' to be a SimpleITK Image, but got {type(image)}.")

    if not isinstance(mask, sitk.Image):
        raise TypeError(f"Expected 'mask' to be a SimpleITK Image, but got {type(mask)}.")

    if not isinstance(patient_id, int):
        raise ValueError(f"Expected 'patient_id' to be a int, but got {type(patient_id)}.")

    return [{
        'PatientID': f"PR{patient_id}",
        'ImageVolume': image,
        'MaskVolume': mask
    }]
 
 
def read_image_and_mask(image_path, mask_path):
    """
    Read an image and its corresponding mask using SimpleITK.

    GIVEN
    -----
    image_path : str
        Path to the image file.
    mask_path : str
        Path to the mask file.

    WHEN
    ----
    The function reads the image and mask from disk.

    THEN
    ----
    Returns a tuple containing the image and mask as SimpleITK images.

    Raises
    ------
    ValueError
        If any input path is empty.
        If the parent directories of the image and mask do not match.
        If the image and mask dimensions do not match.
    TypeError
        If input paths are not strings.
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
    Create a dictionary mapping patient IDs to image-mask data.

    GIVEN
    -----
    imgs_path : list[str]
        List of file paths to image files.
    masks_path : list[str]
        List of file paths to mask files.
    patient_ids : list[int]
        List of patient IDs.
    mode : str
        Either '2D' or '3D' to determine processing type.

    WHEN
    ----
    The function processes each patient's image-mask pair.

    THEN
    ----
    Returns a dictionary where each key is a patient ID and the value is:
        - A list of 2D slices (if mode="2D").
        - A list with a single 3D volume (if mode="3D").

    Raises
    ------
    ValueError
        If `patient_ids` is empty.
        If `imgs_path`, `masks_path`, and `patient_ids` have different lengths.
        If `mode` is not '2D' or '3D'.
    """
    if len(patient_ids) == 0:
        raise ValueError("The patient_ids list cannot be empty.")

    if len(imgs_path) != len(masks_path) or len(imgs_path) != len(patient_ids):
        raise ValueError("The number of images, masks, and patient_ids must be the same.")

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