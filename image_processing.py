from scipy.ndimage import label
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
        If `mask_slice` is an integer and `label_value` is a numpy array.
        If `label_value` is not an integer.
    ValueError
        If `label_value` is negative.
    """

    # Check if inputs are swapped (i.e., label_value is a numpy array and mask_slice is an integer)
    if isinstance(mask_slice, int) and isinstance(label_value, np.ndarray):
        raise TypeError(
            "Inputs appear to be swapped. Expected mask_slice as a numpy array and label_value as an integer.")

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

    # Iterate through all connected components, ignoring the background (0)
    for region_id in range(1, num_labels + 1):
        region = (labeled_region == region_id).astype(mask_slice.dtype) * label_value
        region_area = np.sum(region > 0)

        # Update the largest region if the current one is bigger
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
    
    """

    labels = np.unique(mask_slice)
    labels = labels[labels != 0]  # Exclude background

    for lbl in labels:
        lbl = int(lbl)  # Convert numpy.int16 to native Python int
        largest_region_mask = extract_largest_region(mask_slice, lbl)
        if largest_region_mask is not None:
            return largest_region_mask, lbl
    
    # If no region found
    return None, None

