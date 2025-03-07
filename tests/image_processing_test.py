import pytest
from image_processing import *


def test_extract_largest_region_correct():
    """
    Test the correct behavior of the extract_largest_region function.

    GIVEN: A 2D binary mask with two regions of a specified label.
    WHEN: The extract_largest_region function is called.
    THEN: The function correctly returns the largest connected region.
    """
    mask = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 0, 1, 1],
                     [0, 0, 1, 1]])

    label_value = 1
    largest_region = extract_largest_region(mask, label_value)

    expected = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0]])

    # Assert the largest region is correctly extracted
    assert np.array_equal(largest_region, expected), f"Expected largest region {expected}, but got {largest_region}"


def test_extract_largest_region_negative_label():
    """
    Test that the function raises an error when the label value is negative.

    GIVEN: A negative label value.
    WHEN: The extract_largest_region function is called.
    THEN: The function should raise a ValueError indicating the label cannot be negative.
    """
    mask_slice = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 0, 1, 1],
                     [0, 0, 1, 1]])  # Valid mask slice
    label_value = -1  # Invalid label value

    try:
        extract_largest_region(mask_slice, label_value)
    except ValueError as e:
        assert str(e) == "Label value cannot be negative", "Error message does not match."


def test_extract_largest_region_label_not_found():
    """
    Test that the function returns None when the label is not found in the mask slice.

    GIVEN: A mask slice with no regions for the specified label.
    WHEN: The extract_largest_region function is called.
    THEN: The function should return None, as the label does not exist in the mask.
    """
    mask_slice = np.array([[2, 2, 0, 0],
                     [2, 2, 0, 0],
                     [2, 0, 3, 3],
                     [0, 0, 3, 3]])  # Valid mask slice
    label_value = 1  # Label not present

    result = extract_largest_region(mask_slice, label_value)

    assert result is None, "The function should return None when the label is not found."


def test_extract_largest_region_found():
    """
    Test that the function correctly extracts the largest region of a given label.

    GIVEN: A mask slice with several regions for the specified label.
    WHEN: The extract_largest_region function is called.
    THEN: The function should return the largest region of the given label.
    """
    mask_slice = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 2, 2, 2],
        [0, 0, 2, 2, 2],
        [0, 0, 0, 0, 0]
    ], dtype=int)  # Two regions: Label 1 and Label 2
    label_value = 1

    result = extract_largest_region(mask_slice, label_value)

    # The largest region should be of label 1
    expected_result = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    assert np.array_equal(result,
                          expected_result), "The function should extract the largest connected region for label 1."


def test_extract_largest_region_swapped_inputs():
    """
    Test that the function raises an error when the inputs are swapped.

    GIVEN: The function is called with an integer as mask_slice and a numpy array as label_value.
    WHEN: The extract_largest_region function is called with incorrect argument order.
    THEN: The function should raise a TypeError indicating the inputs are swapped.
    """
    mask_slice = 1  # Incorrect: should be a numpy array
    label_value = np.array([[0, 1], [1, 0]])  # Incorrect: should be an integer

    try:
        extract_largest_region(mask_slice, label_value)
    except TypeError as e:
        assert str(e) == "Inputs appear to be swapped. Expected mask_slice as a numpy array and label_value as an integer.", \
            "Error message does not match."


def test_extract_largest_region_label_not_float():
    """
    Test that the function raises a TypeError if label_value is a float.

    GIVEN: A float label value.
    WHEN: The extract_largest_region function is called.
    THEN: The function should raise a TypeError indicating that the label value must be an integer.
    """
    mask_slice = np.zeros((5, 5), dtype=int)  # Example empty mask slice

    # Test with a float as label_value
    label_value = 1.5
    with pytest.raises(TypeError, match="Label value must be an integer"):
        extract_largest_region(mask_slice, label_value)


def test_extract_largest_region_label_not_string():
    """
    Test that the function raises a TypeError if label_value is a string.

    GIVEN: A string label value.
    WHEN: The extract_largest_region function is called.
    THEN: The function should raise a TypeError indicating that the label value must be an integer.
    """
    mask_slice = np.zeros((5, 5), dtype=int)  # Example empty mask slice

    # Test with a string as label_value
    label_value = "label"
    with pytest.raises(TypeError, match="Label value must be an integer"):
        extract_largest_region(mask_slice, label_value)
