import pytest
import re
from utils import *


@pytest.fixture
def setup_test_files(tmp_path):
    """
    Creates a temporary directory with test image and mask files.

    GIVEN: A temporary directory.
    WHEN: Dummy .nii files (images and masks) are created in the directory.
    THEN: The function returns the temporary directory with image and mask file names.
    """
    
    img_files = ["image1.nii", "image2.nii"]
    mask_files = ["image1_seg.nii", "image2_seg.nii"]

    for file in img_files + mask_files:
        (tmp_path / file).write_text("test")  # Create dummy files

    return tmp_path, img_files, mask_files


def test_get_path_images_masks_images(setup_test_files):
    """
    Test if images are correctly separated into a list.

    GIVEN: A temporary directory with valid .nii image files.
    WHEN: The get_path_images_masks function is called on the directory.
    THEN: The function returns a list of image file paths.
    """
    test_dir, img_files, _ = setup_test_files
    img, _ = get_path_images_masks(str(test_dir))

    expected_img = [str(test_dir / f) for f in img_files]

    assert sorted(img) == sorted(expected_img), f"Expected images: {expected_img}, but got: {img}"


def test_get_path_images_masks_masks(setup_test_files):
    """
    Test if masks are correctly separated into a list.

    GIVEN: A temporary directory with valid .nii mask files.
    WHEN: The get_path_images_masks function is called on the directory.
    THEN: The function returns a list of mask file paths.
    """
    test_dir, _, mask_files = setup_test_files
    _, mask = get_path_images_masks(str(test_dir))

    expected_mask = [str(test_dir / f) for f in mask_files]

    assert sorted(mask) == sorted(expected_mask), f"Expected masks: {expected_mask}, but got: {mask}"


def test_get_path_images_masks_invalid_int():
    """
    Test if passing an integer as path raises TypeError.

    GIVEN: An invalid path type (integer).
    WHEN: The get_path_images_masks function is called with an integer.
    THEN: The function raises a TypeError with the appropriate error message.
    """
    with pytest.raises(TypeError, match="Path must be a string") as exc_info:
        get_path_images_masks(123)  # Pass an integer
    assert str(exc_info.value) == "Path must be a string", f"Expected error message 'Path must be a string', but got: {str(exc_info.value)}"


def test_get_path_images_masks_invalid_none():
    """
    Test if passing None as path raises TypeError.

    GIVEN: An invalid path type (None).
    WHEN: The get_path_images_masks function is called with None.
    THEN: The function raises a TypeError with the appropriate error message.
    """
    with pytest.raises(TypeError, match="Path must be a string") as exc_info:
        get_path_images_masks(None)  # Pass None
    assert str(exc_info.value) == "Path must be a string", f"Expected error message 'Path must be a string', but got: {str(exc_info.value)}"


def test_get_path_images_masks_empty_directory(tmp_path):
    """
    Test if passing an empty directory raises ValueError.

    GIVEN: An empty directory.
    WHEN: The get_path_images_masks function is called on the directory.
    THEN: The function raises a ValueError with the appropriate error message.
    """
    with pytest.raises(ValueError, match="The directory is empty or contains no .nii files") as exc_info:
        get_path_images_masks(str(tmp_path))  # Pass an empty directory
    assert str(exc_info.value) == "The directory is empty or contains no .nii files", f"Expected error message 'The directory is empty or contains no .nii files', but got: {str(exc_info.value)}"


def test_get_path_images_masks_no_nii_files(tmp_path):
    """
    Test if a directory without .nii files raises ValueError.

    GIVEN: A directory with files, but none with .nii extension.
    WHEN: The get_path_images_masks function is called on the directory.
    THEN: The function raises a ValueError with the appropriate error message.
    """
    # Create files with incorrect extensions
    non_nii_files = ["file1.txt", "file2.csv", "file3.jpg"]

    for file in non_nii_files:
        (tmp_path / file).write_text("test")  # Create non-.nii files

    with pytest.raises(ValueError, match="The directory is empty or contains no .nii files") as exc_info:
        get_path_images_masks(str(tmp_path))  # Pass the directory without .nii files

    assert str(
        exc_info.value) == "The directory is empty or contains no .nii files", f"Expected error message 'The directory is empty or contains no .nii files', but got: {str(exc_info.value)}"



def test_get_path_images_masks_mismatched_files(tmp_path):
    """
    Test if a directory with mismatched image and mask files raises ValueError.

    GIVEN: A directory containing an unequal number of image and mask files.
    WHEN: The get_path_images_masks function is called.
    THEN: The function raises a ValueError with the appropriate error message.
    """
    img_files = ["patient1.nii", "patient2.nii"]
    mask_files = ["patient1_seg.nii"]  # Only one mask file for two image files

    for file in img_files + mask_files:
        (tmp_path / file).write_text("test")  # Create both images and masks

    with pytest.raises(ValueError, match="The number of image files does not match the number of mask files") as exc_info:
        get_path_images_masks(str(tmp_path))

    assert str(exc_info.value) == "The number of image files does not match the number of mask files", \
        f"Expected error message 'The number of image files does not match the number of mask files', but got: {str(exc_info.value)}"


def test_get_path_images_masks_multiple_masks_for_one_image(tmp_path):
    """
    Test if a directory with more mask files than image files raises ValueError.

    GIVEN: A directory containing more mask files than image files.
    WHEN: The get_path_images_masks function is called.
    THEN: The function raises a ValueError with the appropriate error message.
    """
    img_files = ["patient1.nii"]  # One image file
    mask_files = ["patient1_seg.nii", "patient2_seg.nii"]  # Two mask files

    for file in img_files + mask_files:
        (tmp_path / file).write_text("test")  # Create one image and multiple masks

    with pytest.raises(ValueError, match="The number of image files does not match the number of mask files") as exc_info:
        get_path_images_masks(str(tmp_path))

    assert str(exc_info.value) == "The number of image files does not match the number of mask files", \
        f"Expected error message 'The number of image files does not match the number of mask files', but got: {str(exc_info.value)}"



def test_extract_id_valid():
    """
    Test that the function correctly extracts the patient ID from a valid file path.

    GIVEN: A file path with a patient ID.
    WHEN: The extract_id function is called.
    THEN: The function returns the correct patient ID.
    """
    valid_path = "/path/to/PR12345_image.nii"
    result = extract_id(valid_path)
    assert result == 12345, f"Expected patient ID: 12345, but got: {result}"

def test_extract_id_invalid_format_with_pr():
    """
    GIVEN: A filename with an incorrectly formatted patient ID containing 'PR'.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/PR_2_image.nii')
    assert result is None, f"Expected None, but got {result}"

def test_extract_id_invalid_number_before_pr():
    """
    GIVEN: A filename where the number precedes 'PR' (e.g., '2PR').
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/2PR_image.nii')
    assert result is None, f"Expected None, but got {result}"

def test_extract_id_no_pr():
    """
    GIVEN: A filename without a patient ID or 'PR' prefix.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/image_without_id.nii')
    assert result is None, f"Expected None, but got {result}"


def test_extract_id_multiple_pr_but_wrong_format():
    """
    GIVEN: A filename containing multiple 'PR' but in an incorrect format.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/PRabc_PR123X_image.nii')
    assert result is None, f"Expected None, but got {result}"


def test_extract_id_no_pr_prefix():
    """
    GIVEN: A filename where the ID is present but lacks the 'PR' prefix.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/12345_image.nii')
    assert result is None, f"Expected None, but got {result}"

def test_extract_id_non_string_path():
    """
    GIVEN: A non-string input as path.
    WHEN: The extract_id function is called.
    THEN: The function raises a TypeError.
    """
    with pytest.raises(TypeError, match="Path must be a string"):
        extract_id(12345)

def test_extract_id_none_input():
    """
    GIVEN: A None input.
    WHEN: The extract_id function is called.
    THEN: The function raises a TypeError.
    """
    with pytest.raises(TypeError, match="Path must be a string"):
        extract_id(None)

def test_extract_id_multiple_valid_pr():
    """
    GIVEN: A filename with multiple valid 'PR<number>' occurrences.
    WHEN: The extract_id function is called.
    THEN: The function returns only the first valid patient ID.
    """
    result = extract_id('path/to/PR12_PR34_image.nii')
    assert result == 12, f"Expected 12, but got {result}"




def test_new_patient_id_empty_set():
    """
    Test that the function returns 1 when the set of patient IDs is empty.

    GIVEN: An empty set of patient IDs.
    WHEN: The new_patient_id function is called.
    THEN: The function returns 1.
    """
    result = new_patient_id(set())
    assert result == 1, f"Expected new ID: 1, but got: {result}"


def test_new_patient_id_sequential():
    """
    Test that the function correctly assigns the next available patient ID when IDs are sequential.

    GIVEN: A set of patient IDs {1, 2, 3, 4, 5}.
    WHEN: The new_patient_id function is called.
    THEN: The function returns 6.
    """
    existing_ids = {1, 2, 3, 4, 5}
    result = new_patient_id(existing_ids)
    assert result == 6, f"Expected new ID: 6, but got: {result}"

def test_new_patient_id_missing_numbers():
    """
    Test that the function assigns the lowest available patient ID when there are missing numbers.

    GIVEN: A set of patient IDs {2, 3, 5}.
    WHEN: The new_patient_id function is called.
    THEN: The function returns 1 as the first available ID.
    """
    existing_ids = {1, 2, 3, 5}
    result = new_patient_id(existing_ids)
    assert result == 4, f"Expected new ID: 4, but got: {result}"


def test_new_patient_id_large_numbers():
    """
    Test that the function correctly assigns 1 if the existing IDs are all large numbers.

    GIVEN: A set of patient IDs {100, 101, 102}.
    WHEN: The new_patient_id function is called.
    THEN: The function returns 1.
    """
    existing_ids = {100, 101, 102}
    result = new_patient_id(existing_ids)
    assert result == 1, f"Expected new ID: 1, but got: {result}"

def test_new_patient_id_invalid_type_list():
    """
    Test that the function raises a TypeError if the input is a list instead of a set.

    GIVEN: A list of patient IDs instead of a set.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a TypeError.
    """
    with pytest.raises(TypeError, match="patients_id must be a set"):
        new_patient_id([1, 2, 3])  # List instead of set


def test_new_patient_id_invalid_type_string():
    """
    Test that the function raises a TypeError if the input is a string instead of a set.

    GIVEN: A string instead of a set.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a TypeError.
    """
    with pytest.raises(TypeError, match="patients_id must be a set"):
        new_patient_id("123")  # String instead of set

def test_new_patient_id_non_integer_values_string():
    """
    Test that the function raises a ValueError if the set contains a string instead of integers.

    GIVEN: A set containing a string value.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a ValueError.
    """
    with pytest.raises(ValueError, match="All patient IDs must be integers"):
        new_patient_id({1, 2, "three"})  # Contains a string

def test_new_patient_id_non_integer_values_float():
    """
    Test that the function raises a ValueError if the set contains a float instead of integers.

    GIVEN: A set containing a float value.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a ValueError.
    """
    with pytest.raises(ValueError, match="All patient IDs must be integers"):
        new_patient_id({1, 2, 3.5})  # Contains a float

def test_new_patient_id_negative_values():
    """
    Test that the function raises a ValueError when the set contains negative patient IDs.

    GIVEN: A set containing negative patient IDs.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a ValueError with an appropriate message.
    """
    existing_ids = {1, -2, 3, -1}
    with pytest.raises(ValueError, match="Patient IDs cannot be negative"):
        new_patient_id(existing_ids)
