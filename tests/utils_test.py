import pytest
from features_extraction.utils import *


# ------------------- Get Path Images Masks tests -------------------


@pytest.fixture
def setup_test_files(tmp_path):
    """
    Creates a temporary directory with test image and mask files.

    GIVEN: A temporary directory.
    WHEN: Dummy.nii files (images and masks) are created in the directory.
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

    assert sorted(img) == sorted(
        expected_img
    ), f"Expected images: {expected_img}, but got: {img}"


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

    assert sorted(mask) == sorted(
        expected_mask
    ), f"Expected masks: {expected_mask}, but got: {mask}"


@pytest.mark.parametrize("invalid_path", [123, None])
def test_get_path_images_masks_invalid_path(invalid_path):
    """
    Test if passing an invalid path type raises a TypeError.

    GIVEN: An invalid path type (e.g., integer or None).
    WHEN: The get_path_images_masks function is called with the invalid path.
    THEN: The function raises a TypeError with the appropriate error message.
    """
    with pytest.raises(TypeError, match="Path must be a string"):
        get_path_images_masks(invalid_path)


@pytest.mark.parametrize(
    "files, expected_error",
    [
        ([], "The directory is empty or contains no .nii files"),
        (
            ["file1.txt", "file2.csv", "file3.jpg"],
            "The directory is empty or contains no .nii files",
        ),
        (
            ["patient1.nii", "patient2.nii", "patient1_seg.nii"],
            "The number of image files does not match the number of mask files",
        ),
        (
            ["patient1.nii", "patient1_seg.nii", "patient2_seg.nii"],
            "The number of image files does not match the number of mask files",
        ),
    ],
)
def test_get_path_images_masks_invalid_cases(tmp_path, files, expected_error):
    """
    Test if passing invalid directory structures raises ValueError.

    GIVEN: A directory with an invalid structure (e.g., empty, no .nii files, or mismatched image/mask counts).
    WHEN: The get_path_images_masks function is called on the directory.
    THEN: The function raises a ValueError with the appropriate error message.
    """
    for file in files:
        (tmp_path / file).write_text("test")

    with pytest.raises(ValueError, match=expected_error):
        get_path_images_masks(str(tmp_path))


# ------------------- Extract ID tests -------------------


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
    Test that the function handles filenames with incorrectly formatted patient IDs.

    GIVEN: A filename with an incorrectly formatted patient ID containing 'PR'.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id("path/to/PR_2_image.nii")
    assert result is None, f"Expected None, but got {result}"


def test_extract_id_invalid_number_before_pr():
    """
    Test that the function handles filenames where the number precedes 'PR'.

    GIVEN: A filename where the number precedes 'PR' (e.g., '2PR').
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id("path/to/2PR_image.nii")
    assert result is None, f"Expected None, but got {result}"


def test_extract_id_no_pr():
    """
    Test that the function handles filenames where the ID lacks the 'PR' prefix.

    GIVEN: A filename without a patient ID or 'PR' prefix.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id("path/to/image_without_id.nii")
    assert result is None, f"Expected None, but got {result}"


@pytest.mark.parametrize("invalid_input", [12345, None])
def test_extract_id_invalid_input(invalid_input):
    """
    Test that extract_id raises a TypeError when provided with a non-string input.

    GIVEN: An invalid input (non-string or None).
    WHEN: The extract_id function is called.
    THEN: The function raises a TypeError with the appropriate error message.
    """
    with pytest.raises(TypeError, match="Path must be a string"):
        extract_id(invalid_input)


def test_extract_id_multiple_valid_pr():
    """
    Test that the function extracts only the first valid patient ID when multiple are present.

    GIVEN: A filename with multiple valid 'PR<number>' occurrences.
    WHEN: The extract_id function is called.
    THEN: The function returns only the first valid patient ID.
    """
    result = extract_id("path/to/PR12_PR34_image.nii")
    assert result == 12, f"Expected 12, but got {result}"


# ------------------- New Patient ID tests -------------------


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

    GIVEN: A set of patient IDs {1, 2, 3, 5}.
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
        new_patient_id([1, 2, 3])


def test_new_patient_id_invalid_type_string():
    """
    Test that the function raises a TypeError if the input is a string instead of a set.

    GIVEN: A string instead of a set.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a TypeError.
    """
    with pytest.raises(TypeError, match="patients_id must be a set"):
        new_patient_id("123")


def test_new_patient_id_non_integer_values_string():
    """
    Test that the function raises a ValueError if the set contains a string instead of integers.

    GIVEN: A set containing a string value.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a ValueError.
    """
    with pytest.raises(TypeError, match="All patient IDs must be integers"):
        new_patient_id({1, 2, "three"})


def test_new_patient_id_non_integer_values_float():
    """
    Test that the function raises a ValueError if the set contains a float instead of integers.

    GIVEN: A set containing a float value.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a ValueError.
    """
    with pytest.raises(TypeError, match="All patient IDs must be integers"):
        new_patient_id({1, 2, 3.5})


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


# ------------------- Assign Patient IDs tests -------------------


def test_assign_patient_ids_type_error():
    """
    Test that the function raises a TypeError when images_path is not a list.

    GIVEN: A non-list input for images_path.
    WHEN: The assign_patient_ids function is called.
    THEN: The function raises a TypeError with the appropriate message.
    """
    invalid_input = "not_a_list"
    with pytest.raises(TypeError, match="images_path must be a list"):
        assign_patient_ids(invalid_input)


def test_assign_patient_ids_empty_list_error():
    """
    Test that the function raises an error when an empty list is passed as images_path.

    GIVEN: An empty list of image file paths.
    WHEN: The assign_patient_ids function is called.
    THEN: The function raises a ValueError with an appropriate message.
    """
    images_path = []

    with pytest.raises(ValueError, match="The list of image paths cannot be empty"):
        assign_patient_ids(images_path)


def test_assign_patient_ids_existing_ids():
    """
    Test that the function extracts existing patient IDs correctly.

    GIVEN: A list of image file paths containing patient IDs in the format 'PR<number>'.
    WHEN: The assign_patient_ids function is called.
    THEN: The function extracts the correct patient IDs for each image path.
    """
    images_path = [
        "../Radiomic_Features_Extraction/data/PR1/PR1_T2W_TSE_AX.nii",
        "../Radiomic_Features_Extraction/data/PR2/PR2_T2W_TSE_AX.nii",
    ]

    patient_ids = assign_patient_ids(images_path)
    assert patient_ids == {
        1,
        2,
    }, f" Expected patient IDs {1, 2}, but got {patient_ids} "


def test_assign_patient_ids_no_existing_ids():
    """
    Test that the function assigns new patient IDs when no existing IDs are found in the file names.

    GIVEN: A list of image file paths without valid patient IDs.
    WHEN: The assign_patient_ids function is called.
    THEN: The function assigns new patient IDs to each image path.
    """
    images_path = [
        "../Radiomic_Features_Extraction/data/image1_T2W_TSE_AX.nii",
        "../Radiomic_Features_Extraction/data/image2_T2W_TSE_AX.nii",
    ]

    patient_ids = assign_patient_ids(images_path)
    assert patient_ids == {
        1,
        2,
    }, f" Expected new patient IDs {1, 2}, but got {patient_ids} "


def test_assign_patient_ids_mixed_existing_and_new_ids():
    """
    Test that the function handles a mix of image paths with existing and new patient IDs.

    GIVEN: A list of image file paths, some with existing patient IDs and others without.
    WHEN: The assign_patient_ids function is called.
    THEN: The function correctly extracts existing IDs and assigns new IDs where necessary.
    """
    images_path = [
        "../Radiomic_Features_Extraction/data/PR1/PR1_T2W_TSE_AX.nii",
        "../Radiomic_Features_Extraction/data/PR2/PR2_T2W_TSE_AX.nii",
        "../Radiomic_Features_Extraction/data/image4_T2W_TSE_AX.nii",
    ]

    patient_ids = assign_patient_ids(images_path)
    assert patient_ids == {
        1,
        2,
        3,
    }, f" Expected patient IDs {1, 2, 3}, but got {patient_ids} "


def test_assign_patient_ids_invalid_id_format():
    """
    Test that the function handles invalid patient ID formats in image paths.

    GIVEN: A list of image file paths with invalid or missing patient IDs.
    WHEN: The assign_patient_ids function is called.
    THEN: The function assigns new patient IDs to each image path.
    """
    images_path = [
        "../Radiomic_Features_Extraction/data/PR_invalid/PR_T2W_TSE_AX.nii",  # Invalid ID format
        "../Radiomic_Features_Extraction/data/PR_invalid/PR_T2W_TSE_AX.nii",  # Invalid ID format
    ]

    patient_ids = assign_patient_ids(images_path)

    assert patient_ids == {
        1,
        2,
    }, f" Expected new patient IDs {1, 2}, but got {patient_ids} "
