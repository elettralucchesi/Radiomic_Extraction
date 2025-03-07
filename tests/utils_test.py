import pytest
from utils import get_path_images_masks


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
