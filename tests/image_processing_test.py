import pytest
from unittest.mock import patch
from features_extraction.image_processing import *
from features_extraction.utils import *


# ---------------- Extract Largest Region Tests ----------------


def test_extract_largest_region_correct():
    """
    Test the correct behavior of the extract_largest_region function.

    GIVEN: A 2D mask with two regions of a specified label.
    WHEN: The extract_largest_region function is called.
    THEN: The function correctly returns the largest connected region.
    """

    mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 1], [0, 0, 1, 1]])

    label_value = 1
    largest_region = extract_largest_region(mask, label_value)
    expected = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    assert np.array_equal(
        largest_region, expected
    ), f"Expected largest region {expected}, but got {largest_region}"


def test_extract_largest_region_found():
    """
    Test that the function correctly extracts the largest region of a given label.

    GIVEN: A mask slice with several regions for the specified label.
    WHEN: The extract_largest_region function is called.
    THEN: The function should return the largest region of the given label.
    """

    mask_slice = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 2, 2],
            [0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    label_value = 1

    result = extract_largest_region(mask_slice, label_value)

    expected_result = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=int,
    )

    assert np.array_equal(
        result, expected_result
    ), "The function should extract the largest connected region for label 1."


def test_extract_largest_region_label_not_found():
    """
    Test that the function returns None when the label is not found in the mask slice.

    GIVEN: A mask slice with no regions for the specified label.
    WHEN: The extract_largest_region function is called.
    THEN: The function should return None, as the label does not exist in the mask.
    """

    mask_slice = np.array([[2, 2, 0, 0], [2, 2, 0, 0], [2, 0, 3, 3], [0, 0, 3, 3]])
    label_value = 1

    result = extract_largest_region(mask_slice, label_value)

    assert (
        result is None
    ), "The function should return None when the label is not found."


@pytest.mark.parametrize(
    "mask_slice, label_value, expected_exception, expected_message",
    [
        (
            1,
            np.array([[0, 1], [1, 0]]),
            TypeError,
            "Inputs appear to be swapped. Expected mask_slice as a numpy array and label_value as an integer.",
        ),
        (np.zeros((5, 5)), 1.5, TypeError, "Label value must be an integer"),
        (np.zeros((5, 5)), "label", TypeError, "Label value must be an integer"),
        ("not an array", 1, TypeError, "mask_slice must be a numpy array"),
    ],
)
def test_extract_largest_region_type_errors(
    mask_slice, label_value, expected_exception, expected_message
):
    """
    Test that the function raises TypeError for invalid input types.

    GIVEN: An invalid type for mask_slice or label_value.
    WHEN: The extract_largest_region function is called.
    THEN: The function raises the appropriate TypeError.
    """

    with pytest.raises(expected_exception, match=expected_message):
        extract_largest_region(mask_slice, label_value)


@pytest.mark.parametrize(
    "mask_slice, label_value, expected_exception, expected_message",
    [
        (
            np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 1], [0, 0, 1, 1]]),
            -1,
            ValueError,
            "Label value cannot be negative",
        ),
        (
            np.zeros((3, 3, 3), dtype=np.uint8),
            1,
            ValueError,
            "mask_slice must be a 2D array",
        ),
    ],
)
def test_extract_largest_region_value_errors(
    mask_slice, label_value, expected_exception, expected_message
):
    """

    Test that the function raises ValueError for invalid input values.

    GIVEN invalid inputs for extract_largest_region:
        - A negative label value
        - A mask slice that is not 2D
    WHEN the function is called
    THEN it should raise the expected exception with the correct error message.
    """

    with pytest.raises(expected_exception, match=expected_message):
        extract_largest_region(mask_slice, label_value)


# ---------------- Process Slice Tests ----------------


def test_process_slice_single_label():
    """
    Test process_slice with a mask containing a single labeled region.

    GIVEN: A mask with a single labeled region.
    WHEN: The function is called.
    THEN: It should return the largest region mask and its corresponding label.
    """
    mask_slice = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 1], [0, 0, 1, 1]])

    _, label = process_slice(mask_slice)
    assert label == 1, "Expected label 1, but got a different label."


def test_process_slice_single_mask():
    """
    Test process_slice with a mask containing a single labeled region.

    GIVEN: A mask with a single labeled region.
    WHEN: The function is called.
    THEN: It should return the largest region mask and its corresponding label.
    """
    mask_slice = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 1], [0, 0, 1, 1]])

    largest_region_mask, _ = process_slice(mask_slice)

    expected_region_mask = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    )

    assert np.array_equal(
        largest_region_mask, expected_region_mask
    ), "The largest region mask does not match the expected result."


def test_process_slice_multiple_labels_label():
    """
    Test that the function correctly returns the largest region mask and its corresponding label when there are multiple labeled regions.

    GIVEN: A mask slice with multiple labeled regions.
    WHEN: The process_slice function is called.
    THEN: The function should return the largest valid region and its corresponding label.
    """
    mask_slice = np.array(
        [[0, 1, 1, 0, 2, 2, 2], [0, 1, 1, 0, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0]]
    )

    _, label = process_slice(mask_slice)
    assert label in [1, 2], "Expected label 1 or 2, but got a different label."


def test_process_slice_multiple_labels_mask():
    """
    Test that the function correctly returns the largest region mask and its corresponding label when there are multiple labeled regions.

    GIVEN: A mask slice with multiple labeled regions.
    WHEN: The process_slice function is called.
    THEN: The function should return the largest valid region and its corresponding label.
    """
    mask_slice = np.array(
        [[0, 1, 1, 0, 2, 2, 2], [0, 1, 1, 0, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0]]
    )

    largest_region_mask, label = process_slice(mask_slice)

    expected_region_mask_1 = np.array(
        [[0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    )
    expected_region_mask_2 = np.array(
        [[0, 0, 0, 0, 2, 2, 2], [0, 0, 0, 0, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0]]
    )

    assert np.array_equal(
        largest_region_mask, expected_region_mask_1
    ) or np.array_equal(
        largest_region_mask, expected_region_mask_2
    ), f"Unexpected largest region mask for label {label}."


def test_process_slice_returns_none_none():
    """

    Test that the function returns (None, None) when no labeled regions exist.

    GIVEN a mask slice with no labeled regions (all zeros)
    WHEN process_slice is called
    THEN it should return (None, None)
    """

    mask_slice = np.zeros((10, 10), dtype=np.uint16)

    region_mask, label = process_slice(mask_slice)

    assert (region_mask, label) == (
        None,
        None,
    ), f"Expected (None, None), but got ({region_mask}, {label})"


def test_process_slice_raises_type_error_if_not_array():
    """
    Test that the function raises TypeError when receiving non-array input.


    GIVEN an input that is not a numpy array
    WHEN process_slice is called
    THEN it should raise a TypeError
    """

    invalid_input = [[0, 1], [1, 0]]

    with pytest.raises(TypeError, match="mask_slice must be a numpy array"):
        process_slice(invalid_input)


def test_process_slice_raises_value_error_if_not_2d():
    """
    Test that the function raises ValueError when receiving non-2D array.

    GIVEN an input that is not a 2D array
    WHEN process_slice is called
    THEN it should raise a ValueError
    """

    invalid_input = np.zeros((10, 10, 10))

    with pytest.raises(ValueError, match="mask_slice must be a 2D array"):
        process_slice(invalid_input)


# ---------------- Get Slices 2D Tests ----------------


@pytest.mark.parametrize(
    "image, mask",
    [
        (None, sitk.Image(10, 10, 10, sitk.sitkUInt8)),
        (sitk.Image(10, 10, 10, sitk.sitkUInt8), None),
        ("not an image", sitk.Image(10, 10, 10, sitk.sitkUInt8)),
        (sitk.Image(10, 10, 10, sitk.sitkUInt8), "not an image"),
    ],
)
def test_get_slices_2D_raises_typeerror_for_invalid_inputs(image, mask):
    """
    Test that get_slices_2D raises TypeError for invalid image or mask inputs.

    GIVEN: An invalid image or mask.
    WHEN: The get_slices_2D function is called.
    THEN: A TypeError should be raised.
    """
    with pytest.raises(TypeError):
        get_slices_2D(image, mask, patient_id=1)


@pytest.mark.parametrize("patient_id", ["string", 3.5, None, [1, 2, 3]])
def test_get_slices_2D_raises_valueerror_for_invalid_patient_id(patient_id):
    """
    Test that get_slices_2D raises ValueError for invalid patient_id.

    GIVEN: An invalid patient_id.
    WHEN: The get_slices_2D function is called.
    THEN: A ValueError should be raised.
    """
    image = sitk.Image(10, 10, 10, sitk.sitkUInt8)
    mask = sitk.Image(10, 10, 10, sitk.sitkUInt8)
    with pytest.raises(ValueError):
        get_slices_2D(image, mask, patient_id)


def test_get_slices_2D_valid_length():
    """
    Test that get_slices_2D returns the expected number of patient slices for a valid input.

    GIVEN: A valid image and mask.
    WHEN: The function get_slices_2D is called.
    THEN: It should return a list with the correct number of slices.
    """

    image_array = np.random.rand(3, 4, 4)
    mask_array = np.array(
        [
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    patient_id = 1234

    patient_slices = get_slices_2D(image, mask, patient_id)

    assert (
        len(patient_slices) == 3
    ), f"Expected 3 slices, but got {len(patient_slices)}."


def test_get_slices_2D_patient_id():
    """
    Test that the PatientID is correctly set in the patient slice data.

    GIVEN: A valid image, mask, and PatientID.
    WHEN: The function get_slices_2D is called.
    THEN: The PatientID should be included correctly in each slice data.
    """
    image_array = np.random.rand(3, 4, 4)
    mask_array = np.array(
        [
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    patient_id = 1234

    patient_slices = get_slices_2D(image, mask, patient_id)

    for slice_data in patient_slices:
        assert (
            slice_data["PatientID"] == f"PR{patient_id}"
        ), f"Expected PatientID 'PR{patient_id}', but got {slice_data['PatientID']}."


def test_get_slices_2D_slice_index():
    """
    Test that the SliceIndex is correctly set in the patient slice data.

    GIVEN: A valid image, mask, and PatientID.
    WHEN: The function get_slices_2D is called.
    THEN: The SliceIndex should be correctly set for each slice.
    """
    image_array = np.random.rand(3, 4, 4)
    mask_array = np.array(
        [
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    patient_id = 1234

    patient_slices = get_slices_2D(image, mask, patient_id)
    c = 0
    for slice_data in patient_slices:
        assert (
            slice_data["SliceIndex"] == c
        ), f"Expected slice index {c}, but got {slice_data['SliceIndex']}."
        c = c + 1


def test_get_slices_2D_image_slice():
    """
    Test that the image slice is correctly converted into a SimpleITK Image.

    GIVEN: A valid image and mask.
    WHEN: The function get_slices_2D is called.
    THEN: The 'ImageSlice' in the returned data should be a SimpleITK Image.
    """

    image_array = np.random.rand(3, 4, 4)
    mask_array = np.array(
        [
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    patient_id = 1234

    patient_slices = get_slices_2D(image, mask, patient_id)

    for slice_data in patient_slices:
        assert isinstance(
            slice_data["ImageSlice"], sitk.Image
        ), "Expected 'ImageSlice' to be a SimpleITK Image."


def test_get_slices_2D_mask_slice():
    """
    Test that the mask slice is correctly converted into a SimpleITK Image.

    GIVEN: A valid image and mask.
    WHEN: The function get_slices_2D is called.
    THEN: The 'MaskSlice' in the returned data should be a SimpleITK Image.
    """
    image_array = np.random.rand(3, 4, 4)
    mask_array = np.array(
        [
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    patient_id = 1234

    patient_slices = get_slices_2D(image, mask, patient_id)

    for slice_data in patient_slices:
        assert isinstance(
            slice_data["MaskSlice"], sitk.Image
        ), "Expected 'MaskSlice' to be a SimpleITK Image."


def test_get_slices_2D_labels():
    """
    Test that the correct labels are assigned to the slices.

    GIVEN: A valid image and mask.
    WHEN: The function get_slices_2D is called.
    THEN: The label should be correctly assigned to each slice.
    """
    image_array = np.random.rand(3, 4, 4)
    mask_array = np.array(
        [
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    patient_id = 1234
    patient_slices = get_slices_2D(image, mask, patient_id)
    for slice_data in patient_slices:
        assert slice_data["Label"] in [
            1,
            2,
        ], f"Expected label 1 or 2, but got {slice_data['Label']}."


def test_get_slices_2D_skip_slice_on_none():
    """
    Test that slices without valid regions are excluded from processing results.

    GIVEN a mask slice where process_slice returns None, None (no region found)
    WHEN get_slices_2D is called
    THEN it should skip that slice and not include it in the results
    """

    img = sitk.GetImageFromArray(np.random.rand(3, 10, 10))
    mask = sitk.GetImageFromArray(
        np.array(
            [np.zeros((10, 10)), np.zeros((10, 10)), np.ones((10, 10))], dtype=np.uint16
        )
    )

    patient_id = 123

    # Mock the process_slice function to return (None, None) for the first two slices and valid results for the last slice
    with patch(
        "features_extraction.image_processing.process_slice",
        side_effect=[(None, None), (None, None), (np.ones((10, 10)), 1)],
    ):
        result = get_slices_2D(img, mask, patient_id)
    assert len(result) == 1, f"Expected 1 slice, but got {len(result)}"


# ---------------- Get Volume 3D Tests ----------------


def test_get_volume_3D_return_type():
    """
    Test that the get_volume_3D function returns a list.
    
    GIVEN: A 3D image and mask.
    WHEN: The get_volume_3D function is called.
    THEN: The function should return a list containing a dictionary with the correct keys.
    """
    image_3d = sitk.Image(3, 3, 3, sitk.sitkUInt8)
    mask_3d = sitk.Image(3, 3, 3, sitk.sitkUInt8)
    patient_id = 1234

    result = get_volume_3D(image_3d, mask_3d, patient_id)

    assert isinstance(
        result, list
    ), f"Expected result to be a list, but got {type(result)}."


@pytest.mark.parametrize(
    "image, mask, patient_id, expected_message",
    [
        (
            "not_an_image",
            sitk.Image(3, 3, 3, sitk.sitkUInt8),
            123,
            "Expected 'image' to be a SimpleITK Image.",
        ),
        (
            sitk.Image(3, 3, 3, sitk.sitkUInt8),
            "not_a_mask",
            123,
            "Expected 'mask' to be a SimpleITK Image.",
        ),
    ],
)
def test_get_volume_3D_type_error(image, mask, patient_id, expected_message):
    """
    Test that get_volume_3D raises TypeError for invalid image or mask inputs.

    GIVEN: An invalid image or mask (e.g., a string instead of an image object).
    WHEN: The get_volume_3D function is called with these invalid inputs.
    THEN: A TypeError should be raised indicating the type mismatch.
    """
    with pytest.raises(TypeError, match=expected_message):
        get_volume_3D(image, mask, patient_id)


def test_get_volume_3D_invalid_patient_id():
    """
    Test that get_volume_3D raises ValueError for invalid patient_id.

    GIVEN: A string patient_id.
    WHEN: The get_volume_3D function is called.
    THEN: The function should raise a ValueError indicating that patient_id must be int.
    """
    image_3d = sitk.Image(3, 3, 3, sitk.sitkUInt8)
    mask_3d = sitk.Image(3, 3, 3, sitk.sitkUInt8)
    patient_id = "123"

    with pytest.raises(ValueError, match="Expected 'patient_id' to be a int"):
        get_volume_3D(image_3d, mask_3d, patient_id)


# ---------------- Read Image and Mask Tests ----------------


@pytest.mark.parametrize(
    "image_path, mask_path, expected_message",
    [
        ("", "valid_mask_path", "Image and mask paths cannot be empty."),
        ("valid_image_path", "", "Image and mask paths cannot be empty."),
        (
            "path/to/image/image.nii",
            "path/to/mask/different_mask.nii",
            "Image and mask must be in the same directory.",
        ),
    ],
)
def test_read_image_and_mask_value_error(image_path, mask_path, expected_message):
    """
    Test that the read_image_and_mask function raises ValueError with the correct message.

    GIVEN: Invalid paths.
    WHEN: The read_image_and_mask function is called.
    THEN: ValueError is raised with the correct message.
    """
    with pytest.raises(ValueError, match=expected_message):
        read_image_and_mask(image_path, mask_path)


@pytest.mark.parametrize(
    "image_path, mask_path, expected_message",
    [
        (123, "valid_mask_path", "Image and mask paths must be strings."),
        ("valid_image_path", 123, "Image and mask paths must be strings."),
    ],
)
def test_read_image_and_mask_type_error(image_path, mask_path, expected_message):
    """
    Test that the read_image_and_mask function raises TypeError with the correct message.

    GIVEN: Non-string paths.
    WHEN: The read_image_and_mask function is called.
    THEN: TypeError is raised with the correct message.
    """
    with pytest.raises(TypeError, match=expected_message):
        read_image_and_mask(image_path, mask_path)


@pytest.fixture
def mock_read_image_and_mask_different_size():
    """
    Mocked function that simulates reading an image and a mask with different dimensions.

    GIVEN: A mocked function that returns an image and a mask with different dimensions.
    WHEN: It is used in place of the actual function.
    THEN: The returned image and mask will have mismatched dimensions.
    """

    def _mock(img_path, mask_path):
        img = sitk.Image(3, 3, 3, sitk.sitkUInt8)
        mask = sitk.Image(4, 4, 4, sitk.sitkUInt8)
        return img, mask

    return _mock


def test_read_image_and_mask_dimension_mismatch(
    mock_read_image_and_mask_different_size, monkeypatch: pytest.MonkeyPatch
):
    """
    Test that read_image_and_mask raises ValueError for mismatched image and mask dimensions.

    GIVEN: An image and a mask with different dimensions.
    WHEN: The read_image_and_mask function is called.
    THEN: A ValueError should be raised.
    """
    monkeypatch.setattr(
        "SimpleITK.ReadImage",
        lambda path: mock_read_image_and_mask_different_size(path, path)[
            0 if "image" in path else 1
        ],
    )

    with pytest.raises(ValueError, match="Image and mask dimensions do not match."):
        read_image_and_mask("image.nii", "mask.nii")


# ---------------- Get Patient Image Mask Dict Tests ----------------


@pytest.mark.parametrize(
    "imgs_path, masks_path, patient_ids, mode, expected_message",
    [
        (
            [],
            ["mask1.nii", "mask2.nii"],
            {1, 2},
            "2D",
            "The imgs_path, masks_path, and patient_ids cannot be empty.",
        ),
        (
            ["image1.nii", "image2.nii"],
            [],
            {1, 2},
            "2D",
            "The imgs_path, masks_path, and patient_ids cannot be empty.",
        ),
        (
            ["image1.nii", "image2.nii"],
            ["mask1.nii", "mask2.nii"],
            set(),
            "2D",
            "The imgs_path, masks_path, and patient_ids cannot be empty.",
        ),
        (
            ["image1.nii", "image2.nii"],
            ["mask1.nii", "mask2.nii"],
            {1},
            "2D",
            "The number of images, masks, and patient_ids must be the same.",
        ),
    ],
)
def test_get_patient_image_mask_dict_value_error(
    imgs_path, masks_path, patient_ids, mode, expected_message
):
    """
    Test that the function get_patient_image_mask_dict raises ValueError with the correct message.

    GIVEN: Invalid paths and mode or empty list of patient_ids
    WHEN: The function get_patient_image_mask_dict is called
    THEN: ValueError is raised with the correct message
    """
    with pytest.raises(ValueError, match=expected_message):
        get_patient_image_mask_dict(imgs_path, masks_path, patient_ids, mode)


@pytest.mark.parametrize(
    "imgs_path, masks_path, patient_ids, mode, expected_message",
    [
        (
            ["image1.nii", 123],
            ["mask1.nii", "mask2.nii"],
            {1, 2},
            "2D",
            "imgs_path must be a list of strings.",
        ),
        (
            ["image1.nii", "image2.nii"],
            ["mask1.nii", 123],
            {1, 2},
            "2D",
            "masks_path must be a list of strings.",
        ),
        (
            ["image1.nii", "image2.nii"],
            ["mask1.nii", "mask2.nii"],
            [1, 2],
            "2D",
            "patient_ids must be a set of integers.",
        ),
        (
            123,
            ["mask1.nii", "mask2.nii"],
            {1, 2},
            "2D",
            "imgs_path must be a list of strings.",
        ),
        (
            ["image1.nii", "image2.nii"],
            123,
            {1, 2},
            "2D",
            "masks_path must be a list of strings.",
        ),
        (
            ["image1.nii", "image2.nii"],
            ["mask1.nii", "mask2.nii"],
            "1, 2",
            "2D",
            "patient_ids must be a set of integers.",
        ),
    ],
)
def test_get_patient_image_mask_dict_type_error(
    imgs_path, masks_path, patient_ids, mode, expected_message
):
    """
    Test that the function get_patient_image_mask_dict raises TypeError with the correct message.

    GIVEN: Invalid paths or incorrect types for imgs_path, masks_path, or patient_ids
    WHEN: The function get_patient_image_mask_dict is called
    THEN: TypeError is raised with the correct message
    """
    with pytest.raises(TypeError, match=expected_message):
        get_patient_image_mask_dict(imgs_path, masks_path, patient_ids, mode)


@patch("features_extraction.image_processing.read_image_and_mask")
def test_get_patient_image_mask_dict_invalid_mode(mock_read_image):
    """
    Test to verify that the function raises a ValueError for invalid mode.

    GIVEN: A mode that is not '2D' or '3D'.
    WHEN: The get_patient_image_mask_dict function is called.
    THEN: It should raise a ValueError indicating that only '2D' and '3D' modes are allowed.
    """
    mock_read_image.return_value = (None, None)

    imgs_path = ["img1.nii", "img2.nii"]
    masks_path = ["mask1.nii", "mask2.nii"]
    patient_ids = {1, 2}
    mode = "4D"

    with pytest.raises(ValueError, match="Mode should be '2D' or '3D'"):
        get_patient_image_mask_dict(imgs_path, masks_path, patient_ids, mode)


@pytest.mark.parametrize(
    "imgs_path, masks_path, patient_ids, mode, expected_output",
    [
        (
            ["image1.nii", "image2.nii"],
            ["mask1.nii", "mask2.nii"],
            {1, 2},
            "2D",
            {1: ["slice_1_1", "slice_1_2"], 2: ["slice_2_1", "slice_2_2"]},
        ),
        (
            ["image1.nii", "image2.nii"],
            ["mask1.nii", "mask2.nii"],
            {1, 2},
            "3D",
            {1: ["volume_1"], 2: ["volume_2"]},
        ),
    ],
)
@patch("features_extraction.image_processing.read_image_and_mask")
@patch("features_extraction.image_processing.get_slices_2D")
@patch("features_extraction.image_processing.get_volume_3D")
def test_get_patient_image_mask_dict(
    mock_get_volume_3D,
    mock_get_slices_2D,
    mock_read_image_and_mask,
    imgs_path,
    masks_path,
    patient_ids,
    mode,
    expected_output,
):
    """
    Test to verify that the get_patient_image_mask_dict function returns the correct dictionary.

    GIVEN: img_path, mask_path, patient_ids and valid mode
    WHEN: The function get_patient_image_mask_dict is called
    THEN: The result should be a dictionary with the correct structure.
    """

    mock_read_image_and_mask.return_value = ("image_data", "mask_data")

    if mode == "2D":

        def get_mocked_slices(image, mask, patient_id):
            if patient_id == 1:
                return ["slice_1_1", "slice_1_2"]
            elif patient_id == 2:
                return ["slice_2_1", "slice_2_2"]

        mock_get_slices_2D.side_effect = get_mocked_slices

    elif mode == "3D":

        def get_mocked_volume(image, mask, patient_id):
            return [f"volume_{patient_id}"]

        mock_get_volume_3D.side_effect = get_mocked_volume

    result = get_patient_image_mask_dict(imgs_path, masks_path, patient_ids, mode)

    assert result == expected_output, f"Expected {expected_output}, but got {result}"
