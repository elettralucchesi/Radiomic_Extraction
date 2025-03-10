import pytest
from features_extraction.image_feature_extractor import *
from unittest.mock import Mock
import SimpleITK as sitk



# ---------------- Get Extractor Test ----------------


def test_get_extractor_invalid_type():
    """
    Test that a TypeError is raised when the input type is invalid.

    GIVEN a non-string yaml_path
    WHEN get_extractor is called
    THEN it should raise a TypeError.
    """
    with pytest.raises(TypeError, match="yaml_path must be a string."):
        get_extractor(123)


def test_get_extractor_empty_string():
    """
    Test that a ValueError is raised for empty yaml_path.

    GIVEN an empty string as yaml_path
    WHEN get_extractor is called
    THEN it should raise a ValueError.
    """
    with pytest.raises(ValueError, match="yaml_path cannot be empty."):
        get_extractor("")


def test_get_extractor_file_not_found():
    """
    Test that a FileNotFoundError is raised for non-existent YAML files.


    GIVEN a non-existent file path
    WHEN get_extractor is called
    THEN it should raise a FileNotFoundError.
    """
    with pytest.raises(
        FileNotFoundError, match="The file 'non_existent.yaml' does not exist."
    ):
        get_extractor("non_existent.yaml")


# ---------------- Radiomic Extractor 3D Test ----------------


@pytest.mark.parametrize(
    "patient_dict_3D, extractor, expected_message",
    [
        (
            123,
            featureextractor.RadiomicsFeatureExtractor(),
            "patient_dict_3D must be a dictionary.",
        ),
        (
            {},
            "invalid_extractor",
            "extractor must be an instance of RadiomicsFeatureExtractor.",
        ),
    ],
)
def test_radiomic_extractor_3D_type_errors(
    patient_dict_3D, extractor, expected_message
):
    """
    Test that the radiomic_extractor_3D function raises the correct TypeErrors.

    GIVEN: Various invalid inputs for patient_dict_3D and extractor.
    WHEN: The radiomic_extractor_3D function is called.
    THEN: The expected TypeError should be raised with the correct message.
    """

    with pytest.raises(TypeError, match=expected_message):
        radiomic_extractor_3D(patient_dict_3D, extractor)


def test_radiomic_extractor_3D_empty_patient_dict():
    """
    Test that a ValueError is raised when patient_dict_3D is empty.

    GIVEN: An empty dictionary for patient_dict_3D.
    WHEN: The radiomic_extractor_3D function is called.
    THEN: A ValueError should be raised indicating that patient_dict_3D cannot be empty.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()

    with pytest.raises(ValueError, match="patient_dict_3D cannot be empty."):
        radiomic_extractor_3D({}, extractor)


def test_radiomic_extractor_3D_no_labels_in_mask():
    """
    Test that a ValueError is raised when no labels are found in the mask for a given patient.

    GIVEN: A patient_dict_3D with an empty mask (no labels).
    WHEN: The radiomic_extractor_3D function is called with these inputs.
    THEN: A ValueError should be raised indicating no labels in the mask.
    """
    img_1 = sitk.GetImageFromArray(np.random.rand(3, 3, 3))
    mask_1 = sitk.GetImageFromArray(np.zeros((3, 3, 3)))

    patient_dict_3D = {
        123: [{"ImageVolume": img_1, "MaskVolume": mask_1}],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor()

    with pytest.raises(ValueError, match="No labels found in mask for patient 123"):
        radiomic_extractor_3D(patient_dict_3D, extractor)


def test_radiomic_extractor_3D_valid_input_patient_key():
    """
    Test that the radiomic_extractor_3D function returns a dictionary containing the expected patient key.

    GIVEN a valid patient_dict_3D and extractor
    WHEN radiomic_extractor_3D is called with these valid inputs
    THEN it should return a dictionary with the expected patient key
    """

    img_1 = sitk.GetImageFromArray(np.random.rand(3, 3, 3))
    mask_1 = sitk.GetImageFromArray(np.full((3, 3, 3), fill_value=1, dtype=np.uint16))

    patient_dict_3D = {
        123: [{"ImageVolume": img_1, "MaskVolume": mask_1}],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.execute = Mock(return_value={"Feature1": 0.5, "Feature2": 0.8})

    result = radiomic_extractor_3D(patient_dict_3D, extractor)
    assert (
        "PR123 - 1" in result
    ), f"Expected patient key 'PR123 - 1' not found in the result."


def test_radiomic_extractor_3D_valid_input_feature1():
    """
    Test that the radiomic_extractor_3D function returns the correct value for Feature1.

    GIVEN a valid patient_dict_3D and extractor
    WHEN radiomic_extractor_3D is called with these valid inputs
    THEN it should return the correct value for Feature1
    """

    img_1 = sitk.GetImageFromArray(np.random.rand(3, 3, 3))
    mask_1 = sitk.GetImageFromArray(
        np.full((3, 3, 3), fill_value=1, dtype=np.uint16)
    )  # Assuming this will result in a label of 1

    patient_dict_3D = {
        123: [{"ImageVolume": img_1, "MaskVolume": mask_1}],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.execute = Mock(return_value={"Feature1": 0.5, "Feature2": 0.8})

    result = radiomic_extractor_3D(patient_dict_3D, extractor)
    assert (
        result["PR123 - 1"]["Feature1"] == 0.5
    ), f"Expected Feature1 value of 0.5, but got {result['PR123 - 1']['Feature1']}"


def test_radiomic_extractor_3D_valid_input_feature2():
    """
    Test that the radiomic_extractor_3D function returns the correct value for Feature2.

    GIVEN a valid patient_dict_3D and extractor
    WHEN radiomic_extractor_3D is called with these valid inputs
    THEN it should return the correct value for Feature2
    """

    img_1 = sitk.GetImageFromArray(np.random.rand(3, 3, 3))
    mask_1 = sitk.GetImageFromArray(np.full((3, 3, 3), fill_value=1, dtype=np.uint16))

    patient_dict_3D = {
        123: [{"ImageVolume": img_1, "MaskVolume": mask_1}],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.execute = Mock(return_value={"Feature1": 0.5, "Feature2": 0.8})

    result = radiomic_extractor_3D(patient_dict_3D, extractor)
    assert (
        result["PR123 - 1"]["Feature2"] == 0.8
    ), f"Expected Feature2 value of 0.8, but got {result['PR123 - 1']['Feature2']}"


# ---------------- Radiomic Extractor 2D Test ----------------


@pytest.mark.parametrize(
    "patient_dict_2D, extractor, expected_message",
    [
        (
            123,
            featureextractor.RadiomicsFeatureExtractor(),
            "patient_dict_2D must be a dictionary.",
        ),
        (
            {},
            "invalid_extractor",
            "extractor must be an instance of RadiomicsFeatureExtractor.",
        ),
    ],
)
def test_radiomic_extractor_2D_type_errors(
    patient_dict_2D, extractor, expected_message
):
    """
    Test that the radiomic_extractor_2D function raises the correct TypeErrors.

    GIVEN: Various invalid inputs for patient_dict_2D and extractor.
    WHEN: The radiomic_extractor_2D function is called.
    THEN: The expected TypeError should be raised with the correct message.
    """
    with pytest.raises(TypeError, match=expected_message):
        radiomic_extractor_2D(patient_dict_2D, extractor)


def test_radiomic_extractor_2D_empty_patient_dict():
    """
    Test that a ValueError is raised when patient_dict_2D is empty.

    GIVEN: An empty dictionary for patient_dict_2D.
    WHEN: The radiomic_extractor_2D function is called.
    THEN: A ValueError should be raised indicating that patient_dict_2D cannot be empty.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()

    with pytest.raises(ValueError, match="patient_dict_2D cannot be empty."):
        radiomic_extractor_2D({}, extractor)


def test_radiomic_extractor_2D_no_labels_in_mask():
    """
    Test that a ValueError is raised when no labels are found in the mask for a given patient.

    GIVEN: A patient_dict_2D with an empty mask (no labels).
    WHEN: The radiomic_extractor_2D function is called with these inputs.
    THEN: A ValueError should be raised indicating no labels in the mask.
    """
    img_1 = sitk.GetImageFromArray(np.random.rand(2, 2))
    mask_1 = sitk.GetImageFromArray(np.zeros((2, 2)))

    patient_dict_2D = {
        123: [
            {"ImageVolume": img_1, "MaskVolume": mask_1, "Label": 0, "SliceIndex": 0}
        ],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor()

    with pytest.raises(ValueError, match="No labels found in mask for patient 123"):
        radiomic_extractor_2D(patient_dict_2D, extractor)


def test_radiomic_extractor_2D_valid_input_patient_key():
    """
    Test that the function returns a dictionary containing the expected patient key.

    GIVEN a valid patient_dict_2D and extractor
    WHEN radiomic_extractor_2D is called with these valid inputs
    THEN it should return a dictionary with the expected patient key
    """

    img_1 = sitk.GetImageFromArray(np.random.rand(3, 3))
    mask_1 = sitk.GetImageFromArray(
        np.full((3, 3), fill_value=1, dtype=np.uint16)
    )  # Label 1

    patient_dict_2D = {
        123: [{"ImageSlice": img_1, "MaskSlice": mask_1, "Label": 1, "SliceIndex": 0}],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.execute = Mock(return_value={"Feature1": 0.5, "Feature2": 0.8})
    result = radiomic_extractor_2D(patient_dict_2D, extractor)

    assert (
        "123-0-1" in result
    ), f"Expected patient key '123-0-1' not found in the result."


def test_radiomic_extractor_2D_valid_input_feature1():
    """
    Test that the function returns the correct value for Feature1.

    GIVEN a valid patient_dict_2D and extractor
    WHEN radiomic_extractor_2D is called with these valid inputs
    THEN it should return the correct value for Feature1
    """

    img_1 = sitk.GetImageFromArray(np.random.rand(3, 3))
    mask_1 = sitk.GetImageFromArray(
        np.full((3, 3), fill_value=1, dtype=np.uint16)
    )  # Label 1

    patient_dict_2D = {
        123: [{"ImageSlice": img_1, "MaskSlice": mask_1, "Label": 1, "SliceIndex": 0}],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.execute = Mock(return_value={"Feature1": 0.5, "Feature2": 0.8})
    result = radiomic_extractor_2D(patient_dict_2D, extractor)

    assert (
        result["123-0-1"]["Feature1"] == 0.5
    ), f"Expected Feature1 value of 0.5, but got {result['123-0-1']['Feature1']}"


def test_radiomic_extractor_2D_valid_input_feature2():
    """
    Test that the function returns the correct value for Feature2.

    GIVEN a valid patient_dict_2D and extractor
    WHEN radiomic_extractor_2D is called with these valid inputs
    THEN it should return the correct value for Feature2
    """

    img_1 = sitk.GetImageFromArray(np.random.rand(3, 3))
    mask_1 = sitk.GetImageFromArray(
        np.full((3, 3), fill_value=1, dtype=np.uint16)
    )  # Label 1

    patient_dict_2D = {
        123: [{"ImageSlice": img_1, "MaskSlice": mask_1, "Label": 1, "SliceIndex": 0}],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.execute = Mock(return_value={"Feature1": 0.5, "Feature2": 0.8})
    result = radiomic_extractor_2D(patient_dict_2D, extractor)

    result = radiomic_extractor_2D(patient_dict_2D, extractor)
    assert (
        result["123-0-1"]["Feature2"] == 0.8
    ), f"Expected Feature2 value of 0.8, but got {result['123-0-1']['Feature2']}"


# ---------------- Extract Radiomic Features Test ----------------


@pytest.mark.parametrize(
    "patient_dict, extractor, expected_message",
    [
        (
            ["invalid_patient_data"],
            featureextractor.RadiomicsFeatureExtractor(),
            "patient_dict must be a dictionary.",
        ),
        (
            {},
            "invalid_extractor",
            "extractor must be an instance of RadiomicsFeatureExtractor.",
        ),
    ],
)
def test_extract_radiomic_features_type_errors(
    patient_dict, extractor, expected_message
):
    """
    Test that the extract_radiomic_features function raises the correct TypeErrors.

    GIVEN: Various invalid inputs for patient_dict and extractor.
    WHEN: The extract_radiomic_features function is called.
    THEN: The expected TypeError should be raised with the correct message.
    """
    with pytest.raises(TypeError, match=expected_message):
        extract_radiomic_features(patient_dict, extractor)


def test_extract_radiomic_features_invalid_mode():
    """
    Testing that a ValueError is raised when an invalid mode is provided

    GIVEN an invalid mode (not '2D' or '3D')
    WHEN calling extract_radiomic_features
    THEN it should raise a ValueError with the message 'Invalid mode. Choose either '2D' or '3D'.'
    """

    patient_dict = {"patient_id": 1, "image_data": "some_data"}
    extractor = featureextractor.RadiomicsFeatureExtractor()

    with pytest.raises(ValueError, match="Invalid mode. Choose either '2D' or '3D'."):
        extract_radiomic_features(patient_dict, extractor, mode="invalid_mode")
