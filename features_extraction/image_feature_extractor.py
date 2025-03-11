import os
from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import warnings

def get_extractor(yaml_path):
    """
    Initialize a RadiomicsFeatureExtractor using a specified YAML configuration file.

    This function loads configuration parameters from a YAML file and creates a 
    `RadiomicsFeatureExtractor` object for extracting radiomic features.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file containing configuration parameters.

    Returns
    -------
    radiomics.featureextractor.RadiomicsFeatureExtractor
        A configured feature extractor ready for use.

    Raises
    ------
    TypeError
        If `yaml_path` is not a string.
    ValueError
        If `yaml_path` is an empty string.
    FileNotFoundError
        If the specified `yaml_path` does not exist.
    """

    if not isinstance(yaml_path, str):
        raise TypeError("yaml_path must be a string.")

    if not yaml_path:
        raise ValueError("yaml_path cannot be empty.")

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"The file '{yaml_path}' does not exist.")

    extractor = featureextractor.RadiomicsFeatureExtractor(yaml_path)
    return extractor


def radiomic_extractor_3D(patient_dict_3D, extractor):
    """
    Extract radiomic features from 3D medical images using a configured extractor.

    This function processes 3D images and their corresponding masks for each patient,
    extracting radiomic features from labeled regions in the mask.

    Parameters
    ----------
    patient_dict_3D : dict
        Dictionary where each key is a patient ID and the value is a list containing 
        a single dictionary with the 3D image (`"ImageVolume"`) and the mask (`"MaskVolume"`).
    extractor : radiomics.featureextractor.RadiomicsFeatureExtractor
        A configured feature extractor for radiomic feature computation.

    Returns
    -------
    dict
        A dictionary where each key follows the format `"PR{patient_id} - {label}"`,
        and the value is a dictionary of extracted features, including `"MaskLabel"` and `"PatientID"`.

    Raises
    ------
    TypeError
        If `patient_dict_3D` is not a dictionary.
        If `extractor` is not an instance of `RadiomicsFeatureExtractor`.
    ValueError
        If `patient_dict_3D` is empty.
        If no labels are found in the mask for a given patient.

    Warns
    -----
    UserWarning
        If feature extraction fails for a patient, a warning is issued.
    """

    if not isinstance(patient_dict_3D, dict):
        raise TypeError("patient_dict_3D must be a dictionary.")

    if not isinstance(extractor, featureextractor.RadiomicsFeatureExtractor):
        raise TypeError("extractor must be an instance of RadiomicsFeatureExtractor.")

    if not patient_dict_3D:
        raise ValueError("patient_dict_3D cannot be empty.")

    all_features_3D = {}

    for pr_id, patient_data in patient_dict_3D.items():
        patient_volume = patient_data[0]
        img = patient_volume["ImageVolume"]
        mask = patient_volume["MaskVolume"]

        mask_array = sitk.GetArrayFromImage(mask)
        labels = np.unique(mask_array)
        labels = labels[labels != 0]

        if len(labels) == 0:
            raise ValueError(f"No labels found in mask for patient {pr_id}")

        for lbl in labels:
            try:
                features = extractor.execute(img, mask, label=int(lbl))
                features = {"MaskLabel": lbl, "PatientID": pr_id, **features}
                all_features_3D[f"PR{pr_id} - {lbl:d}"] = features
            except Exception as e:
                warning_message = f"Invalid Feature for patient PR{pr_id}, label {lbl}: {e}"
                warnings.warn(warning_message, category=UserWarning)

    return all_features_3D


def radiomic_extractor_2D(patient_dict_2D, extractor):
    """
    Extract radiomic features from 2D medical image slices using a configured extractor.

    This function processes 2D image slices and their corresponding masks for each patient,
    extracting radiomic features from labeled regions in the mask.

    Parameters
    ----------
    patient_dict_2D : dict
        Dictionary where each key is a patient ID and the value is a list of dictionaries.
        Each dictionary represents a single 2D slice and contains:
            - `"ImageSlice"`: The 2D medical image.
            - `"MaskSlice"`: The corresponding 2D mask.
            - `"Label"`: The segmentation label in the mask.
            - `"SliceIndex"`: The index of the slice in the patient volume.
    extractor : radiomics.featureextractor.RadiomicsFeatureExtractor
        A configured feature extractor for radiomic feature computation.

    Returns
    -------
    dict
        A dictionary where each key follows the format `"{patient_id}-{slice_index}-{label}"`,
        and the value is a dictionary of extracted features, including `"MaskLabel"`, `"SliceIndex"`, and `"PatientID"`.

    Raises
    ------
    TypeError
        If `patient_dict_2D` is not a dictionary.
        If `extractor` is not an instance of `RadiomicsFeatureExtractor`.
    ValueError
        If `patient_dict_2D` is empty.
        If no labels are found in the mask for a given patient slice.

    Warns
    -----
    UserWarning
        If feature extraction fails for a patient slice, a warning is issued.
    """

    if not isinstance(patient_dict_2D, dict):
        raise TypeError("patient_dict_2D must be a dictionary.")

    if not isinstance(extractor, featureextractor.RadiomicsFeatureExtractor):
        raise TypeError("extractor must be an instance of RadiomicsFeatureExtractor.")

    if not patient_dict_2D:
        raise ValueError("patient_dict_2D cannot be empty.")

    all_features_2D = {}

    for patient_id, patient_slices in patient_dict_2D.items():
        for slice_data in patient_slices:
            lbl = slice_data["Label"]
            index = slice_data["SliceIndex"]

            if lbl == 0:
                raise ValueError(f"No labels found in mask for patient {patient_id}")

            try:
                img_slice = slice_data["ImageSlice"]
                mask_slice = slice_data["MaskSlice"]

                features = extractor.execute(img_slice, mask_slice, label=int(lbl))

                features = {
                    "MaskLabel": lbl,
                    "SliceIndex": index,
                    "PatientID": patient_id,
                    **features,
                }
                key = f"{patient_id}-{index}-{lbl}"
                all_features_2D[key] = features
            except Exception as e:
                warning_message = f"Invalid Feature for patient PR{patient_id}, label {lbl}: {e}"
                warnings.warn(warning_message, category=UserWarning)

    return all_features_2D


def extract_radiomic_features(patient_dict, extractor, mode="3D"):
    """
    Extract radiomic features from medical images, supporting both 2D and 3D processing modes.

    This function processes a patient dataset and extracts radiomic features using a configured
    feature extractor, either in 2D (slice-wise) or 3D (volume-wise) mode.

    Parameters
    ----------
    patient_dict : dict
        Dictionary containing patient imaging data.
        - If `mode="3D"`, it should contain 3D volumes and corresponding masks.
        - If `mode="2D"`, it should contain lists of 2D slices and their corresponding masks.
    extractor : radiomics.featureextractor.RadiomicsFeatureExtractor
        A configured feature extractor for radiomic feature computation.
    mode : str, optional
        Processing mode, either `"2D"` or `"3D"`. Defaults to `"3D"`.

    Returns
    -------
    dict
        A dictionary containing the extracted radiomic features.
        The structure of the output depends on the processing mode:
        - In `"3D"` mode, features are extracted for entire volumes.
        - In `"2D"` mode, features are extracted for individual slices.

    Raises
    ------
    TypeError
        If `patient_dict` is not a dictionary.
        If `extractor` is not an instance of `RadiomicsFeatureExtractor`.
    ValueError
        If `mode` is not `"2D"` or `"3D"`.
    """

    if not isinstance(patient_dict, dict):
        raise TypeError("patient_dict must be a dictionary.")
    if mode not in ["2D", "3D"]:
        raise ValueError("Invalid mode. Choose either '2D' or '3D'.")
    if not isinstance(extractor, featureextractor.RadiomicsFeatureExtractor):
        raise TypeError("extractor must be an instance of RadiomicsFeatureExtractor.")

    if mode == "3D":
        return radiomic_extractor_3D(patient_dict, extractor)
    else:
        return radiomic_extractor_2D(patient_dict, extractor)
