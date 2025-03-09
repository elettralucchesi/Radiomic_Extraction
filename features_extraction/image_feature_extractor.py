import os
from radiomics import featureextractor
import logging
import SimpleITK as sitk
import numpy as np


def get_extractor(yaml_path):
    """
    Create a RadiomicsFeatureExtractor with a specified configuration file.

    GIVEN
    -----
    yaml_path : str
        Path to the YAML file containing configuration parameters.

    WHEN
    ----
    The function initializes the feature extractor using the given YAML configuration.

    THEN
    ----
    Returns a configured RadiomicsFeatureExtractor object.

    Raises
    ------
    TypeError
        If `yaml_path` is not a string.
    ValueError
        If `yaml_path` is empty.
    FileNotFoundError
        If `yaml_path` does not exist.
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
    Extract radiomic features from 3D medical images.

    GIVEN
    -----
    patient_dict_3D : dict
        Dictionary containing patient 3D images and masks.
    extractor : RadiomicsFeatureExtractor
        Configured feature extractor.

    WHEN
    ----
    The function iterates through each patient, extracting features from labeled mask regions.

    THEN
    ----
    Returns a dictionary containing extracted features for each patient and label.

    Raises
    ------
    TypeError
        If `patient_dict_3D` is not a dictionary.
        If `extractor` is not an instance of `RadiomicsFeatureExtractor`.
    ValueError
        If `patient_dict_3D` is empty
        If no labels are found in the mask for a given patient.
    """
    
    if not isinstance(patient_dict_3D, dict):
        raise TypeError("patient_dict_3D must be a dictionary.")

    if not isinstance(extractor, featureextractor.RadiomicsFeatureExtractor):
        raise TypeError("extractor must be an instance of RadiomicsFeatureExtractor.")

    if not patient_dict_3D:
        raise ValueError("patient_dict_3D cannot be empty.")

    
    all_features_3D = {}
    errors = []

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
                errors.append(f"Invalid Feature for patient PR{pr_id}, label {lbl}: {e}")
                print(f"Invalid Feature for patient PR{pr_id}, label {lbl}: {e}")
                
    return all_features_3D



def radiomic_extractor_2D(patient_dict_2D, extractor):
    """
    Extract radiomic features from 2D medical image slices.

    GIVEN
    -----
    patient_dict_2D : dict
        Dictionary containing patient 2D slices.
    extractor : RadiomicsFeatureExtractor
        Configured feature extractor.

    WHEN
    ----
    The function iterates through each patient slice, extracting features from labeled mask regions.

    THEN
    ----
    Returns a dictionary containing extracted features for each patient slice and label.

    Raises
    ------
    ValueError
        If no labels are found in the mask for a given patient slice.
    """

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

                features = {"MaskLabel": lbl, "SliceIndex": index, "PatientID": patient_id, **features}
                key = f"{patient_id}-{index}-{lbl}"
                all_features_2D[key] = features
                # Debug log to check if the key is being added
                logging.debug(f"Added features for {key}: {features}")
            except Exception as e:
                logging.error(f"[Invalid Feature] for patient {patient_id}, Slice {index}, Label {lbl}: {e}")
    return all_features_2D


def extract_radiomic_features(patient_dict, extractor, mode="3D"):
    """
    Extract radiomic features from medical images in either 2D or 3D mode.

    GIVEN
    -----
    patient_dict : dict
        Dictionary containing patient data.
    extractor : RadiomicsFeatureExtractor
        Configured feature extractor.
    mode : str
        Processing mode, either "2D" or "3D". Defaults to "3D".

    WHEN
    ----
    The function processes the patient dictionary based on the specified mode.

    THEN
    ----
    Returns a dictionary with extracted radiomic features.

    Raises
    ------
    TypeError
        If `patient_dict` is not a dictionary.
    ValueError
        If `mode` is not "2D" or "3D".
        If `extractor` is not properly configured.
    """
    if not isinstance(patient_dict, dict):
        raise TypeError("patient_dict must be a dictionary.")
    if mode not in ["2D", "3D"]:
        raise ValueError("Invalid mode. Choose either '2D' or '3D'.")
    if not hasattr(extractor, 'execute'):
        raise ValueError("Extractor is not configured properly. Ensure it has the necessary methods.")

    if mode == "3D":
        return radiomic_extractor_3D(patient_dict, extractor)
    else:
        return radiomic_extractor_2D(patient_dict, extractor)
