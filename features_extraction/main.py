import os
import pandas as pd
import configparser
from utils import get_path_images_masks, assign_patient_ids
from image_processing import get_patient_image_mask_dict
from image_feature_extractor import get_extractor, extract_radiomic_features

# Read the configuration .ini file
config = configparser.ConfigParser()
config.read("../config.ini")

data_path = config["paths"]["data_path"]
output_path = config["paths"]["output_path"]
mode = config["settings"]["mode"]
extractor_config = config["settings"]["extractor_config"]


os.makedirs(output_path, exist_ok=True)


images_path, masks_path = get_path_images_masks(data_path)
patient_ids = assign_patient_ids(images_path)
patient_dict = get_patient_image_mask_dict(images_path, masks_path, patient_ids, mode)

extractor = get_extractor(extractor_config)
radiomic_dictionary = extract_radiomic_features(patient_dict, extractor, mode)
radiomic_dataframe = pd.DataFrame(radiomic_dictionary).T.reset_index()


if mode == "2D":
    radiomic_dataframe.rename(
        columns={"index": "PatientID - Slice - Label"}, inplace=True
    )
else:
    radiomic_dataframe.rename(columns={"index": "PatientID - Label"}, inplace=True)

output_file = os.path.join(output_path, f"{mode}_Radiomic_Features.csv")
radiomic_dataframe.to_csv(output_file, sep=",", header=True, index=False)

print(f"Feature extraction completed successfully! Results saved in {output_file}")
