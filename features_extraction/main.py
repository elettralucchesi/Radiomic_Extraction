import os
import pandas as pd
import configparser
import utils
from image_processing import get_patient_image_mask_dict
from image_feature_extractor import get_extractor, extract_radiomic_features

# Read the configuration .ini file
config = configparser.ConfigParser()
config.read("../config.ini")


print("Sections found:", config.sections())


data_path = config["paths"]["data_path"]
output_path = config["paths"]["output_path"]
mode = config["settings"]["mode"]
extractor_config = config["settings"]["extractor_config"]

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Get image and mask paths
images_path, masks_path = utils.get_path_images_masks(data_path)
patient_ids = utils.assign_patient_ids(images_path)

# Create patient dictionary
patient_dict = get_patient_image_mask_dict(images_path, masks_path, patient_ids, mode)

# Create extractor
extractor = get_extractor(extractor_config)

# Extract radiomic features
radiomic_dictionary = extract_radiomic_features(patient_dict, extractor, mode)

# Convert to DataFrame and save
radiomic_dataframe = pd.DataFrame(radiomic_dictionary).T.reset_index()

# Rename columns based on mode
if mode == "2D":
    radiomic_dataframe.rename(columns={'index': 'PatientID - Slice - Label'}, inplace=True)
else:
    radiomic_dataframe.rename(columns={'index': 'PatientID - Label'}, inplace=True)

output_file = os.path.join(output_path, f"{mode}_Radiomic_Features.csv")
radiomic_dataframe.to_csv(output_file, sep=",", header=True, index=False)

print(f"Feature extraction completed successfully! Results saved in {output_file}") 
