# Radiomic Analysis of 3D MRI Images: Feature Extraction from Lesion Volumes and Individual Slices
---
## Overview
This project enables the extraction of radiomic features from 3D MRI images, particularly focusing on segmented lesions. The extraction can be performed both on:
- **3D segmented lesion volumes**
- **2D slices of the segmented volumes**

Radiomic features are key in medical imaging for quantifying the texture and patterns in lesions that might be indicative of disease. These features are extracted from NIfTI (.nii) format images, and the results are saved in structured CSV files, allowing for easy downstream analysis.
### Extraction Modes
- **3D Mode**: Features are computed from the entire segmented lesion volume.
- **2D Mode**: Features are computed from each individual slice of the segmented lesion volume.

This allows users to choose between extracting features for each individual slice or for the entire lesion volume, depending on their specific needs.

##### ⚠️  2D Mode Considerations
When choosing **2D Mode**, keep in mind that in some slices, multiple regions with the same label may appear. In such cases, **only the largest region for that label is retained for feature extraction**. This ensures consistency and avoids potential bias caused by multiple smaller segmented regions within the same slice

---
## List of Contents
- [Features](#features)
- [Installation](#installation) 
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Utilities](#utilities)
- [Testing](#testing)
- [Results](#results) 
- [License](license)

---
## Features

- **Input**: Accepts 3D MRI images and corresponding segmentation masks in NIfTI (.nii) format.
- **Radiomic Features**: The extraction process relies on a configuration file (.yaml) to define which features (e.g., texture, shape, intensity) should be computed using Pyradiomics. Users can create their own configuration file by referring to the official Pyradiomics documentation: [Radiomic Features- pyradiomics](https://pyradiomics.readthedocs.io/en/latest/features.html)
- Modes: Supports extraction in both **3D** and **2D**:
    - **3D Extraction**: Extracts features from the full 3D volume of a lesion.
    - **2D Extraction**: Extracts features from each individual 2D slice of the lesion.
- **Outputs**:  Results are saved as **CSV datasets**, containing the extracted features for all patients and lesions (per slice in 2D mode), stored in the `output_files/` directory.

---
## Installation

### Prerequisites
Ensure you have **Python 3.11** installed. The project also uses several Python libraries, which are listed in the `requirements.txt` file.
### Clone the Repository
To get started, clone the repository to your local machine. Open a _terminal_ and run the following commands:
```shell
git clone https://github.com/elettralucchesi/Radiomic_Extraction.git
cd Radiomic_Extraction
```
---
## Install Dependecies 

Install the required dependencies by running:
```shell
pip install -r requirements.txt
```
This will install all the necessary Python packages, including Pyradiomics, which is used for feature extraction.

---
## Usage

### Input Data Format
For feature extraction to work correctly, the input **MRI images** and **segmentation masks** must be in **NIfTI (.nii)** format. Additionally, the segmentation masks should have `seg` included in their filename to differentiate them from the MRI image files.
##### Directory Structure
Organize you files in the following structure:
```
📂 data/
├── 📂 PR1/
│   ├── PR1.nii          # MRI Image
│   ├── PR1_seg.nii      # Segmentation Mask
├── 📂 PR2/
│   ├── PR2.nii          # MRI Image
│   ├── PR2_seg.nii      # Segmentation Mask
├── pyradiomics_config.yaml     # Pyradiomics configuration file
```
- A **YAML configuration file** for feature extraction should be provided in the `data/` folder (e.g., `pyradiomics_config.yaml`).

⚠️ **Important**: Inside each patient folder (e.g., `PR1`, `PR2`), there should be **only one MRI image and one segmentation mask**. The mask file must contain the word `seg` in its name (e.g., `PR1_seg.nii`). This ensures that the image and its corresponding mask are correctly paired for feature extraction.

### Configuration
You need to configure the path to the input data and the output directory in the `config.ini` file. It should be modified as follows:
Edit the  file to specify:
```shell
[paths]
data_path = ./data/*           # Path to patient folders containing MRI images and segmentation masks folders
output_path = ./output_files/  # Directory to save extracted features .csv files
[settings]
mode = 3D                      # Mode for feature extraction: '3D' or '2D'
radiomic_config_file = ./data/pyradiomics_config.yaml  # Path to Pyradiomics configuration YAML file
```
This configuration file will specify:
- Where the MRI images and segmentation masks are located (`data_path`).
- Where to save the extracted features (`output_path`).
- The modality of extraction, either 3D or 2D (`mode`).
- The configuration file for Pyradiomics (`radiomic_config_file`), which defines which features to extract.

### Run the Feature Extraction
Once the configuration is set up, you can execute the main script to start the feature extraction process:
```bash
python -m features_extraction.main 
```
---
## Project Structure

```
Radiomic_Features_Extraction/
├── 📂 data                     #  Input MRI images and masks
├── 📂 features_extraction      #  Python code for feature extraction
├── 📂 output files             #  Extracted features in CSV format
├── 📂 tests                    #  Unit tests for validation
├── .gitignore                  #  Git ignore file
├── LICENSE                     #  License file
├── README.md                   #  Project documentation
├── config.ini                  #  Configuration file
├── requirements.txt            #  Python dependencies
```

---
## Utilities

- **The** `utils.py` **file** contains functions to manage MRI image and mask data, ensuring correct pairing of images and masks, extracting and assigning patient IDs, and maintaining consistency in data preparation for feature extraction.
- **The** `image_processing.py` **file** defines a set of functions for processing medical image slices and corresponding masks. It includes functions to extract the largest connected region of a specific label from a binary mask slice, process slices to find the largest region for each label, and extract 2D slices or 3D volumes of images and masks. The functions also handle reading images and masks and organizing them into a dictionary, with the ability to process either 2D slices or full 3D volumes, based on the specified mode.
- **The** `image_feature_extractor.py` **file** extracts radiomic features from 3D and 2D medical images using the radiomics library. It provides functions to initialize the feature extractor and process patient data to extract features from labeled mask regions. The script supports both 2D and 3D image data and handles errors and warnings during feature extraction.
- **The** `main.py` **file** reads a configuration file to set paths and parameters, extracts radiomic features from 2D or 3D medical images, and saves the results in a CSV file. It integrates functions to load images and masks, process patient data, and extract features using a specified configuration. The output is saved to the specified directory, with the mode (2D or 3D) determining the format of the results.

---
## Testing

Unit tests are located in the `tests` directory. To run the tests, use the following command:
```bash
pytest tests/
```
---
## Results

The extracted radiomic features are stored in `output_files/` directory as CSV files. The structure of the CSV files depends on the extraction mode:
- **3D Mode**: Each row corresponds to a segmented lesion from a patient.
- **2D Mode**:  Each row corresponds to a single slice of a segmented lesion from a patient.

The extracted features can then be used for further analysis or machine learning tasks.

The files you see in the `output_files/` directory serve as an example result, generated using the segmentation masks and MRI images stored in the `data/` directory. 
**Please note that the example in the output files is based on the default configuration of 3D mode, which is specified in the** `config.ini` **file. Additionally, the radiomic features are extracted using the provided** `pyradiomics_config.yaml` **file included as an example.**
This configuration file is set up for a comprehensive radiomic feature extraction, including texture features (GLCM, GLRLM, GLSZM, GLDM, NGTDM), first-order statistics, and shape features. The extraction is performed on the original image and also on transformed images using Wavelet and Laplacian of Gaussian (LoG) filters. 
For more detailed information on the extracted features, please refer to this website: [Radiomic Features- pyradiomics](https://pyradiomics.readthedocs.io/en/latest/features.html)

---
## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---
## Author

This project was developed by Elettra Lucchesi
