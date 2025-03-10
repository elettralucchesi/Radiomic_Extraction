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
___
## List of Contents
- [Features](#features)
- [Installation](#installation) 
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Results](#results) 
- [License](license)

___
## Features
___
- **Input**: Accepts 3D MRI images and corresponding segmentation masks in NIfTI (.nii) format.
- **Radiomic Features**: The extraction process relies on a configuration file (.yaml) to define which features (e.g., texture, shape, intensity) should be computed using Pyradiomics. Users can create their own configuration file by referring to the official Pyradiomics documentation: [Radiomic Features- pyradiomics](https://pyradiomics.readthedocs.io/en/latest/features.html)
- Modes: Supports extraction in both **3D** and **2D**:
    - **3D Extraction**: Extracts features from the full 3D volume of a lesion.
    - **2D Extraction**: Extracts features from each individual 2D slice of the lesion.
- **Outputs**:  Results are saved as **CSV datasets**, containing the extracted features for all patients and lesions (per slice in 2D mode), stored in the `output_files/` directory.

---
## Installation
---
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
---
Install the required dependencies by running:
```shell
pip install -r requirements.txt
```
This will install all the necessary Python packages, including Pyradiomics, which is used for feature extraction.
____
## Usage
___
### Input Data Format
For feature extraction to work correctly, the input **MRI images** and **segmentation masks** must be in **NIfTI (.nii)** format. Additionally, the segmentation masks should have `seg` included in their filename to differentiate them from the MRI image files.
##### Directory Structure
Organize you files in the following structure:
```
📂 data/
├── 📂 Patient_1/
│   ├── PR1.nii          # MRI Image
│   ├── PR1_seg.nii      # Segmentation Mask
├── 📂 Patient_2/
│   ├── PR2.nii          # MRI Image
│   ├── PR2_seg.nii      # Segmentation Mask
├── pyradiomics_config.yaml     # Pyradiomics configuration file
```
- A **YAML configuration file** for feature extraction should be provided in the `data/` folder (e.g., `pyradiomics_config.yaml`).

### Configuration
You need to configure the path to the input data and the output directory in the `config.ini` file. It should be modified as follows:
Edit the  file to specify:
```shell
[data]
data_path = ./data/*           # Path to patient folders containing MRI images and segmentation masks folders
output_path = ./output_files/  # Directory to save extracted features .csv files
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
---
```
Radiomic_Features_Extraction/
├── 📂 data                     #  Input MRI images and masks
├── 📂 features_extraction      #  Python code for feature extraction
├── 📂 output files             #  Extracted features in CSV format
├── 📂 tests                    #  Unit tests for validation
├── .gitignore                  #  Git ignore file
├── LICENSE                     #  License file
├── README.md                   # Project documentation
├── config.ini                  # Configuration file
├── requirements.txt            # Python dependencies
```
---
## Testing
---
Unit tests are located in the `tests` directory. To run the tests, use the following command:
```bash
pytest test/
```
---
## Results
---
The extracted radiomic features are stored in `output_files/` directory as CSV files. The structure of the CSV files depends on the extraction mode:
- **3D Mode**: Each row corresponds to a segmented lesion from a patient.
- **2D Mode**:  Each row corresponds to a single slice of a segmented lesion from a patient.

The extracted features can then be used for further analysis or machine learning tasks.

The files you see in the `output_files/` directory serve as an example result, generated using the segmentation masks and MRI images stored in the `data/` directory.
### License
This project is licensed under the **MIT License**. See the `LICENSE` file for more details.