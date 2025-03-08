import os
from radiomics import featureextractor
import logging

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
    # Configure logging for Pyradiomics
    logger = logging.getLogger('radiomics')
    logger.setLevel(logging.ERROR)

    return extractor
