import pytest
from features_extraction.image_feature_extractor import get_extractor

def test_get_extractor_invalid_type():
    """
    GIVEN a non-string yaml_path
    WHEN get_extractor is called
    THEN it should raise a TypeError.
    """
    with pytest.raises(TypeError, match="yaml_path must be a string."):
        get_extractor(123)

def test_get_extractor_empty_string():
    """
    GIVEN an empty string as yaml_path
    WHEN get_extractor is called
    THEN it should raise a ValueError.
    """
    with pytest.raises(ValueError, match="yaml_path cannot be empty."):
        get_extractor("")

def test_get_extractor_file_not_found():
    """
    GIVEN a non-existent file path
    WHEN get_extractor is called
    THEN it should raise a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError, match="The file 'non_existent.yaml' does not exist."):
        get_extractor("non_existent.yaml")
