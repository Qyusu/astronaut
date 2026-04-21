from astronaut.logics.common.code_utils import insert_molecule_name_to_path


class TestInsertMoleculeNameToPath:
    """Test class for insert_molecule_name_to_path function."""

    def test_basic_yaml_file(self):
        """Test basic YAML file path conversion."""
        result = insert_molecule_name_to_path(
            "../hoge/hoge/astronaut.yaml", "h2"
        )
        expected = "../hoge/hoge/astronaut-h2.yaml"
        assert result == expected

    def test_absolute_path(self):
        """Test with absolute path."""
        result = insert_molecule_name_to_path("/path/to/config.yaml", "h2")
        expected = "/path/to/config-h2.yaml"
        assert result == expected

    def test_multiple_dots_in_filename(self):
        """Test with filename that contains multiple dots."""
        result = insert_molecule_name_to_path(
            "../configs/app.config.yaml", "h2"
        )
        expected = "../configs/app.config-h2.yaml"
        assert result == expected

    def test_different_extensions(self):
        """Test with different file extensions."""
        test_cases = [
            ("config.json", "h2", "config-h2.json"),
            ("settings.ini", "h2", "settings-h2.ini"),
            ("data.csv", "h2", "data-h2.csv"),
        ]

        for input_path, mole_name, expected in test_cases:
            result = insert_molecule_name_to_path(input_path, mole_name)
            assert result == expected

    def test_empty_mole_name(self):
        """Test with empty mole_name."""
        result = insert_molecule_name_to_path("../configs/astronaut.yaml", "")
        expected = "../configs/astronaut.yaml"
        assert result == expected

    def test_mole_name_with_special_chars(self):
        """Test with mole_name containing special characters."""
        result = insert_molecule_name_to_path("config.yaml", "h2_specific-1")
        expected = "config-h2_specific-1.yaml"
        assert result == expected
