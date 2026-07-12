import json

import pytest

from tokenizer_analysis.config.language_metadata import LanguageMetadata
from tokenizer_analysis.constants import PACKAGE_ROOT


def test_relative_data_paths_resolve_from_package_root(tmp_path):
    config_path = tmp_path / "langs.json"
    config_path.write_text(
        json.dumps(
            {
                "languages": {
                    "eng_Latn": {
                        "name": "English",
                        "iso_code": "en",
                        "data_path": "parallel/eng_Latn.txt",
                    }
                },
                "analysis_groups": {},
            }
        ),
        encoding="utf-8",
    )

    metadata = LanguageMetadata(str(config_path))

    expected = str((PACKAGE_ROOT / "parallel/eng_Latn.txt").resolve())
    assert metadata.get_data_path("eng_Latn") == expected
    assert metadata.get_language_paths()["eng_Latn"] == expected


def test_absolute_data_paths_are_preserved(tmp_path):
    data_file = tmp_path / "eng_Latn.txt"
    data_file.write_text("hello\n", encoding="utf-8")

    config_path = tmp_path / "langs.json"
    config_path.write_text(
        json.dumps(
            {
                "languages": {
                    "eng_Latn": {
                        "name": "English",
                        "iso_code": "en",
                        "data_path": str(data_file.resolve()),
                    }
                },
                "analysis_groups": {},
            }
        ),
        encoding="utf-8",
    )

    metadata = LanguageMetadata(str(config_path))

    assert metadata.get_data_path("eng_Latn") == str(data_file.resolve())


def test_missing_data_path_raises_key_error(tmp_path):
    config_path = tmp_path / "langs.json"
    config_path.write_text(
        json.dumps(
            {
                "languages": {
                    "eng_Latn": {
                        "name": "English",
                        "iso_code": "en",
                    }
                },
                "analysis_groups": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="data_path"):
        LanguageMetadata(str(config_path))
