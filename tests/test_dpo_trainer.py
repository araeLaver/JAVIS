"""Tests for javis.training.dpo_trainer module."""

import json
import tempfile
from pathlib import Path

import pytest

# Check if required ML modules are available
try:
    from javis.training.dpo_trainer import load_preference_data
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@pytest.mark.skipif(not ML_AVAILABLE, reason="Requires trl and other ML dependencies")
class TestLoadPreferenceData:
    """Tests for load_preference_data function."""

    def test_load_valid_jsonl(self):
        """Test loading valid JSONL preference data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            data = [
                {"prompt": "Hello", "chosen": "Hi there!", "rejected": "Go away"},
                {"prompt": "How are you?", "chosen": "I'm fine", "rejected": "Bad"},
            ]
            for item in data:
                f.write(json.dumps(item) + "\n")
            f.flush()

            dataset = load_preference_data(Path(f.name))

            assert len(dataset) == 2
            assert dataset[0]["prompt"] == "Hello"
            assert dataset[0]["chosen"] == "Hi there!"
            assert dataset[0]["rejected"] == "Go away"

    def test_load_with_empty_lines(self):
        """Test loading JSONL with empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prompt": "Q1", "chosen": "A1", "rejected": "B1"}\n')
            f.write("\n")  # Empty line
            f.write('{"prompt": "Q2", "chosen": "A2", "rejected": "B2"}\n')
            f.write("   \n")  # Whitespace line
            f.flush()

            dataset = load_preference_data(Path(f.name))

            assert len(dataset) == 2

    def test_load_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.flush()

            dataset = load_preference_data(Path(f.name))

            assert len(dataset) == 0


@pytest.mark.skipif(True, reason="Requires trl and heavy ML dependencies")
class TestMainFunction:
    """Tests for main function (mocked dependencies)."""

    def test_main_runs(self):
        """Test that main can be called (requires ML dependencies)."""
        pass
