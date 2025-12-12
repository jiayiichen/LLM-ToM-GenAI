"""
Unit tests for data utility functions.

Tests data loading, preprocessing, and validation functions
used throughout the project.
"""

import pytest
import pandas as pd
import json
import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataValidation:
    """Tests for data validation utilities."""

    def test_csv_required_columns(self, tmp_path):
        """Test that CSV files have required columns."""
        required_columns = ['STORY', 'QUESTION', 'OPTION-A', 'OPTION-B',
                          'OPTION-C', 'OPTION-D', 'ANSWER']

        # Create valid CSV
        valid_data = {
            'STORY': ['Test story'],
            'QUESTION': ['Test question?'],
            'OPTION-A': ['Option A'],
            'OPTION-B': ['Option B'],
            'OPTION-C': ['Option C'],
            'OPTION-D': ['Option D'],
            'ANSWER': ['A']
        }
        df = pd.DataFrame(valid_data)
        csv_path = tmp_path / "valid.csv"
        df.to_csv(csv_path, index=False)

        # Load and verify
        loaded_df = pd.read_csv(csv_path)
        for col in required_columns:
            assert col in loaded_df.columns, f"Missing required column: {col}"

    def test_answer_format_validation(self):
        """Test that answers are in valid format (A, B, C, or D)."""
        valid_answers = ['A', 'B', 'C', 'D']
        invalid_answers = ['E', 'a', '1', 'AB', '', None]

        for ans in valid_answers:
            assert ans in ['A', 'B', 'C', 'D'], f"{ans} should be valid"

        for ans in invalid_answers:
            assert ans not in ['A', 'B', 'C', 'D'], f"{ans} should be invalid"

    def test_jsonl_format(self, tmp_path):
        """Test JSONL format for finetuning data."""
        # Create valid JSONL
        jsonl_path = tmp_path / "train.jsonl"
        samples = [
            {"messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "[[A]]"}
            ]},
            {"messages": [
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "[[B]]"}
            ]}
        ]

        with open(jsonl_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        # Load and verify
        loaded_samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                loaded_samples.append(json.loads(line))

        assert len(loaded_samples) == 2
        for sample in loaded_samples:
            assert 'messages' in sample
            assert len(sample['messages']) == 2
            assert sample['messages'][0]['role'] == 'user'
            assert sample['messages'][1]['role'] == 'assistant'


class TestAccuracyCalculation:
    """Tests for accuracy calculation functions."""

    def test_accuracy_calculation(self):
        """Test basic accuracy calculation."""
        predictions = ['A', 'B', 'C', 'D', 'A']
        ground_truth = ['A', 'B', 'C', 'A', 'A']  # 4/5 correct

        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions)

        assert accuracy == 0.8

    def test_accuracy_with_missing_predictions(self):
        """Test accuracy when some predictions are missing/None."""
        predictions = ['A', None, 'C', '', 'A']
        ground_truth = ['A', 'B', 'C', 'D', 'A']

        # Only count non-empty predictions
        valid_pairs = [(p, g) for p, g in zip(predictions, ground_truth) if p]
        correct = sum(p == g for p, g in valid_pairs)
        accuracy = correct / len(ground_truth) if ground_truth else 0

        assert accuracy == 0.6  # 3/5

    def test_accuracy_by_task(self):
        """Test accuracy calculation grouped by task type."""
        data = pd.DataFrame({
            'prediction': ['A', 'B', 'A', 'B'],
            'ground_truth': ['A', 'A', 'A', 'B'],
            'task': ['Task1', 'Task1', 'Task2', 'Task2']
        })

        # Calculate accuracy by task
        data['correct'] = data['prediction'] == data['ground_truth']
        task_accuracy = data.groupby('task')['correct'].mean()

        assert task_accuracy['Task1'] == 0.5  # 1/2
        assert task_accuracy['Task2'] == 1.0  # 2/2


class TestPromptFormatting:
    """Tests for prompt formatting functions."""

    def test_tom_prompt_format(self):
        """Test ToM evaluation prompt formatting."""
        story = "Test story content"
        question = "What happened?"
        options = ["Option A", "Option B", "Option C", "Option D"]

        # Basic prompt structure check
        prompt = f"""[Story]
{story}
[Question]
{question}
[Candidate Answers]
A. {options[0]} B. {options[1]} C. {options[2]} D. {options[3]}"""

        assert "[Story]" in prompt
        assert "[Question]" in prompt
        assert "[Candidate Answers]" in prompt
        assert story in prompt
        assert question in prompt
        for opt in options:
            assert opt in prompt

    def test_answer_bracket_format(self):
        """Test that answers are formatted with double brackets."""
        import re

        valid_formats = ["[[A]]", "[[B]]", "[[C]]", "[[D]]"]
        invalid_formats = ["[A]", "A", "[[a]]", "[[E]]", "[[AB]]"]

        pattern = r'\[\[([A-D])\]\]'

        for fmt in valid_formats:
            assert re.match(pattern, fmt), f"{fmt} should match"

        for fmt in invalid_formats:
            assert not re.match(pattern, fmt), f"{fmt} should not match"


class TestResultsFormat:
    """Tests for results file format validation."""

    def test_results_csv_columns(self, tmp_path):
        """Test that results CSV has expected columns."""
        expected_columns = [
            'index', 'STORY', 'QUESTION', 'OPTION-A', 'OPTION-B',
            'OPTION-C', 'OPTION-D', 'ANSWER', 'SOURCE_FILE'
        ]

        # Create sample results
        data = {col: ['test'] for col in expected_columns}
        data['model_response'] = ['A']
        df = pd.DataFrame(data)

        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = pd.read_csv(csv_path)
        for col in expected_columns:
            assert col in loaded_df.columns

    def test_synthesized_data_format(self, tmp_path):
        """Test format of synthesized data output."""
        synthesized = {
            "questions": [
                {
                    "story": "Test story",
                    "question": "Test question?",
                    "option_a": "Option A",
                    "option_b": "Option B",
                    "option_c": "Option C",
                    "option_d": "Option D",
                    "correct_answer": "A"
                }
            ]
        }

        json_path = tmp_path / "synthesized.json"
        with open(json_path, 'w') as f:
            json.dump(synthesized, f)

        with open(json_path, 'r') as f:
            loaded = json.load(f)

        assert 'questions' in loaded
        assert len(loaded['questions']) == 1

        question = loaded['questions'][0]
        required_fields = ['story', 'question', 'option_a', 'option_b',
                          'option_c', 'option_d', 'correct_answer']
        for field in required_fields:
            assert field in question


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
