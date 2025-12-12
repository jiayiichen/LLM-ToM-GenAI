"""
Unit tests for parse_inference module.

Tests the streaming response parser functionality including:
- Parsing valid streaming responses
- Extracting answers in [[X]] format
- Handling edge cases and malformed input
"""

import pytest
import json
import os
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parse_inference.parse_inference import parse_streaming_response, process_directory


class TestParseStreamingResponse:
    """Tests for parse_streaming_response function."""

    def test_parse_valid_response(self, tmp_path):
        """Test parsing a valid streaming response with answer."""
        # Create test file with valid streaming format
        test_content = '''data: {"choices":[{"delta":{"content":"Let me analyze "}}]}
data: {"choices":[{"delta":{"content":"this step by step. "}}]}
data: {"choices":[{"delta":{"content":"The answer is [[B]]"}}]}
data: [DONE]'''

        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)

        result = parse_streaming_response(str(test_file))

        assert result is not None
        assert result['answer'] == 'B'
        assert 'step by step' in result['content']
        assert '[[B]]' in result['content']

    def test_parse_response_with_reasoning_field(self, tmp_path):
        """Test parsing response with separate reasoning_content field."""
        test_content = '''data: {"choices":[{"delta":{"reasoning_content":"Thinking about this..."}}]}
data: {"choices":[{"delta":{"reasoning_content":" The key insight is..."}}]}
data: {"choices":[{"delta":{"content":"[[A]]"}}]}
data: [DONE]'''

        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)

        result = parse_streaming_response(str(test_file))

        assert result is not None
        assert result['answer'] == 'A'
        assert 'Thinking about this' in result['reasoning']
        assert 'key insight' in result['reasoning']

    def test_parse_no_answer(self, tmp_path):
        """Test parsing response without a valid answer."""
        test_content = '''data: {"choices":[{"delta":{"content":"I'm not sure about this question."}}]}
data: [DONE]'''

        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)

        result = parse_streaming_response(str(test_file))

        assert result is not None
        assert result['answer'] is None
        assert "not sure" in result['content']

    def test_parse_file_not_found(self):
        """Test handling of non-existent file."""
        result = parse_streaming_response("/nonexistent/path/file.txt")
        assert result is None

    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = parse_streaming_response(str(test_file))

        assert result is not None
        assert result['answer'] is None
        assert result['content'] == ""

    def test_parse_malformed_json(self, tmp_path):
        """Test handling of malformed JSON in response."""
        test_content = '''data: not valid json
data: {"choices":[{"delta":{"content":"Valid content [[C]]"}}]}
data: [DONE]'''

        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)

        result = parse_streaming_response(str(test_file))

        assert result is not None
        assert result['answer'] == 'C'

    def test_parse_all_answer_options(self, tmp_path):
        """Test that all answer options A, B, C, D are correctly parsed."""
        for answer in ['A', 'B', 'C', 'D']:
            test_content = f'data: {{"choices":[{{"delta":{{"content":"[[{answer}]]"}}}}]}}\ndata: [DONE]'
            test_file = tmp_path / f"test_{answer}.txt"
            test_file.write_text(test_content)

            result = parse_streaming_response(str(test_file))

            assert result['answer'] == answer, f"Failed to parse answer {answer}"


class TestProcessDirectory:
    """Tests for process_directory function."""

    def test_process_multiple_files(self, tmp_path):
        """Test processing multiple files in a directory."""
        # Create test files
        for i in range(3):
            content = f'data: {{"choices":[{{"delta":{{"content":"Response {i} [[{"ABC"[i]}]]"}}}}]}}\ndata: [DONE]'
            (tmp_path / f"{i}.txt").write_text(content)

        output_file = tmp_path / "output.json"
        results = process_directory(str(tmp_path), str(output_file))

        assert len(results) == 3
        assert results[0]['answer'] == 'A'
        assert results[1]['answer'] == 'B'
        assert results[2]['answer'] == 'C'

        # Verify output file was created
        assert output_file.exists()
        with open(output_file) as f:
            saved_results = json.load(f)
        assert len(saved_results) == 3

    def test_process_empty_directory(self, tmp_path):
        """Test processing an empty directory."""
        output_file = tmp_path / "output.json"
        results = process_directory(str(tmp_path), str(output_file))

        assert len(results) == 0

    def test_process_nonexistent_directory(self, tmp_path):
        """Test handling of non-existent directory."""
        output_file = tmp_path / "output.json"
        results = process_directory("/nonexistent/directory", str(output_file))

        assert len(results) == 0

    def test_numerical_sorting(self, tmp_path):
        """Test that files are sorted numerically, not alphabetically."""
        # Create files out of order
        for i in [10, 2, 1, 20]:
            content = f'data: {{"choices":[{{"delta":{{"content":"File {i} [[A]]"}}}}]}}\ndata: [DONE]'
            (tmp_path / f"{i}.txt").write_text(content)

        output_file = tmp_path / "output.json"
        results = process_directory(str(tmp_path), str(output_file))

        # Should be sorted as 1, 2, 10, 20 (not 1, 10, 2, 20)
        assert results[0]['id'] == '1'
        assert results[1]['id'] == '2'
        assert results[2]['id'] == '10'
        assert results[3]['id'] == '20'


class TestAnswerExtraction:
    """Tests for answer extraction edge cases."""

    def test_multiple_answers_takes_first(self, tmp_path):
        """Test that when multiple answers exist, the first one is taken."""
        test_content = '''data: {"choices":[{"delta":{"content":"Maybe [[A]] or maybe [[B]]"}}]}
data: [DONE]'''

        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)

        result = parse_streaming_response(str(test_file))

        assert result['answer'] == 'A'

    def test_lowercase_answer_not_matched(self, tmp_path):
        """Test that lowercase answers like [[a]] are not matched."""
        test_content = '''data: {"choices":[{"delta":{"content":"The answer is [[a]]"}}]}
data: [DONE]'''

        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)

        result = parse_streaming_response(str(test_file))

        assert result['answer'] is None

    def test_invalid_answer_letter_not_matched(self, tmp_path):
        """Test that invalid letters like [[E]] are not matched."""
        test_content = '''data: {"choices":[{"delta":{"content":"The answer is [[E]]"}}]}
data: [DONE]'''

        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)

        result = parse_streaming_response(str(test_file))

        assert result['answer'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
