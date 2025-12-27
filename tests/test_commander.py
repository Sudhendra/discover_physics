"""
Test Commander Module
"""
import unittest
from unittest.mock import patch, MagicMock
from src.commander.uplink import Commander

class TestCommander(unittest.TestCase):
    def setUp(self):
        self.mock_hypothesis = {
            "equation": "I = 1000 / d^2",
            "r_squared": 0.995,
            "complexity": 5
        }

    @patch("src.commander.uplink.completion")
    def test_review_hypothesis_success(self, mock_completion):
        # Mock LLM response for a discovery
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REASONING: This matches the inverse square law perfectly.\nSTATUS: DISCOVERY"
        mock_completion.return_value = mock_response
        
        commander = Commander(api_key="test-key")
        result = commander.review_hypothesis(self.mock_hypothesis, sample_size=100)
        
        self.assertEqual(result["status"], "DISCOVERY")
        self.assertIn("matches the inverse square law", result["reasoning"])

    @patch("src.commander.uplink.completion")
    def test_review_hypothesis_continue(self, mock_completion):
        # Mock LLM response for rejection
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REASONING: This is likely overfitting noise.\nSTATUS: CONTINUE"
        mock_completion.return_value = mock_response
        
        commander = Commander(api_key="test-key")
        result = commander.review_hypothesis(self.mock_hypothesis, sample_size=100)
        
        self.assertEqual(result["status"], "CONTINUE")
        self.assertIn("overfitting noise", result["reasoning"])

    @patch("src.commander.uplink.completion")
    def test_no_api_key(self, mock_completion):
        # Should gracefully fail if no key
        commander = Commander(api_key="")
        # Force api_key to be empty (in case env var is set)
        commander.api_key = None
        
        result = commander.review_hypothesis(self.mock_hypothesis, sample_size=100)
        self.assertEqual(result["status"], "CONTINUE")
        self.assertIn("Commander disabled", result["reasoning"])

if __name__ == '__main__':
    unittest.main()
