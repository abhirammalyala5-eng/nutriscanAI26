# -*- coding: utf-8 -*-
"""
tests/test_vlm_regression_fix.py
================================
Unit test suite verifying the VLM regression fix:
- Thinking text is strictly suppressed and NEVER parsed as ingredient data.
- Empty content triggers controlled failure.
- Malformed JSON is rejected.
- Valid structured JSON is parsed and area fractions normalized.
- Conversational phrases ("Let's check: 0.05", "Then the flatbreads: 40%", "S Check") are strictly rejected.
- Legitimate Indian food names ("Chicken Dum Biryani", "Paneer Butter Masala", "Dal Tadka", "Lemon Rice") are preserved.
- Both dual-view and single-view payload construction include explicit JSON Schema and thinking suppression.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from vision.vlm import (
    is_valid_food_component_name,
    extract_universal_components,
    parse_component_response,
    query_qwen_vlm,
    VLM_OUTPUT_SCHEMA,
    ELEVATION_FACTORS,
)


class TestVLMComponentValidation(unittest.TestCase):

    def test_conversational_phrases_strictly_rejected(self):
        """Verify conversational reasoning phrases can never become food ingredients."""
        invalid_examples = [
            "Let's check: 0.05",
            "Then the flatbreads: 40%",
            "Then the vegetables: 45%",
            "S Check",
            "Then The Flatbreads",
            "Then The Vegetables",
            "First the curry",
            "Also some garnish",
            "Finally the rice",
            "Let's check",
            "Check",
            "Flatbreads",
            "Vegetables",
        ]
        for phrase in invalid_examples:
            self.assertFalse(
                is_valid_food_component_name(phrase),
                f"Expected {phrase!r} to be rejected as an ingredient name",
            )

    def test_legitimate_indian_food_names_accepted(self):
        """Verify legitimate Indian food names and common meal components are accepted."""
        valid_examples = [
            "Chicken Dum Biryani",
            "Paneer Butter Masala",
            "Dal Tadka",
            "Lemon Rice",
            "Steamed Rice",
            "Phulka",
            "Roti",
            "Sambar",
            "Palak Paneer",
            "Chicken",
            "Duck",
            "Onion",
            "Lime",
            "Salad",
        ]
        for name in valid_examples:
            self.assertTrue(
                is_valid_food_component_name(name),
                f"Expected {name!r} to be accepted as a valid food name",
            )

    def test_universal_components_filters_conversational_monologue(self):
        """Verify extract_universal_components never yields conversational items."""
        monologue = """
        Let's check: 0.05 for lime and onion.
        - Then the flatbreads: 40%
        - Then the vegetables: 45%
        - Steamed Rice: 10%
        """
        extracted = extract_universal_components(monologue)
        names = [c["name"] for c in extracted.get("components", [])]
        self.assertNotIn("S Check", names)
        self.assertNotIn("Then The Flatbreads", names)
        self.assertNotIn("Then The Vegetables", names)
        self.assertNotIn("Flatbreads", names)
        # Legitimate component is kept
        self.assertIn("Steamed Rice", names)


class TestVLMResponseParsing(unittest.TestCase):

    def test_thinking_text_never_used_as_components(self):
        """Verify passing raw thinking monologue raises ValueError rather than returning garbage ingredients."""
        thinking_text = (
            "Okay, the user uploaded an overhead plate. Let's check: 0.05 for lime and onion. "
            "Then the flatbreads: 40%. Then the vegetables: 45%."
        )
        with self.assertRaises(ValueError):
            parse_component_response(thinking_text, plate_diameter_cm=25.0)

    def test_empty_content_triggers_controlled_failure(self):
        """Verify empty string or whitespace raises ValueError."""
        for empty in ("", "   ", "\n\t"):
            with self.assertRaises(ValueError):
                parse_component_response(empty, plate_diameter_cm=25.0)

    def test_malformed_json_rejected(self):
        """Verify malformed JSON without component structures is rejected."""
        malformed = '{"dish": "Meal", "wrong_field": 123}'
        with self.assertRaises(ValueError):
            parse_component_response(malformed, plate_diameter_cm=25.0)

    def test_structured_json_parsed_and_fractions_normalized(self):
        """Verify valid structured JSON parses cleanly and normalizes area fractions to 1.0."""
        valid_json = """
        {
            "dish": "Chicken Dum Biryani with Raita",
            "side_view_observation": "Biryani mound is heaped tall; raita is in a low bowl.",
            "components": [
                {"name": "Chicken Biryani", "fraction": 0.6, "elevation": "heaped"},
                {"name": "Raita", "fraction": 0.2, "elevation": "flat"},
                {"name": "Sliced Onions", "fraction": 0.1, "elevation": "flat"}
            ]
        }
        """
        parsed = parse_component_response(valid_json, plate_diameter_cm=25.0, has_side_view=True)
        self.assertEqual(parsed["dish"].lower(), "chicken dum biryani with raita")
        self.assertIn("Biryani mound is heaped tall", parsed["side_view_observation"])

        comps = parsed["components"]
        self.assertEqual(len(comps), 3)

        # Fractions must sum to 1.0 (normalized from 0.6 + 0.2 + 0.1 = 0.9)
        total_frac = sum(c["food_area_fraction"] for c in comps)
        self.assertAlmostEqual(total_frac, 1.0, places=2)

        # Check elevations
        self.assertEqual(comps[0]["name"], "Chicken Biryani")
        self.assertEqual(comps[0]["elevation"], "heaped")
        self.assertTrue(comps[0]["side_view_used"])

        self.assertEqual(comps[1]["name"], "Raita")
        self.assertEqual(comps[1]["elevation"], "flat")


class TestVLMOllamaPayloadAndInference(unittest.IsolatedAsyncioTestCase):

    async def test_ollama_payload_enforces_schema_and_prefill(self):
        """Verify query_qwen_vlm passes format=VLM_OUTPUT_SCHEMA, think=False, and assistant prefill."""
        mock_reply = {
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"dish": "Dal Tadka with Rice", "components": [{"name": "Rice", "fraction": 0.6, "elevation": "medium"}, {"name": "Dal Tadka", "fraction": 0.4, "elevation": "medium"}]}',
                "thinking": ""
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_reply
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await query_qwen_vlm(
                top_thumb_b64="TOP_B64",
                plate_diameter_cm=25.0,
                ollama_url="http://127.0.0.1:11434",
                qwen_model="qwen3-vl:2b",
                side_thumb_b64="SIDE_B64",
            )

            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]

            # Assertions on structured payload
            self.assertFalse(payload["think"])
            self.assertEqual(payload["format"], VLM_OUTPUT_SCHEMA)
            self.assertEqual(payload["options"]["temperature"], 0)
            self.assertEqual(payload["options"]["num_predict"], 768)

            # Assert assistant prefill suppresses thinking
            messages = payload["messages"]
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[1]["role"], "assistant")
            self.assertEqual(messages[1]["content"], "<think>\n</think>\n")

            # Assert parsed result
            self.assertEqual(result["dish"].lower(), "dal tadka with rice")
            self.assertEqual(len(result["components"]), 2)

    async def test_empty_content_raises_value_error_without_using_thinking(self):
        """Verify when Ollama returns empty content, query_qwen_vlm raises ValueError without exposing thinking."""
        mock_reply = {
            "done_reason": "length",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Let's check the food: - Then the flatbreads: 40%..."
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_reply
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            with self.assertRaises(ValueError) as ctx:
                await query_qwen_vlm(
                    top_thumb_b64="TOP_B64",
                    plate_diameter_cm=25.0,
                    ollama_url="http://127.0.0.1:11434",
                    qwen_model="qwen3-vl:2b",
                )
            self.assertIn("empty content", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
