"""Validation of the labelling model's JSON output."""

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class BackchannelClass(Enum):
    """Classes the labelling model may output."""

    NEUTRAL = "Neutral"
    POSITIVE = "Positive"
    EMOTIONAL = "Emotional"
    ANTICIPATION = "Anticipation"
    CONFIRMATION = "Confirmation"
    REPETITION = "Repetition"
    NOT_BACKCHANNEL = "NotBackchannel"


class BackchannelValidator:
    """Checks that a response is ``{"classification": <class>, "word": <non-empty str>}``.

    A response is valid only if both keys are present, no other key is, the
    class is one of :class:`BackchannelClass` and ``word`` is a non-empty
    string. This holds for ``NotBackchannel`` too: a ``NotBackchannel`` answer
    with an empty word is rejected (the extraction treats rejected and
    ``NotBackchannel`` answers the same way).
    """

    REQUIRED_FIELDS = {"classification", "word"}
    LONG_WORD_COUNT = 5

    def __init__(self):
        self.valid_classes = {cls.value for cls in BackchannelClass}

    def validate(self, response: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Return ``(is_valid, parsed_data, error_message)``."""
        try:
            if not response or not response.strip():
                return False, None, "Empty response"
            try:
                data = self._parse_json(response)
            except json.JSONDecodeError as e:
                return False, None, f"Invalid JSON: {str(e)}"
            for check in (self._validate_structure, self._validate_classification, self._validate_word):
                ok, error_msg = check(data)
                if not ok:
                    return False, None, error_msg
            return True, data, None
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse the span from the first ``{`` to the last ``}``; on failure
        retry with single quotes replaced by double quotes."""
        cleaned = response.strip()
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return json.loads(cleaned.replace("'", '"'))

    def _validate_structure(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, dict):
            return False, "Response must be a JSON object"
        missing_fields = self.REQUIRED_FIELDS - set(data.keys())
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        extra_fields = set(data.keys()) - self.REQUIRED_FIELDS
        if extra_fields:
            return False, f"Unexpected fields: {extra_fields}"
        return True, None

    def _validate_classification(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        classification = data.get("classification")
        if not isinstance(classification, str):
            return False, f"Classification must be a string, got {type(classification).__name__}"
        if classification not in self.valid_classes:
            return False, (
                f"Invalid classification '{classification}'. "
                f"Must be one of: {sorted(self.valid_classes)}"
            )
        return True, None

    def _validate_word(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        word = data.get("word")
        classification = data.get("classification")
        if not isinstance(word, str):
            return False, f"Word must be a string, got {type(word).__name__}"
        if not word or not word.strip():
            return False, f"Backchannel classification '{classification}' must have non-empty word field"
        word_count = len(word.split())
        if word_count > self.LONG_WORD_COUNT:
            print(f"Warning: Backchannel word has {word_count} words, which is unusually long: '{word}'")
        return True, None

    def validate_batch(self, responses: List[str]) -> Dict[str, Any]:
        """Validate many responses.

        Returns ``{total, valid, invalid, errors: [{index, response, error}],
        valid_data: [...], classification_counts}``; ``valid_data`` holds the
        parsed valid responses in input order.
        """
        results = {
            "total": len(responses),
            "valid": 0,
            "invalid": 0,
            "errors": [],
            "valid_data": [],
            "classification_counts": {cls.value: 0 for cls in BackchannelClass},
        }
        for i, response in enumerate(responses):
            is_valid, data, error = self.validate(response)
            if is_valid:
                results["valid"] += 1
                results["valid_data"].append(data)
                results["classification_counts"][data["classification"]] += 1
            else:
                results["invalid"] += 1
                results["errors"].append({"index": i, "response": response[:100], "error": error})
        return results
