"""JSON serialization/deserialization helpers for the Retirement Planner."""

from __future__ import annotations

import json
import re

from retirement_planner.models import FinancialProfile, RetirementAssessment


def serialize_profile(profile: FinancialProfile) -> str:
    """Serialize FinancialProfile to JSON string.

    Uses Pydantic's built-in model_dump_json() for reliable serialization.
    """
    return profile.model_dump_json()


def deserialize_profile(json_str: str) -> FinancialProfile:
    """Deserialize JSON string back to FinancialProfile.

    Uses Pydantic's built-in model_validate_json() which also runs validators.
    """
    return FinancialProfile.model_validate_json(json_str)


def parse_assessment_response(response: str) -> RetirementAssessment:
    """Parse the agent's response into a RetirementAssessment object.

    The response may be a raw JSON string or contain JSON embedded in
    markdown code fences or surrounding text. This function extracts the
    first valid JSON object that conforms to the RetirementAssessment schema.
    """
    # First, try parsing the entire response as JSON directly.
    stripped = response.strip()
    try:
        return RetirementAssessment.model_validate_json(stripped)
    except Exception:
        pass

    # Try extracting JSON from markdown code fences (```json ... ``` or ``` ... ```).
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    for match in fence_pattern.finditer(response):
        candidate = match.group(1).strip()
        try:
            return RetirementAssessment.model_validate_json(candidate)
        except Exception:
            continue

    # Try to find a JSON object in the raw text by locating balanced braces.
    brace_pattern = re.compile(r"\{.*\}", re.DOTALL)
    for match in brace_pattern.finditer(response):
        candidate = match.group(0).strip()
        try:
            # Validate it's actual JSON first, then validate against the model.
            json.loads(candidate)
            return RetirementAssessment.model_validate_json(candidate)
        except Exception:
            continue

    raise ValueError(
        "Could not parse a valid RetirementAssessment from the agent response."
    )
