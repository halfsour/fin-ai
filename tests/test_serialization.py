"""Unit tests for serialization module — serialize/deserialize profiles and parse assessment responses."""

import json

import pytest

from retirement_planner.models import (
    BankAccount,
    CreditCard,
    FinancialProfile,
    InvestmentAccount,
    MonthlyBudgetItem,
    RetirementAssessment,
)
from retirement_planner.serialization import (
    deserialize_profile,
    parse_assessment_response,
    serialize_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_profile() -> FinancialProfile:
    return FinancialProfile(
        investments=[
            InvestmentAccount(account_type="401k", balance=50000.0, expected_annual_return=7.0),
            InvestmentAccount(account_type="IRA", balance=25000.0, expected_annual_return=6.5),
        ],
        bank_accounts=[
            BankAccount(account_type="checking", balance=10000.0, monthly_income_deposits=5000.0),
        ],
        credit_cards=[
            CreditCard(outstanding_balance=3000.0, credit_limit=10000.0, monthly_payment=500.0),
        ],
    )


@pytest.fixture
def sample_assessment() -> RetirementAssessment:
    return RetirementAssessment(
        can_retire=True,
        retirement_readiness_summary="You are on track for retirement.",
        recommended_monthly_budget=[
            MonthlyBudgetItem(category="Housing", amount=1500.0),
            MonthlyBudgetItem(category="Food", amount=600.0),
        ],
        net_worth=82000.0,
        monthly_cash_flow=4500.0,
        assumptions={"inflation_rate": 0.03, "retirement_age": 65},
        disclaimer="This is AI-generated and does not constitute professional financial advice.",
    )


# ---------------------------------------------------------------------------
# serialize_profile / deserialize_profile
# ---------------------------------------------------------------------------

class TestSerializeProfile:
    def test_round_trip(self, sample_profile: FinancialProfile) -> None:
        json_str = serialize_profile(sample_profile)
        restored = deserialize_profile(json_str)
        assert restored == sample_profile

    def test_output_is_valid_json(self, sample_profile: FinancialProfile) -> None:
        json_str = serialize_profile(sample_profile)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "investments" in parsed
        assert "bank_accounts" in parsed
        assert "credit_cards" in parsed

    def test_empty_profile_round_trip(self) -> None:
        profile = FinancialProfile(investments=[], bank_accounts=[], credit_cards=[])
        json_str = serialize_profile(profile)
        restored = deserialize_profile(json_str)
        assert restored == profile

    def test_deserialize_invalid_json_raises(self) -> None:
        with pytest.raises(Exception):
            deserialize_profile("not valid json")

    def test_deserialize_wrong_schema_raises(self) -> None:
        with pytest.raises(Exception):
            deserialize_profile('{"foo": "bar"}')


# ---------------------------------------------------------------------------
# parse_assessment_response
# ---------------------------------------------------------------------------

class TestParseAssessmentResponse:
    def test_parse_raw_json(self, sample_assessment: RetirementAssessment) -> None:
        raw = sample_assessment.model_dump_json()
        result = parse_assessment_response(raw)
        assert result == sample_assessment

    def test_parse_json_in_code_fence(self, sample_assessment: RetirementAssessment) -> None:
        raw_json = sample_assessment.model_dump_json()
        response = f"Here is the assessment:\n```json\n{raw_json}\n```\nHope this helps!"
        result = parse_assessment_response(response)
        assert result == sample_assessment

    def test_parse_json_in_plain_code_fence(self, sample_assessment: RetirementAssessment) -> None:
        raw_json = sample_assessment.model_dump_json()
        response = f"```\n{raw_json}\n```"
        result = parse_assessment_response(response)
        assert result == sample_assessment

    def test_parse_json_embedded_in_text(self, sample_assessment: RetirementAssessment) -> None:
        raw_json = sample_assessment.model_dump_json()
        response = f"Based on my analysis, here is the result: {raw_json} Let me know if you have questions."
        result = parse_assessment_response(response)
        assert result == sample_assessment

    def test_parse_invalid_response_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_assessment_response("No JSON here at all.")

    def test_parse_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_assessment_response("")

    def test_parse_json_missing_required_fields_raises(self) -> None:
        incomplete = '{"can_retire": true}'
        with pytest.raises(ValueError, match="Could not parse"):
            parse_assessment_response(incomplete)
