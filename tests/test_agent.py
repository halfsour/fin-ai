"""Unit tests for retirement_planner.agent module.

All tests mock the Strands Agent — no real Bedrock calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from retirement_planner.agent import (
    _build_system_prompt,
    _handle_aws_error,
    create_agent,
    normalize_raw_data,
    run_follow_up,
    run_initial_assessment,
)
from retirement_planner.models import (
    BankAccount,
    BedrockError,
    CredentialError,
    CreditCard,
    FinancialProfile,
    InvestmentAccount,
    MonthlyBudgetItem,
    PersonalInfo,
    RetirementAssessment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_profile():
    return FinancialProfile(
        investments=[InvestmentAccount(account_type="401k", balance=100000, expected_annual_return=7.0)],
        bank_accounts=[BankAccount(account_type="checking", balance=25000, monthly_income_deposits=5000)],
        credit_cards=[CreditCard(outstanding_balance=3000, credit_limit=10000, monthly_payment=500)],
    )


@pytest.fixture
def sample_personal_info():
    return PersonalInfo(husband_age=55, wife_age=52, children_ages=[20, 17])


@pytest.fixture
def sample_assessment_json():
    return json.dumps({
        "can_retire": True,
        "retirement_readiness_summary": "You are on track for retirement.",
        "recommended_monthly_budget": [
            {"category": "Housing", "amount": 2000},
            {"category": "Food", "amount": 800},
        ],
        "net_worth": 122000,
        "monthly_cash_flow": 4500,
        "assumptions": {
            "inflation_rate": 0.03,
            "investment_return": 0.07,
            "retirement_age": 65,
        },
        "disclaimer": "This analysis is AI-generated and does not constitute professional financial advice.",
    })


# ---------------------------------------------------------------------------
# System Prompt Tests
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def setup_method(self):
        self.prompt = _build_system_prompt()

    def test_contains_cpa_keyword(self):
        assert "CPA" in self.prompt

    def test_contains_cfa_keyword(self):
        assert "CFA" in self.prompt

    def test_contains_social_security(self):
        assert "SS" in self.prompt or "Social Security" in self.prompt

    def test_contains_inflation(self):
        assert "inflation" in self.prompt.lower()

    def test_contains_investment_growth(self):
        assert "returns" in self.prompt.lower()

    def test_contains_ages_instruction(self):
        assert "birthdates" in self.prompt.lower() or "ages" in self.prompt.lower()

    def test_contains_json_schema_instruction(self):
        assert "can_retire" in self.prompt
        assert "recommended_monthly_budget" in self.prompt

    def test_contains_assumption_change_instruction(self):
        assert "assumption" in self.prompt.lower()

    def test_contains_disclaimer(self):
        assert "disclaimer" in self.prompt.lower()


# ---------------------------------------------------------------------------
# create_agent Tests
# ---------------------------------------------------------------------------

class TestCreateAgent:
    def test_creates_agent_with_correct_model(self):
        mock_bedrock_model_cls = MagicMock()
        mock_agent_cls = MagicMock()

        mock_strands = MagicMock()
        mock_strands.Agent = mock_agent_cls
        mock_strands_bedrock = MagicMock()
        mock_strands_bedrock.BedrockModel = mock_bedrock_model_cls

        with patch.dict("sys.modules", {
            "strands": mock_strands,
            "strands.models": MagicMock(),
            "strands.models.bedrock": mock_strands_bedrock,
        }):
            # Re-import to pick up the patched modules
            import importlib
            import retirement_planner.agent as agent_mod
            importlib.reload(agent_mod)
            agent_mod.create_agent()

        mock_bedrock_model_cls.assert_called_once_with(
            model_id="us.anthropic.claude-sonnet-4-6",
            region_name="us-east-1",
        )
        mock_agent_cls.assert_called_once()

    def test_registers_tools(self):
        mock_bedrock_model_cls = MagicMock()
        mock_agent_cls = MagicMock()

        mock_strands = MagicMock()
        mock_strands.Agent = mock_agent_cls
        mock_strands_bedrock = MagicMock()
        mock_strands_bedrock.BedrockModel = mock_bedrock_model_cls

        with patch.dict("sys.modules", {
            "strands": mock_strands,
            "strands.models": MagicMock(),
            "strands.models.bedrock": mock_strands_bedrock,
        }):
            import importlib
            import retirement_planner.agent as agent_mod
            importlib.reload(agent_mod)
            agent_mod.create_agent()

        call_kwargs = mock_agent_cls.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert tools is not None
        assert len(tools) == 7

    def test_sets_system_prompt(self):
        mock_bedrock_model_cls = MagicMock()
        mock_agent_cls = MagicMock()

        mock_strands = MagicMock()
        mock_strands.Agent = mock_agent_cls
        mock_strands_bedrock = MagicMock()
        mock_strands_bedrock.BedrockModel = mock_bedrock_model_cls

        with patch.dict("sys.modules", {
            "strands": mock_strands,
            "strands.models": MagicMock(),
            "strands.models.bedrock": mock_strands_bedrock,
        }):
            import importlib
            import retirement_planner.agent as agent_mod
            importlib.reload(agent_mod)
            agent_mod.create_agent()

        call_kwargs = mock_agent_cls.call_args
        prompt = call_kwargs.kwargs.get("system_prompt") or call_kwargs[1].get("system_prompt")
        assert "CPA" in prompt
        assert "CFA" in prompt


# ---------------------------------------------------------------------------
# _handle_aws_error Tests
# ---------------------------------------------------------------------------

class TestHandleAwsError:
    def test_no_credentials_error(self):
        from botocore.exceptions import NoCredentialsError
        with pytest.raises(CredentialError, match="AWS credentials not found"):
            _handle_aws_error(NoCredentialsError())

    def test_credential_error_lists_methods(self):
        from botocore.exceptions import NoCredentialsError
        with pytest.raises(CredentialError) as exc_info:
            _handle_aws_error(NoCredentialsError())
        msg = str(exc_info.value)
        assert "AWS_ACCESS_KEY_ID" in msg
        assert "AWS_SECRET_ACCESS_KEY" in msg
        assert "AWS_PROFILE" in msg

    def test_access_denied_error(self):
        from botocore.exceptions import ClientError
        error_response = {
            "Error": {"Code": "AccessDeniedException", "Message": "Access denied"}
        }
        exc = ClientError(error_response, "InvokeModel")
        with pytest.raises(BedrockError) as exc_info:
            _handle_aws_error(exc)
        assert exc_info.value.retryable is False
        assert "Access denied" in str(exc_info.value)

    def test_other_client_error_is_retryable(self):
        from botocore.exceptions import ClientError
        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
        }
        exc = ClientError(error_response, "InvokeModel")
        with pytest.raises(BedrockError) as exc_info:
            _handle_aws_error(exc)
        assert exc_info.value.retryable is True

    def test_generic_exception_is_retryable(self):
        with pytest.raises(BedrockError) as exc_info:
            _handle_aws_error(ConnectionError("network down"))
        assert exc_info.value.retryable is True
        assert "network down" in str(exc_info.value)


# ---------------------------------------------------------------------------
# run_initial_assessment Tests
# ---------------------------------------------------------------------------

class TestRunInitialAssessment:
    def test_returns_valid_assessment(self, sample_profile, sample_personal_info, sample_assessment_json):
        mock_agent = MagicMock()
        mock_agent.return_value = sample_assessment_json

        result = run_initial_assessment(mock_agent, sample_profile, sample_personal_info)

        assert isinstance(result, RetirementAssessment)
        assert result.can_retire is True
        assert result.net_worth == 122000

    def test_passes_profile_and_personal_info_in_prompt(self, sample_profile, sample_personal_info, sample_assessment_json):
        mock_agent = MagicMock()
        mock_agent.return_value = sample_assessment_json

        run_initial_assessment(mock_agent, sample_profile, sample_personal_info)

        call_args = mock_agent.call_args[0][0]
        assert "Financial Profile" in call_args
        assert "Personal Information" in call_args
        assert "55" in call_args  # husband_age
        assert "52" in call_args  # wife_age

    def test_raises_credential_error_on_no_credentials(self, sample_profile, sample_personal_info):
        from botocore.exceptions import NoCredentialsError
        mock_agent = MagicMock()
        mock_agent.side_effect = NoCredentialsError()

        with pytest.raises(CredentialError):
            run_initial_assessment(mock_agent, sample_profile, sample_personal_info)

    def test_raises_bedrock_error_on_api_failure(self, sample_profile, sample_personal_info):
        mock_agent = MagicMock()
        mock_agent.side_effect = ConnectionError("timeout")

        with pytest.raises(BedrockError):
            run_initial_assessment(mock_agent, sample_profile, sample_personal_info)


# ---------------------------------------------------------------------------
# run_follow_up Tests
# ---------------------------------------------------------------------------

class TestRunFollowUp:
    def test_returns_assessment_when_assumptions_change(self, sample_assessment_json):
        mock_agent = MagicMock()
        mock_agent.return_value = sample_assessment_json

        result = run_follow_up(mock_agent, "What if I retire at 67 instead?")

        assert isinstance(result, RetirementAssessment)

    def test_returns_string_for_informational_question(self):
        mock_agent = MagicMock()
        mock_agent.return_value = "Social Security benefits typically start at age 62."

        result = run_follow_up(mock_agent, "When can I start collecting Social Security?")

        assert isinstance(result, str)
        assert "Social Security" in result

    def test_passes_question_in_prompt(self, sample_assessment_json):
        mock_agent = MagicMock()
        mock_agent.return_value = sample_assessment_json

        run_follow_up(mock_agent, "What if I save 20% more?")

        call_args = mock_agent.call_args[0][0]
        assert "What if I save 20% more?" in call_args

    def test_raises_credential_error_on_no_credentials(self):
        from botocore.exceptions import NoCredentialsError
        mock_agent = MagicMock()
        mock_agent.side_effect = NoCredentialsError()

        with pytest.raises(CredentialError):
            run_follow_up(mock_agent, "What if I retire later?")

    def test_raises_bedrock_error_on_api_failure(self):
        mock_agent = MagicMock()
        mock_agent.side_effect = RuntimeError("service unavailable")

        with pytest.raises(BedrockError):
            run_follow_up(mock_agent, "What if I retire later?")


# ---------------------------------------------------------------------------
# normalize_raw_data Tests
# ---------------------------------------------------------------------------

class TestNormalizeRawData:
    def test_extracts_investment_data(self):
        mock_agent = MagicMock()
        mock_agent.return_value = json.dumps([
            {"account_type": "401k", "balance": 50000, "expected_annual_return": 7.0}
        ])

        result = normalize_raw_data(mock_agent, "My 401k has $50,000", "investments")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["account_type"] == "401k"
        assert result[0]["balance"] == 50000

    def test_extracts_banking_data(self):
        mock_agent = MagicMock()
        mock_agent.return_value = json.dumps([
            {"account_type": "checking", "balance": 10000, "monthly_income_deposits": 5000}
        ])

        result = normalize_raw_data(mock_agent, "Checking: $10k, income $5k/mo", "banking")

        assert isinstance(result, list)
        assert result[0]["account_type"] == "checking"

    def test_extracts_credit_card_data(self):
        mock_agent = MagicMock()
        mock_agent.return_value = json.dumps([
            {"outstanding_balance": 2000, "credit_limit": 10000, "monthly_payment": 200}
        ])

        result = normalize_raw_data(mock_agent, "Visa: $2k balance", "credit_cards")

        assert isinstance(result, list)
        assert result[0]["outstanding_balance"] == 2000

    def test_handles_markdown_wrapped_json(self):
        mock_agent = MagicMock()
        mock_agent.return_value = '```json\n[{"account_type": "IRA", "balance": 75000, "expected_annual_return": 6.5}]\n```'

        result = normalize_raw_data(mock_agent, "IRA with 75k", "investments")

        assert isinstance(result, list)
        assert result[0]["balance"] == 75000

    def test_raises_credential_error_on_no_credentials(self):
        from botocore.exceptions import NoCredentialsError
        mock_agent = MagicMock()
        mock_agent.side_effect = NoCredentialsError()

        with pytest.raises(CredentialError):
            normalize_raw_data(mock_agent, "some data", "investments")

    def test_raises_bedrock_error_on_api_failure(self):
        mock_agent = MagicMock()
        mock_agent.side_effect = RuntimeError("model error")

        with pytest.raises(BedrockError):
            normalize_raw_data(mock_agent, "some data", "investments")

    def test_passes_category_fields_in_prompt(self):
        mock_agent = MagicMock()
        mock_agent.return_value = json.dumps([
            {"outstanding_balance": 1000, "credit_limit": 5000, "monthly_payment": 100}
        ])

        normalize_raw_data(mock_agent, "card data", "credit_cards")

        call_args = mock_agent.call_args[0][0]
        assert "outstanding_balance" in call_args
        assert "credit_limit" in call_args
        assert "monthly_payment" in call_args
