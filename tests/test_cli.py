"""Unit tests for the CLI module.

Tests argument parsing, interactive collection flows, assessment display,
conversation session loop, and keyboard interrupt handling.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch, call

import pytest

from retirement_planner.cli import (
    collect_financial_data,
    collect_personal_info,
    display_assessment,
    display_assumption_summary,
    main,
    parse_args,
    prompt_assumption_confirmation,
    run_conversation_session,
)
from retirement_planner.models import (
    BankAccount,
    BedrockError,
    CreditCard,
    CredentialError,
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
        bank_accounts=[BankAccount(account_type="checking", balance=5000, monthly_income_deposits=4000)],
        credit_cards=[CreditCard(outstanding_balance=2000, credit_limit=10000, monthly_payment=500)],
    )


@pytest.fixture
def sample_personal_info():
    return PersonalInfo(husband_age=45, wife_age=43, children_ages=[12, 8])


@pytest.fixture
def sample_assessment():
    return RetirementAssessment(
        can_retire=True,
        retirement_readiness_summary="You are on track for retirement.",
        recommended_monthly_budget=[
            MonthlyBudgetItem(category="Housing", amount=1500),
            MonthlyBudgetItem(category="Food", amount=600),
        ],
        net_worth=103000.0,
        monthly_cash_flow=3500.0,
        assumptions={"retirement_age": 65, "inflation_rate": 3.0},
        disclaimer="This is AI-generated and does not constitute professional financial advice.",
    )


# ---------------------------------------------------------------------------
# Argument Parsing Tests (Requirement 9.2)
# ---------------------------------------------------------------------------

class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_no_arguments(self):
        args = parse_args([])
        assert args.files is None
        assert args.investments is None
        assert args.banking is None
        assert args.credit_cards is None

    def test_files_single(self):
        args = parse_args(["--files", "all_data.txt"])
        assert args.files == ["all_data.txt"]

    def test_files_multiple(self):
        args = parse_args(["--files", "file1.csv", "file2.json"])
        assert args.files == ["file1.csv", "file2.json"]

    def test_investments_argument(self):
        args = parse_args(["--investments", "data/investments.csv"])
        assert args.investments == "data/investments.csv"
        assert args.banking is None
        assert args.credit_cards is None

    def test_banking_argument(self):
        args = parse_args(["--banking", "data/banking.json"])
        assert args.banking == "data/banking.json"

    def test_credit_cards_argument(self):
        args = parse_args(["--credit-cards", "data/cards.txt"])
        assert args.credit_cards == "data/cards.txt"

    def test_all_arguments(self):
        args = parse_args([
            "--investments", "inv.csv",
            "--banking", "bank.json",
            "--credit-cards", "cards.txt",
        ])
        assert args.investments == "inv.csv"
        assert args.banking == "bank.json"
        assert args.credit_cards == "cards.txt"


# ---------------------------------------------------------------------------
# Interactive Collection Tests
# ---------------------------------------------------------------------------

class TestCollectFinancialData:
    """Tests for collect_financial_data with file and interactive paths."""

    def test_loads_from_files_when_paths_provided(self):
        agent = MagicMock()
        args = argparse.Namespace(
            files=None,
            investments="inv.csv",
            banking="bank.json",
            credit_cards="cards.txt",
        )
        inv = [InvestmentAccount(account_type="IRA", balance=50000, expected_annual_return=6.0)]
        bank = [BankAccount(account_type="savings", balance=10000, monthly_income_deposits=3000)]
        cards = [CreditCard(outstanding_balance=1000, credit_limit=5000, monthly_payment=200)]

        with patch("retirement_planner.cli.parse_investments_file", return_value=inv) as mock_inv, \
             patch("retirement_planner.cli.parse_banking_file", return_value=bank) as mock_bank, \
             patch("retirement_planner.cli.parse_credit_cards_file", return_value=cards) as mock_cards:
            profile = collect_financial_data(agent, args)

        mock_inv.assert_called_once_with(agent, "inv.csv")
        mock_bank.assert_called_once_with(agent, "bank.json")
        mock_cards.assert_called_once_with(agent, "cards.txt")
        assert len(profile.investments) == 1
        assert len(profile.bank_accounts) == 1
        assert len(profile.credit_cards) == 1

    def test_prompts_interactively_when_no_files(self):
        agent = MagicMock()
        args = argparse.Namespace(files=None, investments=None, banking=None, credit_cards=None)

        inv = [InvestmentAccount(account_type="401k", balance=100000, expected_annual_return=7.0)]
        bank = [BankAccount(account_type="checking", balance=5000, monthly_income_deposits=4000)]
        cards = [CreditCard(outstanding_balance=2000, credit_limit=10000, monthly_payment=500)]

        with patch("retirement_planner.cli.collect_investments_interactive", return_value=inv), \
             patch("retirement_planner.cli.collect_banking_interactive", return_value=bank), \
             patch("retirement_planner.cli.collect_credit_cards_interactive", return_value=cards):
            profile = collect_financial_data(agent, args)

        assert len(profile.investments) == 1
        assert len(profile.bank_accounts) == 1
        assert len(profile.credit_cards) == 1

    def test_mixed_file_and_interactive(self):
        agent = MagicMock()
        args = argparse.Namespace(files=None, investments="inv.csv", banking=None, credit_cards=None)

        inv = [InvestmentAccount(account_type="IRA", balance=50000, expected_annual_return=6.0)]
        bank = [BankAccount(account_type="checking", balance=5000, monthly_income_deposits=4000)]
        cards = [CreditCard(outstanding_balance=2000, credit_limit=10000, monthly_payment=500)]

        with patch("retirement_planner.cli.parse_investments_file", return_value=inv), \
             patch("retirement_planner.cli.collect_banking_interactive", return_value=bank), \
             patch("retirement_planner.cli.collect_credit_cards_interactive", return_value=cards):
            profile = collect_financial_data(agent, args)

        assert profile.investments[0].account_type == "IRA"
        assert profile.bank_accounts[0].account_type == "checking"

    def test_files_flag_extracts_all_categories(self):
        agent = MagicMock()
        args = argparse.Namespace(files=["all_data.txt"], investments=None, banking=None, credit_cards=None)

        full_profile = FinancialProfile(
            investments=[InvestmentAccount(account_type="401k", balance=100000, expected_annual_return=7.0)],
            bank_accounts=[BankAccount(account_type="checking", balance=5000, monthly_income_deposits=4000)],
            credit_cards=[CreditCard(outstanding_balance=2000, credit_limit=10000, monthly_payment=500)],
        )

        with patch("retirement_planner.cli.parse_all_from_file", return_value=full_profile) as mock_parse:
            profile = collect_financial_data(agent, args)

        mock_parse.assert_called_once_with(agent, "all_data.txt")
        assert len(profile.investments) == 1
        assert len(profile.bank_accounts) == 1
        assert len(profile.credit_cards) == 1

    def test_files_flag_multiple_files_merged(self):
        agent = MagicMock()
        args = argparse.Namespace(files=["inv.csv", "bank.json"], investments=None, banking=None, credit_cards=None)

        profile1 = FinancialProfile(
            investments=[InvestmentAccount(account_type="401k", balance=100000, expected_annual_return=7.0)],
            bank_accounts=[],
            credit_cards=[CreditCard(outstanding_balance=2000, credit_limit=10000, monthly_payment=500)],
        )
        profile2 = FinancialProfile(
            investments=[],
            bank_accounts=[BankAccount(account_type="checking", balance=5000, monthly_income_deposits=4000)],
            credit_cards=[],
        )

        with patch("retirement_planner.cli.parse_all_from_file", side_effect=[profile1, profile2]):
            profile = collect_financial_data(agent, args)

        assert len(profile.investments) == 1
        assert len(profile.bank_accounts) == 1
        assert len(profile.credit_cards) == 1


class TestCollectPersonalInfo:
    """Tests for interactive personal info collection."""

    def test_collects_valid_info(self):
        # husband_age, wife_age, num_children, child1_age
        inputs = ["45", "43", "1", "12"]
        with patch("builtins.input", side_effect=inputs):
            info = collect_personal_info()
        assert info.husband_age == 45
        assert info.wife_age == 43
        assert info.children_ages == [12]

    def test_no_children(self):
        inputs = ["60", "58", "0"]
        with patch("builtins.input", side_effect=inputs):
            info = collect_personal_info()
        assert info.children_ages == []

    def test_under_18_warning_printed(self, capsys):
        inputs = ["17", "16", "0"]
        with patch("builtins.input", side_effect=inputs):
            info = collect_personal_info()
        output = capsys.readouterr().out
        assert "Warning" in output
        assert "below 18" in output
        assert info.husband_age == 17
        assert info.wife_age == 16

    def test_reprompts_on_invalid_age(self):
        # First input is invalid text, then valid
        inputs = ["abc", "45", "43", "0"]
        with patch("builtins.input", side_effect=inputs):
            info = collect_personal_info()
        assert info.husband_age == 45


# ---------------------------------------------------------------------------
# Display Tests
# ---------------------------------------------------------------------------

class TestDisplayAssessment:
    """Tests for assessment display."""

    def test_prints_formatted_assessment(self, sample_assessment, capsys):
        display_assessment(sample_assessment)
        output = capsys.readouterr().out
        assert "RETIREMENT ASSESSMENT" in output
        assert "Housing" in output
        assert "disclaimer" in output.lower() or "AI-generated" in output


# ---------------------------------------------------------------------------
# Assumption Summary Tests (Requirements 19.5, 19.6, 19.7)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_summary():
    return {
        "extracted_data": {
            "accounts_found": 3,
            "total_investment_balance": 100000.0,
            "total_bank_balance": 5000.0,
            "total_credit_card_balance": 2000.0,
            "monthly_income": 4000.0,
            "monthly_expenses": 500.0,
        },
        "assumptions": {
            "retirement_age": 65,
            "inflation_rate": 3.0,
            "expected_investment_return": 7.0,
            "social_security_start_age": 67,
            "life_expectancy": 85,
        },
    }


class TestDisplayAssumptionSummary:
    """Tests for display_assumption_summary."""

    def test_prints_formatted_summary(self, sample_summary, capsys):
        display_assumption_summary(sample_summary)
        output = capsys.readouterr().out
        assert "ASSUMPTION SUMMARY" in output
        assert "Retirement Age" in output
        assert "Inflation Rate" in output

    def test_prints_file_interpretations_when_present(self, sample_summary, capsys):
        sample_summary["file_interpretations"] = {
            "investments": {"account_types": ["401k"], "count": 1, "estimated_total": 100000}
        }
        display_assumption_summary(sample_summary)
        output = capsys.readouterr().out
        assert "File Interpretations" in output


class TestPromptAssumptionConfirmation:
    """Tests for prompt_assumption_confirmation."""

    def test_confirms_immediately_with_yes(self, sample_summary, capsys):
        with patch("builtins.input", return_value="yes"):
            result = prompt_assumption_confirmation(sample_summary)
        assert result is sample_summary
        output = capsys.readouterr().out
        assert "ASSUMPTION SUMMARY" in output

    def test_confirms_with_y(self, sample_summary, capsys):
        with patch("builtins.input", return_value="y"):
            result = prompt_assumption_confirmation(sample_summary)
        assert result is sample_summary

    def test_applies_single_correction_then_confirms(self, sample_summary, capsys):
        with patch("builtins.input", side_effect=["inflation_rate=2.5", "yes"]):
            result = prompt_assumption_confirmation(sample_summary)
        assert result["assumptions"]["inflation_rate"] == 2.5
        # Other keys unchanged
        assert result["assumptions"]["retirement_age"] == 65

    def test_applies_multiple_corrections(self, sample_summary, capsys):
        with patch("builtins.input", side_effect=["inflation_rate=2.5, retirement_age=67", "y"]):
            result = prompt_assumption_confirmation(sample_summary)
        assert result["assumptions"]["inflation_rate"] == 2.5
        assert result["assumptions"]["retirement_age"] == 67

    def test_redisplays_after_correction(self, sample_summary, capsys):
        with patch("builtins.input", side_effect=["inflation_rate=2.5", "yes"]):
            prompt_assumption_confirmation(sample_summary)
        output = capsys.readouterr().out
        # Should display summary twice: initial + after correction
        assert output.count("ASSUMPTION SUMMARY") == 2

    def test_rejects_unknown_key(self, sample_summary, capsys):
        with patch("builtins.input", side_effect=["unknown_key=42", "yes"]):
            result = prompt_assumption_confirmation(sample_summary)
        output = capsys.readouterr().out
        assert "Unknown assumption key" in output

    def test_rejects_invalid_format(self, sample_summary, capsys):
        with patch("builtins.input", side_effect=["not_a_correction", "yes"]):
            result = prompt_assumption_confirmation(sample_summary)
        output = capsys.readouterr().out
        assert "Skipping invalid correction" in output

    def test_empty_input_reprompts(self, sample_summary, capsys):
        with patch("builtins.input", side_effect=["", "yes"]):
            result = prompt_assumption_confirmation(sample_summary)
        output = capsys.readouterr().out
        assert "Please type 'yes'" in output

    def test_invalid_value_type_rejected(self, sample_summary, capsys):
        with patch("builtins.input", side_effect=["inflation_rate=abc", "yes"]):
            result = prompt_assumption_confirmation(sample_summary)
        output = capsys.readouterr().out
        assert "Invalid value" in output
        # Original value unchanged
        assert result["assumptions"]["inflation_rate"] == 3.0


# ---------------------------------------------------------------------------
# Conversation Session Tests (Requirements 9.6, 9.7)
# ---------------------------------------------------------------------------

class TestRunConversationSession:
    """Tests for the conversation session loop."""

    def test_exit_command_ends_session(self, sample_profile, sample_personal_info, capsys):
        agent = MagicMock()
        with patch("builtins.input", side_effect=["exit"]):
            run_conversation_session(agent, sample_profile, sample_personal_info)
        output = capsys.readouterr().out
        assert "Goodbye" in output

    def test_quit_command_ends_session(self, sample_profile, sample_personal_info, capsys):
        agent = MagicMock()
        with patch("builtins.input", side_effect=["quit"]):
            run_conversation_session(agent, sample_profile, sample_personal_info)
        output = capsys.readouterr().out
        assert "Goodbye" in output

    def test_follow_up_text_response(self, sample_profile, sample_personal_info, capsys):
        agent = MagicMock()
        with patch("builtins.input", side_effect=["What is my net worth?", "exit"]), \
             patch("retirement_planner.cli.run_follow_up", return_value="Your net worth is $103,000."):
            run_conversation_session(agent, sample_profile, sample_personal_info)
        output = capsys.readouterr().out
        assert "$103,000" in output

    def test_follow_up_assessment_response(self, sample_profile, sample_personal_info, sample_assessment, capsys):
        agent = MagicMock()
        with patch("builtins.input", side_effect=["What if I retire at 70?", "exit"]), \
             patch("retirement_planner.cli.run_follow_up", return_value=sample_assessment):
            run_conversation_session(agent, sample_profile, sample_personal_info)
        output = capsys.readouterr().out
        assert "RETIREMENT ASSESSMENT" in output or "UPDATED PROJECTION" in output

    def test_handles_agent_error_gracefully(self, sample_profile, sample_personal_info, capsys):
        agent = MagicMock()
        with patch("builtins.input", side_effect=["question", "exit"]), \
             patch("retirement_planner.cli.run_follow_up", side_effect=BedrockError("API error")):
            run_conversation_session(agent, sample_profile, sample_personal_info)
        output = capsys.readouterr().out
        assert "Error" in output
        assert "try asking another question" in output.lower() or "another question" in output.lower()

    def test_empty_input_skipped(self, sample_profile, sample_personal_info):
        agent = MagicMock()
        with patch("builtins.input", side_effect=["", "exit"]), \
             patch("retirement_planner.cli.run_follow_up") as mock_follow:
            run_conversation_session(agent, sample_profile, sample_personal_info)
        mock_follow.assert_not_called()

    def test_eof_ends_session(self, sample_profile, sample_personal_info, capsys):
        agent = MagicMock()
        with patch("builtins.input", side_effect=EOFError):
            run_conversation_session(agent, sample_profile, sample_personal_info)
        output = capsys.readouterr().out
        assert "Goodbye" in output


# ---------------------------------------------------------------------------
# Main Function Tests (Requirement 9.8)
# ---------------------------------------------------------------------------

class TestMain:
    """Tests for the main entry point."""

    def test_keyboard_interrupt_exits_gracefully(self, capsys):
        with patch("retirement_planner.cli.parse_args", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as exc_info:
                main([])
            assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "Goodbye" in output

    def test_credential_error_exits(self, capsys):
        with patch("retirement_planner.cli.parse_args") as mock_parse, \
             patch("retirement_planner.cli.create_agent", side_effect=CredentialError("No AWS credentials found")):
            mock_parse.return_value = argparse.Namespace(files=None, investments=None, banking=None, credit_cards=None, history=False, resume=None, model=None)
            with pytest.raises(SystemExit) as exc_info:
                main([])
            assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "No AWS credentials" in output

    def test_full_flow_with_mocked_agent(self, sample_profile, sample_personal_info, sample_assessment, capsys):
        mock_agent = MagicMock()
        sample_summary = {
            "extracted_data": {"accounts_found": 3, "total_investment_balance": 100000.0,
                               "total_bank_balance": 5000.0, "total_credit_card_balance": 2000.0,
                               "monthly_income": 4000.0, "monthly_expenses": 500.0},
            "assumptions": {"retirement_age": 65, "inflation_rate": 3.0,
                            "expected_investment_return": 7.0, "social_security_start_age": 67,
                            "life_expectancy": 85},
        }
        with patch("retirement_planner.cli.parse_args") as mock_parse, \
             patch("retirement_planner.cli.create_agent", return_value=mock_agent), \
             patch("retirement_planner.cli.collect_financial_data", return_value=sample_profile), \
             patch("retirement_planner.cli.collect_personal_info", return_value=sample_personal_info), \
             patch("retirement_planner.cli.generate_assumption_summary", return_value=sample_summary), \
             patch("retirement_planner.cli.prompt_assumption_confirmation", return_value=sample_summary), \
             patch("retirement_planner.cli.run_initial_assessment", return_value=sample_assessment), \
             patch("retirement_planner.cli.run_conversation_session") as mock_session, \
             patch("retirement_planner.cli.save_session"), \
             patch("retirement_planner.cli.new_session_id", return_value="test_session"):
            mock_parse.return_value = argparse.Namespace(files=None, investments=None, banking=None, credit_cards=None, history=False, resume=None, model=None)
            main([])

        output = capsys.readouterr().out
        assert "Generating assumption summary" in output
        assert "Analyzing your financial data" in output
        assert "RETIREMENT ASSESSMENT" in output
        mock_session.assert_called_once()
