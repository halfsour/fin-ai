"""Unit tests for Pydantic data models and custom exceptions."""

import pytest
from pydantic import ValidationError

from retirement_planner.models import (
    BankAccount,
    BedrockError,
    CreditCard,
    CredentialError,
    FileParseError,
    FinancialProfile,
    InvestmentAccount,
    MonthlyBudgetItem,
    NormalizationError,
    PersonalInfo,
    RetirementAssessment,
    RetirementPlannerError,
)


# ---------------------------------------------------------------------------
# InvestmentAccount
# ---------------------------------------------------------------------------


class TestInvestmentAccount:
    def test_valid_construction(self):
        acct = InvestmentAccount(account_type="401k", balance=50000.0, expected_annual_return=7.0)
        assert acct.account_type == "401k"
        assert acct.balance == 50000.0
        assert acct.expected_annual_return == 7.0

    def test_zero_balance_allowed(self):
        acct = InvestmentAccount(account_type="IRA", balance=0.0, expected_annual_return=5.0)
        assert acct.balance == 0.0

    def test_negative_balance_rejected(self):
        with pytest.raises(ValidationError, match="balance must be >= 0"):
            InvestmentAccount(account_type="401k", balance=-100.0, expected_annual_return=7.0)

    def test_empty_account_type_rejected(self):
        with pytest.raises(ValidationError, match="account_type must be a non-empty string"):
            InvestmentAccount(account_type="", balance=1000.0, expected_annual_return=7.0)

    def test_whitespace_account_type_rejected(self):
        with pytest.raises(ValidationError, match="account_type must be a non-empty string"):
            InvestmentAccount(account_type="   ", balance=1000.0, expected_annual_return=7.0)

    def test_negative_return_allowed(self):
        acct = InvestmentAccount(account_type="brokerage", balance=1000.0, expected_annual_return=-2.0)
        assert acct.expected_annual_return == -2.0


# ---------------------------------------------------------------------------
# BankAccount
# ---------------------------------------------------------------------------


class TestBankAccount:
    def test_valid_construction(self):
        acct = BankAccount(account_type="checking", balance=5000.0, monthly_income_deposits=3000.0)
        assert acct.account_type == "checking"
        assert acct.balance == 5000.0
        assert acct.monthly_income_deposits == 3000.0

    def test_negative_balance_rejected(self):
        with pytest.raises(ValidationError, match="balance must be >= 0"):
            BankAccount(account_type="savings", balance=-1.0, monthly_income_deposits=0.0)

    def test_negative_deposits_rejected(self):
        with pytest.raises(ValidationError, match="monthly_income_deposits must be >= 0"):
            BankAccount(account_type="checking", balance=100.0, monthly_income_deposits=-500.0)

    def test_zero_values_allowed(self):
        acct = BankAccount(account_type="savings", balance=0.0, monthly_income_deposits=0.0)
        assert acct.balance == 0.0
        assert acct.monthly_income_deposits == 0.0


# ---------------------------------------------------------------------------
# CreditCard
# ---------------------------------------------------------------------------


class TestCreditCard:
    def test_valid_construction(self):
        cc = CreditCard(outstanding_balance=2000.0, credit_limit=10000.0, monthly_payment=200.0)
        assert cc.outstanding_balance == 2000.0
        assert cc.credit_limit == 10000.0
        assert cc.monthly_payment == 200.0

    def test_negative_outstanding_balance_rejected(self):
        with pytest.raises(ValidationError, match="outstanding_balance must be >= 0"):
            CreditCard(outstanding_balance=-1.0, credit_limit=5000.0, monthly_payment=100.0)

    def test_negative_credit_limit_rejected(self):
        with pytest.raises(ValidationError, match="credit_limit must be >= 0"):
            CreditCard(outstanding_balance=0.0, credit_limit=-1.0, monthly_payment=100.0)

    def test_negative_monthly_payment_rejected(self):
        with pytest.raises(ValidationError, match="monthly_payment must be >= 0"):
            CreditCard(outstanding_balance=0.0, credit_limit=5000.0, monthly_payment=-50.0)

    def test_all_zeros_allowed(self):
        cc = CreditCard(outstanding_balance=0.0, credit_limit=0.0, monthly_payment=0.0)
        assert cc.outstanding_balance == 0.0


# ---------------------------------------------------------------------------
# FinancialProfile
# ---------------------------------------------------------------------------


class TestFinancialProfile:
    def test_valid_construction(self):
        profile = FinancialProfile(
            investments=[InvestmentAccount(account_type="401k", balance=100000.0, expected_annual_return=7.0)],
            bank_accounts=[BankAccount(account_type="checking", balance=5000.0, monthly_income_deposits=3000.0)],
            credit_cards=[CreditCard(outstanding_balance=1000.0, credit_limit=10000.0, monthly_payment=200.0)],
        )
        assert len(profile.investments) == 1
        assert len(profile.bank_accounts) == 1
        assert len(profile.credit_cards) == 1

    def test_empty_lists_allowed(self):
        profile = FinancialProfile(investments=[], bank_accounts=[], credit_cards=[])
        assert profile.investments == []

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            FinancialProfile(investments=[], bank_accounts=[])  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# PersonalInfo
# ---------------------------------------------------------------------------


class TestPersonalInfo:
    def test_valid_construction(self):
        info = PersonalInfo(husband_age=45, wife_age=42, children_ages=[12, 8])
        assert info.husband_age == 45
        assert info.wife_age == 42
        assert info.children_ages == [12, 8]

    def test_boundary_ages_valid(self):
        info = PersonalInfo(husband_age=0, wife_age=120, children_ages=[0, 120])
        assert info.husband_age == 0
        assert info.wife_age == 120

    def test_husband_age_too_high(self):
        with pytest.raises(ValidationError, match="husband_age must be between 0 and 120"):
            PersonalInfo(husband_age=121, wife_age=40, children_ages=[])

    def test_wife_age_negative(self):
        with pytest.raises(ValidationError, match="wife_age must be between 0 and 120"):
            PersonalInfo(husband_age=40, wife_age=-1, children_ages=[])

    def test_child_age_out_of_range(self):
        with pytest.raises(ValidationError, match="children_ages"):
            PersonalInfo(husband_age=40, wife_age=38, children_ages=[5, 200])

    def test_empty_children_allowed(self):
        info = PersonalInfo(husband_age=50, wife_age=48, children_ages=[])
        assert info.children_ages == []


# ---------------------------------------------------------------------------
# MonthlyBudgetItem & RetirementAssessment
# ---------------------------------------------------------------------------


class TestMonthlyBudgetItem:
    def test_valid_construction(self):
        item = MonthlyBudgetItem(category="Housing", amount=2000.0)
        assert item.category == "Housing"
        assert item.amount == 2000.0


class TestRetirementAssessment:
    def test_valid_construction(self):
        assessment = RetirementAssessment(
            can_retire=True,
            retirement_readiness_summary="You are on track.",
            recommended_monthly_budget=[MonthlyBudgetItem(category="Housing", amount=2000.0)],
            net_worth=500000.0,
            monthly_cash_flow=3000.0,
            assumptions={"inflation_rate": 3.0, "retirement_age": 65},
            disclaimer="This is AI-generated and not professional financial advice.",
        )
        assert assessment.can_retire is True
        assert len(assessment.recommended_monthly_budget) == 1
        assert assessment.assumptions["inflation_rate"] == 3.0


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class TestCustomExceptions:
    def test_retirement_planner_error(self):
        err = RetirementPlannerError("something went wrong")
        assert str(err) == "something went wrong"

    def test_file_parse_error(self):
        err = FileParseError("/tmp/data.csv", "encoding error")
        assert err.file_path == "/tmp/data.csv"
        assert err.reason == "encoding error"
        assert "Failed to read /tmp/data.csv: encoding error" in str(err)
        assert isinstance(err, RetirementPlannerError)

    def test_normalization_error(self):
        err = NormalizationError("/tmp/data.csv", ["balance", "account_type"])
        assert err.file_path == "/tmp/data.csv"
        assert err.missing_fields == ["balance", "account_type"]
        assert "balance" in str(err)
        assert isinstance(err, RetirementPlannerError)

    def test_credential_error(self):
        err = CredentialError("no creds")
        assert isinstance(err, RetirementPlannerError)

    def test_bedrock_error_retryable(self):
        err = BedrockError("timeout", retryable=True)
        assert err.retryable is True
        assert str(err) == "timeout"
        assert isinstance(err, RetirementPlannerError)

    def test_bedrock_error_not_retryable(self):
        err = BedrockError("bad request", retryable=False)
        assert err.retryable is False
