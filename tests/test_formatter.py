"""Unit tests for the formatter module — assessment display and projection updates."""

import pytest

from retirement_planner.formatter import (
    format_assessment,
    format_assumption_summary,
    format_projection_update,
    _format_currency,
    _find_changed_assumptions,
)
from retirement_planner.models import (
    MonthlyBudgetItem,
    RetirementAssessment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_assessment() -> RetirementAssessment:
    return RetirementAssessment(
        can_retire=True,
        retirement_readiness_summary="You are on track for retirement at age 65.",
        recommended_monthly_budget=[
            MonthlyBudgetItem(category="Housing", amount=1500.0),
            MonthlyBudgetItem(category="Food", amount=600.0),
            MonthlyBudgetItem(category="Healthcare", amount=400.0),
        ],
        net_worth=500000.0,
        monthly_cash_flow=3500.0,
        assumptions={"inflation_rate": 0.03, "retirement_age": 65, "life_expectancy": 85},
        disclaimer="This is AI-generated and does not constitute professional financial advice.",
    )


@pytest.fixture
def negative_cash_flow_assessment() -> RetirementAssessment:
    return RetirementAssessment(
        can_retire=False,
        retirement_readiness_summary="You are not ready to retire.",
        recommended_monthly_budget=[
            MonthlyBudgetItem(category="Housing", amount=1000.0),
        ],
        net_worth=-5000.0,
        monthly_cash_flow=-200.0,
        assumptions={"inflation_rate": 0.03},
        disclaimer="AI-generated analysis. Not professional advice.",
    )


@pytest.fixture
def updated_assessment(sample_assessment: RetirementAssessment) -> RetirementAssessment:
    return RetirementAssessment(
        can_retire=True,
        retirement_readiness_summary="With delayed retirement, you are in a stronger position.",
        recommended_monthly_budget=[
            MonthlyBudgetItem(category="Housing", amount=1600.0),
            MonthlyBudgetItem(category="Food", amount=650.0),
            MonthlyBudgetItem(category="Healthcare", amount=450.0),
        ],
        net_worth=550000.0,
        monthly_cash_flow=4000.0,
        assumptions={"inflation_rate": 0.03, "retirement_age": 68, "life_expectancy": 85},
        disclaimer="This is AI-generated and does not constitute professional financial advice.",
    )


# ---------------------------------------------------------------------------
# _format_currency
# ---------------------------------------------------------------------------


class TestFormatCurrency:
    def test_positive_value(self) -> None:
        assert _format_currency(1234.56) == "$1,234.56"

    def test_zero(self) -> None:
        assert _format_currency(0.0) == "$0.00"

    def test_negative_value(self) -> None:
        assert _format_currency(-5000.0) == "-$5,000.00"

    def test_large_value(self) -> None:
        assert _format_currency(1000000.0) == "$1,000,000.00"


# ---------------------------------------------------------------------------
# format_assessment
# ---------------------------------------------------------------------------


class TestFormatAssessment:
    def test_contains_header(self, sample_assessment: RetirementAssessment) -> None:
        output = format_assessment(sample_assessment)
        assert "RETIREMENT ASSESSMENT" in output

    def test_contains_net_worth(self, sample_assessment: RetirementAssessment) -> None:
        output = format_assessment(sample_assessment)
        assert "$500,000.00" in output
        assert "Net Worth" in output

    def test_contains_cash_flow(self, sample_assessment: RetirementAssessment) -> None:
        output = format_assessment(sample_assessment)
        assert "$3,500.00" in output
        assert "Monthly Cash Flow" in output

    def test_contains_readiness(self, sample_assessment: RetirementAssessment) -> None:
        output = format_assessment(sample_assessment)
        assert "Retirement Ready: YES" in output
        assert "on track for retirement" in output

    def test_contains_not_ready(self, negative_cash_flow_assessment: RetirementAssessment) -> None:
        output = format_assessment(negative_cash_flow_assessment)
        assert "Retirement Ready: NO" in output

    def test_contains_budget_categories(self, sample_assessment: RetirementAssessment) -> None:
        output = format_assessment(sample_assessment)
        assert "Housing" in output
        assert "$1,500.00" in output
        assert "Food" in output
        assert "$600.00" in output
        assert "Healthcare" in output
        assert "$400.00" in output

    def test_contains_budget_total(self, sample_assessment: RetirementAssessment) -> None:
        output = format_assessment(sample_assessment)
        assert "Total" in output
        assert "$2,500.00" in output

    def test_contains_disclaimer(self, sample_assessment: RetirementAssessment) -> None:
        output = format_assessment(sample_assessment)
        assert "AI-generated" in output
        assert "does not constitute professional financial advice" in output

    def test_negative_values_formatted(self, negative_cash_flow_assessment: RetirementAssessment) -> None:
        output = format_assessment(negative_cash_flow_assessment)
        assert "-$5,000.00" in output
        assert "-$200.00" in output

    def test_empty_budget(self) -> None:
        assessment = RetirementAssessment(
            can_retire=False,
            retirement_readiness_summary="Insufficient data.",
            recommended_monthly_budget=[],
            net_worth=0.0,
            monthly_cash_flow=0.0,
            assumptions={},
            disclaimer="AI-generated.",
        )
        output = format_assessment(assessment)
        assert "Recommended Monthly Budget" not in output
        assert "AI-generated." in output

    def test_readiness_summary_present(self, sample_assessment: RetirementAssessment) -> None:
        output = format_assessment(sample_assessment)
        assert sample_assessment.retirement_readiness_summary in output


# ---------------------------------------------------------------------------
# format_projection_update
# ---------------------------------------------------------------------------


class TestFormatProjectionUpdate:
    def test_contains_header(
        self,
        sample_assessment: RetirementAssessment,
        updated_assessment: RetirementAssessment,
    ) -> None:
        output = format_projection_update(sample_assessment, updated_assessment)
        assert "UPDATED PROJECTION" in output

    def test_shows_changed_assumptions(
        self,
        sample_assessment: RetirementAssessment,
        updated_assessment: RetirementAssessment,
    ) -> None:
        output = format_projection_update(sample_assessment, updated_assessment)
        assert "Changed Assumptions" in output
        assert "retirement_age" in output
        assert "65" in output
        assert "68" in output
        assert "->" in output

    def test_unchanged_assumptions_not_shown(
        self,
        sample_assessment: RetirementAssessment,
        updated_assessment: RetirementAssessment,
    ) -> None:
        output = format_projection_update(sample_assessment, updated_assessment)
        # inflation_rate and life_expectancy didn't change
        assert "inflation_rate" not in output
        assert "life_expectancy" not in output

    def test_contains_updated_readiness(
        self,
        sample_assessment: RetirementAssessment,
        updated_assessment: RetirementAssessment,
    ) -> None:
        output = format_projection_update(sample_assessment, updated_assessment)
        assert "Retirement Ready: YES" in output
        assert "delayed retirement" in output

    def test_contains_updated_budget(
        self,
        sample_assessment: RetirementAssessment,
        updated_assessment: RetirementAssessment,
    ) -> None:
        output = format_projection_update(sample_assessment, updated_assessment)
        assert "Housing" in output
        assert "$1,600.00" in output

    def test_contains_disclaimer(
        self,
        sample_assessment: RetirementAssessment,
        updated_assessment: RetirementAssessment,
    ) -> None:
        output = format_projection_update(sample_assessment, updated_assessment)
        assert "AI-generated" in output

    def test_no_changes_section_when_same_assumptions(
        self, sample_assessment: RetirementAssessment
    ) -> None:
        output = format_projection_update(sample_assessment, sample_assessment)
        assert "Changed Assumptions" not in output

    def test_new_assumption_added(self, sample_assessment: RetirementAssessment) -> None:
        updated = RetirementAssessment(
            can_retire=True,
            retirement_readiness_summary="Updated.",
            recommended_monthly_budget=[],
            net_worth=500000.0,
            monthly_cash_flow=3500.0,
            assumptions={
                "inflation_rate": 0.03,
                "retirement_age": 65,
                "life_expectancy": 85,
                "savings_rate": 0.20,
            },
            disclaimer="AI-generated.",
        )
        output = format_projection_update(sample_assessment, updated)
        assert "savings_rate" in output
        assert "None" in output  # old value was None (didn't exist)
        assert "0.2" in output

    def test_assumption_removed(self, sample_assessment: RetirementAssessment) -> None:
        updated = RetirementAssessment(
            can_retire=True,
            retirement_readiness_summary="Updated.",
            recommended_monthly_budget=[],
            net_worth=500000.0,
            monthly_cash_flow=3500.0,
            assumptions={"inflation_rate": 0.03, "retirement_age": 65},
            disclaimer="AI-generated.",
        )
        output = format_projection_update(sample_assessment, updated)
        assert "life_expectancy" in output
        assert "None" in output  # new value is None (removed)


# ---------------------------------------------------------------------------
# _find_changed_assumptions
# ---------------------------------------------------------------------------


class TestFindChangedAssumptions:
    def test_no_changes(self) -> None:
        d = {"a": 1, "b": 2}
        assert _find_changed_assumptions(d, d) == {}

    def test_value_changed(self) -> None:
        result = _find_changed_assumptions({"a": 1}, {"a": 2})
        assert result == {"a": (1, 2)}

    def test_key_added(self) -> None:
        result = _find_changed_assumptions({}, {"a": 1})
        assert result == {"a": (None, 1)}

    def test_key_removed(self) -> None:
        result = _find_changed_assumptions({"a": 1}, {})
        assert result == {"a": (1, None)}

    def test_multiple_changes(self) -> None:
        prev = {"a": 1, "b": 2, "c": 3}
        upd = {"a": 1, "b": 5, "d": 4}
        result = _find_changed_assumptions(prev, upd)
        assert "a" not in result
        assert result["b"] == (2, 5)
        assert result["c"] == (3, None)
        assert result["d"] == (None, 4)


# ---------------------------------------------------------------------------
# format_assumption_summary
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_assumption_summary() -> dict:
    return {
        "extracted_data": {
            "accounts_found": 5,
            "total_investment_balance": 350000.0,
            "total_bank_balance": 45000.0,
            "total_credit_card_balance": 8000.0,
            "monthly_income": 12000.0,
            "monthly_expenses": 4500.0,
        },
        "assumptions": {
            "retirement_age": 65,
            "inflation_rate": 3.0,
            "expected_investment_return": 7.0,
            "social_security_start_age": 67,
            "life_expectancy": 85,
        },
    }


@pytest.fixture
def summary_with_files(sample_assumption_summary: dict) -> dict:
    summary = dict(sample_assumption_summary)
    summary["file_interpretations"] = {
        "investments.csv": {
            "account_types": "401k, IRA, brokerage",
            "account_count": 3,
        },
        "banking.txt": {
            "account_types": "checking, savings",
            "estimated_monthly_income": "$12,000",
        },
    }
    return summary


class TestFormatAssumptionSummary:
    def test_contains_header(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "ASSUMPTION SUMMARY" in output

    def test_contains_extracted_data_section(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Extracted Financial Data" in output

    def test_contains_accounts_found(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Accounts Found: 5" in output

    def test_contains_investment_balance(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Total Investment Balance: $350,000.00" in output

    def test_contains_bank_balance(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Total Bank Balance: $45,000.00" in output

    def test_contains_credit_card_balance(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Total Credit Card Balance: $8,000.00" in output

    def test_contains_monthly_income(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Monthly Income: $12,000.00" in output

    def test_contains_monthly_expenses(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Monthly Expenses: $4,500.00" in output

    def test_contains_assumptions_section(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Key Assumptions" in output

    def test_contains_retirement_age(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Retirement Age: 65" in output

    def test_contains_inflation_rate(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Inflation Rate: 3.0%" in output

    def test_contains_investment_return(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Expected Investment Return: 7.0%" in output

    def test_contains_social_security_age(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Social Security Start Age: 67" in output

    def test_contains_life_expectancy(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "Life Expectancy: 85" in output

    def test_no_file_interpretations_when_absent(self, sample_assumption_summary: dict) -> None:
        output = format_assumption_summary(sample_assumption_summary)
        assert "File Interpretations" not in output

    def test_contains_file_interpretations_when_present(self, summary_with_files: dict) -> None:
        output = format_assumption_summary(summary_with_files)
        assert "File Interpretations" in output
        assert "investments.csv" in output
        assert "banking.txt" in output

    def test_file_interpretation_details(self, summary_with_files: dict) -> None:
        output = format_assumption_summary(summary_with_files)
        assert "account_types: 401k, IRA, brokerage" in output
        assert "account_count: 3" in output
        assert "estimated_monthly_income: $12,000" in output

    def test_empty_summary(self) -> None:
        output = format_assumption_summary({})
        assert "ASSUMPTION SUMMARY" in output
        assert "Extracted Financial Data" not in output
        assert "Key Assumptions" not in output
        assert "File Interpretations" not in output

    def test_defaults_for_missing_extracted_fields(self) -> None:
        summary = {"extracted_data": {}}
        output = format_assumption_summary(summary)
        assert "Accounts Found: 0" in output
        assert "Total Investment Balance: $0.00" in output

    def test_defaults_for_missing_assumption_fields(self) -> None:
        summary = {"assumptions": {}}
        output = format_assumption_summary(summary)
        assert "Retirement Age: N/A" in output
        assert "Inflation Rate: N/A%" in output
