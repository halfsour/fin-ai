"""Terminal output formatting for retirement assessments and projection updates."""

from __future__ import annotations

from retirement_planner.models import RetirementAssessment


def _format_currency(value: float) -> str:
    """Format a float as a currency string (e.g. $1,234.56)."""
    if value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"


def format_assessment(assessment: RetirementAssessment) -> str:
    """Format the full assessment for terminal display.

    Includes net worth, monthly cash flow, retirement readiness,
    recommended monthly budget by category, and the disclaimer.
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append("RETIREMENT ASSESSMENT")
    lines.append("=" * 60)

    # Retirement readiness
    status = "YES" if assessment.can_retire else "NO"
    lines.append(f"\nRetirement Ready: {status}")
    lines.append(f"\n{assessment.retirement_readiness_summary}")

    # Financial summary
    lines.append("\n--- Financial Summary ---")
    lines.append(f"Net Worth: {_format_currency(assessment.net_worth)}")
    lines.append(f"Monthly Cash Flow: {_format_currency(assessment.monthly_cash_flow)}")

    # Budget breakdown
    if assessment.recommended_monthly_budget:
        lines.append("\n--- Recommended Monthly Budget ---")
        total = 0.0
        for item in assessment.recommended_monthly_budget:
            lines.append(f"  {item.category}: {_format_currency(item.amount)}")
            total += item.amount
        lines.append(f"  {'Total':}: {_format_currency(total)}")

    # Disclaimer
    lines.append(f"\n{assessment.disclaimer}")

    return "\n".join(lines)


def format_projection_update(
    previous: RetirementAssessment,
    updated: RetirementAssessment,
) -> str:
    """Format an updated projection, highlighting changed assumptions.

    Compares the previous and updated assessments to show which
    assumptions changed, then displays the updated readiness and budget.
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append("UPDATED PROJECTION")
    lines.append("=" * 60)

    # Changed assumptions
    changed = _find_changed_assumptions(previous.assumptions, updated.assumptions)
    if changed:
        lines.append("\n--- Changed Assumptions ---")
        for key, (old_val, new_val) in changed.items():
            lines.append(f"  {key}: {old_val} -> {new_val}")

    # Updated readiness
    status = "YES" if updated.can_retire else "NO"
    lines.append(f"\nRetirement Ready: {status}")
    lines.append(f"\n{updated.retirement_readiness_summary}")

    # Updated financial summary
    lines.append("\n--- Financial Summary ---")
    lines.append(f"Net Worth: {_format_currency(updated.net_worth)}")
    lines.append(f"Monthly Cash Flow: {_format_currency(updated.monthly_cash_flow)}")

    # Updated budget
    if updated.recommended_monthly_budget:
        lines.append("\n--- Recommended Monthly Budget ---")
        total = 0.0
        for item in updated.recommended_monthly_budget:
            lines.append(f"  {item.category}: {_format_currency(item.amount)}")
            total += item.amount
        lines.append(f"  {'Total':}: {_format_currency(total)}")

    # Disclaimer
    lines.append(f"\n{updated.disclaimer}")

    return "\n".join(lines)

def format_assumption_summary(summary: dict) -> str:
    """Format the assumption summary for terminal display.

    Includes extracted financial data, key assumptions, and
    file interpretations (when present).
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append("ASSUMPTION SUMMARY")
    lines.append("=" * 60)

    # Extracted financial data
    extracted = summary.get("extracted_data")
    if extracted is not None:
        lines.append("\n--- Extracted Financial Data ---")
        lines.append(f"  Accounts Found: {extracted.get('accounts_found', 0)}")
        lines.append(f"  Total Investment Balance: {_format_currency(extracted.get('total_investment_balance', 0.0))}")
        lines.append(f"  Total Bank Balance: {_format_currency(extracted.get('total_bank_balance', 0.0))}")
        lines.append(f"  Total Credit Card Balance: {_format_currency(extracted.get('total_credit_card_balance', 0.0))}")
        lines.append(f"  Monthly Income: {_format_currency(extracted.get('monthly_income', 0.0))}")
        lines.append(f"  Monthly Expenses: {_format_currency(extracted.get('monthly_expenses', 0.0))}")

    # Key assumptions
    assumptions = summary.get("assumptions")
    if assumptions is not None:
        lines.append("\n--- Key Assumptions ---")
        lines.append(f"  Retirement Age: {assumptions.get('retirement_age', 'N/A')}")
        lines.append(f"  Inflation Rate: {assumptions.get('inflation_rate', 'N/A')}%")
        lines.append(f"  Expected Investment Return: {assumptions.get('expected_investment_return', 'N/A')}%")
        lines.append(f"  Social Security Start Age: {assumptions.get('social_security_start_age', 'N/A')}")
        lines.append(f"  Life Expectancy: {assumptions.get('life_expectancy', 'N/A')}")

    # File interpretations (only when present)
    file_interps = summary.get("file_interpretations", {})
    if file_interps:
        lines.append("\n--- File Interpretations ---")
        for filename, details in file_interps.items():
            lines.append(f"  {filename}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    lines.append(f"    {key}: {value}")
            else:
                lines.append(f"    {details}")

    return "\n".join(lines)




def _find_changed_assumptions(
    previous: dict, updated: dict
) -> dict[str, tuple]:
    """Compare two assumption dicts and return changed keys with old/new values."""
    changed: dict[str, tuple] = {}

    all_keys = set(previous.keys()) | set(updated.keys())
    for key in sorted(all_keys):
        old_val = previous.get(key)
        new_val = updated.get(key)
        if old_val != new_val:
            changed[key] = (old_val, new_val)

    return changed
