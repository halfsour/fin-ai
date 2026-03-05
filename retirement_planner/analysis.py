"""Pre-computed financial analysis for the Retirement Planner.

Pure Python calculations for net worth, cash flow, withdrawal rates,
milestone dates, and account classification. Builds a structured brief
that is passed to the LLM for narrative generation only.
"""

from __future__ import annotations

from datetime import date
from retirement_planner.models import FinancialProfile, PersonalInfo


def compute_net_worth(profile: FinancialProfile) -> dict:
    """Compute net worth from profile data."""
    inv_total = sum(a.balance for a in profile.investments)
    bank_total = sum(a.balance for a in profile.bank_accounts)
    cc_debt = sum(c.outstanding_balance for c in profile.credit_cards)
    return {
        "investment_total": round(inv_total, 2),
        "bank_total": round(bank_total, 2),
        "credit_card_debt": round(cc_debt, 2),
        "net_worth": round(inv_total + bank_total - cc_debt, 2),
    }


def compute_cash_flow(profile: FinancialProfile) -> dict:
    """Compute monthly cash flow from profile data."""
    income = sum(a.monthly_income_deposits for a in profile.bank_accounts)
    if profile.spending:
        expenses = sum(s.monthly_amount for s in profile.spending)
        source = "spending"
    else:
        expenses = sum(c.monthly_payment for c in profile.credit_cards)
        source = "credit_cards"
    return {
        "monthly_income": round(income, 2),
        "monthly_expenses": round(expenses, 2),
        "monthly_cash_flow": round(income - expenses, 2),
        "annual_expenses": round(expenses * 12, 2),
        "expense_source": source,
    }


def compute_withdrawal_rates(investable_total: float) -> dict:
    """Compute safe withdrawal amounts at various rates."""
    return {
        "swr_3pct": {"annual": round(investable_total * 0.03, 2), "monthly": round(investable_total * 0.03 / 12, 2)},
        "swr_4pct": {"annual": round(investable_total * 0.04, 2), "monthly": round(investable_total * 0.04 / 12, 2)},
        "swr_5pct": {"annual": round(investable_total * 0.05, 2), "monthly": round(investable_total * 0.05 / 12, 2)},
    }


def _parse_date(dob_str: str | None) -> date | None:
    if not dob_str:
        return None
    try:
        parts = dob_str.strip().split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


def _age_at(dob: date, target: date) -> int:
    return target.year - dob.year - ((target.month, target.day) < (dob.month, dob.day))


def compute_milestones(personal_info: PersonalInfo, retire_ages: dict | None = None) -> dict:
    """Compute key retirement milestone dates from birthdates."""
    today = date.today()
    milestones = {}
    retire_ages = retire_ages or {}

    for label, dob_str in [("spouse_1", personal_info.husband_birthdate), ("spouse_2", personal_info.wife_birthdate)]:
        dob = _parse_date(dob_str)
        if not dob:
            continue
        current_age = _age_at(dob, today)
        target_retire = retire_ages.get(label, 65)
        m = {
            "birthdate": str(dob),
            "current_age": current_age,
            "target_retirement_age": target_retire,
            "medicare_eligible": date(dob.year + 65, dob.month, dob.day).isoformat(),
            "medicare_years_away": max(0, 65 - current_age),
            "ss_fra": date(dob.year + 67, dob.month, dob.day).isoformat(),
            "ss_early_62": date(dob.year + 62, dob.month, dob.day).isoformat(),
            "ss_delayed_70": date(dob.year + 70, dob.month, dob.day).isoformat(),
            "rule_of_55": date(dob.year + 55, dob.month, dob.day).isoformat(),
            "rmd_start_75": date(dob.year + 75, dob.month, dob.day).isoformat(),
            "penalty_free_59_5": date(dob.year + 59, dob.month + 6 if dob.month <= 6 else dob.month - 6, dob.day).isoformat() if dob.month != 7 else date(dob.year + 60, 1, dob.day).isoformat(),
        }
        milestones[label] = m

    # Children
    for i, cdob_str in enumerate(personal_info.children_birthdates or []):
        cdob = _parse_date(cdob_str)
        if cdob:
            milestones[f"child_{i+1}"] = {
                "birthdate": str(cdob),
                "current_age": _age_at(cdob, today),
                "college_start_age_18": date(cdob.year + 18, cdob.month, cdob.day).isoformat(),
            }

    return milestones


_TAX_DEFERRED_KEYWORDS = ["401k", "401(k)", "403b", "403(b)", "457", "traditional ira", "ira", "pension", "tsp", "sep", "retirement", "savings plan", "thrift", "plan "]
_TAX_FREE_KEYWORDS = ["roth", "hsa", "health savings"]
_EDUCATION_KEYWORDS = ["529", "education", "coverdell"]

def classify_accounts(profile: FinancialProfile) -> list[dict]:
    """Classify each investment account by tax treatment."""
    result = []
    for a in profile.investments:
        at = a.account_type.lower()
        holdings = (a.holdings or "").lower()
        combined = at + " " + holdings
        if any(k in combined for k in _TAX_FREE_KEYWORDS):
            tax_type = "tax-free"
            restrictions = "Roth: tax/penalty-free after 59.5 + 5yr rule. HSA: tax-free for medical."
        elif any(k in combined for k in _EDUCATION_KEYWORDS):
            tax_type = "education"
            restrictions = "529: tax-free for qualified education expenses. 10% penalty + tax on non-qualified."
        elif any(k in combined for k in _TAX_DEFERRED_KEYWORDS):
            tax_type = "tax-deferred"
            restrictions = "Taxed as ordinary income on withdrawal. 10% penalty before 59.5 (Rule of 55 exception for 401k)."
        else:
            tax_type = "taxable"
            restrictions = "No withdrawal restrictions. Long-term capital gains taxed at 0/15/20%."
        result.append({
            "account_type": a.account_type,
            "balance": a.balance,
            "holdings": a.holdings,
            "expected_return": a.expected_annual_return,
            "tax_treatment": tax_type,
            "restrictions": restrictions,
        })
    return result


def build_analysis_brief(
    profile: FinancialProfile,
    personal_info: PersonalInfo | None = None,
    retire_ages: dict | None = None,
    additional_context: str = "",
) -> str:
    """Build a structured analysis brief for the LLM."""
    nw = compute_net_worth(profile)
    cf = compute_cash_flow(profile)
    accounts = classify_accounts(profile)
    milestones = compute_milestones(personal_info, retire_ages) if personal_info else {}

    # Investable total (exclude education accounts)
    investable = sum(a["balance"] for a in accounts if a["tax_treatment"] != "education")
    wr = compute_withdrawal_rates(investable)

    lines = ["=== PRE-COMPUTED FINANCIAL ANALYSIS ===", ""]

    # Net worth
    lines.append(f"NET WORTH: ${nw['net_worth']:,.0f}")
    lines.append(f"  Investments: ${nw['investment_total']:,.0f} | Banks: ${nw['bank_total']:,.0f} | CC Debt: ${nw['credit_card_debt']:,.0f}")
    lines.append("")

    # Cash flow
    lines.append(f"MONTHLY CASH FLOW: ${cf['monthly_cash_flow']:,.0f}")
    lines.append(f"  Income: ${cf['monthly_income']:,.0f}/mo | Expenses: ${cf['monthly_expenses']:,.0f}/mo ({cf['expense_source']})")
    lines.append(f"  Annual expenses: ${cf['annual_expenses']:,.0f}")
    lines.append("")

    # Accounts
    lines.append("ACCOUNTS BY TAX TREATMENT:")
    for tax_type in ["tax-deferred", "tax-free", "taxable", "education"]:
        group = [a for a in accounts if a["tax_treatment"] == tax_type]
        if group:
            total = sum(a["balance"] for a in group)
            lines.append(f"  {tax_type.upper()} (${total:,.0f} total):")
            for a in group:
                lines.append(f"    - {a['account_type']}: ${a['balance']:,.0f} [{a['holdings']}] @{a['expected_return']:.0%}")
                lines.append(f"      {a['restrictions']}")
    lines.append("")

    # Withdrawal rates
    lines.append(f"SAFE WITHDRAWAL RATES (investable ${investable:,.0f}):")
    lines.append(f"  3%: ${wr['swr_3pct']['monthly']:,.0f}/mo (${wr['swr_3pct']['annual']:,.0f}/yr)")
    lines.append(f"  4%: ${wr['swr_4pct']['monthly']:,.0f}/mo (${wr['swr_4pct']['annual']:,.0f}/yr)")
    lines.append(f"  5%: ${wr['swr_5pct']['monthly']:,.0f}/mo (${wr['swr_5pct']['annual']:,.0f}/yr)")
    lines.append("")

    # Milestones
    lines.append("KEY MILESTONES:")
    for label, m in milestones.items():
        if label.startswith("child"):
            lines.append(f"  {label}: age {m['current_age']}, college at 18 ({m['college_start_age_18']})")
        else:
            lines.append(f"  {label}: age {m['current_age']}, target retire {m['target_retirement_age']}")
            lines.append(f"    Medicare: {m['medicare_eligible']} ({m['medicare_years_away']}yr away)")
            lines.append(f"    SS: early {m['ss_early_62']}, FRA {m['ss_fra']}, delayed {m['ss_delayed_70']}")
            lines.append(f"    401k penalty-free: Rule of 55 {m['rule_of_55']}, age 59.5 {m['penalty_free_59_5']}")
            lines.append(f"    RMDs start: {m['rmd_start_75']}")
    lines.append("")

    # Spending breakdown
    if profile.spending:
        lines.append("MONTHLY SPENDING BREAKDOWN:")
        for s in sorted(profile.spending, key=lambda x: x.monthly_amount, reverse=True):
            lines.append(f"  {s.category}: ${s.monthly_amount:,.0f}")
        lines.append("")

    if additional_context:
        lines.append(f"USER-PROVIDED CONTEXT: {additional_context}")
        lines.append("")

    lines.append("=== END ANALYSIS ===")
    return "\n".join(lines)
