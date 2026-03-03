"""Pydantic data models and custom exceptions for the Retirement Planner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class RetirementPlannerError(Exception):
    """Base exception for the retirement planner."""


class FileParseError(RetirementPlannerError):
    """Raised when a data file cannot be read."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to read {file_path}: {reason}")


class NormalizationError(RetirementPlannerError):
    """Raised when the agent cannot extract required financial fields from raw file content."""

    def __init__(self, file_path: str, missing_fields: list[str]):
        self.file_path = file_path
        self.missing_fields = missing_fields
        super().__init__(
            f"Could not extract fields from {file_path}: {', '.join(missing_fields)}"
        )


class CredentialError(RetirementPlannerError):
    """Raised when AWS credentials are missing or invalid."""


class BedrockError(RetirementPlannerError):
    """Raised when Amazon Bedrock API calls fail."""

    def __init__(self, message: str, retryable: bool = True):
        self.retryable = retryable
        super().__init__(message)


# ---------------------------------------------------------------------------
# Financial Data Models
# ---------------------------------------------------------------------------


class InvestmentAccount(BaseModel):
    """An investment account (e.g. 401k, IRA, brokerage)."""

    account_type: str
    balance: float
    expected_annual_return: float
    holdings: str = ""  # e.g. "VTSAX", "S&P 500 index fund", "target date 2045"

    @field_validator("account_type")
    @classmethod
    def account_type_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("account_type must be a non-empty string")
        return v

    @field_validator("balance")
    @classmethod
    def balance_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("balance must be >= 0")
        return v


class BankAccount(BaseModel):
    """A bank account (checking or savings)."""

    account_type: str
    balance: float
    monthly_income_deposits: float

    @field_validator("balance")
    @classmethod
    def balance_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("balance must be >= 0")
        return v

    @field_validator("monthly_income_deposits")
    @classmethod
    def deposits_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("monthly_income_deposits must be >= 0")
        return v


class CreditCard(BaseModel):
    """A credit card with balance, limit, and payment info."""

    outstanding_balance: float
    credit_limit: float
    monthly_payment: float

    @field_validator("outstanding_balance")
    @classmethod
    def outstanding_balance_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("outstanding_balance must be >= 0")
        return v

    @field_validator("credit_limit")
    @classmethod
    def credit_limit_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("credit_limit must be >= 0")
        return v

    @field_validator("monthly_payment")
    @classmethod
    def monthly_payment_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("monthly_payment must be >= 0")
        return v


class MonthlySpending(BaseModel):
    """A category-level monthly spending summary (e.g., Groceries: $800/mo)."""

    category: str
    monthly_amount: float

    @field_validator("category")
    @classmethod
    def category_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("category must be a non-empty string")
        return v

    @field_validator("monthly_amount")
    @classmethod
    def monthly_amount_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("monthly_amount must be >= 0")
        return v


class FinancialProfile(BaseModel):
    """Combined financial data for the couple."""

    investments: list[InvestmentAccount]
    bank_accounts: list[BankAccount]
    credit_cards: list[CreditCard]
    spending: list[MonthlySpending] = []


# ---------------------------------------------------------------------------
# Personal Information
# ---------------------------------------------------------------------------


def _validate_age(v: int, field_name: str) -> int:
    if v < 0 or v > 120:
        raise ValueError(f"{field_name} must be between 0 and 120")
    return v


class PersonalInfo(BaseModel):
    """Personal information for the couple and their children.

    Accepts either birthdates (preferred for accuracy) or ages.
    When birthdates are provided, ages are computed automatically.
    """

    husband_age: int = 0
    wife_age: int = 0
    children_ages: list[int] = []
    husband_birthdate: str | None = None  # YYYY-MM-DD
    wife_birthdate: str | None = None  # YYYY-MM-DD
    children_birthdates: list[str] = []  # YYYY-MM-DD

    @model_validator(mode="after")
    def compute_ages_from_birthdates(self):
        """If birthdates are provided, compute ages from them."""
        from datetime import date

        today = date.today()

        def age_from_dob(dob_str: str) -> int:
            parts = dob_str.strip().split("-")
            if len(parts) == 3:
                dob = date(int(parts[0]), int(parts[1]), int(parts[2]))
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                return age
            return 0

        if self.husband_birthdate:
            self.husband_age = age_from_dob(self.husband_birthdate)
        if self.wife_birthdate:
            self.wife_age = age_from_dob(self.wife_birthdate)
        if self.children_birthdates:
            self.children_ages = [age_from_dob(d) for d in self.children_birthdates]

        return self

    @field_validator("husband_age")
    @classmethod
    def validate_husband_age(cls, v: int) -> int:
        return _validate_age(v, "husband_age")

    @field_validator("wife_age")
    @classmethod
    def validate_wife_age(cls, v: int) -> int:
        return _validate_age(v, "wife_age")

    @field_validator("children_ages")
    @classmethod
    def validate_children_ages(cls, v: list[int]) -> list[int]:
        for i, age in enumerate(v):
            _validate_age(age, f"children_ages[{i}]")
        return v


# ---------------------------------------------------------------------------
# Assessment Models
# ---------------------------------------------------------------------------


class MonthlyBudgetItem(BaseModel):
    """A single line item in the recommended monthly budget."""

    category: str
    amount: float


class RetirementAssessment(BaseModel):
    """The AI-generated retirement readiness assessment."""

    can_retire: bool
    retirement_readiness_summary: str
    recommended_monthly_budget: list[MonthlyBudgetItem]
    net_worth: float
    monthly_cash_flow: float
    assumptions: dict[str, Any]
    disclaimer: str
