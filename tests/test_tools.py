"""Unit tests for retirement_planner.tools module."""

from __future__ import annotations

from retirement_planner.tools import (
    calculate_cash_flow,
    calculate_net_worth,
    read_financial_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    investments=None,
    bank_accounts=None,
    credit_cards=None,
) -> dict:
    return {
        "investments": investments or [],
        "bank_accounts": bank_accounts or [],
        "credit_cards": credit_cards or [],
    }


# ---------------------------------------------------------------------------
# calculate_net_worth
# ---------------------------------------------------------------------------


class TestCalculateNetWorth:
    def test_empty_profile(self):
        result = calculate_net_worth(profile=_make_profile())
        assert result == {"net_worth": 0, "total_assets": 0, "total_liabilities": 0}

    def test_assets_only(self):
        profile = _make_profile(
            investments=[{"balance": 100_000}],
            bank_accounts=[{"balance": 50_000}],
        )
        result = calculate_net_worth(profile=profile)
        assert result["net_worth"] == 150_000
        assert result["total_assets"] == 150_000
        assert result["total_liabilities"] == 0

    def test_liabilities_only(self):
        profile = _make_profile(credit_cards=[{"outstanding_balance": 5_000}])
        result = calculate_net_worth(profile=profile)
        assert result["net_worth"] == -5_000
        assert result["total_liabilities"] == 5_000

    def test_mixed(self):
        profile = _make_profile(
            investments=[{"balance": 200_000}, {"balance": 50_000}],
            bank_accounts=[{"balance": 10_000}],
            credit_cards=[{"outstanding_balance": 3_000}, {"outstanding_balance": 2_000}],
        )
        result = calculate_net_worth(profile=profile)
        assert result["total_assets"] == 260_000
        assert result["total_liabilities"] == 5_000
        assert result["net_worth"] == 255_000

    def test_missing_fields_returns_error(self):
        result = calculate_net_worth(profile={"investments": []})
        assert "error" in result

    def test_non_dict_returns_error(self):
        result = calculate_net_worth(profile="not a dict")
        assert "error" in result

    def test_non_list_values_returns_error(self):
        result = calculate_net_worth(
            profile={"investments": "bad", "bank_accounts": [], "credit_cards": []},
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# calculate_cash_flow
# ---------------------------------------------------------------------------


class TestCalculateCashFlow:
    def test_empty_profile(self):
        result = calculate_cash_flow(profile=_make_profile())
        assert result == {"monthly_cash_flow": 0, "total_income": 0, "total_expenses": 0, "expense_source": "credit_cards"}

    def test_income_only(self):
        profile = _make_profile(
            bank_accounts=[{"monthly_income_deposits": 5_000}],
        )
        result = calculate_cash_flow(profile=profile)
        assert result["total_income"] == 5_000
        assert result["total_expenses"] == 0
        assert result["monthly_cash_flow"] == 5_000
        assert result["expense_source"] == "credit_cards"

    def test_expenses_only(self):
        profile = _make_profile(
            credit_cards=[{"monthly_payment": 1_500}],
        )
        result = calculate_cash_flow(profile=profile)
        assert result["total_income"] == 0
        assert result["total_expenses"] == 1_500
        assert result["monthly_cash_flow"] == -1_500
        assert result["expense_source"] == "credit_cards"

    def test_mixed(self):
        profile = _make_profile(
            bank_accounts=[
                {"monthly_income_deposits": 4_000},
                {"monthly_income_deposits": 3_000},
            ],
            credit_cards=[
                {"monthly_payment": 500},
                {"monthly_payment": 200},
            ],
        )
        result = calculate_cash_flow(profile=profile)
        assert result["total_income"] == 7_000
        assert result["total_expenses"] == 700
        assert result["monthly_cash_flow"] == 6_300
        assert result["expense_source"] == "credit_cards"

    def test_spending_data_preferred_over_credit_cards(self):
        profile = _make_profile(
            bank_accounts=[{"monthly_income_deposits": 10_000}],
            credit_cards=[{"monthly_payment": 500}],
        )
        profile["spending"] = [
            {"category": "Groceries", "monthly_amount": 800},
            {"category": "Housing", "monthly_amount": 3_000},
        ]
        result = calculate_cash_flow(profile=profile)
        assert result["total_expenses"] == 3_800
        assert result["monthly_cash_flow"] == 6_200
        assert result["expense_source"] == "spending"

    def test_missing_fields_returns_error(self):
        result = calculate_cash_flow(profile={"bank_accounts": []})
        assert "error" in result

    def test_non_dict_returns_error(self):
        result = calculate_cash_flow(profile=42)
        assert "error" in result


# ---------------------------------------------------------------------------
# read_financial_file
# ---------------------------------------------------------------------------


class TestReadFinancialFile:
    def test_read_valid_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\n1,2\n", encoding="utf-8")
        result = read_financial_file(file_path=str(f))
        assert result["content"] == "col1,col2\n1,2\n"
        assert result["file_path"] == str(f)

    def test_file_not_found(self):
        result = read_financial_file(file_path="/nonexistent/path.csv")
        assert "error" in result
        assert "not found" in result["error"].lower() or "File not found" in result["error"]

    def test_directory_path(self, tmp_path):
        result = read_financial_file(file_path=str(tmp_path))
        assert "error" in result

    def test_empty_path_returns_error(self):
        result = read_financial_file(file_path="")
        assert "error" in result

    def test_binary_file(self, tmp_path):
        f = tmp_path / "binary.dat"
        f.write_bytes(b"\x80\x81\x82\xff\xfe")
        result = read_financial_file(file_path=str(f))
        # May or may not error depending on content; just ensure no exception
        assert "content" in result or "error" in result

    def test_json_file(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"accounts": []}', encoding="utf-8")
        result = read_financial_file(file_path=str(f))
        assert result["content"] == '{"accounts": []}'
