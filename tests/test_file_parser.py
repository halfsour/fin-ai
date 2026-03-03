"""Unit tests for retirement_planner.file_parser module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from retirement_planner.file_parser import (
    normalize_file_data,
    parse_banking_file,
    parse_credit_cards_file,
    parse_investments_file,
    read_file_contents,
)
from retirement_planner.models import (
    BankAccount,
    CreditCard,
    FileParseError,
    InvestmentAccount,
    NormalizationError,
)


# ---------------------------------------------------------------------------
# read_file_contents
# ---------------------------------------------------------------------------


class TestReadFileContents:
    def test_read_valid_text_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\nval1,val2\n", encoding="utf-8")
        result = read_file_contents(str(f))
        assert result == "col1,col2\nval1,val2\n"

    def test_read_json_file(self, tmp_path):
        f = tmp_path / "data.json"
        content = json.dumps([{"account_type": "401k", "balance": 50000}])
        f.write_text(content, encoding="utf-8")
        result = read_file_contents(str(f))
        assert result == content

    def test_file_not_found_raises(self):
        with pytest.raises(FileParseError) as exc_info:
            read_file_contents("/nonexistent/path/file.csv")
        assert "not found" in exc_info.value.reason.lower()
        assert exc_info.value.file_path == "/nonexistent/path/file.csv"

    def test_directory_raises(self, tmp_path):
        with pytest.raises(FileParseError) as exc_info:
            read_file_contents(str(tmp_path))
        assert "directory" in exc_info.value.reason.lower()

    def test_binary_file_raises(self, tmp_path):
        f = tmp_path / "binary.dat"
        # Write bytes that are invalid UTF-8
        f.write_bytes(b"\x80\x81\x82\xff\xfe\x00\x01")
        with pytest.raises(FileParseError) as exc_info:
            read_file_contents(str(f))
        assert "binary" in exc_info.value.reason.lower() or "encoding" in exc_info.value.reason.lower()

    def test_empty_file_returns_empty_string(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = read_file_contents(str(f))
        assert result == ""

    def test_error_includes_file_path(self, tmp_path):
        path = "/no/such/file.txt"
        with pytest.raises(FileParseError) as exc_info:
            read_file_contents(path)
        assert exc_info.value.file_path == path


# ---------------------------------------------------------------------------
# normalize_file_data
# ---------------------------------------------------------------------------


def _mock_agent(response_json):
    """Create a mock agent that returns the given JSON when called."""
    agent = MagicMock()
    agent.return_value = json.dumps(response_json)
    return agent


class TestNormalizeFileData:
    def test_normalize_investments(self):
        data = [
            {"account_type": "401k", "balance": 100000.0, "expected_annual_return": 7.0},
            {"account_type": "IRA", "balance": 50000.0, "expected_annual_return": 6.5},
        ]
        agent = _mock_agent(data)
        result = normalize_file_data(agent, "some raw content", "investments")
        assert len(result) == 2
        assert all(isinstance(r, InvestmentAccount) for r in result)
        assert result[0].account_type == "401k"
        assert result[0].balance == 100000.0
        assert result[1].expected_annual_return == 6.5

    def test_normalize_banking(self):
        data = [
            {"account_type": "checking", "balance": 5000.0, "monthly_income_deposits": 4000.0},
        ]
        agent = _mock_agent(data)
        result = normalize_file_data(agent, "raw banking data", "banking")
        assert len(result) == 1
        assert isinstance(result[0], BankAccount)
        assert result[0].monthly_income_deposits == 4000.0

    def test_normalize_credit_cards(self):
        data = [
            {"outstanding_balance": 2000.0, "credit_limit": 10000.0, "monthly_payment": 500.0},
        ]
        agent = _mock_agent(data)
        result = normalize_file_data(agent, "raw cc data", "credit_cards")
        assert len(result) == 1
        assert isinstance(result[0], CreditCard)
        assert result[0].outstanding_balance == 2000.0

    def test_invalid_category_raises(self):
        agent = MagicMock()
        with pytest.raises(FileParseError) as exc_info:
            normalize_file_data(agent, "content", "invalid_category")
        assert "Invalid category" in exc_info.value.reason

    def test_agent_returns_markdown_fenced_json(self):
        data = [{"account_type": "savings", "balance": 1000.0, "monthly_income_deposits": 500.0}]
        agent = MagicMock()
        agent.return_value = f"Here is the data:\n```json\n{json.dumps(data)}\n```"
        result = normalize_file_data(agent, "raw content", "banking")
        assert len(result) == 1
        assert isinstance(result[0], BankAccount)

    def test_agent_returns_single_object(self):
        data = {"account_type": "brokerage", "balance": 75000.0, "expected_annual_return": 8.0}
        agent = _mock_agent(data)
        result = normalize_file_data(agent, "raw content", "investments")
        assert len(result) == 1
        assert result[0].account_type == "brokerage"

    def test_agent_failure_raises(self):
        agent = MagicMock(side_effect=RuntimeError("Bedrock timeout"))
        with pytest.raises(FileParseError) as exc_info:
            normalize_file_data(agent, "content", "investments")
        assert "Agent failed" in exc_info.value.reason

    def test_agent_returns_non_json_raises(self):
        agent = MagicMock()
        agent.return_value = "I could not understand the file contents."
        with pytest.raises(FileParseError) as exc_info:
            normalize_file_data(agent, "content", "investments")
        assert "valid JSON" in exc_info.value.reason

    def test_agent_returns_empty_array_raises(self):
        agent = _mock_agent([])
        with pytest.raises(FileParseError) as exc_info:
            normalize_file_data(agent, "content", "banking")
        assert "no" in exc_info.value.reason.lower()

    def test_validation_failure_raises_normalization_error(self):
        # Negative balance should fail Pydantic validation
        data = [{"account_type": "401k", "balance": -100.0, "expected_annual_return": 7.0}]
        agent = _mock_agent(data)
        with pytest.raises(NormalizationError) as exc_info:
            normalize_file_data(agent, "content", "investments")
        assert "balance" in exc_info.value.missing_fields


# ---------------------------------------------------------------------------
# Category-specific parse functions
# ---------------------------------------------------------------------------


class TestParseInvestmentsFile:
    def test_parses_valid_file(self, tmp_path):
        f = tmp_path / "investments.csv"
        f.write_text("account_type,balance,return\n401k,100000,7.0\n", encoding="utf-8")
        data = [{"account_type": "401k", "balance": 100000.0, "expected_annual_return": 7.0}]
        agent = _mock_agent(data)
        result = parse_investments_file(agent, str(f))
        assert len(result) == 1
        assert isinstance(result[0], InvestmentAccount)

    def test_nonexistent_file_raises(self):
        agent = MagicMock()
        with pytest.raises(FileParseError):
            parse_investments_file(agent, "/no/such/file.csv")


class TestParseBankingFile:
    def test_parses_valid_file(self, tmp_path):
        f = tmp_path / "banking.json"
        f.write_text('{"accounts": []}', encoding="utf-8")
        data = [{"account_type": "checking", "balance": 5000.0, "monthly_income_deposits": 3000.0}]
        agent = _mock_agent(data)
        result = parse_banking_file(agent, str(f))
        assert len(result) == 1
        assert isinstance(result[0], BankAccount)

    def test_nonexistent_file_raises(self):
        agent = MagicMock()
        with pytest.raises(FileParseError):
            parse_banking_file(agent, "/no/such/file.json")


class TestParseCreditCardsFile:
    def test_parses_valid_file(self, tmp_path):
        f = tmp_path / "cards.txt"
        f.write_text("Visa: balance $2000, limit $10000, payment $500\n", encoding="utf-8")
        data = [{"outstanding_balance": 2000.0, "credit_limit": 10000.0, "monthly_payment": 500.0}]
        agent = _mock_agent(data)
        result = parse_credit_cards_file(agent, str(f))
        assert len(result) == 1
        assert isinstance(result[0], CreditCard)

    def test_nonexistent_file_raises(self):
        agent = MagicMock()
        with pytest.raises(FileParseError):
            parse_credit_cards_file(agent, "/no/such/file.txt")
