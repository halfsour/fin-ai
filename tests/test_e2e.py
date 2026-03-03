"""End-to-end test using sample data files.

Exercises the full web flow: upload files → get assumptions → confirm → follow-up.

Run modes:
    pytest tests/test_e2e.py              # mocked (fast, no credentials)
    pytest tests/test_e2e.py --live       # real Bedrock (slow, needs AWS creds)
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from retirement_planner.models import (
    BankAccount,
    CreditCard,
    FinancialProfile,
    InvestmentAccount,
    MonthlyBudgetItem,
    MonthlySpending,
    RetirementAssessment,
)
from retirement_planner.web import _active_sessions, app

client = TestClient(app)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(raw: str) -> list[dict]:
    events = []
    current_event = None
    current_data = None
    for line in raw.split("\n"):
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            current_data = line[len("data: "):]
        elif line == "" and current_event is not None:
            events.append({"event": current_event, "data": json.loads(current_data)})
            current_event = None
            current_data = None
    return events


def _sample_assessment() -> RetirementAssessment:
    return RetirementAssessment(
        can_retire=True,
        retirement_readiness_summary="James and Sarah are well-positioned for retirement.",
        recommended_monthly_budget=[
            MonthlyBudgetItem(category="Housing", amount=3450),
            MonthlyBudgetItem(category="Food", amount=900),
            MonthlyBudgetItem(category="Healthcare", amount=1200),
            MonthlyBudgetItem(category="Transportation", amount=600),
            MonthlyBudgetItem(category="Utilities", amount=400),
        ],
        net_worth=1_800_000,
        monthly_cash_flow=12_000,
        assumptions={
            "retirement_age": 65,
            "inflation_rate": 0.03,
            "expected_investment_return": 0.07,
            "social_security_start_age": 67,
            "life_expectancy": 90,
        },
        disclaimer="This analysis is AI-generated and does not constitute professional financial advice.",
    )


def _mock_profile() -> FinancialProfile:
    return FinancialProfile(
        investments=[
            InvestmentAccount(account_type="401k", balance=850_000, expected_annual_return=7.0, holdings="FXAIX, FSPSX"),
            InvestmentAccount(account_type="IRA", balance=320_000, expected_annual_return=7.0, holdings="VTSAX, VXUS"),
            InvestmentAccount(account_type="Taxable Brokerage", balance=475_000, expected_annual_return=8.0, holdings="AAPL, MSFT, VOO"),
            InvestmentAccount(account_type="HSA", balance=45_000, expected_annual_return=7.0, holdings="VFIAX"),
        ],
        bank_accounts=[
            BankAccount(account_type="checking", balance=28_500, monthly_income_deposits=18_500),
            BankAccount(account_type="savings", balance=65_000, monthly_income_deposits=0),
            BankAccount(account_type="savings", balance=42_000, monthly_income_deposits=180),
        ],
        credit_cards=[
            CreditCard(outstanding_balance=4_200, credit_limit=30_000, monthly_payment=4_200),
            CreditCard(outstanding_balance=1_800, credit_limit=15_000, monthly_payment=1_800),
            CreditCard(outstanding_balance=0, credit_limit=20_000, monthly_payment=0),
        ],
        spending=[
            MonthlySpending(category="Housing", monthly_amount=3_450),
            MonthlySpending(category="Groceries", monthly_amount=800),
            MonthlySpending(category="Transportation", monthly_amount=500),
            MonthlySpending(category="Utilities", monthly_amount=350),
            MonthlySpending(category="Insurance", monthly_amount=400),
            MonthlySpending(category="Dining", monthly_amount=300),
        ],
    )


def _mock_assumption_summary() -> dict:
    return {
        "extracted_data": {
            "accounts_found": 10,
            "total_investment_balance": 1_690_000,
            "total_bank_balance": 135_500,
            "total_credit_card_balance": 6_000,
            "monthly_income": 18_680,
            "monthly_expenses": 5_800,
            "spending_breakdown": [
                {"category": "Housing", "monthly_amount": 3_450},
                {"category": "Groceries", "monthly_amount": 800},
                {"category": "Transportation", "monthly_amount": 500},
            ],
        },
        "assumptions": {
            "retirement_age": 65,
            "inflation_rate": 3.0,
            "expected_investment_return": 7.0,
            "social_security_start_age": 67,
            "life_expectancy": 90,
        },
        "missing_data_questions": [
            "Do you own a home?",
            "Do you have any student loans?",
        ],
    }


@contextmanager
def _mock_all():
    """Context manager that patches all agent/Bedrock calls for fast offline tests."""
    with (
        patch("retirement_planner.web.history") as mock_history,
        patch("retirement_planner.web.generate_assumption_summary") as mock_assumptions,
        patch("retirement_planner.web.run_initial_assessment") as mock_assess,
        patch("retirement_planner.web.create_agent") as mock_create_agent,
        patch("retirement_planner.web._parse_all_from_file") as mock_parse,
        patch("retirement_planner.web.stream_follow_up") as mock_stream,
    ):
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_parse.return_value = _mock_profile()
        mock_assumptions.return_value = _mock_assumption_summary()
        mock_assess.return_value = _sample_assessment()
        mock_history.new_session_id.return_value = "e2e_session_001"
        mock_history.load_session.return_value = {"conversation": []}

        async def fake_stream(agent, question):
            yield {"text": "If you retire at 62, you would need to bridge 3 years."}
            yield {"done": True, "full_text": "If you retire at 62, you would need to bridge 3 years."}
        mock_stream.side_effect = fake_stream

        yield {
            "agent": mock_agent,
            "history": mock_history,
            "stream": mock_stream,
        }


@contextmanager
def _live_passthrough():
    """Context manager that lets all calls go through to real Bedrock."""
    # Only patch history to avoid writing to the user's real session dir
    with patch("retirement_planner.web.history") as mock_history:
        mock_history.new_session_id.return_value = "e2e_live_001"
        mock_history.load_session.return_value = {"conversation": []}
        yield {"history": mock_history}


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full flow: upload sample files → assumptions → confirm → follow-up."""

    def setup_method(self):
        _active_sessions.clear()

    def teardown_method(self):
        _active_sessions.clear()

    def test_full_flow_with_sample_files(self, request):
        live = request.config.getoption("--live")
        ctx = _live_passthrough if live else _mock_all

        with ctx() as mocks:
            # --- Step 1: Upload sample files via /upload-smart ---
            sample_files = [
                "investment_statement.csv",
                "bank_statements.csv",
                "credit_cards.csv",
                "spending_transactions.csv",
            ]
            uploaded = {"investments": [], "bank_accounts": [], "credit_cards": [], "spending": []}

            for filename in sample_files:
                content = (SAMPLES_DIR / filename).read_bytes()
                resp = client.post(
                    "/upload-smart",
                    files={"file": (filename, io.BytesIO(content), "text/csv")},
                )
                assert resp.status_code == 200, f"Upload failed for {filename}: {resp.text}"
                data = resp.json()
                assert "filename" in data
                for key in uploaded:
                    uploaded[key].extend(data.get(key, []))

            assert len(uploaded["investments"]) > 0
            assert len(uploaded["bank_accounts"]) > 0
            assert len(uploaded["credit_cards"]) > 0
            assert len(uploaded["spending"]) > 0

            # --- Step 2: Build profile and personal info ---
            personal_info_raw = json.loads((SAMPLES_DIR / "personal_info.json").read_text())
            personal_info = {
                "husband_birthdate": personal_info_raw["husband_birthdate"],
                "wife_birthdate": personal_info_raw["wife_birthdate"],
                "children_birthdates": personal_info_raw["children_birthdates"],
            }

            if live:
                # Use the real data the agent extracted
                profile = {
                    "investments": uploaded["investments"],
                    "bank_accounts": uploaded["bank_accounts"],
                    "credit_cards": uploaded["credit_cards"],
                    "spending": uploaded["spending"],
                }
            else:
                profile = _mock_profile().model_dump()

            # --- Step 3: Get assumption summary ---
            resp = client.post("/assumptions", json={
                "profile": profile,
                "personal_info": personal_info,
            })
            assert resp.status_code == 200
            summary = resp.json()
            assert "extracted_data" in summary
            assert "assumptions" in summary
            assert "retirement_age" in summary["assumptions"]

            # --- Step 4: Confirm assumptions (SSE stream) ---
            resp = client.post("/confirm-assumptions", json={
                "profile": profile,
                "personal_info": personal_info,
                "summary": summary,
            })
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            events = _parse_sse(resp.text)
            event_types = [e["event"] for e in events]
            assert "done" in event_types, f"No done event. Events: {event_types}"
            assert "error" not in event_types, f"Got error: {events}"

            done = next(e for e in events if e["event"] == "done")
            assessment = done["data"]["assessment"]
            session_id = done["data"]["session_id"]

            # Validate assessment structure regardless of mode
            assert "can_retire" in assessment
            assert isinstance(assessment["can_retire"], bool)
            assert "recommended_monthly_budget" in assessment
            assert len(assessment["recommended_monthly_budget"]) > 0
            assert "net_worth" in assessment
            assert assessment["net_worth"] > 0
            assert "disclaimer" in assessment

            if live:
                # Live-specific: the agent should recognize ~$1.8M net worth
                assert assessment["net_worth"] > 1_000_000
                print(f"\n  ✓ Live assessment: can_retire={assessment['can_retire']}, "
                      f"net_worth=${assessment['net_worth']:,.0f}")
            else:
                assert assessment["net_worth"] == 1_800_000

            # --- Step 5: Follow-up question ---
            _active_sessions[session_id] = mocks.get("agent") or _active_sessions.get(session_id)

            resp = client.post("/followup", json={
                "session_id": session_id,
                "question": "What if I retire at 62 instead?",
            })
            assert resp.status_code == 200

            events = _parse_sse(resp.text)
            event_types = [e["event"] for e in events]
            assert "done" in event_types, f"No done event. Events: {event_types}"

            done = next(e for e in events if e["event"] == "done")
            followup_response = done["data"].get("response") or json.dumps(done["data"].get("assessment", ""))
            assert len(followup_response) > 0

            if live:
                print(f"  ✓ Follow-up response: {followup_response[:120]}...")

    def test_assumption_corrections_applied(self, request):
        live = request.config.getoption("--live")
        ctx = _live_passthrough if live else _mock_all

        with ctx():
            profile = _mock_profile().model_dump()
            personal_info = {"husband_birthdate": "1974-06-15", "wife_birthdate": "1976-11-22"}

            # Get initial assumptions
            resp = client.post("/assumptions", json={
                "profile": profile,
                "personal_info": personal_info,
            })
            assert resp.status_code == 200
            summary = resp.json()

            # Confirm with corrections
            resp = client.post("/confirm-assumptions", json={
                "profile": profile,
                "personal_info": personal_info,
                "summary": summary,
                "corrections": {"retirement_age": 62, "inflation_rate": 2.5},
            })
            assert resp.status_code == 200
            events = _parse_sse(resp.text)
            assert any(e["event"] == "done" for e in events)
            assert not any(e["event"] == "error" for e in events)
