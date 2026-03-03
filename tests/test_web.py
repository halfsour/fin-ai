"""Tests for the FastAPI web server endpoints.

Mocks the Strands Agent and backend modules to avoid real Bedrock calls.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from retirement_planner.models import (
    BedrockError,
    BankAccount,
    CreditCard,
    CredentialError,
    FileParseError,
    InvestmentAccount,
    MonthlyBudgetItem,
    NormalizationError,
    RetirementAssessment,
)
from retirement_planner.web import _active_sessions, app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _sample_assessment() -> RetirementAssessment:
    return RetirementAssessment(
        can_retire=True,
        retirement_readiness_summary="You are on track.",
        recommended_monthly_budget=[
            MonthlyBudgetItem(category="Housing", amount=2000.0),
        ],
        net_worth=500000.0,
        monthly_cash_flow=3000.0,
        assumptions={"inflation_rate": 0.03},
        disclaimer="This is not financial advice.",
    )


def _sample_profile_dict() -> dict:
    return {
        "investments": [
            {"account_type": "401k", "balance": 100000.0, "expected_annual_return": 0.07}
        ],
        "bank_accounts": [
            {"account_type": "checking", "balance": 5000.0, "monthly_income_deposits": 4000.0}
        ],
        "credit_cards": [
            {"outstanding_balance": 1000.0, "credit_limit": 10000.0, "monthly_payment": 200.0}
        ],
    }


def _sample_personal_info_dict() -> dict:
    return {"husband_age": 55, "wife_age": 52, "children_ages": [20, 18]}


def _sample_assess_payload() -> dict:
    return {
        "profile": _sample_profile_dict(),
        "personal_info": _sample_personal_info_dict(),
    }


def _parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts."""
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


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestServeIndex:
    def test_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    @patch("retirement_planner.web.normalize_file_data")
    @patch("retirement_planner.web.read_file_contents")
    @patch("retirement_planner.web.create_agent")
    def test_valid_upload_returns_parsed_data(self, mock_agent, mock_read, mock_normalize):
        mock_read.return_value = "raw csv content"
        mock_normalize.return_value = [
            InvestmentAccount(account_type="401k", balance=50000.0, expected_annual_return=0.07),
        ]
        resp = client.post(
            "/upload",
            files={"file": ("portfolio.csv", io.BytesIO(b"some,csv,data"), "text/csv")},
            data={"category": "investments"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert body[0]["account_type"] == "401k"
        assert body[0]["balance"] == 50000.0

    def test_invalid_category_returns_422(self):
        resp = client.post(
            "/upload",
            files={"file": ("data.csv", io.BytesIO(b"data"), "text/csv")},
            data={"category": "invalid_cat"},
        )
        assert resp.status_code == 422
        assert "invalid_cat" in resp.json()["detail"].lower()

    @patch("retirement_planner.web.read_file_contents")
    def test_unreadable_file_returns_500(self, mock_read):
        mock_read.side_effect = FileParseError("/tmp/bad.bin", "binary file")
        resp = client.post(
            "/upload",
            files={"file": ("bad.bin", io.BytesIO(b"\x00\x01"), "application/octet-stream")},
            data={"category": "banking"},
        )
        assert resp.status_code == 500
        assert "binary file" in resp.json()["detail"]

    @patch("retirement_planner.web.normalize_file_data")
    @patch("retirement_planner.web.read_file_contents")
    @patch("retirement_planner.web.create_agent")
    def test_normalization_failure_returns_500(self, mock_agent, mock_read, mock_normalize):
        mock_read.return_value = "raw content"
        mock_normalize.side_effect = NormalizationError("<file>", ["balance"])
        resp = client.post(
            "/upload",
            files={"file": ("data.csv", io.BytesIO(b"data"), "text/csv")},
            data={"category": "credit_cards"},
        )
        assert resp.status_code == 500
        assert "balance" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /assess
# ---------------------------------------------------------------------------


class TestAssessEndpoint:
    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.run_initial_assessment")
    @patch("retirement_planner.web.create_agent")
    def test_valid_assess_returns_sse_stream(self, mock_agent, mock_run, mock_history):
        assessment = _sample_assessment()
        mock_run.return_value = assessment
        mock_history.new_session_id.return_value = "20250101_120000"

        resp = client.post("/assess", json=_sample_assess_payload())
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = _parse_sse(resp.text)
        event_types = [e["event"] for e in events]
        assert "chunk" in event_types
        assert "done" in event_types

        done_event = next(e for e in events if e["event"] == "done")
        assert done_event["data"]["session_id"] == "20250101_120000"
        assert done_event["data"]["assessment"]["can_retire"] is True

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.run_initial_assessment")
    @patch("retirement_planner.web.create_agent")
    def test_sse_chunk_contains_summary_text(self, mock_agent, mock_run, mock_history):
        assessment = _sample_assessment()
        mock_run.return_value = assessment
        mock_history.new_session_id.return_value = "sess1"

        resp = client.post("/assess", json=_sample_assess_payload())
        events = _parse_sse(resp.text)
        chunk = next(e for e in events if e["event"] == "chunk")
        assert chunk["data"]["text"] == "You are on track."

    def test_missing_fields_returns_422(self):
        # Missing personal_info entirely
        resp = client.post("/assess", json={"profile": _sample_profile_dict()})
        assert resp.status_code == 422

    def test_invalid_profile_returns_422(self):
        payload = _sample_assess_payload()
        payload["profile"]["investments"][0]["balance"] = -100  # negative balance
        resp = client.post("/assess", json=payload)
        assert resp.status_code == 422

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.run_initial_assessment")
    @patch("retirement_planner.web.create_agent")
    def test_agent_failure_produces_sse_error(self, mock_agent, mock_run, mock_history):
        mock_run.side_effect = BedrockError("Model unavailable")
        mock_history.new_session_id.return_value = "sess_err"

        resp = client.post("/assess", json=_sample_assess_payload())
        assert resp.status_code == 200  # SSE stream still returns 200
        events = _parse_sse(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert "Model unavailable" in error_events[0]["data"]["message"]

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.run_initial_assessment")
    @patch("retirement_planner.web.create_agent")
    def test_credential_error_produces_sse_error(self, mock_agent, mock_run, mock_history):
        mock_run.side_effect = CredentialError("No AWS credentials")
        mock_history.new_session_id.return_value = "sess_cred"

        resp = client.post("/assess", json=_sample_assess_payload())
        events = _parse_sse(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert "credentials" in error_events[0]["data"]["message"].lower()

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.run_initial_assessment")
    @patch("retirement_planner.web.create_agent")
    def test_saves_session_after_assessment(self, mock_agent, mock_run, mock_history):
        mock_run.return_value = _sample_assessment()
        mock_history.new_session_id.return_value = "sess_save"

        client.post("/assess", json=_sample_assess_payload())
        mock_history.save_session.assert_called_once()
        call_args = mock_history.save_session.call_args
        assert call_args[0][0] == "sess_save"


# ---------------------------------------------------------------------------
# POST /followup
# ---------------------------------------------------------------------------


class TestFollowUpEndpoint:
    def setup_method(self):
        """Inject a fake agent into _active_sessions for follow-up tests."""
        _active_sessions.clear()
        self.mock_agent = MagicMock()
        _active_sessions["test_session"] = self.mock_agent

    def teardown_method(self):
        _active_sessions.clear()

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.stream_follow_up")
    def test_valid_followup_returns_sse_stream(self, mock_stream, mock_history):
        async def fake_stream(agent, question):
            yield {"text": "Here is your updated projection."}
            yield {"done": True, "full_text": "Here is your updated projection."}
        mock_stream.side_effect = fake_stream
        mock_history.load_session.return_value = {"conversation": []}

        resp = client.post(
            "/followup",
            json={"session_id": "test_session", "question": "What if I retire at 62?"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = _parse_sse(resp.text)
        event_types = [e["event"] for e in events]
        assert "chunk" in event_types
        assert "done" in event_types

        done = next(e for e in events if e["event"] == "done")
        assert done["data"]["session_id"] == "test_session"
        assert done["data"]["response"] == "Here is your updated projection."

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.stream_follow_up")
    def test_followup_with_assessment_result(self, mock_stream, mock_history):
        assessment = _sample_assessment()
        assessment_json = json.dumps(assessment.model_dump())
        async def fake_stream(agent, question):
            yield {"text": assessment_json}
            yield {"done": True, "full_text": assessment_json}
        mock_stream.side_effect = fake_stream
        mock_history.load_session.return_value = {"conversation": []}

        resp = client.post(
            "/followup",
            json={"session_id": "test_session", "question": "Change inflation to 4%"},
        )
        events = _parse_sse(resp.text)
        done = next(e for e in events if e["event"] == "done")
        assert "assessment" in done["data"]
        assert done["data"]["assessment"]["can_retire"] is True

    def test_unknown_session_returns_404(self):
        resp = client.post(
            "/followup",
            json={"session_id": "nonexistent_session", "question": "Hello?"},
        )
        assert resp.status_code == 404
        assert "nonexistent_session" in resp.json()["detail"]

    def test_missing_fields_returns_422(self):
        # Missing question field
        resp = client.post("/followup", json={"session_id": "test_session"})
        assert resp.status_code == 422

    def test_missing_session_id_returns_422(self):
        resp = client.post("/followup", json={"question": "Hello?"})
        assert resp.status_code == 422

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.stream_follow_up")
    def test_agent_failure_produces_sse_error(self, mock_stream, mock_history):
        async def fake_stream(agent, question):
            raise BedrockError("Throttled")
            yield  # make it a generator  # noqa: unreachable
        mock_stream.side_effect = fake_stream

        resp = client.post(
            "/followup",
            json={"session_id": "test_session", "question": "What if?"},
        )
        events = _parse_sse(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert "Throttled" in error_events[0]["data"]["message"]

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.stream_follow_up")
    def test_updates_session_after_followup(self, mock_stream, mock_history):
        async def fake_stream(agent, question):
            yield {"text": "Answer"}
            yield {"done": True, "full_text": "Answer"}
        mock_stream.side_effect = fake_stream
        mock_history.load_session.return_value = {"conversation": []}

        client.post(
            "/followup",
            json={"session_id": "test_session", "question": "Q?"},
        )
        mock_history.save_session.assert_called_once()
        saved_data = mock_history.save_session.call_args[0][1]
        assert len(saved_data["conversation"]) == 1
        assert saved_data["conversation"][0]["question"] == "Q?"


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------


class TestSessionsEndpoint:
    @patch("retirement_planner.web.history")
    def test_list_sessions_returns_list(self, mock_history):
        mock_history.list_sessions.return_value = [
            {"session_id": "s1", "created_at": "20250101_120000", "can_retire": True, "net_worth": 500000, "exchanges": 2},
            {"session_id": "s2", "created_at": "20250102_120000", "can_retire": False, "net_worth": 100000, "exchanges": 0},
        ]
        resp = client.get("/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 2
        assert body[0]["session_id"] == "s1"

    @patch("retirement_planner.web.history")
    def test_list_sessions_empty(self, mock_history):
        mock_history.list_sessions.return_value = []
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}
# ---------------------------------------------------------------------------


class TestGetSessionEndpoint:
    @patch("retirement_planner.web.history")
    def test_valid_session_returns_data(self, mock_history):
        session_data = {
            "created_at": "20250101_120000",
            "profile": _sample_profile_dict(),
            "personal_info": _sample_personal_info_dict(),
            "assessment": _sample_assessment().model_dump(),
            "conversation": [],
        }
        mock_history.load_session.return_value = session_data

        resp = client.get("/sessions/s1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["created_at"] == "20250101_120000"
        assert body["assessment"]["can_retire"] is True

    @patch("retirement_planner.web.history")
    def test_unknown_session_returns_404(self, mock_history):
        mock_history.load_session.side_effect = FileNotFoundError("not found")
        resp = client.get("/sessions/nonexistent")
        assert resp.status_code == 404
        assert "nonexistent" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# SSE format validation
# ---------------------------------------------------------------------------


class TestSSEFormat:
    """Verify that SSE events have correct event: and data: fields."""

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.run_initial_assessment")
    @patch("retirement_planner.web.create_agent")
    def test_sse_events_have_correct_format(self, mock_agent, mock_run, mock_history):
        mock_run.return_value = _sample_assessment()
        mock_history.new_session_id.return_value = "fmt_sess"

        resp = client.post("/assess", json=_sample_assess_payload())
        raw = resp.text

        # Each SSE block should have "event:" and "data:" lines
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.split("\n")
            event_lines = [l for l in lines if l.startswith("event: ")]
            data_lines = [l for l in lines if l.startswith("data: ")]
            assert len(event_lines) == 1, f"Expected 1 event line, got {len(event_lines)} in: {block}"
            assert len(data_lines) == 1, f"Expected 1 data line, got {len(data_lines)} in: {block}"

            # data should be valid JSON
            data_str = data_lines[0][len("data: "):]
            parsed = json.loads(data_str)
            assert isinstance(parsed, dict)

    @patch("retirement_planner.web.history")
    @patch("retirement_planner.web.run_initial_assessment")
    @patch("retirement_planner.web.create_agent")
    def test_sse_event_types_are_valid(self, mock_agent, mock_run, mock_history):
        mock_run.return_value = _sample_assessment()
        mock_history.new_session_id.return_value = "type_sess"

        resp = client.post("/assess", json=_sample_assess_payload())
        events = _parse_sse(resp.text)
        valid_types = {"chunk", "done", "error"}
        for e in events:
            assert e["event"] in valid_types, f"Unexpected event type: {e['event']}"
