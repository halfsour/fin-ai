"""FastAPI web server for the Retirement Planner.

Exposes REST API endpoints for file upload, assessment, follow-up questions,
and session management. Streams agent responses via Server-Sent Events (SSE).
Delegates all business logic to the shared backend modules.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from retirement_planner import history
from retirement_planner.agent import (
    create_agent,
    generate_assumption_summary,
    restore_agent_from_session,
    run_follow_up,
    run_initial_assessment,
    stream_follow_up,
)
from retirement_planner.file_parser import normalize_file_data, read_file_contents
from retirement_planner.file_parser import parse_all_from_file as _parse_all_from_file
from retirement_planner.models import (
    BedrockError,
    CredentialError,
    FileParseError,
    FinancialProfile,
    NormalizationError,
    PersonalInfo,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Retirement Planner", version="0.1.0")

# In-memory store of active agent instances keyed by session_id
_active_sessions: dict[str, Any] = {}

# Path to the static directory
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AssessRequest(BaseModel):
    """Request body for POST /assess."""
    profile: FinancialProfile
    personal_info: PersonalInfo


class FollowUpRequest(BaseModel):
    """Request body for POST /followup."""
    session_id: str
    question: str


class AssumptionSummaryRequest(BaseModel):
    """Request body for POST /assumptions."""
    profile: FinancialProfile
    personal_info: PersonalInfo
    file_sources: dict[str, str] | None = None


class ConfirmAssumptionsRequest(BaseModel):
    """Request body for POST /confirm-assumptions."""
    profile: FinancialProfile
    personal_info: PersonalInfo
    summary: dict[str, Any]
    corrections: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"investments", "banking", "credit_cards"}


def _split_into_sections(text: str) -> list[str]:
    """Split response text into sections at major boundaries.

    Splits on double-newline followed by a header-like pattern (═══, ---,
    ## markdown headers, or ALL CAPS lines).
    """
    import re
    # Split on double-newline before section markers
    parts = re.split(r'(\n\n(?=[═─#\-]{2,}|\n[A-Z][A-Z\s—\-]{4,}\n|#{1,4}\s))', text)
    sections: list[str] = []
    current = ""
    for part in parts:
        current += part
        # If current section is substantial, emit it
        if len(current.strip()) > 50 and current.count('\n') >= 2:
            # Check if next part starts a new section
            if current.rstrip().endswith('\n') or len(current) > 500:
                sections.append(current)
                current = ""
    if current.strip():
        sections.append(current)
    # If splitting produced nothing useful, fall back to paragraph splits
    if len(sections) <= 1:
        paragraphs = text.split('\n\n')
        sections = []
        current = ""
        for p in paragraphs:
            current += p + '\n\n'
            if len(current) > 300:
                sections.append(current)
                current = ""
        if current.strip():
            sections.append(current)
def _split_into_sections(text: str) -> list[str]:
    """Split response text into sections at major boundaries."""
    import re
    parts = re.split(r'(\n\n(?=[═─#\-]{2,}|[A-Z][A-Z\s—\-]{4,}\n|#{1,4}\s))', text)
    sections: list[str] = []
    current = ""
    for part in parts:
        current += part
        if len(current.strip()) > 50 and current.count('\n') >= 2:
            if current.rstrip().endswith('\n') or len(current) > 500:
                sections.append(current)
                current = ""
    if current.strip():
        sections.append(current)
    if len(sections) <= 1:
        paragraphs = text.split('\n\n')
        sections = []
        current = ""
        for p in paragraphs:
            current += p + '\n\n'
            if len(current) > 300:
                sections.append(current)
                current = ""
        if current.strip():
            sections.append(current)
    return sections if sections else [text]


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_index():
    """Serve the static index.html page."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path, media_type="text/html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), category: str = Form(...)):
    """Accept a Data_File upload with a financial category.

    Reads raw contents, passes to the agent for normalization,
    validates against Pydantic models, and returns parsed data as JSON.
    """
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}",
        )

    # Save uploaded file to a temp path
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}")

    # Read raw contents via file_parser
    try:
        raw_content = read_file_contents(tmp_path)
    except FileParseError as exc:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # Clean up temp file after reading
    Path(tmp_path).unlink(missing_ok=True)

    if not raw_content or not raw_content.strip():
        raise HTTPException(
            status_code=500,
            detail=f"Could not extract text from {file.filename}. "
                   "If this is a scanned PDF, try using the CLI with --files instead.",
        )

    # Normalize via agent
    try:
        agent = create_agent()
        validated = normalize_file_data(agent, raw_content, category)
    except (FileParseError, NormalizationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except (CredentialError, BedrockError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content=[item.model_dump() for item in validated])


@app.post("/upload-smart")
async def upload_file_smart(file: UploadFile = File(...)):
    """Upload a file and auto-detect all financial data categories.

    Handles text files (CSV, JSON, plain text) and PDFs (including
    scanned/image-based PDFs). Uses the agent to extract investments,
    bank accounts, and credit cards from whatever is in the file.

    Returns a JSON object with investments, bank_accounts, and credit_cards arrays.
    """
    # Save uploaded file to a temp path, preserving extension
    try:
        contents = await file.read()
        suffix = ""
        if file.filename:
            import os
            _, ext = os.path.splitext(file.filename)
            suffix = ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}")

    try:
        loop = asyncio.get_event_loop()
        agent = create_agent()
        print(f"[upload-smart] Processing {file.filename} from {tmp_path}")
        profile = await loop.run_in_executor(None, _parse_all_from_file, agent, tmp_path)
        print(f"[upload-smart] Result: {len(profile.investments)} investments, {len(profile.bank_accounts)} bank accounts, {len(profile.credit_cards)} credit cards")

        total = len(profile.investments) + len(profile.bank_accounts) + len(profile.credit_cards)
        if total == 0:
            raise HTTPException(
                status_code=422,
                detail=f"No financial data could be extracted from {file.filename}. "
                       "This can happen with PDFs generated from web pages (like Fidelity NetBenefits) "
                       "where the content is rendered by JavaScript and not captured in the PDF. "
                       "Try one of these alternatives:\n"
                       "• Use the site's Download/Export option to get a CSV or Excel file\n"
                       "• Take a screenshot of the page and upload the image (PNG/JPG)\n"
                       "• Copy the text from the page and paste it into a .txt file, then upload that",
            )

        result = {
            "filename": file.filename,
            "investments": [item.model_dump() for item in profile.investments],
            "bank_accounts": [item.model_dump() for item in profile.bank_accounts],
            "credit_cards": [item.model_dump() for item in profile.credit_cards],
            "spending": [item.model_dump() for item in profile.spending],
        }
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except (FileParseError, NormalizationError) as exc:
        print(f"[upload-smart] Parse error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    except (CredentialError, BedrockError) as exc:
        print(f"[upload-smart] Credential/Bedrock error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        print(f"[upload-smart] Unexpected error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/assess")
async def run_assessment(request: AssessRequest):
    """Run the initial retirement assessment and stream the result via SSE.

    Creates a new agent instance and session, invokes the agent, and streams
    incremental text chunks followed by a final done event with the structured
    assessment JSON and session_id.
    """

    async def _stream():
        try:
            agent = create_agent()
            session_id = history.new_session_id()

            # Run the assessment in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            assessment = await loop.run_in_executor(
                None, run_initial_assessment, agent, request.profile, request.personal_info
            )

            # Stream the assessment summary in chunks
            summary_text = assessment.retirement_readiness_summary
            pos = 0
            while pos < len(summary_text):
                end = min(pos + 200, len(summary_text))
                if end < len(summary_text):
                    nl = summary_text.rfind('\n', pos, end + 50)
                    if nl > pos:
                        end = nl + 1
                yield _sse_event("chunk", {"text": summary_text[pos:end]})
                pos = end
                await asyncio.sleep(0.05)

            # Build session data
            session_data = {
                "created_at": session_id,
                "profile": request.profile.model_dump(),
                "personal_info": request.personal_info.model_dump(),
                "assessment": assessment.model_dump(),
                "conversation": [],
            }

            # Save session
            history.save_session(session_id, session_data)

            # Store agent for follow-up reuse
            _active_sessions[session_id] = agent

            # Final done event
            yield _sse_event("done", {
                "assessment": assessment.model_dump(),
                "session_id": session_id,
            })

        except (CredentialError, BedrockError) as exc:
            yield _sse_event("error", {"message": str(exc)})
        except Exception as exc:
            yield _sse_event("error", {"message": f"Assessment failed: {exc}"})

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/assumptions")
async def get_assumptions(request: AssumptionSummaryRequest):
    """Generate an assumption summary for review before running the assessment.

    Creates a new agent, invokes ``generate_assumption_summary``, and returns
    the structured summary JSON so the user can review and optionally correct
    assumptions before confirming.
    """
    try:
        agent = create_agent()
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            None,
            generate_assumption_summary,
            agent,
            request.profile,
            request.personal_info,
            request.file_sources,
        )
    except (CredentialError, BedrockError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate assumptions: {exc}")

    return JSONResponse(content=summary)


@app.post("/confirm-assumptions")
async def confirm_assumptions(request: ConfirmAssumptionsRequest):
    """Confirm assumptions (with optional corrections) and run the assessment.

    If corrections are provided they are applied to the summary's
    ``assumptions`` dict.  Then the agent is invoked for the initial
    assessment and the result is streamed back via SSE, following the same
    pattern as ``POST /assess``.
    """

    async def _stream():
        try:
            # Apply corrections to the summary if provided
            summary = request.summary
            if request.corrections:
                assumptions = summary.get("assumptions", {})
                assumptions.update(request.corrections)
                summary["assumptions"] = assumptions

            agent = create_agent()
            session_id = history.new_session_id()

            # Extract additional context from the summary (user's answers to missing data questions)
            additional_context = summary.get("additional_context", "")

            loop = asyncio.get_event_loop()
            assessment = await loop.run_in_executor(
                None, run_initial_assessment, agent, request.profile, request.personal_info, additional_context
            )

            # Stream the assessment summary in chunks
            summary_text = assessment.retirement_readiness_summary
            pos = 0
            while pos < len(summary_text):
                end = min(pos + 200, len(summary_text))
                if end < len(summary_text):
                    nl = summary_text.rfind('\n', pos, end + 50)
                    if nl > pos:
                        end = nl + 1
                yield _sse_event("chunk", {"text": summary_text[pos:end]})
                pos = end
                await asyncio.sleep(0.05)

            # Build session data
            session_data = {
                "created_at": session_id,
                "profile": request.profile.model_dump(),
                "personal_info": request.personal_info.model_dump(),
                "assessment": assessment.model_dump(),
                "assumptions_summary": summary,
                "conversation": [],
            }

            # Save session
            history.save_session(session_id, session_data)

            # Store agent for follow-up reuse
            _active_sessions[session_id] = agent

            # Final done event
            yield _sse_event("done", {
                "assessment": assessment.model_dump(),
                "session_id": session_id,
            })

        except (CredentialError, BedrockError) as exc:
            yield _sse_event("error", {"message": str(exc)})
        except Exception as exc:
            yield _sse_event("error", {"message": f"Assessment failed: {exc}"})

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/followup")
async def follow_up(request: FollowUpRequest):
    """Process a follow-up question and stream the response via SSE.

    Looks up the existing agent instance by session_id, invokes the agent
    with conversation context, and streams the response.
    """
    if request.session_id not in _active_sessions:
        # Session may exist on disk but agent was lost (e.g. server restart).
        # Restore the agent with full session context.
        try:
            session_data = history.load_session(request.session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found")

        loop = asyncio.get_event_loop()
        try:
            agent = await loop.run_in_executor(
                None, restore_agent_from_session, session_data
            )
            _active_sessions[request.session_id] = agent
        except (CredentialError, BedrockError) as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    agent = _active_sessions[request.session_id]

    async def _stream():
        try:
            full_text = ""
            async for event in stream_follow_up(agent, request.question):
                if "text" in event:
                    yield _sse_event("chunk", {"text": event["text"]})
                elif event.get("done"):
                    full_text = event.get("full_text", "")

            # Try to parse as assessment
            result: Any = full_text
            try:
                from retirement_planner.serialization import parse_assessment_response
                result = parse_assessment_response(full_text)
            except Exception:
                pass

            if isinstance(result, str):
                done_data: dict[str, Any] = {"response": result, "session_id": request.session_id}
            else:
                done_data = {"assessment": result.model_dump(), "session_id": request.session_id}

            # Update saved session
            try:
                session_data = history.load_session(request.session_id)
            except FileNotFoundError:
                session_data = {"conversation": []}

            session_data.setdefault("conversation", []).append({
                "question": request.question,
                "response": result.model_dump() if not isinstance(result, str) else result,
            })
            history.save_session(request.session_id, session_data)

            yield _sse_event("done", done_data)

        except (CredentialError, BedrockError) as exc:
            yield _sse_event("error", {"message": str(exc)})
        except Exception as exc:
            yield _sse_event("error", {"message": f"Follow-up failed: {exc}"})

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/sessions")
async def list_sessions():
    """Return a list of saved conversation sessions."""
    return JSONResponse(content=history.list_sessions())


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Return the full data for a specific saved conversation session."""
    try:
        data = history.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return JSONResponse(content=data)
