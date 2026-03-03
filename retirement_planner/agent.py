"""Strands Agent setup, system prompt, tool registration, and conversation management.

Configures a Strands Agent with Claude Opus 4.6 on Amazon Bedrock, registers
financial calculation tools, and provides functions for initial assessment,
follow-up questions, and raw data normalization.
"""

from __future__ import annotations

import json
import os
import re
from typing import Union

from retirement_planner.models import (
    BedrockError,
    CredentialError,
    FinancialProfile,
    PersonalInfo,
    RetirementAssessment,
)
from retirement_planner.serialization import parse_assessment_response, serialize_profile
from retirement_planner.tools import (
    calculate_cash_flow,
    calculate_net_worth,
    estimate_aca_premiums,
    get_inflation_data,
    get_treasury_yields,
    lookup_tax_data,
    read_financial_file,
)

def _build_system_prompt() -> str:
    """Build the system prompt with the current date injected."""
    from datetime import date
    today = date.today()
    date_str = today.strftime("%B %d, %Y")
    return f"""\
You are a CPA/CFA retirement planning expert. Today is {date_str}. Anchor all projections to this date.

CRITICAL OUTPUT RULES — FOLLOW EXACTLY:
- NEVER explain your reasoning or show your work. Output ONLY what is requested.
- For file data extraction: return ONLY the JSON. No text before or after.
- For assessments: write a SHORT summary (max 300 words, bullets only) then the JSON block. No tables in the summary.
- For follow-ups: max 200 words. Short bullets. One small table max. No preamble.
- End every response with SUGGESTED_FOLLOWUPS: followed by exactly 3 short questions (one line each).
- Do NOT call read_financial_file when file content is already provided in the prompt.

ANALYSIS RULES:
- Cite IRS.gov, SSA.gov, Healthcare.gov, Medicare.gov when referencing rules. Never fabricate URLs.
- Use birthdates for precise milestone calculations (SS, Medicare, Rule of 55, RMDs).
- Estimate returns from holdings (e.g. VTSAX ~7-10%, bonds ~3-5%).
- Use lookup_tax_data, estimate_aca_premiums, get_inflation_data, get_treasury_yields tools for current data.

ASSESSMENT JSON SCHEMA:
{{
  "can_retire": true/false,
  "retirement_readiness_summary": "2-3 sentence summary only",
  "recommended_monthly_budget": [{{"category": "string", "amount": number}}, ...],
  "net_worth": number,
  "monthly_cash_flow": number,
  "assumptions": {{"key": "value", ...}},
  "disclaimer": "This analysis is AI-generated and does not constitute professional financial advice."
}}

Budget categories minimum: Housing, Food, Healthcare, Transportation, Utilities, Insurance, Entertainment, Savings, Miscellaneous.
Track assumptions (inflation, returns, retirement age, SS start, life expectancy) for projection updates.
For assumption changes in follow-ups, return updated JSON. For informational questions, respond as text.
"""


DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"

MODEL_ALIASES: dict[str, str] = {
    "opus": "us.anthropic.claude-opus-4-6-v1",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "llama4-maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "llama4-scout": "us.meta.llama4-scout-17b-instruct-v1:0",
}


def _resolve_model_id(model_id: str | None = None) -> str:
    """Resolve model_id from argument, env var, or default. Supports aliases."""
    raw = model_id or os.environ.get("RETIREMENT_PLANNER_MODEL") or DEFAULT_MODEL_ID
    return MODEL_ALIASES.get(raw, raw)


def create_agent(model_id: str | None = None):
    """Create and configure the Strands Agent with tools and system prompt.

    Args:
        model_id: Bedrock model ID or alias. Falls back to RETIREMENT_PLANNER_MODEL
                  env var, then the default (Claude Opus 4.6).
                  Aliases: opus, sonnet, llama4-maverick, llama4-scout.

    Returns:
        A configured Strands Agent instance.

    Raises:
        CredentialError: If AWS credentials are not found.
        BedrockError: If the Bedrock model cannot be initialized.
    """
    resolved = _resolve_model_id(model_id)

    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise BedrockError(
            f"Strands Agents SDK is not installed: {exc}", retryable=False
        )

    try:
        model = BedrockModel(
            model_id=resolved,
            region_name="us-east-1",
        )
    except Exception as exc:
        _handle_aws_error(exc)

    try:
        agent = Agent(
            model=model,
            system_prompt=_build_system_prompt(),
            tools=[
                calculate_net_worth,
                calculate_cash_flow,
                read_financial_file,
                estimate_aca_premiums,
                lookup_tax_data,
                get_inflation_data,
                get_treasury_yields,
            ],
        )
    except Exception as exc:
        _handle_aws_error(exc)

    return agent


def run_initial_assessment(
    agent,
    profile: FinancialProfile,
    personal_info: PersonalInfo,
    additional_context: str = "",
) -> RetirementAssessment:
    """Invoke the agent to produce the initial retirement assessment.

    Args:
        agent: A configured Strands Agent instance.
        profile: The couple's financial profile.
        personal_info: Personal information (ages of spouses and children).
        additional_context: Optional user-provided context about missing data
            (e.g. property, mortgages, 529 plans, etc.).

    Returns:
        A validated RetirementAssessment.

    Raises:
        CredentialError: If AWS credentials are invalid or missing.
        BedrockError: If the Bedrock API call fails.
    """
    profile_json = serialize_profile(profile)

    personal_data = {
        "husband_age": personal_info.husband_age,
        "wife_age": personal_info.wife_age,
        "children_ages": personal_info.children_ages,
        "husband_birthdate": personal_info.husband_birthdate,
        "wife_birthdate": personal_info.wife_birthdate,
        "children_birthdates": personal_info.children_birthdates,
    }

    context_section = ""
    if additional_context:
        context_section = (
            f"\n\nAdditional context provided by the user:\n{additional_context}\n"
            "Factor this information into the assessment.\n"
        )

    prompt = (
        "Analyze the following financial and personal data to produce a "
        "retirement readiness assessment.\n\n"
        f"Financial Profile:\n{profile_json}\n\n"
        f"Personal Information:\n{json.dumps(personal_data)}\n"
        f"{context_section}\n"
        "Use birthdates for precise milestone calculations. "
        "Use the calculate_net_worth and calculate_cash_flow tools. "
        "Produce a retirement assessment as JSON matching the required schema."
    )

    try:
        result = agent(prompt)
        response_text = str(result)
    except Exception as exc:
        _handle_aws_error(exc)

    return parse_assessment_response(response_text)


def run_follow_up(
    agent,
    question: str,
) -> Union[RetirementAssessment, str]:
    """Process a follow-up question using the agent's conversation history.

    The agent maintains conversation context from prior exchanges. If the
    question implies assumption changes, the agent produces an updated
    assessment. Otherwise it returns a text response.

    Args:
        agent: A configured Strands Agent instance (with prior conversation).
        question: The user's follow-up question.

    Returns:
        A RetirementAssessment if the question triggers a projection update,
        or a plain string response otherwise.

    Raises:
        CredentialError: If AWS credentials are invalid or missing.
        BedrockError: If the Bedrock API call fails.
    """
    prompt = (
        f"{question}\n\n"
        "If this question implies changes to financial assumptions, produce an "
        "updated retirement assessment as a JSON object matching the required schema. "
        "If it does not imply assumption changes, respond with helpful information "
        "as plain text."
    )

    try:
        result = agent(prompt)
        response_text = str(result)
    except Exception as exc:
        _handle_aws_error(exc)

    # Try to parse as an assessment; if that fails, return the raw text
    try:
        return parse_assessment_response(response_text)
    except (ValueError, Exception):
        return response_text


async def stream_follow_up(agent, question: str):
    """Stream a follow-up response, yielding text chunks as they're generated.

    Uses the Strands SDK's stream_async to yield text data events in real-time.

    Args:
        agent: A configured Strands Agent instance (with prior conversation).
        question: The user's follow-up question.

    Yields:
        Dicts with either {"text": str} for text chunks or
        {"done": True, "full_text": str} when complete.
    """
    prompt = (
        f"{question}\n\n"
        "If this question implies changes to financial assumptions, produce an "
        "updated retirement assessment as a JSON object matching the required schema. "
        "If it does not imply assumption changes, respond with helpful information "
        "as plain text."
    )

    full_text = ""
    try:
        async for event in agent.stream_async(prompt):
            if "data" in event:
                chunk = event["data"]
                full_text += chunk
                yield {"text": chunk}
    except Exception as exc:
        _handle_aws_error(exc)

    yield {"done": True, "full_text": full_text}


def restore_agent_from_session(session_data: dict):
    """Create a new agent and prime it with saved session context.

    Feeds the agent a condensed summary of the original profile, assessment,
    and conversation history so that follow-up questions have proper context
    even after a server restart.

    Args:
        session_data: The session dict loaded from disk (must contain at least
            ``profile`` and ``personal_info``; ``assessment`` and
            ``conversation`` are optional).

    Returns:
        A configured Strands Agent instance with session context loaded.

    Raises:
        CredentialError: If AWS credentials are not found.
        BedrockError: If the Bedrock model cannot be initialized.
    """
    agent = create_agent()

    # Build a compact context priming message from the saved session
    parts = ["Resuming a previous session. Context:\n"]

    if session_data.get("personal_info"):
        parts.append(f"Personal: {json.dumps(session_data['personal_info'])}\n")

    if session_data.get("assessment"):
        a = session_data["assessment"]
        parts.append(f"Previous assessment: can_retire={a.get('can_retire')}, "
                     f"net_worth={a.get('net_worth')}, "
                     f"summary={a.get('retirement_readiness_summary', '')[:300]}\n")

    if session_data.get("conversation"):
        # Only include last 5 exchanges to save context
        recent = session_data["conversation"][-5:]
        parts.append(f"Recent exchanges ({len(recent)} of {len(session_data['conversation'])}):")
        for i, exchange in enumerate(recent, 1):
            q = exchange.get("question", "")[:100]
            resp = exchange.get("response", "")
            if isinstance(resp, dict):
                resp = resp.get("retirement_readiness_summary", str(resp))[:200]
            elif len(resp) > 200:
                resp = resp[:200] + "..."
            parts.append(f"\n  Q{i}: {q}\n  A{i}: {resp}")

    context_prompt = "\n".join(parts) + (
        "\n\nAcknowledge that you have this context and are ready for follow-up questions. "
        "Respond with a brief confirmation only."
    )

    try:
        agent(context_prompt)
    except Exception:
        # If priming fails, the agent still works — just without history context
        pass

    return agent


def normalize_raw_data(
    agent,
    raw_content: str,
    category: str,
) -> dict:
    """Invoke the agent to interpret raw file content and extract structured financial data.

    The agent uses its language understanding to identify relevant fields
    regardless of the file's format, column names, or structure.

    Args:
        agent: A configured Strands Agent instance.
        raw_content: The raw text content read from a file.
        category: One of 'investments', 'banking', or 'credit_cards'.

    Returns:
        A dict (or list of dicts) conforming to the expected Pydantic model
        schema for the category.

    Raises:
        CredentialError: If AWS credentials are invalid or missing.
        BedrockError: If the Bedrock API call fails.
    """
    field_map = {
        "investments": ["account_type", "balance", "expected_annual_return"],
        "banking": ["account_type", "balance", "monthly_income_deposits"],
        "credit_cards": ["outstanding_balance", "credit_limit", "monthly_payment"],
        "spending": ["category", "monthly_amount"],
    }

    fields = field_map.get(category, [])

    prompt = (
        f"Extract {category} financial data from the following raw file content.\n"
        f"Return ONLY a JSON array of objects. Each object must have these fields: "
        f"{', '.join(fields)}.\n"
        f"Do not include any explanation, just the JSON array.\n\n"
        f"Raw content:\n{raw_content}"
    )

    try:
        result = agent(prompt)
        response_text = str(result)
    except Exception as exc:
        _handle_aws_error(exc)

    return _extract_json_from_agent_response(response_text)

def generate_assumption_summary(
    agent,
    profile: FinancialProfile,
    personal_info: PersonalInfo,
    file_sources: dict[str, str] | None = None,
) -> dict:
    """Invoke the agent to produce a structured summary of extracted data and assumptions.

    Before running the full retirement assessment, this function asks the agent
    to summarize what it sees in the financial data and what assumptions it will
    use, giving the user a chance to review and correct before proceeding.

    Args:
        agent: A configured Strands Agent instance.
        profile: The couple's financial profile.
        personal_info: Personal information (ages of spouses and children).
        file_sources: Optional mapping of category to file path for data that
            was loaded from files (e.g. ``{"investments": "portfolio.csv"}``).

    Returns:
        A dict with keys ``extracted_data``, ``assumptions``, and optionally
        ``file_interpretations``.

    Raises:
        BedrockError: If the agent invocation fails.
    """
    profile_json = serialize_profile(profile)

    personal_data = {
        "husband_age": personal_info.husband_age,
        "wife_age": personal_info.wife_age,
        "children_ages": personal_info.children_ages,
        "husband_birthdate": personal_info.husband_birthdate,
        "wife_birthdate": personal_info.wife_birthdate,
        "children_birthdates": personal_info.children_birthdates,
    }

    file_section = ""
    if file_sources:
        file_section = (
            "\n\nThe following data categories were loaded from files:\n"
            + "\n".join(f"  - {cat}: {path}" for cat, path in file_sources.items())
            + "\n\nInclude a 'file_interpretations' key in your response showing "
            "how each file's data was interpreted (account types/counts, estimated figures)."
        )

    prompt = (
        "Before running the full retirement assessment, summarize the extracted "
        "financial data, key assumptions, and identify any MISSING information.\n\n"
        f"Financial Profile:\n{profile_json}\n\n"
        f"Personal Information:\n{json.dumps(personal_data)}\n"
        f"{file_section}\n\n"
        "Analyze what data is present and what common retirement-relevant data is MISSING. "
        "Check for these common items and ask about any that are NOT in the uploaded data:\n"
        "- Real estate / property owned (home value, mortgage balance, monthly payment, years remaining)\n"
        "- Car loans or leases (monthly payment, balance, years remaining)\n"
        "- 529 college savings plans (if children are pre-college age)\n"
        "- HSA (Health Savings Account) balance\n"
        "- Pension or defined benefit plan\n"
        "- Social Security earnings history or estimated benefits\n"
        "- Life insurance policies\n"
        "- Student loans or other debt\n"
        "- Expected inheritance\n"
        "- Employer match details for 401(k)\n"
        "- Current annual salary/income (if not clear from deposits)\n"
        "- Target retirement age\n"
        "- Desired retirement lifestyle / annual spending target\n\n"
        "Return ONLY a JSON object with this structure:\n"
        "{\n"
        '  "extracted_data": {\n'
        '    "accounts_found": <number>,\n'
        '    "total_investment_balance": <sum>,\n'
        '    "total_bank_balance": <sum>,\n'
        '    "total_credit_card_balance": <sum>,\n'
        '    "monthly_income": <sum of deposits>,\n'
        '    "monthly_expenses": <sum of spending categories if available, otherwise sum of credit card payments>,\n'
        '    "spending_breakdown": [{"category": "<name>", "monthly_amount": <amount>}, ...] or [] if no spending data\n'
        "  },\n"
        '  "assumptions": {\n'
        '    "retirement_age": <age>,\n'
        '    "inflation_rate": <pct>,\n'
        '    "expected_investment_return": <pct>,\n'
        '    "social_security_start_age": <age>,\n'
        '    "life_expectancy": <age>\n'
        "  },\n"
        '  "missing_data_questions": [\n'
        '    "Do you own a home? If so, what is the estimated value, mortgage balance, and monthly payment?",\n'
        '    "<other relevant questions based on what is missing from the data>"\n'
        "  ]"
        + (
            ',\n  "file_interpretations": { <category>: { "account_types": [...], '
            '"count": <n>, "estimated_total": <amount> } }'
            if file_sources
            else ""
        )
        + "\n}\n\n"
        "Include 3-6 questions in missing_data_questions, prioritized by impact on the retirement plan. "
        "Only ask about items NOT already present in the data. RESPOND WITH ONLY THE JSON. No explanation."
    )

    try:
        result = agent(prompt)
        response_text = str(result)
    except Exception as exc:
        _handle_aws_error(exc)

    try:
        summary = _extract_json_from_agent_response(response_text)
    except ValueError as exc:
        raise BedrockError(
            f"Failed to parse assumption summary from agent response: {exc}",
            retryable=True,
        ) from exc

    return summary




def _extract_json_from_agent_response(response_text: str):
    """Extract JSON data from an agent response that may contain markdown or prose."""
    stripped = response_text.strip()

    # Try direct parse
    try:
        parsed = json.loads(stripped)
        return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Try markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    for match in fence_pattern.finditer(response_text):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    # Try to find a JSON array in the raw text
    bracket_pattern = re.compile(r"\[.*\]", re.DOTALL)
    for match in bracket_pattern.finditer(response_text):
        candidate = match.group(0).strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    # Try to find a JSON object
    brace_pattern = re.compile(r"\{.*\}", re.DOTALL)
    for match in brace_pattern.finditer(response_text):
        candidate = match.group(0).strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    raise ValueError("Could not extract JSON from agent response")


def _handle_aws_error(exc: Exception) -> None:
    """Inspect an exception and raise the appropriate custom error.

    Raises:
        CredentialError: For missing AWS credentials.
        BedrockError: For Bedrock API / permission / connection errors.
    """
    # Import botocore lazily so the module loads even without boto3 installed
    try:
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        # If botocore isn't available, wrap as a generic BedrockError
        raise BedrockError(str(exc), retryable=True) from exc

    if isinstance(exc, NoCredentialsError):
        raise CredentialError(
            "AWS credentials not found. Please configure credentials using one of:\n"
            "  - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables\n"
            "  - AWS_SESSION_TOKEN (for temporary credentials)\n"
            "  - AWS_PROFILE environment variable\n"
            "  - ~/.aws/credentials file"
        ) from exc

    if isinstance(exc, ClientError):
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code == "AccessDeniedException":
            raise BedrockError(
                "Access denied to Amazon Bedrock. Ensure your AWS credentials have "
                "permission to invoke the Bedrock model (bedrock:InvokeModel).",
                retryable=False,
            ) from exc
        raise BedrockError(
            f"Amazon Bedrock API error ({error_code}): {exc}",
            retryable=True,
        ) from exc

    # For any other exception, wrap as retryable BedrockError
    raise BedrockError(
        f"Amazon Bedrock request failed: {exc}",
        retryable=True,
    ) from exc
