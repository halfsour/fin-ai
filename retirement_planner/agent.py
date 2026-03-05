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
    search_web,
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
- For file data extraction: return ONLY the JSON. No text before or after. NO SUGGESTED_FOLLOWUPS.
- For assumption summaries: return ONLY the JSON. NO SUGGESTED_FOLLOWUPS.
- For assessments: Call calculate_net_worth and calculate_cash_flow tools ONCE, then IMMEDIATELY write a DETAILED summary (max 500 words, bullets organized by topic) followed by the JSON block. Do NOT call lookup_tax_data or other tools repeatedly. End with SUGGESTED_FOLLOWUPS: followed by exactly 3 short requests phrased as the USER speaking. Example: "Show me a budget that gets spending under $15K/month" NOT "Would you like me to create a reduced budget?"
- For follow-ups: max 200 words. Short bullets. One small table max. No preamble. End with SUGGESTED_FOLLOWUPS: followed by exactly 3 short questions (one line each). Phrase them as USER requests, not agent questions. Example: "Show me what happens if I retire at 62" NOT "Would you like me to model retiring at 62?"
- Do NOT call read_financial_file when file content is already provided in the prompt.
- Do NOT call read_financial_file during follow-up questions. All financial data including holdings/tickers is already in the conversation history from the initial assessment. Use that data directly.
- STOP after producing the assessment JSON. Do NOT continue calling tools or explaining.

ANALYSIS RULES:
- Cite IRS.gov, SSA.gov, Healthcare.gov, Medicare.gov when referencing rules. Never fabricate URLs.
- Use birthdates for precise milestone calculations (SS, Medicare, Rule of 55, RMDs).
- Estimate returns from holdings (e.g. VTSAX ~7-10%, bonds ~3-5%).
- Use lookup_tax_data, estimate_aca_premiums, get_inflation_data, get_treasury_yields tools for reference data.
- Use search_web for current data not covered by built-in tools (e.g., current ACA premiums by state, fed funds rate, specific fund performance). Prefer built-in tools when they have the data.

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


DEFAULT_MODEL_ID = "moonshotai.kimi-k2.5"

MODEL_ALIASES: dict[str, str] = {
    "opus": "us.anthropic.claude-opus-4-6-v1",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "llama4-maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "llama4-scout": "us.meta.llama4-scout-17b-instruct-v1:0",
    "kimi": "moonshotai.kimi-k2.5",
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "nova-lite": "us.amazon.nova-lite-v1:0",
    "nova-pro": "us.amazon.nova-pro-v1:0",
    "nova-micro": "us.amazon.nova-micro-v1:0",
    "deepseek-v3": "deepseek.v3.2",
}

# Task-based model routing defaults
MODEL_ROUTING: dict[str, str] = {
    "extraction": "nova-lite",
    "assumptions": "llama4-scout",
    "assessment": "haiku",
    "followup": "haiku",
}


def _resolve_model_id(model_id: str | None = None, task: str | None = None) -> str:
    """Resolve model_id from argument, env var, routing, or default. Supports aliases."""
    # Explicit model_id overrides everything
    if model_id:
        return MODEL_ALIASES.get(model_id, model_id)
    # Env var overrides routing
    env = os.environ.get("RETIREMENT_PLANNER_MODEL")
    if env:
        return MODEL_ALIASES.get(env, env)
    # Task-based routing
    if task and task in MODEL_ROUTING:
        alias = MODEL_ROUTING[task]
        return MODEL_ALIASES.get(alias, alias)
    # Default
    return MODEL_ALIASES.get(DEFAULT_MODEL_ID, DEFAULT_MODEL_ID)


def create_agent(model_id: str | None = None, task: str | None = None):
    """Create and configure the Strands Agent with tools and system prompt.

    Args:
        model_id: Bedrock model ID or alias. Overrides all routing.
        task: Task type for model routing (extraction, assumptions, assessment, followup).
              Ignored if model_id is provided.

    Returns:
        A configured Strands Agent instance.

    Raises:
        CredentialError: If AWS credentials are not found.
        BedrockError: If the Bedrock model cannot be initialized.
    """
    resolved = _resolve_model_id(model_id, task=task)
    print(f"[agent] Using model: {resolved} (task={task or 'default'})")

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

    # Extraction and assumptions agents don't need tools
    if task in ("extraction", "assumptions"):
        tools = []
    else:
        tools = [
            calculate_net_worth,
            calculate_cash_flow,
            read_financial_file,
            estimate_aca_premiums,
            lookup_tax_data,
            get_inflation_data,
            get_treasury_yields,
            search_web,
        ]

    try:
        agent = Agent(
            model=model,
            system_prompt=_build_system_prompt(),
            tools=tools,
        )
    except Exception as exc:
        _handle_aws_error(exc)

    agent._model_id = resolved
    return agent


def run_initial_assessment(
    agent,
    profile: FinancialProfile,
    personal_info: PersonalInfo,
    additional_context: str = "",
    retire_ages: dict | None = None,
) -> RetirementAssessment:
    """Invoke the agent to produce the initial retirement assessment.

    Uses a pre-computed analysis brief with all financial calculations done
    in Python. The LLM only writes the narrative and recommended budget.

    Args:
        agent: A configured Strands Agent instance.
        profile: The couple's financial profile.
        personal_info: Personal information (ages of spouses and children).
        additional_context: Optional user-provided context.
        retire_ages: Optional dict with target retirement ages per spouse.

    Returns:
        A validated RetirementAssessment.
    """
    from retirement_planner.analysis import build_analysis_brief

    brief = build_analysis_brief(profile, personal_info, retire_ages, additional_context)
    print(f"[agent] Analysis brief: {len(brief)} chars")

    prompt = (
        "You are given a pre-computed financial analysis. All numbers are accurate — do NOT recalculate them.\n"
        "Write a retirement readiness assessment based on this data.\n\n"
        f"{brief}\n\n"
        "YOUR ANALYSIS MUST COVER:\n"
        "1. Account-by-account breakdown with tax treatment and withdrawal rules\n"
        "2. Withdrawal sequencing strategy (which accounts to draw from first for tax optimization)\n"
        "3. Healthcare bridge: ACA costs from retirement to Medicare, HSA strategy\n"
        "4. Social Security optimization: optimal claiming ages for each spouse\n"
        "5. Tax bracket management: Roth conversions, capital gains harvesting\n"
        "6. Risk factors: concentration risk, sequence-of-returns risk, longevity risk\n\n"
        "\nEXAMPLE OUTPUT FORMAT (do not copy these numbers, use the pre-computed data above):\n"
        "**Account Breakdown:**\n"
        "- 401(k) ($500K): Tax-deferred. Penalty-free at 59.5 or Rule of 55 if separated from employer. RMDs at 73.\n"
        "- Roth IRA ($200K): Tax-free withdrawals after 59.5 + 5yr rule. No RMDs.\n"
        "- Taxable Brokerage ($300K): No restrictions. LTCG at 0/15/20%.\n\n"
        "**Withdrawal Sequence:**\n"
        "1. Ages 55-59: Taxable brokerage (harvest gains in 0% bracket)\n"
        "2. Ages 59-72: Mix of 401(k) + Roth conversions to fill low brackets\n"
        "3. Ages 73+: RMDs from 401(k), supplement with Roth\n\n"
        "**Healthcare Bridge:**\n"
        "- 12 years to Medicare. ACA marketplace ~$1,500/mo for family. HSA covers out-of-pocket.\n\n"
        "**Social Security:**\n"
        "- Spouse 1: Delay to 70 for max benefit ($3,800/mo). Spouse 2: Claim at FRA 67 ($2,400/mo).\n\n"
        "Produce a DETAILED summary (max 500 words, bullets organized by topic) followed by the JSON assessment.\n"
        "Use the pre-computed net_worth and monthly_cash_flow values in the JSON.\n"
        "STOP after producing the JSON."
    )

    try:
        result = agent(prompt)
        # Log token usage if available
        try:
            metrics = getattr(result, 'metrics', None) or {}
            usage = metrics.get('usage', {}) if isinstance(metrics, dict) else {}
            if usage:
                print(f"[agent] Tokens: input={usage.get('inputTokens', '?')}, output={usage.get('outputTokens', '?')}")
        except Exception:
            pass
        response_text = str(result)
        print(f"[agent] Assessment response length: {len(response_text)} chars")
        print(f"[agent] Response preview: {response_text[:300]}...")
    except Exception as exc:
        _handle_aws_error(exc)

    try:
        assessment = parse_assessment_response(response_text)
    except ValueError as exc:
        # Retry once with error feedback
        print(f"[agent] JSON parse failed, retrying: {exc}")
        retry_prompt = (
            f"Your previous response could not be parsed as valid JSON. Error: {exc}\n"
            "Please produce ONLY the JSON assessment object matching the required schema. "
            "No text before or after the JSON."
        )
        try:
            retry_result = agent(retry_prompt)
            response_text = str(retry_result)
        except Exception as retry_exc:
            _handle_aws_error(retry_exc)
        assessment = parse_assessment_response(response_text)
    assessment._raw_response = response_text
    return assessment


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
            elif hasattr(event, 'data'):
                chunk = str(event.data)
                full_text += chunk
                yield {"text": chunk}
    except Exception as exc:
        _handle_aws_error(exc)

    if not full_text:
        # stream_async didn't yield data events — fall back to sync
        try:
            result = agent(prompt)
            full_text = str(result)
            yield {"text": full_text}
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

    if session_data.get("profile"):
        # Include only investments (with holdings/tickers) — skip raw bank/cc/spending
        profile = session_data["profile"]
        inv = profile.get("investments", [])
        if inv:
            inv_summary = [f"{a['account_type']}: ${a['balance']:,.0f} [{a.get('holdings','')}]" for a in inv]
            parts.append("Investments:\n" + "\n".join(inv_summary) + "\n")

    if session_data.get("assessment"):
        a = session_data["assessment"]
        parts.append(f"Previous assessment: can_retire={a.get('can_retire')}, "
                     f"net_worth={a.get('net_worth')}, "
                     f"monthly_cash_flow={a.get('monthly_cash_flow')}, "
                     f"summary={a.get('retirement_readiness_summary', '')[:500]}\n"
                     f"assumptions={json.dumps(a.get('assumptions', {}))}\n")

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
        '    "monthly_income": <sum of monthly_income_deposits from bank accounts>,\n'
        '    "spending_breakdown": <COPY the spending array from the profile as-is. It already contains averaged monthly amounts per category. If the spending array is empty, return []>\n'
        "  },\n"
        '  "assumptions": {\n'
        '    "retirement_age": <age>,\n'
        '    "inflation_rate": <pct>,\n'
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
        # Log token usage if available
        try:
            metrics = getattr(result, 'metrics', None) or {}
            usage = metrics.get('usage', {}) if isinstance(metrics, dict) else {}
            if usage:
                print(f"[agent] Tokens: input={usage.get('inputTokens', '?')}, output={usage.get('outputTokens', '?')}")
        except Exception:
            pass
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

    # Override extracted_data with actual computed values from the profile
    # so we don't rely on the agent to copy numbers correctly
    ed = summary.get("extracted_data", {})
    ed["accounts_found"] = len(profile.investments) + len(profile.bank_accounts) + len(profile.credit_cards)
    ed["total_investment_balance"] = sum(a.balance for a in profile.investments)
    ed["total_bank_balance"] = sum(a.balance for a in profile.bank_accounts)
    ed["total_credit_card_balance"] = sum(c.outstanding_balance for c in profile.credit_cards)
    ed["monthly_income"] = sum(a.monthly_income_deposits for a in profile.bank_accounts)
    if profile.spending:
        # serialize_profile already aggregates, but profile.spending may still be raw
        from collections import defaultdict
        totals = defaultdict(lambda: [0.0, 0])
        for item in profile.spending:
            totals[item.category][0] += item.monthly_amount
            totals[item.category][1] += 1
        ed["spending_breakdown"] = [
            {"category": cat, "monthly_amount": round(total / count, 2)}
            for cat, (total, count) in sorted(totals.items())
        ]
    else:
        ed["spending_breakdown"] = []
    summary["extracted_data"] = ed

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
