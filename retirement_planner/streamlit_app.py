"""Streamlit web interface for the Retirement Planner.

Calls backend modules directly — no HTTP layer needed.
Run with: streamlit run retirement_planner/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
from retirement_planner.agent import (
    create_agent,
    generate_assumption_summary,
    run_initial_assessment,
    stream_follow_up,
)
from retirement_planner.file_parser import parse_all_from_file
from retirement_planner.models import (
    BedrockError,
    CredentialError,
    FinancialProfile,
    MonthlySpending,
    PersonalInfo,
)
from retirement_planner import history
import asyncio
import json
from collections import defaultdict

def _clean_response(text: str) -> str:
    """Clean agent response for Streamlit display: strip JSON blocks."""
    import re
    # Strip JSON code blocks
    text = re.sub(r'```json[\s\S]*?```', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Strip bare JSON objects with can_retire key
    text = re.sub(r'\{[^{}]*"can_retire"[\s\S]*?\n\}', '', text)
    # Strip SUGGESTED_FOLLOWUPS section (we render these separately)
    text = re.sub(r'SUGGESTED_FOLLOWUPS:.*', '', text, flags=re.DOTALL)
    # Escape $ for Streamlit (use HTML entity)
    text = text.replace('$', '&#36;')
    return text.strip()


def _extract_followups(text: str) -> list[str]:
    """Extract suggested follow-up questions from agent response."""
    import re
    match = re.search(r'SUGGESTED_FOLLOWUPS:(.*)', text, re.DOTALL)
    if not match:
        return []
    lines = match.group(1).strip().split('\n')
    followups = []
    for line in lines:
        line = line.strip().lstrip('-•* ')
        if line and len(line) > 10:
            followups.append(line)
    return followups[:3]


def _render_message(content: str):
    """Render a message using HTML to avoid LaTeX interpretation."""
    st.markdown(content, unsafe_allow_html=True)


def _render_assessment(assessment: dict):
    """Render a structured assessment card with metrics and budget."""
    cols = st.columns(3)
    cols[0].metric("Can Retire?", "✅ Yes" if assessment.get("can_retire") else "❌ Not Yet")
    nw = assessment.get("net_worth", 0)
    cols[1].metric("Net Worth", f"${nw:,.0f}")
    cf = assessment.get("monthly_cash_flow", 0)
    cols[2].metric("Monthly Cash Flow", f"${cf:,.0f}")

    # Budget pie chart
    budget = assessment.get("recommended_monthly_budget", [])
    if budget:
        st.subheader("Recommended Monthly Budget")
        import pandas as pd
        df = pd.DataFrame(budget)
        df.columns = ["Category", "Amount"]
        total = df["Amount"].sum()

        import altair as alt
        chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("Amount:Q"),
            color=alt.Color("Category:N", legend=alt.Legend(orient="bottom", columns=3)),
            tooltip=["Category", alt.Tooltip("Amount:Q", format="$,.0f")],
        ).properties(height=350)
        st.altair_chart(chart, width="stretch")
        st.caption(f"Total: ${total:,.0f}/month")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Retirement Planner", page_icon="🏠", layout="wide")

MODEL_OPTIONS = {
    "Auto (cost-optimized)": None,
    "Claude Haiku 4.5": "haiku",
    "Kimi K2.5": "kimi",
    "Claude Sonnet 4.6": "sonnet",
    "Claude Opus 4.6": "opus",
    "Llama 4 Maverick": "llama4-maverick",
}

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = None
if "personal_info" not in st.session_state:
    st.session_state.personal_info = None
if "assessment" not in st.session_state:
    st.session_state.assessment = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "assumptions" not in st.session_state:
    st.session_state.assumptions = None
if "step" not in st.session_state:
    st.session_state.step = "upload"  # upload → review → chat


def _aggregate_spending(spending_list):
    """Aggregate raw spending into average monthly per category."""
    if not spending_list:
        return []
    totals = defaultdict(lambda: [0.0, 0])
    for item in spending_list:
        cat = item.category if hasattr(item, "category") else item["category"]
        amt = item.monthly_amount if hasattr(item, "monthly_amount") else item["monthly_amount"]
        totals[cat][0] += amt
        totals[cat][1] += 1
    return [
        {"category": cat, "monthly_amount": round(total / count, 2)}
        for cat, (total, count) in sorted(totals.items())
    ]


# ---------------------------------------------------------------------------
# Sidebar: model selection + session history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏠 Retirement Planner")
    model_name = st.selectbox("AI Model", list(MODEL_OPTIONS.keys()), index=0)
    model_id = MODEL_OPTIONS[model_name]

    st.divider()
    if st.button("➕ New Session", use_container_width=True):
        for key in ["messages", "profile", "personal_info", "assessment", "agent", "session_id", "assumptions"]:
            st.session_state[key] = None if key != "messages" else []
        st.session_state.step = "upload"
        st.rerun()

    st.subheader("Sessions")
    try:
        sessions = history.list_sessions()
        for s in sessions[:10]:
            sid = s.get("session_id", "")
            label = sid[:13].replace("_", " ") if sid else "Session"
            if st.button(label, key=f"sess_{sid}", use_container_width=True):
                data = history.load_session(sid)
                st.session_state.session_id = sid
                st.session_state.assessment = data.get("assessment")
                # Restore profile from session
                if data.get("profile"):
                    try:
                        st.session_state.profile = FinancialProfile.model_validate(data["profile"])
                    except Exception:
                        st.session_state.profile = None
                if data.get("personal_info"):
                    try:
                        st.session_state.personal_info = PersonalInfo.model_validate(data["personal_info"])
                    except Exception:
                        pass
                st.session_state.messages = []
                if data.get("assessment"):
                    st.session_state.messages.append({"role": "assistant", "content": _clean_response(data["assessment"].get("retirement_readiness_summary", ""))})
                for ex in data.get("conversation", []):
                    st.session_state.messages.append({"role": "user", "content": ex.get("question", "")})
                    resp = ex.get("response", "")
                    if isinstance(resp, dict):
                        resp = resp.get("retirement_readiness_summary", str(resp))
                    st.session_state.messages.append({"role": "assistant", "content": _clean_response(resp)})
                st.session_state.step = "chat"
                st.rerun()
    except Exception:
        st.caption("No sessions yet")


# ---------------------------------------------------------------------------
# Step 1: Upload & Personal Info
# ---------------------------------------------------------------------------
if st.session_state.step == "upload":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload Financial Files")
        uploaded_files = st.file_uploader(
            "Upload CSV, JSON, text, or PDF files",
            type=["csv", "json", "txt", "pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("🔍 Process Files", type="primary"):
            all_investments = []
            all_bank_accounts = []
            all_credit_cards = []
            all_spending = []

            for uf in uploaded_files:
                with st.spinner(f"Processing {uf.name}..."):
                    # Save to temp file
                    import tempfile, os
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    try:
                        agent = create_agent(task="extraction")
                        profile = parse_all_from_file(agent, tmp_path)
                        all_investments.extend(profile.investments)
                        all_bank_accounts.extend(profile.bank_accounts)
                        all_credit_cards.extend(profile.credit_cards)
                        all_spending.extend(profile.spending)
                        st.success(f"✓ {uf.name}: {len(profile.investments)} investments, {len(profile.bank_accounts)} bank accounts, {len(profile.credit_cards)} credit cards, {len(profile.spending)} spending categories")
                    except Exception as e:
                        st.error(f"✗ {uf.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

            agg_spending = _aggregate_spending(all_spending)
            st.session_state.profile = FinancialProfile(
                investments=all_investments,
                bank_accounts=all_bank_accounts,
                credit_cards=all_credit_cards,
                spending=[MonthlySpending(**s) for s in agg_spending],
            )
            st.success(f"Total: {len(all_investments)} investments, {len(all_bank_accounts)} bank accounts, {len(all_credit_cards)} credit cards, {len(agg_spending)} spending categories")

    with col2:
        st.subheader("👤 Personal Information")
        from datetime import date
        
        st.markdown("**Spouse 1**")
        s1_dob = st.date_input("Date of Birth", value=None, min_value=date(1920, 1, 1), max_value=date.today(), key="s1_dob")
        s1_retire_age = st.number_input("Target Retirement Age", min_value=30, max_value=100, value=65, key="s1_retire")

        add_spouse2 = st.checkbox("Add Spouse 2")
        s2_dob = None
        s2_retire_age = 65
        if add_spouse2:
            st.markdown("**Spouse 2**")
            s2_dob = st.date_input("Date of Birth", value=None, min_value=date(1920, 1, 1), max_value=date.today(), key="s2_dob")
            s2_retire_age = st.number_input("Target Retirement Age", min_value=30, max_value=100, value=65, key="s2_retire")

        num_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        child_dobs = []
        for i in range(num_children):
            cd = st.date_input(f"Child {i+1} Date of Birth", value=None, key=f"child_{i}", min_value=date(1920, 1, 1), max_value=date.today())
            if cd:
                child_dobs.append(str(cd))

    # Run Assessment button
    if st.session_state.profile and s1_dob:
        st.session_state.personal_info = PersonalInfo(
            husband_birthdate=str(s1_dob),
            wife_birthdate=str(s2_dob) if s2_dob else None,
            children_birthdates=child_dobs,
        )
        # Store retirement ages for the assessment prompt
        st.session_state.retire_ages = {"spouse_1": s1_retire_age}
        if add_spouse2 and s2_dob:
            st.session_state.retire_ages["spouse_2"] = s2_retire_age

        if st.button("🚀 Run Assessment", type="primary", use_container_width=True):
            with st.spinner("Generating assumption summary..."):
                try:
                    agent = create_agent(task="assumptions")
                    summary = generate_assumption_summary(
                        agent, st.session_state.profile, st.session_state.personal_info
                    )
                    # Override retirement age with user-specified values
                    retire_ages = st.session_state.get("retire_ages", {})
                    if retire_ages:
                        assumptions = summary.get("assumptions", {})
                        assumptions["retirement_age"] = retire_ages.get("spouse_1", 65)
                        if "spouse_2" in retire_ages:
                            assumptions["retirement_age_spouse_2"] = retire_ages["spouse_2"]
                        summary["assumptions"] = assumptions
                    st.session_state.assumptions = summary
                    st.session_state.step = "review"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Step 2: Review Assumptions
# ---------------------------------------------------------------------------
elif st.session_state.step == "review":
    st.subheader("📋 Assumption Summary — Please Review")
    summary = st.session_state.assumptions or {}
    ed = summary.get("extracted_data", {})

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accounts Found", ed.get("accounts_found", 0))
        st.metric("Total Investment Balance", f"${ed.get('total_investment_balance', 0):,.0f}")
        st.metric("Total Bank Balance", f"${ed.get('total_bank_balance', 0):,.0f}")
    with col2:
        st.metric("Total Credit Card Balance", f"${ed.get('total_credit_card_balance', 0):,.0f}")
        st.metric("Monthly Income", f"${ed.get('monthly_income', 0):,.0f}")

    # Investment account breakdown
    if st.session_state.profile and st.session_state.profile.investments:
        st.subheader("Investment Accounts")
        import pandas as pd
        from retirement_planner.analysis import classify_accounts
        classified = classify_accounts(st.session_state.profile)
        
        # Summary by account type
        from collections import defaultdict
        type_totals = defaultdict(float)
        for a in classified:
            type_totals[a["tax_treatment"].replace("-", " ").title()] += a["balance"]
        summary_cols = st.columns(len(type_totals))
        for i, (t, total) in enumerate(sorted(type_totals.items(), key=lambda x: -x[1])):
            summary_cols[i].metric(t, f"${total:,.0f}")
        
        # Detail table
        inv_data = []
        for a in classified:
            inv_data.append({
                "Account": a["account_type"],
                "Type": a["tax_treatment"].replace("-", " ").title(),
                "Balance": f"${a['balance']:,.0f}",
                "Holdings": (a["holdings"] or "")[:60],
            })
        st.dataframe(pd.DataFrame(inv_data), hide_index=True, width=800)

    # Spending breakdown
    breakdown = ed.get("spending_breakdown", [])
    if breakdown:
        st.subheader("Monthly Spending by Category")
        import pandas as pd
        import altair as alt
        df = pd.DataFrame(breakdown).rename(columns={"category": "Category", "monthly_amount": "Amount"})
        chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("Amount:Q"),
            color=alt.Color("Category:N", legend=alt.Legend(orient="bottom", columns=3)),
            tooltip=["Category", alt.Tooltip("Amount:Q", format="$,.0f")],
        ).properties(height=350)
        st.altair_chart(chart, width="stretch")
        total = sum(item["monthly_amount"] for item in breakdown)
        st.markdown(f"**Total Monthly Expenses: ${total:,.2f}**")

    # Assumptions
    assumptions = summary.get("assumptions", {})
    if assumptions:
        st.subheader("Key Assumptions")
        for k, v in assumptions.items():
            label = k.replace("_", " ").title()
            if "rate" in k or "inflation" in k or "return" in k:
                pct = v * 100 if isinstance(v, (int, float)) and v < 1 else v
                st.text(f"{label}: {pct}%")
            else:
                st.text(f"{label}: {v}")

    # Missing data questions
    questions = summary.get("missing_data_questions", [])
    if questions:
        st.subheader("❓ Additional Information Needed")
        for i, q in enumerate(questions, 1):
            st.markdown(f"{i}. {q}")

    # Feedback + confirm
    feedback = st.text_area(
        "Corrections or additional information (optional)",
        placeholder="e.g. We own a home worth $850K with $320K mortgage. Retirement age should be 62.",
    )

    if st.button("✅ Confirm & Run Assessment", type="primary", use_container_width=True):
        # Inject retirement ages into context
        retire_ages = st.session_state.get("retire_ages", {})
        age_context = ""
        if retire_ages:
            parts = [f"Spouse 1 target retirement age: {retire_ages.get('spouse_1', 65)}"]
            if "spouse_2" in retire_ages:
                parts.append(f"Spouse 2 target retirement age: {retire_ages['spouse_2']}")
            age_context = ". ".join(parts) + ". "
        
        combined = age_context + (feedback or "")
        if combined.strip():
            summary["additional_context"] = combined.strip()
        with st.spinner("Running retirement assessment..."):
            try:
                agent = create_agent(model_id=model_id, task="assessment")
                additional_context = summary.get("additional_context", "")
                assessment = run_initial_assessment(
                    agent, st.session_state.profile, st.session_state.personal_info,
                    additional_context, retire_ages=st.session_state.get("retire_ages"),
                )
                st.session_state.assessment = assessment.model_dump()
                st.session_state.agent = agent
                st.session_state.session_id = history.new_session_id()

                # Save session
                session_data = {
                    "created_at": st.session_state.session_id,
                    "profile": st.session_state.profile.model_dump(),
                    "personal_info": st.session_state.personal_info.model_dump(),
                    "assessment": assessment.model_dump(),
                    "assumptions_summary": summary,
                    "conversation": [],
                }
                history.save_session(st.session_state.session_id, session_data)

                # Show assessment
                raw = getattr(assessment, "_raw_response", "") or assessment.retirement_readiness_summary
                followups = _extract_followups(raw)
                st.session_state.messages = [{"role": "assistant", "content": _clean_response(raw)}]
                st.session_state.followups = followups
                st.session_state.step = "chat"
                st.rerun()
            except Exception as e:
                st.error(f"Assessment failed: {e}")


# ---------------------------------------------------------------------------
# Step 3: Chat
# ---------------------------------------------------------------------------
elif st.session_state.step == "chat":
    # Display structured assessment card
    a = st.session_state.assessment
    if a:
        _render_assessment(a)
        st.divider()

    # Chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            _render_message(msg["content"])

    # Show any pending suggested followups
    pending_fu = st.session_state.get("followups", [])
    if pending_fu:
        st.caption("Suggested follow-ups:")
        for fu in pending_fu:
            if st.button(fu, key=f"fu_{hash(fu)}"):
                st.session_state.followups = []
                st.session_state.messages.append({"role": "user", "content": fu})
                st.rerun()

    # Mid-conversation file upload
    with st.expander("📎 Upload additional files"):
        chat_files = st.file_uploader("Add more financial data", type=["csv", "json", "txt", "pdf"], accept_multiple_files=True, key="chat_upload")
        if chat_files and st.button("Process & Update", key="chat_process"):
            import tempfile, os
            new_inv, new_bank, new_cc, new_spend = [], [], [], []
            for uf in chat_files:
                with st.spinner(f"Processing {uf.name}..."):
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    try:
                        agent = create_agent(task="extraction")
                        parsed = parse_all_from_file(agent, tmp_path)
                        new_inv.extend(parsed.investments)
                        new_bank.extend(parsed.bank_accounts)
                        new_cc.extend(parsed.credit_cards)
                        new_spend.extend(parsed.spending)
                        st.success(f"✓ {uf.name}")
                    except Exception as e:
                        st.error(f"✗ {uf.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

            if new_inv or new_bank or new_cc or new_spend:
                p = st.session_state.profile
                if p:
                    merged_inv = list(p.investments) + new_inv
                    merged_bank = list(p.bank_accounts) + new_bank
                    merged_cc = list(p.credit_cards) + new_cc
                    merged_spend = list(new_spend) if new_spend else list(p.spending)
                else:
                    merged_inv, merged_bank, merged_cc = new_inv, new_bank, new_cc
                    merged_spend = new_spend
                st.session_state.profile = FinancialProfile(
                    investments=merged_inv,
                    bank_accounts=merged_bank,
                    credit_cards=merged_cc,
                    spending=merged_spend,
                )
                from retirement_planner.analysis import build_analysis_brief
                brief = build_analysis_brief(
                    st.session_state.profile, st.session_state.personal_info,
                    st.session_state.get("retire_ages"),
                )
                changes = []
                if new_inv: changes.append(f"{len(new_inv)} new investment(s)")
                if new_bank: changes.append(f"{len(new_bank)} new bank account(s)")
                if new_cc: changes.append(f"{len(new_cc)} new credit card(s)")
                if new_spend: changes.append(f"{len(new_spend)} spending categories updated")
                change_msg = "Updated financial data: " + ", ".join(changes)
                st.session_state.messages.append({"role": "user", "content": change_msg})
                st.info(f"Profile updated. {change_msg}")
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask a follow-up question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ensure agent exists
        if not st.session_state.agent:
            try:
                if st.session_state.session_id:
                    from retirement_planner.agent import restore_agent_from_session
                    session_data = history.load_session(st.session_state.session_id)
                    st.session_state.agent = restore_agent_from_session(session_data)
                else:
                    st.session_state.agent = create_agent(model_id=model_id, task="followup")
            except Exception as e:
                st.error(f"Failed to create agent: {e}")
                st.stop()

        # Context injection based on question
        context_hint = ""
        if st.session_state.profile:
            p = st.session_state.profile
            q = prompt.lower()
            if any(w in q for w in ("ticker", "holding", "fund", "stock", "invest", "return", "portfolio", "401k", "ira", "brokerage")):
                lines = [f"- {a.account_type}: ${a.balance:,.0f} [{a.holdings}]" for a in p.investments]
                if lines:
                    context_hint = "\n\n[Investment data:\n" + "\n".join(lines) + "]\n"
            elif any(w in q for w in ("spend", "budget", "expense", "cost", "cut", "reduce")):
                lines = [f"- {s.category}: ${s.monthly_amount:,.0f}/mo" for s in p.spending]
                if lines:
                    context_hint = "\n\n[Spending data:\n" + "\n".join(lines) + "]\n"

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_parts = []

            async def _stream():
                async for event in stream_follow_up(st.session_state.agent, prompt + context_hint):
                    if "text" in event:
                        response_parts.append(event["text"])
                        placeholder.markdown(_clean_response("".join(response_parts)) + "▌", unsafe_allow_html=True)
                    elif event.get("done"):
                        full = event.get("full_text", "")
                        if full:
                            response_parts.clear()
                            response_parts.append(full)

            asyncio.run(_stream())
            raw_full = "".join(response_parts)
            followups = _extract_followups(raw_full)
            full_response = _clean_response(raw_full)
            placeholder.markdown(full_response, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Check if response contains an updated assessment
        try:
            from retirement_planner.serialization import parse_assessment_response
            updated = parse_assessment_response(raw_full)
            st.session_state.assessment = updated.model_dump()
            # Recompute brief with any new assumptions
            from retirement_planner.analysis import build_analysis_brief
            st.session_state._current_brief = build_analysis_brief(
                st.session_state.profile, st.session_state.personal_info,
                st.session_state.get("retire_ages")
            )
        except (ValueError, Exception):
            pass  # Not an assessment update, just informational

        # Show suggested followups as buttons
        st.session_state.followups = followups
        if followups:
            st.rerun()

        # Save to session
        if st.session_state.session_id:
            try:
                session_data = history.load_session(st.session_state.session_id)
                session_data.setdefault("conversation", []).append({
                    "question": prompt,
                    "response": full_response,
                })
                if st.session_state.assessment:
                    session_data["assessment"] = st.session_state.assessment
                history.save_session(st.session_state.session_id, session_data)
            except Exception:
                pass
