"""CLI module for the Retirement Planner.

Handles argument parsing, interactive data collection, assessment display,
and the conversational follow-up session loop.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from pydantic import ValidationError

from retirement_planner.agent import (
    create_agent,
    generate_assumption_summary,
    run_follow_up,
    run_initial_assessment,
)
from retirement_planner.file_parser import (
    parse_all_from_file,
    parse_banking_file,
    parse_credit_cards_file,
    parse_investments_file,
)
from retirement_planner.formatter import (
    format_assessment,
    format_assumption_summary,
    format_projection_update,
)
from retirement_planner.history import (
    list_sessions,
    load_session,
    new_session_id,
    save_session,
)
from retirement_planner.models import (
    BankAccount,
    BedrockError,
    CreditCard,
    CredentialError,
    FinancialProfile,
    InvestmentAccount,
    PersonalInfo,
    RetirementAssessment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with file path options.
    """
    parser = argparse.ArgumentParser(
        description="Retirement Planner — AI-powered retirement readiness assessment",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="One or more files containing financial data in any format. The agent figures out what's in each file.",
    )
    parser.add_argument(
        "--investments",
        type=str,
        default=None,
        help="Path to investments data file (any text format)",
    )
    parser.add_argument(
        "--banking",
        type=str,
        default=None,
        help="Path to banking data file (any text format)",
    )
    parser.add_argument(
        "--credit-cards",
        type=str,
        default=None,
        help="Path to credit cards data file (any text format)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="latest",
        default=None,
        help="Resume a previous session (optionally pass a session ID, or 'latest')",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        default=False,
        help="List previous sessions and exit",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Bedrock model ID or alias (opus, sonnet, llama4-maverick, llama4-scout). "
             "Overrides RETIREMENT_PLANNER_MODEL env var. Default: Claude Opus 4.6",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Interactive data collection helpers
# ---------------------------------------------------------------------------

def _prompt_float(prompt_text: str, *, allow_negative: bool = False) -> float:
    """Prompt the user for a float value, re-prompting on invalid input."""
    while True:
        raw = input(prompt_text)
        try:
            value = float(raw)
        except ValueError:
            print(f"  Invalid number: {raw!r}. Please enter a numeric value.")
            continue
        if not allow_negative and value < 0:
            print("  Value must be non-negative. Please try again.")
            continue
        return value


def _prompt_int(prompt_text: str, *, min_val: int | None = None, max_val: int | None = None) -> int:
    """Prompt the user for an integer value, re-prompting on invalid input."""
    while True:
        raw = input(prompt_text)
        try:
            value = int(raw)
        except ValueError:
            print(f"  Invalid integer: {raw!r}. Please enter a whole number.")
            continue
        if min_val is not None and value < min_val:
            print(f"  Value must be at least {min_val}. Please try again.")
            continue
        if max_val is not None and value > max_val:
            print(f"  Value must be at most {max_val}. Please try again.")
            continue
        return value


def collect_investments_interactive() -> list[InvestmentAccount]:
    """Prompt the user to enter investment accounts one at a time."""
    accounts: list[InvestmentAccount] = []
    print("\n--- Investment Accounts ---")
    while True:
        print(f"\nInvestment account #{len(accounts) + 1}:")
        account_type = input("  Account type (e.g. 401k, IRA, brokerage): ").strip()
        if not account_type:
            print("  Account type cannot be empty.")
            continue
        balance = _prompt_float("  Balance: $")
        holdings = input("  Holdings (e.g. VTSAX, S&P 500 index, target date 2045 — or press Enter to skip): ").strip()
        expected_return = _prompt_float("  Expected annual return (%, or 0 if unsure — the agent will estimate from holdings): ", allow_negative=True)
        try:
            account = InvestmentAccount(
                account_type=account_type,
                balance=balance,
                expected_annual_return=expected_return,
                holdings=holdings,
            )
            accounts.append(account)
        except ValidationError as exc:
            print(f"  Validation error: {exc}")
            continue
        more = input("  Add another investment account? (y/n): ").strip().lower()
        if more != "y":
            break
    return accounts


def collect_banking_interactive() -> list[BankAccount]:
    """Prompt the user to enter bank accounts one at a time."""
    accounts: list[BankAccount] = []
    print("\n--- Bank Accounts ---")
    while True:
        print(f"\nBank account #{len(accounts) + 1}:")
        account_type = input("  Account type (checking/savings): ").strip()
        if not account_type:
            print("  Account type cannot be empty.")
            continue
        balance = _prompt_float("  Balance: $")
        deposits = _prompt_float("  Monthly income deposits: $")
        try:
            account = BankAccount(
                account_type=account_type,
                balance=balance,
                monthly_income_deposits=deposits,
            )
            accounts.append(account)
        except ValidationError as exc:
            print(f"  Validation error: {exc}")
            continue
        more = input("  Add another bank account? (y/n): ").strip().lower()
        if more != "y":
            break
    return accounts


def collect_credit_cards_interactive() -> list[CreditCard]:
    """Prompt the user to enter credit cards one at a time."""
    cards: list[CreditCard] = []
    print("\n--- Credit Cards ---")
    while True:
        print(f"\nCredit card #{len(cards) + 1}:")
        balance = _prompt_float("  Outstanding balance: $")
        limit = _prompt_float("  Credit limit: $")
        payment = _prompt_float("  Monthly payment: $")
        try:
            card = CreditCard(
                outstanding_balance=balance,
                credit_limit=limit,
                monthly_payment=payment,
            )
            cards.append(card)
        except ValidationError as exc:
            print(f"  Validation error: {exc}")
            continue
        more = input("  Add another credit card? (y/n): ").strip().lower()
        if more != "y":
            break
    return cards


def collect_financial_data(agent, args: argparse.Namespace) -> FinancialProfile:
    """Collect financial data from files or interactive prompts.

    Supports three modes:
    - --files: agent classifies and extracts all categories from each file
    - --investments/--banking/--credit-cards: category-specific file parsing
    - Interactive prompts for any category not supplied via files

    Args:
        agent: A configured Strands Agent instance.
        args: Parsed CLI arguments.

    Returns:
        A validated FinancialProfile.
    """
    # --files mode: agent figures out what's in each file
    if args.files:
        all_investments = []
        all_bank_accounts = []
        all_credit_cards = []
        for file_path in args.files:
            print(f"\nAnalyzing {file_path}...")
            profile = parse_all_from_file(agent, file_path)
            all_investments.extend(profile.investments)
            all_bank_accounts.extend(profile.bank_accounts)
            all_credit_cards.extend(profile.credit_cards)
            loaded = []
            if profile.investments:
                loaded.append(f"{len(profile.investments)} investment(s)")
            if profile.bank_accounts:
                loaded.append(f"{len(profile.bank_accounts)} bank account(s)")
            if profile.credit_cards:
                loaded.append(f"{len(profile.credit_cards)} credit card(s)")
            print(f"  Found: {', '.join(loaded) if loaded else 'no financial data'}")

        # Prompt interactively for any missing categories
        if not all_investments:
            print("\nNo investment data found in files.")
            all_investments = collect_investments_interactive()
        if not all_bank_accounts:
            print("\nNo banking data found in files.")
            all_bank_accounts = collect_banking_interactive()
        if not all_credit_cards:
            print("\nNo credit card data found in files.")
            all_credit_cards = collect_credit_cards_interactive()

        return FinancialProfile(
            investments=all_investments,
            bank_accounts=all_bank_accounts,
            credit_cards=all_credit_cards,
        )

    # Category-specific file mode
    if args.investments:
        print(f"\nLoading investments from {args.investments}...")
        investments = parse_investments_file(agent, args.investments)
        print(f"  Loaded {len(investments)} investment account(s).")
    else:
        investments = collect_investments_interactive()

    if args.banking:
        print(f"\nLoading banking data from {args.banking}...")
        bank_accounts = parse_banking_file(agent, args.banking)
        print(f"  Loaded {len(bank_accounts)} bank account(s).")
    else:
        bank_accounts = collect_banking_interactive()

    if args.credit_cards:
        print(f"\nLoading credit card data from {args.credit_cards}...")
        credit_cards = parse_credit_cards_file(agent, args.credit_cards)
        print(f"  Loaded {len(credit_cards)} credit card(s).")
    else:
        credit_cards = collect_credit_cards_interactive()

    profile = FinancialProfile(
        investments=investments,
        bank_accounts=bank_accounts,
        credit_cards=credit_cards,
    )

    if not investments and not bank_accounts and not credit_cards:
        print("\nWarning: No financial data provided. The assessment may be limited.")

    return profile


def collect_personal_info() -> PersonalInfo:
    """Prompt the user for personal information (ages).

    Validates ages are within 0-120 range and warns if either spouse is under 18.

    Returns:
        A validated PersonalInfo instance.
    """
    print("\n--- Personal Information ---")
    while True:
        husband_age = _prompt_int("Husband's age: ", min_val=0, max_val=120)
        wife_age = _prompt_int("Wife's age: ", min_val=0, max_val=120)

        num_children = _prompt_int("Number of children: ", min_val=0)
        children_ages: list[int] = []
        for i in range(num_children):
            age = _prompt_int(f"  Child {i + 1} age: ", min_val=0, max_val=120)
            children_ages.append(age)

        try:
            info = PersonalInfo(
                husband_age=husband_age,
                wife_age=wife_age,
                children_ages=children_ages,
            )
        except ValidationError as exc:
            print(f"Validation error: {exc}")
            print("Please re-enter personal information.\n")
            continue

        # Under-18 warning (Requirement 2.5)
        if info.husband_age < 18:
            print("  Warning: Husband's age is below 18. Retirement planning typically applies to adults.")
        if info.wife_age < 18:
            print("  Warning: Wife's age is below 18. Retirement planning typically applies to adults.")

        return info


def display_assumption_summary(summary: dict) -> None:
    """Print the assumption summary to the terminal.

    Args:
        summary: The assumption summary dict from generate_assumption_summary.
    """
    formatted = format_assumption_summary(summary)
    print(f"\n{formatted}\n")


def prompt_assumption_confirmation(summary: dict) -> dict:
    """Display the assumption summary and prompt the user to confirm or correct.

    The user can type 'yes' or 'y' to confirm, or provide corrections in the
    format 'key=value, key=value' (e.g. 'inflation_rate=2.5, retirement_age=67').
    Corrections are applied to the summary's ``assumptions`` dict, the updated
    summary is re-displayed, and the user is re-prompted until they confirm.

    Args:
        summary: The assumption summary dict.

    Returns:
        The final confirmed (possibly corrected) summary dict.
    """
    display_assumption_summary(summary)

    while True:
        response = input(
            "Confirm assumptions? (yes/y to confirm, or provide corrections "
            "e.g. 'inflation_rate=2.5, retirement_age=67'): "
        ).strip()

        if response.lower() in ("yes", "y"):
            return summary

        if not response:
            print("Please type 'yes' to confirm or provide corrections.")
            continue

        # Parse corrections
        assumptions = summary.get("assumptions", {})
        corrections_applied = False
        for part in response.split(","):
            part = part.strip()
            if "=" not in part:
                print(f"  Skipping invalid correction: {part!r} (expected key=value)")
                continue
            key, _, raw_value = part.partition("=")
            key = key.strip()
            raw_value = raw_value.strip()

            if key not in assumptions:
                print(f"  Unknown assumption key: {key!r}. "
                      f"Valid keys: {', '.join(sorted(assumptions.keys()))}")
                continue

            # Try to cast to the same type as the existing value
            existing = assumptions[key]
            try:
                if isinstance(existing, float):
                    value = float(raw_value)
                elif isinstance(existing, int):
                    value = int(raw_value)
                else:
                    value = raw_value
            except ValueError:
                print(f"  Invalid value for {key}: {raw_value!r}")
                continue

            assumptions[key] = value
            corrections_applied = True

        if corrections_applied:
            summary["assumptions"] = assumptions
            print("\nUpdated assumptions:")
            display_assumption_summary(summary)
        else:
            print("No valid corrections applied. Please try again.")


def display_assessment(assessment: RetirementAssessment) -> None:
    """Print the retirement assessment to the terminal.

    Args:
        assessment: The assessment to display.
    """
    formatted = format_assessment(assessment)
    print(f"\n{formatted}\n")


def run_conversation_session(
    agent, profile: FinancialProfile, info: PersonalInfo,
    session_id: str | None = None, session_data: dict | None = None,
) -> None:
    """Enter the follow-up question loop until user types exit/quit.

    Maintains conversation context through the agent. Displays updated
    projections when the agent detects assumption changes, or plain text
    responses otherwise. Saves each exchange to the session file.

    Args:
        agent: A configured Strands Agent instance (with prior conversation).
        profile: The couple's financial profile.
        info: Personal information.
        session_id: Optional session ID for history tracking.
        session_data: Optional session data dict to append conversation to.
    """
    previous_assessment: RetirementAssessment | None = None

    print("You can now ask follow-up questions about your retirement plan.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            print("\nGoodbye! Best of luck with your retirement planning.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Goodbye! Best of luck with your retirement planning.")
            break

        print("Thinking...")
        try:
            result = run_follow_up(agent, question)
        except (CredentialError, BedrockError) as exc:
            print(f"\nError: {exc}")
            print("You can try asking another question.\n")
            continue
        except Exception as exc:
            print(f"\nError processing your question: {exc}")
            print("You can try asking another question.\n")
            continue

        if isinstance(result, RetirementAssessment):
            if previous_assessment is not None:
                formatted = format_projection_update(previous_assessment, result)
            else:
                formatted = format_assessment(result)
            print(f"\n{formatted}\n")
            previous_assessment = result
            response_text = formatted
        else:
            print(f"\n{result}\n")
            response_text = result

        # Save exchange to session
        if session_id and session_data is not None:
            session_data["conversation"].append({
                "question": question,
                "response": response_text,
                "is_assessment": isinstance(result, RetirementAssessment),
            })
            if isinstance(result, RetirementAssessment):
                session_data["assessment"] = result.model_dump()
            save_session(session_id, session_data)


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse args, collect data, run assessment, enter conversation loop.

    Args:
        argv: Optional argument list for testing (defaults to sys.argv[1:]).
    """
    try:
        args = parse_args(argv)

        # --history: list previous sessions and exit
        if args.history:
            sessions = list_sessions()
            if not sessions:
                print("No previous sessions found.")
            else:
                print(f"{'Session ID':<20} {'Date':<22} {'Retire?':<10} {'Net Worth':<15} {'Exchanges'}")
                print("-" * 75)
                for s in sessions:
                    can_retire = "Yes" if s["can_retire"] else "No" if s["can_retire"] is not None else "—"
                    nw = f"${s['net_worth']:,.0f}" if s["net_worth"] is not None else "—"
                    print(f"{s['session_id']:<20} {s['created_at']:<22} {can_retire:<10} {nw:<15} {s['exchanges']}")
            return

        print("Welcome to the Retirement Planner!")
        print("=" * 40)

        # Create the agent first (needed for file parsing)
        print("\nInitializing AI agent...")
        try:
            agent = create_agent(model_id=args.model)
        except CredentialError as exc:
            print(f"\nError: {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"\nFailed to initialize agent: {exc}")
            sys.exit(1)

        # --resume: load a previous session and jump to conversation
        if args.resume:
            from retirement_planner.history import get_latest_session_id
            sid = args.resume if args.resume != "latest" else get_latest_session_id()
            if not sid:
                print("No previous sessions found to resume.")
                sys.exit(1)
            try:
                session_data = load_session(sid)
            except FileNotFoundError:
                print(f"Session '{sid}' not found.")
                sys.exit(1)
            print(f"\nResuming session {sid}...")
            profile = FinancialProfile.model_validate(session_data["profile"])
            info = PersonalInfo.model_validate(session_data["personal_info"])
            if session_data.get("assessment"):
                assessment = RetirementAssessment.model_validate(session_data["assessment"])
                display_assessment(assessment)
            print(f"\n{len(session_data.get('conversation', []))} previous exchange(s) in this session.\n")
            run_conversation_session(agent, profile, info, sid, session_data)
            return

        # New session
        session_id = new_session_id()

        # Collect financial data
        profile = collect_financial_data(agent, args)

        # Collect personal info
        info = collect_personal_info()

        # Build file_sources for assumption summary
        file_sources: dict[str, str] | None = None
        if args.files:
            file_sources = {f"file_{i}": fp for i, fp in enumerate(args.files)}
        else:
            sources = {}
            if args.investments:
                sources["investments"] = args.investments
            if args.banking:
                sources["banking"] = args.banking
            if args.credit_cards:
                sources["credit_cards"] = args.credit_cards
            if sources:
                file_sources = sources

        # Generate and confirm assumptions before assessment
        print("\nGenerating assumption summary...")
        try:
            summary = generate_assumption_summary(agent, profile, info, file_sources=file_sources)
        except (CredentialError, BedrockError) as exc:
            print(f"\nError generating assumptions: {exc}")
            sys.exit(1)

        summary = prompt_assumption_confirmation(summary)

        # Run initial assessment
        print("\nAnalyzing your financial data...")
        try:
            assessment = run_initial_assessment(agent, profile, info)
        except (CredentialError, BedrockError) as exc:
            print(f"\nError: {exc}")
            if isinstance(exc, BedrockError) and exc.retryable:
                retry = input("Would you like to retry? (y/n): ").strip().lower()
                if retry == "y":
                    try:
                        assessment = run_initial_assessment(agent, profile, info)
                    except Exception as retry_exc:
                        print(f"\nRetry failed: {retry_exc}")
                        sys.exit(1)
                else:
                    sys.exit(1)
            else:
                sys.exit(1)

        # Display the assessment
        display_assessment(assessment)

        # Save initial session
        session_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile.model_dump(),
            "personal_info": info.model_dump(),
            "assessment": assessment.model_dump(),
            "conversation": [],
        }
        path = save_session(session_id, session_data)
        print(f"Session saved. Resume later with: --resume {session_id}\n")

        # Enter conversation loop
        run_conversation_session(agent, profile, info, session_id, session_data)

    except KeyboardInterrupt:
        print("\n\nGoodbye! Best of luck with your retirement planning.")
        sys.exit(0)
