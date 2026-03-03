# Retirement Planner

An AI-powered retirement planning tool that analyzes your financial data and produces a retirement readiness assessment. Uses a Strands Agent with CPA/CFA expertise powered by Claude on Amazon Bedrock.

## Quick Start

### Prerequisites

- Python 3.13+ (or Docker)
- AWS credentials with Amazon Bedrock access (Claude Opus 4.6)
- Credentials via environment variables, AWS profile, or `~/.aws/credentials`

### Option 1: Docker (Recommended)

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). That's it.

AWS credentials are passed through automatically from your environment or `~/.aws/` config. Sessions persist in a Docker volume.

To pass credentials explicitly:

```bash
AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy docker compose up --build
```

### Option 2: Local Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn retirement_planner.web:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

The web interface provides:
- File upload with drag-and-drop for financial data (CSV, JSON, plain text)
- Personal info form for ages of spouses and children
- Assumption review and correction before running the assessment
- Rich formatted results with charts, tables, and budget breakdowns
- Conversational follow-up with suggested questions
- Session history sidebar to revisit previous assessments

## How It Works

1. Upload your financial files or enter data manually
2. Review the extracted data and assumptions the agent will use
3. Confirm or correct assumptions, then run the assessment
4. Get a retirement readiness assessment with a recommended monthly budget
5. Ask follow-up questions to explore scenarios ("What if I retire at 67?")

## CLI (Alternative)

For terminal-based usage:

```bash
# Interactive mode
python -m retirement_planner

# From files
python -m retirement_planner --files finances.csv statement.pdf
python -m retirement_planner --investments inv.csv --banking bank.json --credit-cards cards.txt

# Session history
python -m retirement_planner --history
python -m retirement_planner --resume
```

Sessions are saved to `~/.retirement_planner/sessions/`.

## Sample Data

The `samples/` directory contains realistic fictional financial data for testing — a dual-income couple (James & Sarah Chen) in Redmond, WA with ~$1.8M net worth.

```bash
# CLI with sample files
python -m retirement_planner \
  --investments samples/investment_statement.csv \
  --banking samples/bank_statements.csv \
  --credit-cards samples/credit_cards.csv

# Or load all at once
python -m retirement_planner --files samples/investment_statement.csv samples/bank_statements.csv samples/credit_cards.csv
```

For the web UI, upload the CSV files via the data input panel or the 📎 chat button, then enter birthdates from `samples/personal_info.json`.

See [`samples/README.md`](samples/README.md) for full details on the fictional family profile and data contents.

## Tests

```bash
pytest tests/ -v
```

## License

MIT
