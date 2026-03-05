"""File parsing and agent-driven data normalization for the Retirement Planner.

Reads raw file contents from Data_Files in any text-based format, delegates
interpretation to the Strands Agent, and validates extracted data against
Pydantic model constraints.
"""

from __future__ import annotations

import json
import re
from typing import Union

from pydantic import ValidationError

from retirement_planner.models import (
    BankAccount,
    CreditCard,
    FileParseError,
    InvestmentAccount,
    MonthlySpending,
    NormalizationError,
)

# Max characters per chunk sent to the agent.
# Empirically, CSV data tokenizes at ~2 chars/token. With a 200k token context
# window and ~10k tokens for system prompt + tools + response overhead,
# we target ~150k tokens of content = ~300k chars. But the first chunk at 500k
# chars hit 231k tokens, so the real ratio is ~2.2 chars/token.
# Use 250k chars (~115k tokens) to leave comfortable headroom.
_MAX_CONTENT_CHARS = 250_000




# Maps category names to their Pydantic model class and expected JSON key
_CATEGORY_CONFIG: dict[str, dict] = {
    "investments": {
        "model": InvestmentAccount,
        "fields": ["account_type", "balance", "expected_annual_return"],
    },
    "banking": {
        "model": BankAccount,
        "fields": ["account_type", "balance", "monthly_income_deposits"],
    },
    "credit_cards": {
        "model": CreditCard,
        "fields": ["outstanding_balance", "credit_limit", "monthly_payment"],
    },
    "spending": {
        "model": MonthlySpending,
        "fields": ["category", "monthly_amount"],
    },
}


def read_file_contents(file_path: str) -> str:
    """Read raw text contents from a file.

    Supports text files (CSV, JSON, plain text, etc.) and PDFs.
    For PDFs, extracts text from all pages using pymupdf.
    Returns empty string for image-based PDFs (caller should use image path).

    Args:
        file_path: Path to the file to read.

    Returns:
        The raw text content of the file, or empty string for image-based PDFs.

    Raises:
        FileParseError: If the file doesn't exist, is a directory, can't be
            decoded as text, or any other read failure occurs.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        raise FileParseError(file_path or "<empty>", "file_path must be a non-empty string")

    import os
    if not os.path.exists(file_path):
        raise FileParseError(file_path, "File not found")
    if os.path.isdir(file_path):
        raise FileParseError(file_path, "Path is a directory, not a file")

    # Handle PDFs
    if file_path.lower().endswith(".pdf"):
        return _read_pdf(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        raise FileParseError(
            file_path, "File cannot be read as text (binary or encoding error)"
        )
    except PermissionError:
        raise FileParseError(file_path, "Permission denied")
    except Exception as exc:
        raise FileParseError(file_path, str(exc))


def _read_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pymupdf.

    Returns extracted text, or empty string if the PDF is image-based.
    """
    try:
        import pymupdf
    except ImportError:
        raise FileParseError(
            file_path,
            "PDF support requires pymupdf. Install it with: pip install pymupdf"
        )

    try:
        doc = pymupdf.open(file_path)
        text_parts = []
        for i, page in enumerate(doc):
            # Try multiple text extraction methods
            page_text = page.get_text()
            if not page_text.strip():
                # Try extracting from text dict (catches some edge cases)
                text_dict = page.get_text("dict")
                blocks = text_dict.get("blocks", [])
                block_texts = []
                for block in blocks:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            t = span.get("text", "").strip()
                            if t:
                                block_texts.append(t)
                page_text = " ".join(block_texts)
            if not page_text.strip():
                # Try raw text extraction
                page_text = page.get_text("text")
            print(f"  PDF page {i+1} text length: {len(page_text)} chars")
            text_parts.append(page_text)
        doc.close()
        full_text = "\n".join(text_parts).strip()
        print(f"  PDF total extracted text: {len(full_text)} chars")
        if full_text:
            print(f"  First 200 chars: {full_text[:200]!r}")
        return full_text
    except Exception as exc:
        raise FileParseError(file_path, f"Failed to read PDF: {exc}")


def _pdf_pages_to_images(file_path: str) -> list[bytes]:
    """Convert each page of a PDF to PNG image bytes.

    Tries pymupdf first, then falls back to macOS sips or pdf2image (poppler)
    for PDFs that pymupdf can't render (XFA forms, JavaScript-rendered, etc.).
    """
    # Try pymupdf first
    images = _pdf_pages_to_images_pymupdf(file_path)
    if images:
        # Check if images are actually blank (< 20KB for a full page is suspicious)
        non_blank = [img for img in images if len(img) > 20000]
        if non_blank:
            return non_blank
        print("  pymupdf rendered blank images, trying alternative renderer...")

    # Fallback: try pdf2image (poppler)
    try:
        from pdf2image import convert_from_path
        pil_images = convert_from_path(file_path, dpi=200)
        images = []
        for i, img in enumerate(pil_images):
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            print(f"  pdf2image page {i+1}: {img.width}x{img.height}, {len(png_bytes)} bytes")
            if len(png_bytes) > 1000:
                images.append(png_bytes)
        if images:
            return images
        print("  pdf2image also produced blank images")
    except ImportError:
        print("  pdf2image not available, trying macOS sips...")
    except Exception as exc:
        print(f"  pdf2image failed: {exc}, trying macOS sips...")

    # Fallback: macOS sips (converts first page only)
    try:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_png = tmp.name
        result = subprocess.run(
            ["sips", "-s", "format", "png", "-s", "dpiWidth", "200", "-s", "dpiHeight", "200",
             file_path, "--out", tmp_png],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            import os
            with open(tmp_png, "rb") as f:
                png_bytes = f.read()
            os.unlink(tmp_png)
            print(f"  sips rendered: {len(png_bytes)} bytes")
            if len(png_bytes) > 1000:
                return [png_bytes]
        else:
            print(f"  sips failed: {result.stderr}")
    except Exception as exc:
        print(f"  sips fallback failed: {exc}")

    return []


def _pdf_pages_to_images_pymupdf(file_path: str) -> list[bytes]:
    """Convert PDF pages to images using pymupdf."""
    try:
        import pymupdf
    except ImportError:
        return []

    try:
        doc = pymupdf.open(file_path)
        images = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200, alpha=False)
            png_bytes = pix.tobytes("png")
            print(f"  pymupdf page {i+1}: {pix.width}x{pix.height}, {len(png_bytes)} bytes")
            if pix.width > 50 and pix.height > 50:
                images.append(png_bytes)
        doc.close()
        return images
    except Exception as exc:
        print(f"  pymupdf rendering failed: {exc}")
        return []


def _extract_json_from_response(response_text: str) -> list[dict]:
    """Extract a JSON array from an agent response that may contain markdown or prose.

    Tries, in order:
    1. Parse the whole response as JSON
    2. Extract JSON from markdown code fences
    3. Find a JSON array via brace/bracket matching in the raw text

    Returns a list of dicts.
    """
    stripped = response_text.strip()

    # Try direct parse
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    for match in fence_pattern.finditer(response_text):
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except (json.JSONDecodeError, ValueError):
            continue

    # Try to find a JSON array in the raw text
    bracket_pattern = re.compile(r"\[.*\]", re.DOTALL)
    for match in bracket_pattern.finditer(response_text):
        candidate = match.group(0).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    # Try to find a JSON object in the raw text
    brace_pattern = re.compile(r"\{.*\}", re.DOTALL)
    for match in brace_pattern.finditer(response_text):
        candidate = match.group(0).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return [parsed]
        except (json.JSONDecodeError, ValueError):
            continue

    raise ValueError("Could not extract JSON from agent response")


def _truncate_for_agent(raw_content: str, max_chars: int = 50000) -> str:
    """Truncate file content to fit within the agent's context window.

    For CSV-like content (lines with commas/tabs), keeps the header row
    plus a sample of data rows from the beginning and end.
    For other content, takes the first max_chars characters.

    Args:
        raw_content: The full file content.
        max_chars: Maximum characters to return.

    Returns:
        Truncated content that fits within the limit.
    """
    if len(raw_content) <= max_chars:
        return raw_content

    lines = raw_content.split('\n')

    # Detect CSV-like content (header with commas or tabs)
    if len(lines) > 2 and (',' in lines[0] or '\t' in lines[0]):
        header = lines[0]
        data_lines = [l for l in lines[1:] if l.strip()]

        if len(data_lines) > 200:
            # Keep header + first 100 rows + last 100 rows
            sample = (
                [header]
                + data_lines[:100]
                + [f"\n... ({len(data_lines) - 200} rows omitted) ...\n"]
                + data_lines[-100:]
            )
            truncated = '\n'.join(sample)
            if len(truncated) <= max_chars:
                return truncated

        # Still too big — keep header + fewer rows
        sample = [header] + data_lines[:50]
        truncated = '\n'.join(sample)
        if len(truncated) > max_chars:
            truncated = truncated[:max_chars]
        return truncated + f"\n\n[Truncated: file had {len(data_lines)} data rows, showing first 50]"

    # Non-CSV: just take the first max_chars
    return raw_content[:max_chars] + f"\n\n[Truncated: file was {len(raw_content)} characters]"


def normalize_file_data(
    agent,
    raw_content: str,
    category: str,
) -> Union[list[InvestmentAccount], list[BankAccount], list[CreditCard]]:
    """Pass raw file content to the Strands Agent for interpretation.

    The agent extracts relevant financial fields and returns structured data
    validated against the Pydantic model constraints. Large files are
    automatically truncated to fit within the agent's context window.

    Args:
        agent: A Strands Agent instance.
        raw_content: The raw text content read from a file.
        category: One of 'investments', 'banking', or 'credit_cards'.

    Returns:
        A list of validated Pydantic model instances for the given category.

    Raises:
        FileParseError: If the category is invalid, the agent fails to produce
            valid JSON, or the extracted data fails Pydantic validation.
    """
    if category not in _CATEGORY_CONFIG:
        raise FileParseError(
            "<unknown>",
            f"Invalid category '{category}'. Must be one of: {', '.join(_CATEGORY_CONFIG)}",
        )

    config = _CATEGORY_CONFIG[category]
    model_cls = config["model"]
    expected_fields = config["fields"]

    # Truncate large content to avoid context overflow
    content = _truncate_for_agent(raw_content)

    prompt = (
        f"Extract {category} financial data from the following raw file content.\n"
        f"Return ONLY a JSON array of objects. Each object must have these fields: "
        f"{', '.join(expected_fields)}.\n"
        f"If the file contains transaction data (dates, descriptions, amounts), "
        f"summarize it into account-level totals rather than listing each transaction.\n"
        f"For investment statements, extract each account with its balance and type.\n"
        f"Do not include any explanation or markdown, just the JSON array.\n\n"
        f"Raw content:\n{content}"
    )

    try:
        result = agent(prompt)
        response_text = str(result)
    except Exception as exc:
        raise FileParseError("<file>", f"Agent failed to process content: {exc}")

    # Extract JSON from the agent response
    try:
        records = _extract_json_from_response(response_text)
    except ValueError:
        raise FileParseError(
            "<file>",
            f"Agent response did not contain valid JSON for {category} data",
        )

    if not records:
        raise FileParseError("<file>", f"Agent returned no {category} records")

    # Validate each record against the Pydantic model
    validated: list = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise FileParseError(
                "<file>",
                f"Expected a dict for record {i}, got {type(record).__name__}",
            )
        try:
            validated.append(model_cls.model_validate(record))
        except ValidationError as exc:
            # Collect the field names that failed validation
            missing = [
                err["loc"][0] if err["loc"] else err["msg"]
                for err in exc.errors()
            ]
            raise NormalizationError("<file>", [str(f) for f in missing])

    return validated


def parse_investments_file(agent, file_path: str) -> list[InvestmentAccount]:
    """Read file and normalize investment data via the agent.

    Args:
        agent: A Strands Agent instance.
        file_path: Path to the investments data file.

    Returns:
        A list of validated InvestmentAccount instances.

    Raises:
        FileParseError: If the file can't be read or data can't be normalized.
    """
    raw_content = read_file_contents(file_path)
    return normalize_file_data(agent, raw_content, "investments")


def parse_banking_file(agent, file_path: str) -> list[BankAccount]:
    """Read file and normalize banking data via the agent.

    Args:
        agent: A Strands Agent instance.
        file_path: Path to the banking data file.

    Returns:
        A list of validated BankAccount instances.

    Raises:
        FileParseError: If the file can't be read or data can't be normalized.
    """
    raw_content = read_file_contents(file_path)
    return normalize_file_data(agent, raw_content, "banking")


def parse_credit_cards_file(agent, file_path: str) -> list[CreditCard]:
    """Read file and normalize credit card data via the agent.

    Args:
        agent: A Strands Agent instance.
        file_path: Path to the credit card data file.

    Returns:
        A list of validated CreditCard instances.

    Raises:
        FileParseError: If the file can't be read or data can't be normalized.
    """
    raw_content = read_file_contents(file_path)
    return normalize_file_data(agent, raw_content, "credit_cards")

def _chunk_by_lines(raw_content: str, max_chars: int) -> list[str]:
    """Split file content into chunks that fit within max_chars, preserving the header row.

    For CSV-like files, the first line (header) is prepended to each chunk so the
    agent can understand column names in every chunk.
    """
    lines = raw_content.splitlines(keepends=True)
    if not lines:
        return [raw_content]

    header = lines[0]
    data_lines = lines[1:]

    # If the whole thing fits, just return it
    if len(raw_content) <= max_chars:
        return [raw_content]

    chunks: list[str] = []
    current_chunk = header
    for line in data_lines:
        if len(current_chunk) + len(line) > max_chars and current_chunk != header:
            chunks.append(current_chunk)
            current_chunk = header
        current_chunk += line

    if current_chunk != header:
        chunks.append(current_chunk)

    return chunks if chunks else [raw_content]


def _extract_all_from_chunk(agent, chunk: str, file_path: str) -> dict:
    """Send a single chunk to the agent and extract financial data."""
    import os
    filename = os.path.basename(file_path)
    prompt = (
        f"Extract ALL financial data from the following file: {filename}\n"
        "The file may contain account summaries, transaction ledgers, or any mix.\n"
        "CRITICAL: Treat each unique account number + fund combination as a SEPARATE investment.\n"
        "If the same plan number appears with different funds, list each fund as a separate holding within ONE account.\n"
        "Include the account number in the account_type to distinguish accounts.\n\n"
        "IMPORTANT: If this is transaction-level data (a spending/activity log), you should:\n"
        "- Identify each unique bank account and estimate monthly_income_deposits by summing\n"
        "  income/deposit/payroll transactions per month (use the most recent full month).\n"
        "  Set balance to 0 if no balance info is available.\n"
        "- Identify each unique credit card and estimate monthly_payment by summing\n"
        "  payment amounts per month. Set outstanding_balance and credit_limit to 0\n"
        "  if not available.\n"
        "- For investment accounts, only include them if explicit investment data is present.\n\n"
        "Return ONLY a JSON object with four keys:\n"
        '  "investments": array of objects with fields: account_type, balance, expected_annual_return\n'
        '    (optionally include holdings as a comma-separated string of fund names/tickers)\n'
        '  "bank_accounts": array of objects with fields: account_type, balance, monthly_income_deposits\n'
        '  "credit_cards": array of objects with fields: outstanding_balance, credit_limit, monthly_payment\n'
        '  "spending": array of objects with fields: category, monthly_amount\n'
        '    (aggregate transactions into category-level monthly averages, e.g. {"category": "Groceries", "monthly_amount": 800})\n\n'
        "Use empty arrays for categories not found in the data.\n"
        "RESPOND WITH ONLY THE JSON OBJECT. No explanation, no markdown, no text before or after.\n\n"
        f"Raw content:\n{chunk}"
    )

    try:
        result = agent(prompt)
        response_text = str(result)
    except Exception as exc:
        # If extraction model fails (e.g., Nova Lite tool_use error), retry with default model
        model_id = getattr(agent, '_model_id', '')
        if 'nova' in model_id.lower():
            print(f"  [fallback] {model_id} failed, retrying with Kimi K2.5...")
            from retirement_planner.agent import create_agent as _create_agent
            fallback = _create_agent(model_id="kimi")
            try:
                result = fallback(prompt)
                response_text = str(result)
            except Exception as exc2:
                raise FileParseError(file_path, f"Agent failed to process content: {exc2}")
        else:
            raise FileParseError(file_path, f"Agent failed to process content: {exc}")

    try:
        records = _extract_json_from_response(response_text)
        if len(records) == 1:
            data = records[0]
        else:
            data = records[0] if records else {}
    except ValueError:
        # If extraction model failed to produce JSON, retry with fallback
        _mid = getattr(agent, '_model_id', '')
        if 'nova' in _mid.lower() or 'scout' in _mid.lower():
            print(f"  [fallback] {_mid} produced no JSON, retrying with Kimi...")
            from retirement_planner.agent import create_agent as _create_agent
            _fb = _create_agent(model_id="kimi")
            try:
                result = _fb(prompt)
                records = _extract_json_from_response(str(result))
                data = records[0] if records else {}
            except Exception:
                raise FileParseError(file_path, "Agent response did not contain valid JSON")
        else:
            raise FileParseError(file_path, "Agent response did not contain valid JSON")

    if not isinstance(data, dict):
        raise FileParseError(file_path, "Expected a JSON object with investments, bank_accounts, credit_cards keys")

    return data


def _extract_all_from_images(agent, images: list, file_path: str) -> dict:
    """Send PDF page images to the agent for visual extraction of financial data."""
    instruction = (
        "Extract ALL financial data from these document page images.\n"
        "The pages may contain investments, bank accounts, credit cards, or any mix.\n\n"
        "Return ONLY a JSON object with four keys:\n"
        '  "investments": array of objects with fields: account_type, balance, expected_annual_return, holdings\n'
        '    (holdings should be a comma-separated string of fund names/tickers, e.g. "VTSAX, FXAIX")\n'
        '  "bank_accounts": array of objects with fields: account_type, balance, monthly_income_deposits\n'
        '  "credit_cards": array of objects with fields: outstanding_balance, credit_limit, monthly_payment\n'
        '  "spending": array of objects with fields: category, monthly_amount\n'
        '    (aggregate any visible transactions into category-level monthly averages)\n\n'
        "For investment accounts, set holdings to the fund name/ticker if visible.\n"
        "If expected_annual_return is not shown, estimate from the holdings or set to 0.\n"
        "Use empty arrays for categories not found.\n"
        "RESPOND WITH ONLY THE JSON OBJECT. No explanation, no markdown, no text before or after."
    )

    # Build multi-modal prompt: text instruction + all page images
    prompt_parts: list[dict] = [{"text": instruction}]
    for img_bytes in images:
        prompt_parts.append({
            "image": {
                "format": "png",
                "source": {"bytes": img_bytes},
            }
        })

    try:
        result = agent(prompt_parts)
        response_text = str(result)
    except Exception as exc:
        raise FileParseError(file_path, f"Agent failed to process PDF images: {exc}")

    try:
        records = _extract_json_from_response(response_text)
        if len(records) == 1:
            data = records[0]
        else:
            data = records[0] if records else {}
    except ValueError:
        raise FileParseError(file_path, "Agent response did not contain valid JSON")

    if not isinstance(data, dict):
        raise FileParseError(file_path, "Expected a JSON object with investments, bank_accounts, credit_cards keys")

    return data


def _validate_and_collect(data: dict, file_path: str):
    """Validate extracted data and return lists of model instances."""
    investments = []
    for rec in data.get("investments", []):
        try:
            # Coerce holdings to string if it's a list
            if "holdings" in rec and isinstance(rec["holdings"], list):
                rec["holdings"] = ", ".join(str(h) for h in rec["holdings"])
            investments.append(InvestmentAccount.model_validate(rec))
        except ValidationError as exc:
            # Try without the problematic field rather than failing entirely
            for err in exc.errors():
                field = err["loc"][0] if err["loc"] else None
                if field and field in rec:
                    del rec[field]
            try:
                investments.append(InvestmentAccount.model_validate(rec))
            except ValidationError:
                pass  # skip this record rather than failing the whole file

    bank_accounts = []
    for rec in data.get("bank_accounts", []):
        try:
            bank_accounts.append(BankAccount.model_validate(rec))
        except ValidationError:
            try:
                # Try with defaults for missing fields
                rec.setdefault("balance", 0)
                rec.setdefault("monthly_income_deposits", 0)
                rec.setdefault("account_type", "checking")
                bank_accounts.append(BankAccount.model_validate(rec))
            except ValidationError:
                pass

    credit_cards = []
    for rec in data.get("credit_cards", []):
        try:
            credit_cards.append(CreditCard.model_validate(rec))
        except ValidationError:
            try:
                rec.setdefault("outstanding_balance", 0)
                rec.setdefault("credit_limit", 0)
                rec.setdefault("monthly_payment", 0)
                credit_cards.append(CreditCard.model_validate(rec))
            except ValidationError:
                pass

    spending = []
    for rec in data.get("spending", []):
        try:
            spending.append(MonthlySpending.model_validate(rec))
        except ValidationError:
            try:
                rec.setdefault("monthly_amount", 0)
                spending.append(MonthlySpending.model_validate(rec))
            except ValidationError:
                pass

    return investments, bank_accounts, credit_cards, spending


# ---------------------------------------------------------------------------
# Fuzzy column header matching
# ---------------------------------------------------------------------------

# Map canonical names to common variations (all lowercase)
_COLUMN_ALIASES = {
    "account_type": ["account type", "acct type", "type", "account category"],
    "account_name": ["account name", "acct name", "name", "account", "description"],
    "amount": ["amount", "total", "value", "total value", "balance", "amt"],
    "category": ["category", "cat", "spending category", "expense category"],
    "date": ["date", "transaction date", "posted date", "post date", "trade date"],
    "institution": ["institution name", "institution", "bank", "bank name"],
    "account_number": ["account number", "acct number", "account no", "acct no", "acct #"],
    "symbol": ["symbol", "ticker", "ticker symbol"],
    "shares": ["shares", "quantity", "qty", "units"],
    "share_price": ["share price", "price", "unit price", "nav"],
}


def _match_column(fields: list[str], canonical: str) -> tuple[str | None, str]:
    """Match a canonical column name against actual CSV headers.

    Returns (matched_field_name, confidence) where confidence is HIGH/MEDIUM/LOW.
    """
    aliases = _COLUMN_ALIASES.get(canonical, [canonical])
    lower_fields = {f.lower().strip(): f for f in fields}

    # Exact match
    for alias in aliases:
        if alias in lower_fields:
            return lower_fields[alias], "HIGH"

    # Substring match
    for alias in aliases:
        for lf, orig in lower_fields.items():
            if alias in lf or lf in alias:
                return orig, "MEDIUM"

    return None, "LOW"


def _try_csv_bank_accounts(raw_content: str) -> tuple[list[BankAccount] | None, str]:
    """Try to extract bank account data directly from CSV.

    Returns (list of BankAccount or None, confidence level).
    """
    import csv
    import io
    from collections import defaultdict

    try:
        reader = csv.DictReader(io.StringIO(raw_content))
        fields = [f.strip() for f in (reader.fieldnames or [])]
    except Exception:
        return None, "LOW"

    type_col, type_conf = _match_column(fields, "account_type")
    amt_col, amt_conf = _match_column(fields, "amount")
    date_col, _ = _match_column(fields, "date")
    name_col, _ = _match_column(fields, "account_name")
    inst_col, _ = _match_column(fields, "institution")

    if not type_col or not amt_col:
        return None, "LOW"

    confidence = min(type_conf, amt_conf, key=lambda x: ["HIGH", "MEDIUM", "LOW"].index(x))

    # Aggregate by account: track deposits (negative amounts in some formats, positive in others)
    accounts: dict[str, dict] = defaultdict(lambda: {"deposits": 0.0, "count": 0})
    months: set[str] = set()
    has_cash_type = False

    for row in csv.DictReader(io.StringIO(raw_content)):
        acct_type = (row.get(type_col) or "").strip()
        if not acct_type:
            continue
        # Only process cash/bank accounts
        if not any(k in acct_type.lower() for k in ("cash", "checking", "savings", "money market", "bank", "deposit")):
            continue
        has_cash_type = True

        acct_name = (row.get(name_col) or "").strip() if name_col else ""
        inst = (row.get(inst_col) or "").strip() if inst_col else ""
        key = f"{acct_name} - {inst}".strip(" -") if (acct_name or inst) else acct_type

        try:
            amount = float(row.get(amt_col, 0))
        except (ValueError, TypeError):
            continue

        # Negative amounts are typically income/deposits in transaction exports
        if amount < 0:
            accounts[key]["deposits"] += abs(amount)
        accounts[key]["count"] += 1

        if date_col:
            ds = (row.get(date_col) or "")[:7]
            if ds:
                months.add(ds)

    if not has_cash_type:
        return None, "LOW"

    num_months = max(len(months), 1)
    result = []
    for key, data in sorted(accounts.items()):
        monthly_deposits = round(data["deposits"] / num_months, 2)
        result.append(BankAccount(
            account_type=key,
            balance=0,  # Transaction exports don't show current balance
            monthly_income_deposits=monthly_deposits,
        ))

    return result if result else None, confidence


def _try_csv_credit_cards(raw_content: str) -> tuple[list[CreditCard] | None, str]:
    """Try to extract credit card data directly from CSV.

    Returns (list of CreditCard or None, confidence level).
    """
    import csv
    import io
    from collections import defaultdict

    try:
        reader = csv.DictReader(io.StringIO(raw_content))
        fields = [f.strip() for f in (reader.fieldnames or [])]
    except Exception:
        return None, "LOW"

    type_col, type_conf = _match_column(fields, "account_type")
    amt_col, amt_conf = _match_column(fields, "amount")
    date_col, _ = _match_column(fields, "date")
    name_col, _ = _match_column(fields, "account_name")
    inst_col, _ = _match_column(fields, "institution")

    if not type_col or not amt_col:
        return None, "LOW"

    confidence = min(type_conf, amt_conf, key=lambda x: ["HIGH", "MEDIUM", "LOW"].index(x))

    accounts: dict[str, dict] = defaultdict(lambda: {"payments": 0.0, "count": 0})
    months: set[str] = set()
    has_cc_type = False

    for row in csv.DictReader(io.StringIO(raw_content)):
        acct_type = (row.get(type_col) or "").strip()
        if not acct_type:
            continue
        if "credit" not in acct_type.lower():
            continue
        has_cc_type = True

        acct_name = (row.get(name_col) or "").strip() if name_col else ""
        inst = (row.get(inst_col) or "").strip() if inst_col else ""
        key = f"{acct_name} - {inst}".strip(" -") if (acct_name or inst) else acct_type

        try:
            amount = abs(float(row.get(amt_col, 0)))
        except (ValueError, TypeError):
            continue

        accounts[key]["payments"] += amount
        accounts[key]["count"] += 1

        if date_col:
            ds = (row.get(date_col) or "")[:7]
            if ds:
                months.add(ds)

    if not has_cc_type:
        return None, "LOW"

    num_months = max(len(months), 1)
    result = []
    for key, data in sorted(accounts.items()):
        monthly_payment = round(data["payments"] / num_months, 2)
        result.append(CreditCard(
            outstanding_balance=0,
            credit_limit=0,
            monthly_payment=monthly_payment,
        ))

    return result if result else None, confidence


def _summarize_csv_for_agent(raw_content: str) -> str | None:
    """Summarize a transaction CSV into account-level data for the agent.

    When spending is already extracted directly, the agent only needs to
    identify bank accounts and credit cards. This builds a compact summary
    with monthly totals per account instead of sending thousands of rows.
    """
    import csv
    import io
    from collections import defaultdict

    try:
        reader = csv.DictReader(io.StringIO(raw_content))
        fields = [f.strip() for f in (reader.fieldnames or [])]
    except Exception:
        return None

    # Need Account Type, Account Name, and Amount columns
    type_col = next((f for f in fields if f.lower() in ('account type',)), None)
    name_col = next((f for f in fields if f.lower() in ('account name',)), None)
    amt_col = next((f for f in fields if f.lower() == 'amount'), None)
    date_col = next((f for f in fields if f.lower() == 'date'), None)
    inst_col = next((f for f in fields if f.lower() in ('institution name', 'institution')), None)

    if not type_col or not amt_col:
        return None

    # Aggregate by account
    accounts: dict[str, dict] = defaultdict(lambda: {"type": "", "total": 0.0, "income": 0.0, "expenses": 0.0, "count": 0})
    months: set[str] = set()

    for row in reader:
        acct_type = (row.get(type_col) or '').strip()
        acct_name = (row.get(name_col) or '').strip() if name_col else ''
        inst = (row.get(inst_col) or '').strip() if inst_col else ''
        key = f"{acct_type} - {acct_name} - {inst}" if inst else f"{acct_type} - {acct_name}"

        try:
            amount = float(row.get(amt_col, 0))
        except (ValueError, TypeError):
            continue

        accounts[key]["type"] = acct_type
        accounts[key]["count"] += 1
        if amount > 0:
            accounts[key]["expenses"] += amount
        else:
            accounts[key]["income"] += abs(amount)

        if date_col:
            date_str = (row.get(date_col) or '')[:7]
            if date_str:
                months.add(date_str)

    if not accounts:
        return None

    num_months = max(len(months), 1)

    lines = [f"Account summary ({num_months} months of transaction data):"]
    lines.append("Account | Type | Monthly Income | Monthly Expenses | Transaction Count")
    for key, data in sorted(accounts.items()):
        monthly_inc = round(data["income"] / num_months, 2)
        monthly_exp = round(data["expenses"] / num_months, 2)
        lines.append(f"{key} | {data['type']} | ${monthly_inc} | ${monthly_exp} | {data['count']}")

    lines.append("\nSpending categories have already been extracted. Focus on identifying bank accounts and credit cards only.")
    return "\n".join(lines)


def _try_csv_investments(raw_content: str) -> tuple[list[InvestmentAccount] | None, str]:
    """Try to extract investment data directly from CSV.

    Handles CSVs with columns like Symbol/Fund Name, Shares, Price, Total Value.
    Also handles multi-section CSVs by trying each section separately.
    """
    import csv
    import io

    # Try each section of a multi-section CSV (separated by blank lines
    # where headers change), but also try the whole file as one CSV first
    all_investments = []
    best_confidence = "LOW"

    # First: try parsing the whole file as one CSV (handles blank-line-separated groups with same headers)
    cleaned = "\n".join(line for line in raw_content.split("\n") if line.strip())
    for attempt_content in [cleaned, raw_content]:
        try:
            reader = csv.DictReader(io.StringIO(attempt_content))
            fields = [f.strip() for f in (reader.fieldnames or [])]
        except Exception:
            continue

        value_col, val_conf = _match_column(fields, "amount")
        symbol_col, _ = _match_column(fields, "symbol")
        shares_col, _ = _match_column(fields, "shares")
        price_col, _ = _match_column(fields, "share_price")
        fund_col = next((f for f in fields if any(k in f.lower() for k in ("fund name", "investment name", "fund", "security"))), None)
        acct_col = next((f for f in fields if any(k in f.lower() for k in ("account number", "plan number", "acct", "account"))), None)
        plan_col = next((f for f in fields if "plan name" in f.lower()), None)

        if not value_col or (not symbol_col and not fund_col):
            continue
        has_total_value = any("total value" in f.lower() for f in fields)
        has_price = price_col is not None or any("price" in f.lower() for f in fields)
        if not has_total_value and not has_price:
            continue

        confidence = val_conf
        found = []

        for row in csv.DictReader(io.StringIO(attempt_content)):
            try:
                balance = float(row.get(value_col, 0) or 0)
            except (ValueError, TypeError):
                continue
            if balance <= 0:
                continue

            holdings = (row.get(symbol_col, '') or '').strip() if symbol_col else ''
            fund_name = (row.get(fund_col, '') or '').strip() if fund_col else ''
            acct_num = (row.get(acct_col, '') or '').strip() if acct_col else ''
            plan_name = (row.get(plan_col, '') or '').strip() if plan_col else ''

            if plan_name and acct_num:
                acct_type = f"{plan_name} - Plan {acct_num}"
            elif acct_num:
                acct_type = f"Investment Account {acct_num}"
            else:
                acct_type = "Investment Account"

            if not holdings:
                holdings = fund_name

            h_lower = holdings.lower()
            if any(k in h_lower for k in ('bond', 'fixed', 'income', 'treasury')):
                ret = 0.04
            elif any(k in h_lower for k in ('money market', 'mmkt', 'cash')):
                ret = 0.02
            elif any(k in h_lower for k in ('target', 'balanced', 'wellington')):
                ret = 0.06
            else:
                ret = 0.07

            found.append(InvestmentAccount(
                account_type=acct_type, balance=balance,
                expected_annual_return=ret, holdings=holdings,
            ))
            best_confidence = "HIGH" if confidence == "HIGH" else best_confidence

        if found:
            all_investments = found
            break  # First successful attempt wins

    return all_investments if all_investments else None, best_confidence


def _try_csv_spending(raw_content: str) -> list[MonthlySpending] | None:
    """Try to extract spending data directly from CSV without using the agent.

    Looks for Category and Amount columns. If found, aggregates transactions
    into average monthly spending per category. Returns None if not a CSV
    or columns not found.
    """
    import csv
    import io
    from collections import defaultdict

    lines = raw_content.strip().split('\n')
    if len(lines) < 2:
        return None

    try:
        reader = csv.DictReader(io.StringIO(raw_content))
        fields = [f.strip() for f in (reader.fieldnames or [])]
    except Exception:
        return None

    # Find category and amount columns (case-insensitive)
    cat_col = next((f for f in fields if f.lower() == 'category'), None)
    amt_col = next((f for f in fields if f.lower() == 'amount'), None)
    date_col = next((f for f in fields if f.lower() == 'date'), None)

    if not cat_col or not amt_col:
        return None

    # Collect totals per category and track months
    cat_totals: dict[str, float] = defaultdict(float)
    months: set[str] = set()

    for row in reader:
        category = (row.get(cat_col) or '').strip()
        if not category or category.lower() in ('uncategorized', ''):
            continue
        try:
            amount = abs(float(row.get(amt_col, 0)))
        except (ValueError, TypeError):
            continue
        cat_totals[category] += amount
        if date_col:
            date_str = (row.get(date_col) or '')[:7]  # YYYY-MM
            if date_str:
                months.add(date_str)

    if not cat_totals:
        return None

    num_months = max(len(months), 1)
    return [
        MonthlySpending(category=cat, monthly_amount=round(total / num_months, 2))
        for cat, total in sorted(cat_totals.items())
    ]


def parse_all_from_file(agent, file_path: str) -> FinancialProfile:
    """Read a file and extract all financial data categories via the agent.

    For text files and text-based PDFs, splits into chunks if needed.
    For image-based PDFs (scanned), converts pages to images and sends
    them directly to the agent for visual extraction.

    Args:
        agent: A Strands Agent instance.
        file_path: Path to the financial data file.

    Returns:
        A validated FinancialProfile with whatever categories were found.

    Raises:
        FileParseError: If the file can't be read or data can't be extracted.
    """
    raw_content = read_file_contents(file_path)

    # If PDF returned no text, use image-based extraction
    if not raw_content and file_path.lower().endswith(".pdf"):
        print(f"  PDF text extraction returned empty. Trying image-based extraction...")
        images = _pdf_pages_to_images(file_path)
        if not images:
            raise FileParseError(file_path, "PDF has no pages")
        print(f"  Converted {len(images)} page(s) to images.")
        data = _extract_all_from_images(agent, images, file_path)
        investments, bank_accounts, credit_cards, spending = _validate_and_collect(data, file_path)
        from retirement_planner.models import FinancialProfile
        return FinancialProfile(
            investments=investments,
            bank_accounts=bank_accounts,
            credit_cards=credit_cards,
            spending=spending,
        )

    # Text-based path: try direct CSV parsing first, LLM fallback second
    pre_investments, inv_conf = _try_csv_investments(raw_content)
    pre_spending = _try_csv_spending(raw_content)
    pre_banks, bank_conf = _try_csv_bank_accounts(raw_content)
    pre_cards, cc_conf = _try_csv_credit_cards(raw_content)

    direct_parsed = False
    if pre_investments:
        print(f"  [direct] Extracted {len(pre_investments)} investments from CSV (confidence: {inv_conf}).")
        direct_parsed = True
    if pre_spending:
        print(f"  [direct] Extracted {len(pre_spending)} spending categories from CSV.")
        direct_parsed = True
    if pre_banks:
        print(f"  [direct] Extracted {len(pre_banks)} bank accounts from CSV (confidence: {bank_conf}).")
        direct_parsed = True
    if pre_cards:
        print(f"  [direct] Extracted {len(pre_cards)} credit cards from CSV (confidence: {cc_conf}).")
        direct_parsed = True

    # If everything was parsed directly, skip LLM entirely
    if pre_investments and (pre_spending or pre_banks or pre_cards):
        print(f"  [direct] All data extracted from CSV — skipping LLM.")
        from retirement_planner.models import FinancialProfile
        return FinancialProfile(
            investments=pre_investments,
            bank_accounts=pre_banks or [],
            credit_cards=pre_cards or [],
            spending=pre_spending or [],
        )

    # If we got spending + banks + cards directly, only send investment data to LLM
    if pre_spending and pre_banks and pre_cards:
        summary = _summarize_csv_for_agent(raw_content)
        if summary:
            raw_content = summary
            print(f"  [direct] Summarized CSV to {len(summary)} chars for agent (investment extraction only).")
    elif pre_spending:
        summary = _summarize_csv_for_agent(raw_content)
        if summary:
            raw_content = summary
            print(f"  [direct] Summarized CSV to {len(summary)} chars for agent (account extraction only).")

    chunks = _chunk_by_lines(raw_content, 25000)

    if len(chunks) > 1:
        print(f"  File is large — splitting into {len(chunks)} chunks for processing.")

    all_investments: list[InvestmentAccount] = []
    all_bank_accounts: list[BankAccount] = []
    all_credit_cards: list[CreditCard] = []
    all_spending: list[MonthlySpending] = []

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  Processing chunk {i + 1}/{len(chunks)}...")

        _used_image_fallback = False
        try:
            # Use a fresh agent per chunk to avoid context window overflow
            # from accumulated conversation history
            from retirement_planner.agent import create_agent as _create_agent
            chunk_agent = _create_agent(model_id=getattr(agent, '_model_id', None)) if len(chunks) > 1 else agent
            data = _extract_all_from_chunk(chunk_agent, chunk, file_path)
            inv, bank, cc, spend = _validate_and_collect(data, file_path)
            all_investments.extend(inv)
            all_bank_accounts.extend(bank)
            all_credit_cards.extend(cc)
            all_spending.extend(spend)
        except (FileParseError, NormalizationError) as exc:
            # If text-based extraction fails for a PDF, try image-based as fallback
            if file_path.lower().endswith(".pdf") and i == 0:
                print(f"  Text extraction failed ({exc}), trying image-based extraction...")
                try:
                    images = _pdf_pages_to_images(file_path)
                    if images:
                        print(f"  Converted {len(images)} page(s) to images for visual extraction.")
                        data = _extract_all_from_images(agent, images, file_path)
                        inv, bank, cc, spend = _validate_and_collect(data, file_path)
                        all_investments.extend(inv)
                        all_bank_accounts.extend(bank)
                        all_credit_cards.extend(cc)
                        all_spending.extend(spend)
                        _used_image_fallback = True
                    else:
                        raise exc
                except (FileParseError, NormalizationError):
                    raise
                except Exception as img_exc:
                    print(f"  Image extraction also failed: {img_exc}")
                    raise exc from img_exc
            else:
                raise
        if _used_image_fallback:
            break  # image extraction covers all pages

    from retirement_planner.models import FinancialProfile
    # Use pre-extracted CSV data if available (more accurate than agent extraction)
    final_investments = pre_investments if pre_investments else all_investments
    final_spending = pre_spending if pre_spending else all_spending
    final_banks = pre_banks if pre_banks else all_bank_accounts
    final_cards = pre_cards if pre_cards else all_credit_cards
    return FinancialProfile(
        investments=final_investments,
        bank_accounts=final_banks,
        credit_cards=final_cards,
        spending=final_spending,
    )

