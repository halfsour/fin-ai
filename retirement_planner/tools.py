"""Strands Agent tool functions for financial calculations and file reading."""

from __future__ import annotations

from strands import tool


@tool
def calculate_net_worth(profile: dict) -> dict:
    """Calculate net worth as total assets minus total liabilities.

    Assets include investment account balances and bank account balances.
    Liabilities include credit card outstanding balances.

    Args:
        profile: A dict with keys 'investments', 'bank_accounts', and 'credit_cards'.
                 Each value is a list of dicts with the relevant balance fields.

    Returns:
        A dict with 'net_worth', 'total_assets', and 'total_liabilities',
        or a dict with 'error' describing the problem.
    """
    try:
        if not isinstance(profile, dict):
            return {"error": f"Expected a dict for profile, got {type(profile).__name__}"}

        investments = profile.get("investments")
        bank_accounts = profile.get("bank_accounts")
        credit_cards = profile.get("credit_cards")

        if investments is None or bank_accounts is None or credit_cards is None:
            missing = [
                k
                for k in ("investments", "bank_accounts", "credit_cards")
                if profile.get(k) is None
            ]
            return {"error": f"Missing required profile fields: {', '.join(missing)}"}

        if not isinstance(investments, list) or not isinstance(bank_accounts, list) or not isinstance(credit_cards, list):
            return {"error": "investments, bank_accounts, and credit_cards must be lists"}

        investment_total = sum(inv.get("balance", 0) for inv in investments)
        bank_total = sum(bank.get("balance", 0) for bank in bank_accounts)
        total_assets = investment_total + bank_total

        total_liabilities = sum(cc.get("outstanding_balance", 0) for cc in credit_cards)

        net_worth = total_assets - total_liabilities

        return {
            "net_worth": net_worth,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
        }
    except Exception as exc:
        return {"error": f"Failed to calculate net worth: {exc}"}


@tool
def calculate_cash_flow(profile: dict) -> dict:
    """Calculate monthly cash flow as income minus expenses.

    Income is the sum of monthly_income_deposits across all bank accounts.
    Expenses come from spending data when available, otherwise from credit card payments.

    Args:
        profile: A dict with keys 'bank_accounts', 'credit_cards', and optionally 'spending'.
                 Each value is a list of dicts with the relevant fields.

    Returns:
        A dict with 'monthly_cash_flow', 'total_income', 'total_expenses', and 'expense_source',
        or a dict with 'error' describing the problem.
    """
    try:
        if not isinstance(profile, dict):
            return {"error": f"Expected a dict for profile, got {type(profile).__name__}"}

        bank_accounts = profile.get("bank_accounts")
        credit_cards = profile.get("credit_cards")

        if bank_accounts is None or credit_cards is None:
            missing = [
                k
                for k in ("bank_accounts", "credit_cards")
                if profile.get(k) is None
            ]
            return {"error": f"Missing required profile fields: {', '.join(missing)}"}

        if not isinstance(bank_accounts, list) or not isinstance(credit_cards, list):
            return {"error": "bank_accounts and credit_cards must be lists"}

        total_income = sum(bank.get("monthly_income_deposits", 0) for bank in bank_accounts)

        spending = profile.get("spending") or []
        if spending and isinstance(spending, list):
            total_expenses = sum(item.get("monthly_amount", 0) for item in spending)
            expense_source = "spending"
        else:
            total_expenses = sum(cc.get("monthly_payment", 0) for cc in credit_cards)
            expense_source = "credit_cards"

        monthly_cash_flow = total_income - total_expenses

        return {
            "monthly_cash_flow": monthly_cash_flow,
            "total_income": total_income,
            "total_expenses": total_expenses,
            "expense_source": expense_source,
        }
    except Exception as exc:
        return {"error": f"Failed to calculate cash flow: {exc}"}


@tool
def read_financial_file(file_path: str) -> dict:
    """Read raw text contents from a financial data file.

    Supports any text-based file format (CSV, JSON, plain text, etc.).

    Args:
        file_path: The path to the file to read.

    Returns:
        A dict with 'content' and 'file_path' on success,
        or a dict with 'error' describing the problem.
    """
    try:
        if not isinstance(file_path, str) or not file_path.strip():
            return {"error": "file_path must be a non-empty string"}

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {"content": content, "file_path": file_path}
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except IsADirectoryError:
        return {"error": f"Path is a directory, not a file: {file_path}"}
    except UnicodeDecodeError:
        return {"error": f"File cannot be read as text (binary or encoding issue): {file_path}"}
    except PermissionError:
        return {"error": f"Permission denied reading file: {file_path}"}
    except Exception as exc:
        return {"error": f"Failed to read file {file_path}: {exc}"}


@tool
def estimate_aca_premiums(
    zipcode: str,
    state: str,
    household_income: float,
    people: list[dict],
    year: int = 2025,
) -> dict:
    """Estimate ACA Marketplace health insurance premiums and subsidies.

    Queries the Healthcare.gov Marketplace API to get real plan pricing
    for a household based on location, income, and member ages.

    Args:
        zipcode: 5-digit ZIP code (e.g. "27360").
        state: 2-letter state code (e.g. "NC").
        household_income: Annual household income in dollars.
        people: List of household members. Each dict should have:
            - age (int): Person's age
            - gender (str): "Male" or "Female"
            - relationship (str, optional): "Self", "Spouse", or "Child"
            - uses_tobacco (bool, optional): Defaults to False
        year: Plan year (defaults to 2025).

    Returns:
        A dict with premium estimates including lowest cost plans by
        metal level, estimated tax credits, and net monthly costs,
        or a dict with 'error' describing the problem.
    """
    import json as _json
    import urllib.request
    import urllib.error

    API_KEY = "d687412e7b53146b2631dc01974ad0a4"
    BASE_URL = "https://marketplace.api.healthcare.gov/api/v1"

    try:
        # Step 1: Look up county FIPS code from ZIP
        county_url = f"{BASE_URL}/counties/by/zip/{zipcode}?apikey={API_KEY}"
        req = urllib.request.Request(county_url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            county_data = _json.loads(resp.read().decode())

        counties = county_data.get("counties", [])
        if not counties:
            return {"error": f"No counties found for ZIP code {zipcode}"}

        # Use the first county (most ZIP codes map to one county)
        countyfips = counties[0].get("fips", "")

        # Step 2: Build household and search for plans
        household_people = []
        for i, person in enumerate(people):
            p = {
                "age": person.get("age", 30),
                "gender": person.get("gender", "Female"),
                "aptc_eligible": True,
                "uses_tobacco": person.get("uses_tobacco", False),
            }
            rel = person.get("relationship")
            if rel:
                p["relationship"] = rel
            elif i == 0:
                p["relationship"] = "Self"
            household_people.append(p)

        search_body = {
            "household": {
                "income": household_income,
                "people": household_people,
            },
            "market": "Individual",
            "place": {
                "countyfips": countyfips,
                "state": state,
                "zipcode": zipcode,
            },
            "year": year,
        }

        search_url = f"{BASE_URL}/plans/search?apikey={API_KEY}"
        body_bytes = _json.dumps(search_body).encode("utf-8")
        req = urllib.request.Request(
            search_url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            plan_data = _json.loads(resp.read().decode())

        plans = plan_data.get("plans", [])
        if not plans:
            return {
                "error": "No plans found for this location and household",
                "county": counties[0].get("name", ""),
                "state": state,
            }

        # Step 3: Summarize results by metal level
        metal_summary = {}
        for plan in plans:
            metal = plan.get("metal_level", "Unknown")
            premium = plan.get("premium", 0)
            premium_w_credit = plan.get("premium_w_credit", premium)
            ehb_premium = plan.get("ehb_premium", premium)

            if metal not in metal_summary:
                metal_summary[metal] = {
                    "count": 0,
                    "lowest_premium": premium,
                    "lowest_premium_after_subsidy": premium_w_credit,
                    "highest_premium": premium,
                    "plan_name": plan.get("name", ""),
                }

            entry = metal_summary[metal]
            entry["count"] += 1
            if premium < entry["lowest_premium"]:
                entry["lowest_premium"] = premium
                entry["lowest_premium_after_subsidy"] = premium_w_credit
                entry["plan_name"] = plan.get("name", "")
            if premium > entry["highest_premium"]:
                entry["highest_premium"] = premium

        # Extract APTC (tax credit) from the response
        aptc = plan_data.get("aptc_value", 0)

        return {
            "zipcode": zipcode,
            "state": state,
            "county": counties[0].get("name", ""),
            "countyfips": countyfips,
            "year": year,
            "household_income": household_income,
            "household_size": len(people),
            "estimated_monthly_tax_credit": round(aptc, 2),
            "total_plans_found": len(plans),
            "plans_by_metal_level": metal_summary,
            "note": "Premiums are monthly. Premium after subsidy reflects estimated APTC.",
        }

    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode()
            err_data = _json.loads(err_body)
            return {"error": f"Marketplace API error: {err_data.get('message', err_body)}"}
        except Exception:
            return {"error": f"Marketplace API HTTP {exc.code}: {exc.reason}"}
    except urllib.error.URLError as exc:
        return {"error": f"Network error reaching Marketplace API: {exc.reason}"}
    except Exception as exc:
        return {"error": f"Failed to estimate ACA premiums: {exc}"}


# ---------------------------------------------------------------------------
# Tax reference data (updated annually)
# ---------------------------------------------------------------------------

_TAX_DATA = {
    2025: {
        "income_tax_brackets_married_filing_jointly": [
            {"min": 0, "max": 23850, "rate": 0.10},
            {"min": 23850, "max": 96950, "rate": 0.12},
            {"min": 96950, "max": 206700, "rate": 0.22},
            {"min": 206700, "max": 394600, "rate": 0.24},
            {"min": 394600, "max": 501050, "rate": 0.32},
            {"min": 501050, "max": 751600, "rate": 0.35},
            {"min": 751600, "max": None, "rate": 0.37},
        ],
        "income_tax_brackets_single": [
            {"min": 0, "max": 11925, "rate": 0.10},
            {"min": 11925, "max": 48475, "rate": 0.12},
            {"min": 48475, "max": 103350, "rate": 0.22},
            {"min": 103350, "max": 197300, "rate": 0.24},
            {"min": 197300, "max": 250525, "rate": 0.32},
            {"min": 250525, "max": 626350, "rate": 0.35},
            {"min": 626350, "max": None, "rate": 0.37},
        ],
        "standard_deduction_married": 30000,
        "standard_deduction_single": 15000,
        "standard_deduction_additional_65_plus": 1600,
        "capital_gains_brackets_married": [
            {"min": 0, "max": 96700, "rate": 0.0},
            {"min": 96700, "max": 600050, "rate": 0.15},
            {"min": 600050, "max": None, "rate": 0.20},
        ],
        "niit_threshold_married": 250000,
        "niit_rate": 0.038,
        "social_security": {
            "max_taxable_earnings": 176100,
            "full_retirement_age": 67,
            "early_retirement_age": 62,
            "delayed_credits_per_year": 0.08,
            "max_benefit_at_fra_2025": 4018,
            "cola_2025": 0.025,
        },
        "retirement_contribution_limits": {
            "401k_employee": 23500,
            "401k_catch_up_50_plus": 7500,
            "401k_super_catch_up_60_63": 11250,
            "401k_total_limit": 70000,
            "ira_contribution": 7000,
            "ira_catch_up_50_plus": 1000,
            "hsa_family": 8550,
            "hsa_individual": 4300,
            "hsa_catch_up_55_plus": 1000,
        },
        "rmd": {
            "start_age": 73,
            "uniform_lifetime_table_sample": {
                72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7,
                77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2, 81: 19.4,
                82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
                87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2,
            },
            "note": "RMD = account_balance / distribution_period. Start age increases to 75 in 2033.",
        },
        "irmaa_brackets_married": [
            {"magi_max": 212000, "part_b_surcharge": 0, "part_d_surcharge": 0},
            {"magi_max": 266000, "part_b_surcharge": 70.0, "part_d_surcharge": 13.70},
            {"magi_max": 332000, "part_b_surcharge": 175.0, "part_d_surcharge": 35.50},
            {"magi_max": 398000, "part_b_surcharge": 280.0, "part_d_surcharge": 57.30},
            {"magi_max": 750000, "part_b_surcharge": 384.90, "part_d_surcharge": 79.00},
            {"magi_max": None, "part_b_surcharge": 419.30, "part_d_surcharge": 85.80},
        ],
        "medicare": {
            "part_b_standard_premium": 185.0,
            "part_a_premium": 0,
            "eligibility_age": 65,
        },
        "estate_tax": {
            "exemption_per_person": 13990000,
            "top_rate": 0.40,
            "annual_gift_exclusion": 19000,
            "note": "Exemption scheduled to drop to ~7M in 2026 if TCJA sunsets.",
        },
        "roth_conversion": {
            "no_income_limit": True,
            "5_year_rule": "Converted amounts penalty-free after 5 years or age 59.5",
            "backdoor_roth_allowed": True,
            "mega_backdoor_roth_limit": "Depends on employer plan; 401k total limit minus employee + employer contributions",
        },
        "qlac_limit_per_person": 200000,
        "qcd_limit_per_person": 105000,
        "qcd_eligible_age": 70.5,
    },
}


@tool
def lookup_tax_data(category: str, year: int = 2025, filing_status: str = "married") -> dict:
    """Look up current US tax law data for retirement planning.

    Provides tax brackets, deductions, contribution limits, RMD tables,
    IRMAA thresholds, Social Security parameters, estate tax rules, and more.

    Args:
        category: One of:
            - "brackets" — federal income tax brackets
            - "capital_gains" — long-term capital gains brackets + NIIT
            - "deductions" — standard deduction amounts
            - "social_security" — SS parameters (FRA, max benefit, COLA)
            - "contribution_limits" — 401k, IRA, HSA limits with catch-up
            - "rmd" — required minimum distribution rules and table
            - "irmaa" — Medicare IRMAA surcharge brackets
            - "medicare" — Medicare premiums and eligibility
            - "estate_tax" — estate/gift tax exemptions and rates
            - "roth" — Roth conversion rules and limits
            - "all" — return everything
        year: Tax year (default 2025).
        filing_status: "married" or "single" (affects brackets/deductions).

    Returns:
        A dict with the requested tax data, or an error dict.
    """
    data = _TAX_DATA.get(year)
    if not data:
        available = sorted(_TAX_DATA.keys())
        return {"error": f"No tax data for year {year}. Available: {available}"}

    if category == "all":
        return {"year": year, "filing_status": filing_status, **data}

    mapping = {
        "brackets": lambda: {
            "brackets": data[f"income_tax_brackets_{'married_filing_jointly' if filing_status == 'married' else 'single'}"],
            "filing_status": filing_status,
            "year": year,
        },
        "capital_gains": lambda: {
            "brackets": data.get("capital_gains_brackets_married", []),
            "niit_threshold": data.get("niit_threshold_married"),
            "niit_rate": data.get("niit_rate"),
            "year": year,
        },
        "deductions": lambda: {
            "standard_deduction": data[f"standard_deduction_{'married' if filing_status == 'married' else 'single'}"],
            "additional_65_plus": data["standard_deduction_additional_65_plus"],
            "filing_status": filing_status,
            "year": year,
        },
        "social_security": lambda: {"year": year, **data["social_security"]},
        "contribution_limits": lambda: {"year": year, **data["retirement_contribution_limits"]},
        "rmd": lambda: {"year": year, **data["rmd"]},
        "irmaa": lambda: {"year": year, "brackets": data["irmaa_brackets_married"]},
        "medicare": lambda: {"year": year, **data["medicare"]},
        "estate_tax": lambda: {"year": year, **data["estate_tax"]},
        "roth": lambda: {"year": year, **data["roth_conversion"], "qlac_limit": data["qlac_limit_per_person"], "qcd_limit": data["qcd_limit_per_person"], "qcd_eligible_age": data["qcd_eligible_age"]},
    }

    if category not in mapping:
        return {"error": f"Unknown category '{category}'. Valid: {', '.join(sorted(mapping.keys()))} or 'all'"}

    return mapping[category]()


# ---------------------------------------------------------------------------
# BLS Inflation Data (no API key required for v1)
# ---------------------------------------------------------------------------

@tool
def get_inflation_data(start_year: int = 2020, end_year: int = 2025) -> dict:
    """Get Consumer Price Index (CPI) inflation data from the Bureau of Labor Statistics.

    Uses the BLS Public Data API (v1, no key required) to retrieve CPI-U
    (All Urban Consumers, All Items) monthly data and compute annual inflation rates.

    Args:
        start_year: First year of data (default 2020). Max 3-year span without API key.
        end_year: Last year of data (default 2025).

    Returns:
        A dict with monthly CPI values and computed annual inflation rates,
        or a dict with 'error' describing the problem.
    """
    import json as _json
    import urllib.request
    import urllib.error

    # BLS v1 API limits to 3 years per request; chunk if needed
    BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
    SERIES_ID = "CUUR0000SA0"  # CPI-U All Urban Consumers, All Items, US City Average

    # Clamp span to 3 years per BLS v1 limit
    if end_year - start_year > 2:
        start_year = end_year - 2

    try:
        body = _json.dumps({
            "seriesid": [SERIES_ID],
            "startyear": str(start_year),
            "endyear": str(end_year),
        }).encode("utf-8")

        req = urllib.request.Request(
            BLS_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = _json.loads(resp.read().decode())

        if result.get("status") != "REQUEST_SUCCEEDED":
            return {"error": f"BLS API error: {result.get('message', 'Unknown error')}"}

        series = result.get("Results", {}).get("series", [])
        if not series:
            return {"error": "No CPI data returned from BLS"}

        data_points = series[0].get("data", [])

        # Organize by year
        yearly = {}
        for dp in data_points:
            year = int(dp["year"])
            period = dp["period"]
            if period == "M13":  # annual average
                yearly.setdefault(year, {})["annual_average"] = float(dp["value"])
            elif period.startswith("M"):
                month = int(period[1:])
                yearly.setdefault(year, {})[month] = float(dp["value"])

        # Compute annual inflation rates (Dec-to-Dec)
        annual_rates = {}
        sorted_years = sorted(yearly.keys())
        for i in range(1, len(sorted_years)):
            prev_year = sorted_years[i - 1]
            curr_year = sorted_years[i]
            prev_dec = yearly[prev_year].get(12)
            curr_dec = yearly[curr_year].get(12)
            if prev_dec and curr_dec:
                rate = round((curr_dec - prev_dec) / prev_dec * 100, 2)
                annual_rates[curr_year] = rate

        # Latest month available
        latest_year = sorted_years[-1] if sorted_years else end_year
        latest_months = {k: v for k, v in yearly.get(latest_year, {}).items() if isinstance(k, int)}
        latest_month = max(latest_months.keys()) if latest_months else None
        latest_cpi = latest_months.get(latest_month) if latest_month else None

        return {
            "series": "CPI-U All Urban Consumers, All Items",
            "start_year": start_year,
            "end_year": end_year,
            "annual_inflation_rates": annual_rates,
            "latest_cpi": {"year": latest_year, "month": latest_month, "value": latest_cpi},
            "source": "Bureau of Labor Statistics (bls.gov)",
        }

    except urllib.error.URLError as exc:
        return {"error": f"Network error reaching BLS API: {exc.reason}"}
    except Exception as exc:
        return {"error": f"Failed to get inflation data: {exc}"}


# ---------------------------------------------------------------------------
# US Treasury Yield Data (no API key required)
# ---------------------------------------------------------------------------

@tool
def get_treasury_yields(year: int = 2025) -> dict:
    """Get current US Treasury yield curve data from Treasury.gov.

    Returns daily Treasury yield rates for all maturities (1mo through 30yr).
    No API key required — data comes directly from the US Treasury.

    Useful for retirement planning to determine:
    - Bond allocation return assumptions
    - Risk-free rate for financial modeling
    - TIPS real yields for inflation-adjusted projections
    - Yield curve shape (normal vs inverted)

    Args:
        year: The year to fetch data for (default 2025).

    Returns:
        A dict with the latest yield curve data and summary,
        or a dict with 'error' describing the problem.
    """
    import csv
    import io
    import urllib.request
    import urllib.error

    url = (
        f"https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/daily-treasury-rates.csv/all/{year}"
        f"?type=daily_treasury_yield_curve&page&_format=csv"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RetirementPlanner/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)

        if not rows:
            return {"error": f"No Treasury yield data found for {year}"}

        # Column names vary slightly; map common ones
        maturity_keys = [
            ("1 Mo", "1_month"),
            ("2 Mo", "2_month"),
            ("3 Mo", "3_month"),
            ("4 Mo", "4_month"),
            ("6 Mo", "6_month"),
            ("1 Yr", "1_year"),
            ("2 Yr", "2_year"),
            ("3 Yr", "3_year"),
            ("5 Yr", "5_year"),
            ("7 Yr", "7_year"),
            ("10 Yr", "10_year"),
            ("20 Yr", "20_year"),
            ("30 Yr", "30_year"),
        ]

        def parse_row(row):
            result = {"date": row.get("Date", "")}
            for csv_key, out_key in maturity_keys:
                val = row.get(csv_key, "")
                if val and val.strip():
                    try:
                        result[out_key] = float(val)
                    except ValueError:
                        pass
            return result

        # Most recent data point (last row in CSV is newest)
        latest = parse_row(rows[-1])

        # Get a few recent data points for trend
        recent = [parse_row(r) for r in rows[-5:]]
        recent.reverse()  # newest first

        # Summary of key maturities
        key_yields = {}
        for label, key in [("3-Month", "3_month"), ("2-Year", "2_year"),
                           ("5-Year", "5_year"), ("10-Year", "10_year"),
                           ("30-Year", "30_year")]:
            if key in latest:
                key_yields[label] = latest[key]

        # Detect yield curve inversion
        y2 = latest.get("2_year", 0)
        y10 = latest.get("10_year", 0)
        spread_2_10 = round(y10 - y2, 3) if y2 and y10 else None
        inverted = spread_2_10 is not None and spread_2_10 < 0

        return {
            "year": year,
            "latest_date": latest.get("date", ""),
            "key_yields": key_yields,
            "full_curve": latest,
            "spread_2yr_10yr": spread_2_10,
            "yield_curve_inverted": inverted,
            "recent_trend": recent,
            "total_data_points": len(rows),
            "source": "US Department of the Treasury (treasury.gov)",
            "note": "Yields are annualized percentages. Use 10-Year as benchmark for long-term bond assumptions.",
        }

    except urllib.error.HTTPError as exc:
        return {"error": f"Treasury.gov HTTP {exc.code}: {exc.reason}"}
    except urllib.error.URLError as exc:
        return {"error": f"Network error reaching Treasury.gov: {exc.reason}"}
    except Exception as exc:
        return {"error": f"Failed to get Treasury yield data: {exc}"}



@tool
def search_web(query: str) -> str:
    """Search the web for current financial data, tax rules, market information, or any other topic.

    Use this when built-in tools (lookup_tax_data, get_inflation_data, get_treasury_yields)
    don't have the needed data. Returns top results with titles and snippets.

    Args:
        query: The search query string.

    Returns:
        A string with the top search results including titles, URLs, and snippets.
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results found for: {query}"
        lines = []
        for r in results:
            lines.append(f"**{r.get('title', '')}**")
            lines.append(f"  {r.get('href', '')}")
            lines.append(f"  {r.get('body', '')}")
            lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        return f"Web search failed: {exc}. Please proceed with available data."
