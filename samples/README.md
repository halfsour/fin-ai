# Sample Financial Data

Realistic sample data for testing the Retirement Planner. All data is fictional.

## The Chen Family (as of March 2, 2026)

- **James Chen**, age 51 (born 1974-06-15) — Software engineer
- **Sarah Chen**, age 49 (born 1976-11-22) — Marketing director
- **Emma Chen**, age 17 (born 2008-03-10)
- **Lucas Chen**, age 14 (born 2011-09-28)
- Location: Redmond, WA (ZIP 98052)

## Financial Summary

| Category | Amount |
|----------|--------|
| Combined gross income | ~$20,300/mo ($243,600/yr) |
| Investment accounts | ~$1,691,000 |
| Bank accounts | ~$135,500 |
| Credit card debt | ~$6,000 |
| Estimated net worth | ~$1,820,500 |

## Files

| File | Description |
|------|-------------|
| `personal_info.json` | Birthdates, ZIP code, and state |
| `spending_transactions.csv` | 3 years of transactions (Jan 2023 – Feb 2026), Rocket Money export style |
| `investment_statement.csv` | Brokerage statement with 401(k), IRA, taxable, and HSA accounts |
| `bank_statements.csv` | Checking and savings account summary |
| `credit_cards.csv` | Credit card balances and payment info |

## Usage

### Web UI

1. Start the server: `uvicorn retirement_planner.web:app --reload` (or `docker compose up`)
2. Open http://localhost:8000
3. Upload files using the data input panel or the 📎 chat button
4. Enter birthdates from `personal_info.json` in the personal info form
5. Review assumptions and run the assessment

### CLI

```bash
# Load all financial files at once
python -m retirement_planner \
  --investments samples/investment_statement.csv \
  --banking samples/bank_statements.csv \
  --credit-cards samples/credit_cards.csv

# Or use the smart file loader
python -m retirement_planner --files samples/investment_statement.csv samples/bank_statements.csv samples/credit_cards.csv
```

When prompted for personal info, enter:
- Husband birthdate: 1974-06-15
- Wife birthdate: 1976-11-22
- Children birthdates: 2008-03-10, 2011-09-28

## Data Notes

- Spending transactions show seasonal patterns: higher travel in summer, holiday shopping in December
- Income reflects annual raises (~4-5% per year) across the 3-year period
- Investment accounts include a mix of index funds, individual stocks, bonds, and target-date funds
- The Schwab taxable account holds individual tech stocks (AAPL, MSFT, GOOGL) and ETFs
- The HSA is fully invested in VFIAX (Vanguard 500 Index)
- Credit cards are paid in full monthly (no revolving debt except current statement balances)
