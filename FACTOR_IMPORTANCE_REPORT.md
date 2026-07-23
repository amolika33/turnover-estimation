# Factor Importance Report

What actually drives each model's predictions, and by how much. Covers all
6 models across both pipelines (3 missions × estimation + forecasting),
the raw Beauhurst Financial Statement field inventory, a grant/accelerator
synthesis, and an investigation (not a build) into accelerator/grant
identity as a future feature.

**Methodology notes, read before the tables**:
- *Magnitude* for linear models (Lasso) is `|coefficient|` in log-turnover,
  standardised-feature space (scale-sensitive models pass numeric features
  through `StandardScaler`) — multiplicative effect size, not directly
  comparable to a raw-unit coefficient. For tree models (CatBoost, Extra
  Trees) it's `feature_importances_` (impurity/loss-reduction share,
  magnitude only, sums to ~1 across all features, no sign).
- *Relative magnitude* is each feature's magnitude as a % of the single
  top feature's magnitude, within that mission/model only — not comparable
  across missions or across estimation-vs-forecasting (different scales,
  different models entirely).
- *Direction* is exact (coefficient sign) for Lasso. For CatBoost/Extra
  Trees, there's no signed coefficient — direction below is a **Spearman
  rank correlation** between the feature's raw value and the target,
  computed separately and clearly marked "correlation-based (approximate)"
  — this is a real, verified statistical relationship, but it is not the
  same thing as the tree model's own attribution, and for a handful of
  grant/accelerator features it likely reflects company life-stage
  (smaller, earlier companies seek accelerators/funding more) rather than
  a causal "this helps/hurts turnover" relationship — flagged explicitly
  wherever it applies.
- Only features that reached a mission's own bake-off (i.e. were actually
  offered to that model) appear — the forecasting bake-off doesn't include
  any grant/accelerator/Source 3 feature at all (see Part 3).

---

## Part 1: Full ranked factor list, per mission

### 1.1 ACE — Estimation (`src/model_bakeoff.py`, selected model: **Lasso**)

Lasso's L1 penalty zeroed out every feature except 7 — the rest of the
71-feature design matrix (including every one-hot categorical column:
`sic_code_1`, `company_size`, `value_stream`, `employee_count_source`)
contributes exactly nothing to this model. This is itself a finding: **no
categorical/industry signal survives for ACE** under this model.

| Rank | Feature | What it is | Source | Direction | Relative magnitude |
|---|---|---|---|---|---|
| 1 | Balance sheet total assets | Company's total assets on the balance sheet, that year | Source 2: `Balance Sheet Total Assets {year}` | Increases turnover | **100%** (top factor) |
| 2 | Total employees | Headcount (filed or Beauhurst-estimated) | Source 2: `Total Employees (CH {year})`/`(Est. {year})` | Increases turnover | 20% |
| 3 | Company age | Years since founding | Source 2: `Founded` (derived: year − Founded) | Increases turnover | 10% |
| 4 | Debt fundraising signal | Boolean: company has raised debt financing (ever, as of export) | Source 3: `Growth signals - Debt fundraising` | **Decreases** turnover | 10% |
| 5 | Academic spinout flag | Boolean: company was spun out of a university | Source 3: `Academic Spinout Events 1/2 - Academic Institution Name` (derived: any non-null) | **Decreases** turnover | 8% |
| 6 | Fundraising count | Number of fundraising rounds completed (as of export) | Source 3: `Fundraisings - Number of fundraisings completed by the company` | **Decreases** turnover | 8% |
| 7 | Assets per employee | Efficiency ratio | Source 2 (derived: assets ÷ employees, both Source 2) | **Decreases** turnover | 4% |

**Reading this**: ACE's model is almost entirely a company-scale story
(assets, employees, age) — the 3 negative factors (debt fundraising,
academic spinout, fundraising count) are best read as "smaller/earlier-
stage company" markers, not literal turnover-reducing mechanisms; a
company that's still actively fundraising or spun out of a university
recently is systematically smaller, and Lasso is partly using these as a
proxy for company maturity/scale in the opposite direction from assets.

### 1.2 Beyond Earth — Estimation (`src/model_bakeoff.py`, selected model: **Lasso**)

Same story — 7 nonzero features out of 99; every categorical column
zeroed.

| Rank | Feature | What it is | Source | Direction | Relative magnitude |
|---|---|---|---|---|---|
| 1 | Balance sheet total assets | Total assets, that year | Source 2: `Balance Sheet Total Assets {year}` | Increases turnover | **100%** (top factor) |
| 2 | Assets per employee | Efficiency ratio | Source 2 (derived) | **Decreases** turnover | 15% |
| 3 | Total employees | Headcount | Source 2: `Total Employees (CH {year})`/`(Est. {year})` | Increases turnover | 6% |
| 4 | Company age | Years since founding | Source 2: `Founded` (derived) | Increases turnover | 2% |
| 5 | Accelerator count | Number of accelerator programmes attended | Source 3: `Accelerator Attendances 1-5 - Accelerator Name` (derived: count) | Increases turnover | 2% |
| 6 | Return on capital employed (%) | Operating profit ÷ capital employed — profitability ratio | Source 1: `Financial Statement 1 - Return on capital employed (%)` | Increases turnover | 1% |
| 7 | Academic spinout flag | Spun out of a university | Source 3 (derived) | **Decreases** turnover | 1% |

**Reading this**: dominated even more heavily by assets than ACE (assets
per employee is the #2 factor here, not #7) — Beyond Earth's turnover is
almost entirely explained by company scale, with everything else
(accelerator attendance, one financial ratio, academic-spinout status)
contributing marginal, single-digit-percent effects by comparison.

### 1.3 Resilient Earth — Estimation (`src/model_bakeoff.py`, selected model: **CatBoost**)

CatBoost's native categorical handling means categorical features
(`company_size`, `sic_code_1`, `value_stream`) survive here, unlike the
two Lasso missions — a direct, visible consequence of the earlier CatBoost
addition. Direction below is Spearman correlation (approximate — see
methodology note), not the model's own signed attribution.

| Rank | Feature | What it is | Source | Direction (correlational) | Relative magnitude |
|---|---|---|---|---|---|
| 1 | Balance sheet total assets | Total assets, that year | Source 2: `Balance Sheet Total Assets {year}` | + (ρ=0.90) | **100%** (top factor) |
| 2 | Company size | Beauhurst's own Micro/Small/Medium/Large bucket, that year | Source 2: `Size {year}` | n/a (categorical) | 63% |
| 3 | Company age | Years since founding | Source 2: `Founded` (derived) | + (ρ=0.49) | 19% |
| 4 | Total employees | Headcount | Source 2: `Total Employees (CH/Est. {year})` | + (ρ=0.84) | 18% |
| 5 | Assets per employee | Efficiency ratio | Source 2 (derived) | + (ρ=0.43) | 16% |
| 6 | Value Stream | The company's specific technical sub-domain within Resilient Earth | Source 2: `Value Stream` | n/a (categorical) | 11% |
| 7 | SIC Code 1 | Primary UK industry classification code | Source 2: `SIC Code 1` | n/a (categorical) | 9% |
| 8 | Accelerator count | Number of accelerator programmes attended | Source 3 (derived) | − (ρ=−0.29, likely life-stage, see Part 3) | 8% |
| 9 | Equity fundraising signal | Boolean: raised equity financing (ever) | Source 3: `Growth signals - Equity fundraising` | − (ρ=−0.35, likely life-stage) | 6% |
| 10 | Total export revenue | Export sales, that year | Source 2: `Total Export Revenue {year}` | + (ρ=0.77) | 5% |
| 11 | Grants count | Number of grants received | Source 3: `Grants - Number of grants received by the company` | + (ρ=0.29) | 5% |
| 12 | Export revenue per employee | Efficiency ratio | Source 2 (derived) | + (ρ=0.27) | 5% |
| 13 | Grants total amount | Total £ received via grants | Source 3: `Grants - Total amount received by the company through grants (GBP)` | ≈0 (ρ=−0.03, not significant) | 4% |
| 14 | Year | The panel year itself | Source 2 (panel year index) | ≈0 (ρ=0.00) | 4% |
| 15 | Fundraising count | Number of fundraising rounds | Source 3 (derived) | − (ρ=−0.59, likely life-stage) | 3% |

**Reading this**: the ONLY mission where categorical company classification
(size, value stream, SIC code) genuinely matters — combined, `company_size` +
`value_stream` + `sic_code_1` are worth roughly as much as total employees
and assets-per-employee combined. This is the direct payoff of CatBoost's
native categorical handling for this specific mission.

### 1.4 ACE — Forecasting (`forecast_src/forecast_bakeoff.py`, best non-benchmark ML candidate: **Extra Trees**; NOTE — Persistence, a benchmark with no features at all, is the model actually deployed for ACE's real 2030 trajectory. This table describes the best ML alternative, informative about what covariates carry one-year-ahead signal, not what's in production.)

| Rank | Feature | What it is | Source | Direction (correlational) | Relative magnitude |
|---|---|---|---|---|---|
| 1 | 2-year rolling mean turnover | Average of the last 2 real turnover years | Source 2 + Source 1 (derived from annualized `Total Turnover {year}`) | + (ρ=0.96) | **100%** (top factor) |
| 2 | Historical turnover max | Highest real turnover ever recorded for this company | Source 2 + Source 1 (derived) | + (ρ=0.93) | 89% |
| 3 | Current-year turnover | This year's own turnover | Source 2 + Source 1 (annualized) | + (ρ=0.97) | 78% |
| 4 | 3-year rolling mean turnover | Average of the last 3 real turnover years | Source 2 + Source 1 (derived) | + (ρ=0.96) | 77% |
| 5 | 3-year rolling median turnover | Median of the last 3 real turnover years | Source 2 + Source 1 (derived) | + (ρ=0.95) | 74% |
| 6 | Historical turnover min | Lowest real turnover ever recorded | Source 2 + Source 1 (derived) | + (ρ=0.89) | 44% |
| 7 | Company size = Large (flag) | Beauhurst's own size bucket, one-hot | Source 2: `Size {year}` | n/a (categorical) | 18% |
| 8 | Employees | Headcount | Source 2 | + (ρ=0.73) | 16% |
| 9 | Total assets | Balance sheet total assets | Source 2 | + (ρ=0.77) | 11% |
| 10 | SIC Code 1 = 61900 (flag) | "Other telecommunications activities" | Source 2: `SIC Code 1` | n/a (categorical) | 6% |
| 11 | 1-year-lag turnover | Turnover exactly one year before | Source 2 + Source 1 (annualized) | + (ρ=0.93) | 6% |
| 12 | SIC Code 1 = 72190 (flag) | "Other research and experimental development on natural sciences and engineering" | Source 2: `SIC Code 1` | n/a (categorical) | 3% |
| 13 | Company age | Years since founding | Source 2 (derived) | + (ρ=0.42) | 2% |
| 14 | Asset growth | Year-over-year log-difference in total assets | Source 2 (derived) | ≈0 (ρ=0.04, not significant) | 1% |
| 15 | Growth volatility | Rolling std of year-over-year turnover growth | Source 2 + Source 1 (derived) | − (ρ=−0.25) | 1% |

**Reading this**: an almost pure "recent scale + recent trajectory" story
— the top 6 factors are ALL variations on "how big has this company's
turnover recently been" (current, lag, rolling mean/median, historical
max/min), together worth roughly 4-5× everything else combined. This is
exactly consistent with the project's own finding that Persistence
(turnover next year = turnover this year) wins outright: even the best ML
alternative is mostly re-deriving "recent turnover level" from six
different angles.

### 1.5 Beyond Earth — Forecasting (best non-benchmark ML candidate: **Lasso**)

| Rank | Feature | What it is | Source | Direction | Relative magnitude |
|---|---|---|---|---|---|
| 1 | Current-year turnover | This year's own turnover | Source 2 + Source 1 (annualized) | Increases next-year turnover | **100%** (top factor) |
| 2 | Total assets | Balance sheet total assets | Source 2 | Increases | 12% |
| 3 | Employees | Headcount | Source 2 | Increases | 7% |
| 4 | 1-year-lag turnover | Turnover one year before | Source 2 + Source 1 (annualized) | Increases | 6% |
| 5 | Asset growth | YoY log-difference in total assets | Source 2 (derived) | Increases | 4% |
| 6 | Has employee data (flag) | Whether a real (not imputed) employee figure exists for this row | Source 2 (derived: missingness flag) | Increases | 3% |
| 7 | Historical turnover min | Lowest real turnover ever recorded | Source 2 + Source 1 (derived) | Increases | 2% |
| 8 | Employee growth | YoY log-difference in headcount | Source 2 (derived) | Increases | 2% |
| 9 | Growth volatility | Rolling std of YoY turnover growth | Source 2 + Source 1 (derived) | **Decreases** | 2% |
| 10 | History span (years) | Years between first and latest real turnover | Source 2 (derived) | Increases | 1% |
| 11 | 1-year log growth | Last year's YoY log-growth rate | Source 2 + Source 1 (derived) | Increases | 1% |
| 12 | Export revenue | Export sales, that year | Source 2 | Increases | <1% |
| 13 | Growth acceleration | Change in the growth rate itself | Source 2 + Source 1 (derived) | Increases | <1% |
| 14 | 3-year-lag turnover | Turnover three years before | Source 2 + Source 1 (annualized) | Increases | <1% |

**Reading this**: current-year turnover alone dwarfs everything else
(8× the #2 factor) — Beyond Earth's one-year-ahead forecast is
overwhelmingly "what did you make last year," with total assets a distant
second and every growth/volatility feature contributing only marginally.

### 1.6 Resilient Earth — Forecasting (best non-benchmark ML candidate: **Extra Trees**)

| Rank | Feature | What it is | Source | Direction (correlational) | Relative magnitude |
|---|---|---|---|---|---|
| 1 | Current-year turnover | This year's own turnover | Source 2 + Source 1 (annualized) | + (ρ=0.99) | **100%** (top factor) |
| 2 | 2-year rolling mean turnover | Average of the last 2 real turnover years | Source 2 + Source 1 (derived) | + (ρ=0.99) | 95% |
| 3 | 3-year rolling mean turnover | Average of the last 3 real turnover years | Source 2 + Source 1 (derived) | + (ρ=0.99) | 95% |
| 4 | Historical turnover max | Highest real turnover ever recorded | Source 2 + Source 1 (derived) | + (ρ=0.99) | 78% |
| 5 | 3-year rolling median turnover | Median of the last 3 real turnover years | Source 2 + Source 1 (derived) | + (ρ=0.98) | 76% |
| 6 | Historical turnover min | Lowest real turnover ever recorded | Source 2 + Source 1 (derived) | + (ρ=0.95) | 52% |
| 7 | Total assets | Balance sheet total assets | Source 2 | + (ρ=0.90) | 25% |
| 8 | 1-year-lag turnover | Turnover one year before | Source 2 + Source 1 (annualized) | + (ρ=0.98) | 13% |
| 9 | Employees | Headcount | Source 2 | + (ρ=0.84) | 5% |
| 10 | Has employee data (flag) | Whether a real employee figure exists for this row | Source 2 (derived) | + (ρ=0.38) | 4% |
| 11 | Company size = Large (flag) | Beauhurst's own size bucket | Source 2: `Size {year}` | n/a (categorical) | 4% |
| 12 | Employee growth | YoY log-difference in headcount | Source 2 (derived) | − (ρ=−0.13) | 1% |
| 13 | Assets per employee | Efficiency ratio | Source 2 (derived) | + (ρ=0.42) | 1% |
| 14 | Company size = Medium (flag) | Beauhurst's own size bucket | Source 2: `Size {year}` | n/a (categorical) | 1% |
| 15 | SIC Code 1 = 62090 (flag) | "Other information technology service activities" | Source 2: `SIC Code 1` | n/a (categorical) | 1% |

**Reading this**: virtually identical shape to ACE's forecasting table —
recent-turnover-level features dominate almost completely (top 6 factors
all turnover-derived, worth 4-8× everything else). The two forecasting
missions with an ML runner-up (ACE, Resilient Earth) tell the same story:
one-year-ahead turnover is overwhelmingly explained by recent turnover
itself, which is exactly why Persistence — predicting no change at all —
is so hard to beat.

---

## Part 2: Beauhurst Financial Statement fields — used vs never used

Each Financial Statement block (1-10, repeated per filing) has **102 base
fields** (verified directly against the raw export — not the ~110
originally estimated). Of these:

| Category | Count | Fields |
|---|---|---|
| **Used as a feature** | 9 | `Current ratio`, `Liquidity acid test`, `Gearing (%)`, `Equity (%)`, `Current debt ratio`, `Total debt ratio`, `Return on capital employed (%)`, `Return on total assets employed (%)`, `Return on net assets employed (%)` |
| **Used structurally (not a model feature, but essential)** | 3 | `Turnover` (base value for the annualization fix and the turnover-derivation leakage check every excluded ratio was tested against), `Number of weeks in the accounting year` (the annualization factor itself — **CRITICAL**, see `PROJECT_NOTES.md`'s "Filing-period annualization"), `Date of accounts` (anchors each statement to a calendar year — required for both the annualization merge and the Statement-1-is-most-recent confirmation) |
| **Evaluated and excluded (turnover-derivation leakage or definitional)** | 6 | `Pretax profit margin (%)`, `Debtor days`, `Creditor days`, `Exports turnover ratio (%)`, `Sales networking capital`, `Stock turnover ratio (%)` — see `DATA_SCHEMA.md`'s "Source 1 financial ratios" for the exact reconstruction formula that got each one excluded |
| **Never evaluated at all** | **84** | See full list below |

**The 84 never-used fields** (P&L detail, balance-sheet line items, cash
flow, and administrative fields — none of these have been explored as
feature candidates this pass, positive or negative):

`Accounts currency`, `Accounts are consolidated?`, `Pretax profit`,
`Profit after tax`, `Direct exports from country of incorporation`, `Cost
of sales`, `Gross profit`, `Wages & salaries`, `Director emoluments`,
`Operating profit`, `Depreciation`, `Audit fees`, `Non Audit fees`,
`Interest payments`, `Taxation`, `Dividends payable`, `Retained profit`,
`Employee wages total`, `Social security`, `Capitalised staff costs`,
`Pension costs`, `Other staff costs`, `Amortisation`, `EBITDA`, `Interest
receivable`, `Net interest`, `Cash`, `Tangible assets`, `Intangible
assets`, `Total fixed assets`, `Stock`, `Trade debtors`, `Other debtors`,
`Miscellaneous current assets`, `Total current assets`, `Trade
creditors`, `Short term bank loans`, `Long term bank loans`, `Bank loans
and overdrafts`, `Other short-term finance`, `Miscellaneous current
liabilities`, `Total current liabilities`, `Other non-current
liabilities`, `Bank loans & overdrafts & long-term liabilities`, `Other
long-term finance`, `Pension liabilities`, `Total long-term liabilities`,
`Investments and other fixed assets`, `Investments/Current assets`,
`Director loans`, `Group subsidiary loans`, `Prepayments accrued`, `Total
debtors`, `Deferred tax`, `Called-up share capital`, `P&L account
reserve`, `Revaluation reserve`, `Sundry reserves`, `Shareholder funds`,
`Share premium`, `Other reserves`, `Net worth`, `Working capital`, `Total
assets` (FS-block's own copy — distinct from Source 2's
`Balance Sheet Total Assets`, which IS used), `Total liabilities`, `Net
assets`, `Net cash flow from operations`, `Net cash flow before
financing`, `Net cash flow from financing`, `Increase in cash`,
`Acquisition and disposal`, `Management and liquid resources`, `Capital
expenditure`, `Equity dividends paid`, `Investing activities`, `Net cash
flow from investments`, `R&D expenditure`, `Number of employees` (FS-block's
own copy — distinct from Source 2's `Total Employees`, which IS used),
`Auditors`, `Senior Auditor`, `GVA`, `Contingent liability`, `Capital
employed`, `Auditor comments`.

**Practical implication for the adjacent-company pull**: only 12 of Source
1's 102 base Financial-Statement fields have any active role in either
pipeline (9 features + 3 structural). The other 90 (84 never-used + 6
excluded-for-leakage) can be safely skipped when sourcing adjacent
companies, **except** that `Turnover` and `Number of weeks in the
accounting year` (both in the "used structurally" tier) are absolutely
required regardless — without them, growth calculations for adjacent
companies would carry the same unannualized distortion this project's own
data had before the fix.

---

## Part 3: Grant/accelerator synthesis

Pulling out the 6 specifically-requested features across every mission
they were offered to (all are Source 3, all are ESTIMATION-only —
**the forecasting pipeline's feature set contains no grant/accelerator/
Source 3 feature at all**, confirmed directly against
`forecast_bakeoff.FEATURE_COLUMNS`; this is a real, structural asymmetry
between the two pipelines' feature sets, not an oversight to fix here).

| Feature | ACE (Lasso) | Beyond Earth (Lasso) | Resilient Earth (CatBoost) |
|---|---|---|---|
| `grants_count` | Zeroed out (no effect) | Zeroed out (no effect) | Rank 11, + (ρ=0.29), 5% relative |
| `grants_total_amount` | Zeroed out | Zeroed out | Rank 13, ≈0 (ρ=−0.03, not significant), 4% relative |
| `fundraising_count` | **Rank 6**, decreases turnover, 8% relative | Zeroed out | Rank 15, − (ρ=−0.59), 3% relative |
| `fundraising_total_amount` | Zeroed out | Zeroed out | Not in top 15 (importance 0.89, below rank 15's 0.94 cutoff — present but marginal) |
| `has_attended_accelerator` | Zeroed out | Zeroed out | Not in top 15 (importance 0.90 — present but marginal) |
| `accelerator_count` | Zeroed out | **Rank 5**, increases turnover, 2% relative | Rank 8, − (ρ=−0.29), 8% relative |

**Plain-language summary, per mission**:
- **ACE**: grants/accelerators essentially don't matter — the only
  survivor is fundraising count, and it points the "wrong" way (more
  fundraising rounds → lower predicted turnover). Read this as a
  company-life-stage proxy (companies still actively fundraising tend to
  be smaller/earlier-stage), not "fundraising hurts your turnover."
- **Beyond Earth**: only accelerator count survives, with a small positive
  effect (2% of the top factor's magnitude) — genuinely marginal either
  way.
- **Resilient Earth**: the only mission where grants show a plausible
  positive signal (grants count, +ρ=0.29) — more grants correlates with
  higher turnover here, unlike the other two features in this group
  (accelerator count, fundraising count) which both correlate negatively,
  again likely a life-stage effect (accelerators and multiple fundraising
  rounds both skew toward smaller/younger companies) rather than a genuine
  turnover driver.

**Overall**: across all 3 missions, there is no clear, consistent "having
a grant/accelerator helps" story — where these features survive at all,
roughly half point positive and half negative, and the negative ones are
plausibly explained by company life-stage rather than a causal
relationship. Grants/accelerators are, at most, weak, mission-specific
signals — not a strong, reliable driver of predicted turnover anywhere in
either pipeline.

---

## Part 4: Specific accelerator/grant identity — investigation only, no code changes

### Grant-program identity: does not exist

Checked both Source 1 and Source 3 for a grant-programme/type/name field
(the equivalent of an accelerator's name). **No such field exists in
either file** — Source 1 and Source 3 both carry exactly the same 5
grant-related columns: `Grants - Number of grants received by the
company`, `Grants - Total amount received by the company through grants
(GBP)`, `Grants - Amount received by the company in its latest grant
(GBP)`, `Grants - Date of the company's latest grant`, `Grants - Date of
the company's earliest grant` — all aggregate counts/amounts/dates, never
a specific scheme name (e.g. "Innovate UK Smart Grant", "SBRI"). This
isn't a sparsity problem like `linkedin_industry` — **the data simply
isn't collected at this level of detail by Beauhurst**. Recommend against
pursuing grant-programme identity as a feature: there is nothing to pull.

### Accelerator identity: exists, and looks genuinely promising

Source 3 has 5 numbered "Accelerator Attendances N - Accelerator Name"
slots (Source 1's July 9 export only has a single, un-numbered slot — the
5-slot structure is Source 3-specific). Checked against real data:

- **261 of 1,372 companies (19%)** have attended at least one accelerator.
- **133 distinct accelerator names**, but concentration is real: **6
  accelerators have ≥10 distinct companies each**, accounting for the
  clear majority of all attendances. **23 have ≥5 companies.** 68 (about
  half of the 133) have exactly 1 company ever — a long tail, as expected.

| Accelerator | Distinct companies |
|---|---|
| UK Space Agency Accelerator | 93 |
| European Space Agency Business Incubation Centre (ESA BIC UK) | 89 |
| Seraphim Space Camp | 21 |
| Fusion Connect With Capital | 19 |
| UK Space Agency Accelerator GovBridge | 15 |
| Mayor's International Business Programme | 10 |

**This is NOT as sparse as `linkedin_industry` was** (158 categories,
mostly singleton, no meaningful bucketing possible). Here, a genuinely
small number (6, or generously 23) of categories cover a meaningful share
of the attending population, and — unlike a generic external LinkedIn
tag — these are directly, specifically relevant to the space sector: UK
Space Agency Accelerator and ESA BIC UK together cover **158 distinct
companies** (union, not sum — 24 companies attended both), i.e. over
60% of the 261 attending companies from just these two programmes.

**Proposed feature, if pursued** (not built — this is the investigation
the task asked for): boolean flags for the top 6 (`attended_uksa_accelerator`,
`attended_esa_bic_uk`, `attended_seraphim_space_camp`, etc.), or a single
categorical `primary_accelerator` column bucketing everything below the
top 6 (or top 23) into an "other"/"infrequent" category — the same
`min_frequency`-style bucketing already used for `sic_code_1` (see
`ASSUMPTIONS_REGISTER.md` #12).

**Risks to flag explicitly before building this**:
- **Leakage**: none identified — accelerator attendance is a real-world
  event, not a value computed from turnover, so it doesn't fail this
  project's no-turnover-derivation rule the way the excluded financial
  ratios did.
- **Confounding/selection effect, not leakage but still real**: Part 3
  above already found `accelerator_count` correlates *negatively* with
  turnover in 2 of 3 missions (likely because accelerators skew toward
  smaller/earlier-stage companies) — a specific-accelerator feature would
  likely inherit the same life-stage confound, just split finer. Worth
  checking directly once built, not assumed away.
- **Coverage is still modest**: even the top accelerator (93 companies)
  covers only 6.8% of Source 3's 1,372-company universe — useful as a
  supplementary signal, not a primary driver, given how few companies it
  actually applies to.
- **Multi-attendance companies**: a company can attend more than one
  accelerator (565 attendance-slot records vs 535 distinct company-
  accelerator pairs vs 261 attending companies) — a one-hot/boolean design
  needs to decide whether multiple simultaneous flags are acceptable (yes,
  in principle — a company can genuinely have attended both UK Space
  Agency Accelerator and ESA BIC UK) or whether a single "which one"
  categorical forces an arbitrary pick.

**Recommendation (superseded by the confound check below — see that
section's revised recommendation)**: worth pursuing as a follow-on
feature-engineering pass (not urgent — Part 3 already shows the existing
aggregate accelerator features carry only marginal, inconsistent signal)
— but do it with the top-6-or-top-23 bucketed approach, not a raw
133-category one-hot, and re-check the life-stage confound once built
rather than assuming the finer-grained version behaves differently from
the aggregate version.

### Life-stage confound check, done *before* building (checked directly, not assumed)

The life-stage confound flagged above as a risk to "check directly once
built" was instead checked first, per instruction, before writing any
feature code. For each of the 3 real missions, split all companies
(labelled + inference population together, so the comparison isn't
restricted to the ~367 companies with observed turnover) into "attended
at least one of the 6 well-represented accelerators" (109 distinct
companies total) vs. "did not," and compared `company_age_years`
(`2026 - Founded`) and forecast evidence group (A = 3+ real turnover
years, B = 2, C = 1, D = 0 — the forecasting pipeline's own data-quality
tiering, `data/processed/forecast_evidence_groups.csv`):

| Mission | Attendees: mean / median age | Non-attendees: mean / median age | Attendees in group D (no observed turnover) | Non-attendees in group D |
|---|---|---|---|---|
| ACE | 8.2 / 9 yrs (n=14) | 26.0 / 20 yrs (n=181) | 85.7% | 56.9% |
| Beyond Earth | 8.1 / 7 yrs (n=52) | 39.1 / 27.5 yrs (n=406) | 94.2% | 51.1% |
| Resilient Earth | 6.2 / 5 yrs (n=25) | 24.4 / 13 yrs (n=238) | 92.0% | 65.4% |

**The skew is real, large, and consistent across all 3 missions.**
Attendees of the 6 well-represented accelerators are roughly a **third to
a quarter the age** of non-attendees (medians: 9 vs 20, 7 vs 27.5, 5 vs
13), and are disproportionately concentrated in evidence group D — the
tier with **zero** observed turnover years at all, i.e. the inference-only
population a model never trains against. In Beyond Earth, 94% of
attendees have no observed turnover whatsoever, vs. 51% of non-attendees.

This confirms the exact concern flagged as a risk above, and more sharply
than the aggregate `accelerator_count` feature already showed it: a
specific-accelerator flag built today would be learning almost entirely
from the small, young, evidence-poor slice of each mission, and — because
so few attendees have any real turnover outcome to train against in the
first place — would have very little genuine (accelerator, turnover)
signal available to learn from independent of age. It would very likely
be re-encoding "young, early-stage, no filing history yet" rather than
telling the model something new about that specific programme.

**Revised recommendation: do not build the specific-accelerator feature
now.** The confound isn't a modest risk to monitor after the fact — it's
large enough, and the attendee population thin enough on real turnover
outcomes, that the feature is unlikely to add signal beyond what
`company_age_years` and the existing evidence-group tiering already
capture. Revisit only if the labelled (observed-turnover) population
grows enough for a specific accelerator to have a meaningful number of
non-D-group attendees to learn from.
