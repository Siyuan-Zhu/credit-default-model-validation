# Data dictionary and lineage

Three layers, all in `warehouse/credit.duckdb`:

```
data/raw/default_of_credit_card_clients.xls
  └─ raw.credit_card_clients          (src/cdmv/warehouse.py: parsed with xlrd, names lower-cased)
       └─ staging.stg_accounts         (sql/staging/010_stg_accounts.sql: one row per account)
            └─ staging.stg_account_months   (sql/staging/020_stg_account_months.sql: one row per account-month)
                 └─ features.account_features   (sql/features/010_account_features.sql: one row per account)
                      └─ features.model_split   (sql/features/020_model_split.sql)
                           └─ model.scores      (src/cdmv/train.py)
```

Month index: `m1` = September 2005 (most recent snapshot) through `m6` = April 2005. Amounts are in New Taiwan dollars.

## Raw: `raw.credit_card_clients`

Source column names from the UCI file, lower-cased; `default payment next month` becomes `default_payment_next_month`. Documented codes are from the UCI variable descriptions.

| Raw column | Source variable | Description | Documented codes |
| :--- | :--- | :--- | :--- |
| `id` | ID | Account identifier | |
| `limit_bal` | X1 | Credit limit, including supplementary cards | |
| `sex` | X2 | Sex | 1 male, 2 female |
| `education` | X3 | Education | 1 graduate school, 2 university, 3 high school, 4 others. Observed but undocumented: 0, 5, 6 |
| `marriage` | X4 | Marital status | 1 married, 2 single, 3 others. Observed but undocumented: 0 |
| `age` | X5 | Age in years | |
| `pay_0`, `pay_2` ... `pay_6` | X6-X11 | Repayment status, September (`pay_0`) back to April (`pay_6`); there is no `pay_1` | -1 pay duly, 1-8 months of delay, 9 nine or more months. Observed but undocumented: -2, 0 |
| `bill_amt1` ... `bill_amt6` | X12-X17 | Bill statement amount, September (`1`) back to April (`6`) | Negative values are credit balances |
| `pay_amt1` ... `pay_amt6` | X18-X23 | Amount paid, September (`1`) back to April (`6`) | |
| `default_payment_next_month` | Y | Default on the next payment | 1 yes, 0 no |

## Staging: `staging.stg_accounts`

| Column | Lineage | Rule |
| :--- | :--- | :--- |
| `account_id` | `id` | renamed |
| `credit_limit` | `limit_bal` | cast to double |
| `sex_code`, `age` | `sex`, `age` | passed through; not model inputs |
| `education_code_raw` | `education` | passed through |
| `education_code` | `education` | 0, 5, 6 mapped to 4 (others) |
| `education_code_undocumented` | `education` | true when the raw code is outside 1-4 |
| `marriage_code_raw` | `marriage` | passed through |
| `marriage_code` | `marriage` | 0 mapped to 3 (others) |
| `marriage_code_undocumented` | `marriage` | true when the raw code is outside 1-3 |
| `pay_status_m1` ... `pay_status_m6` | `pay_0`, `pay_2` ... `pay_6` | renamed to a month index |
| `bill_amt_m1` ... `bill_amt_m6` | `bill_amt1` ... `bill_amt6` | renamed, cast to double |
| `pay_amt_m1` ... `pay_amt_m6` | `pay_amt1` ... `pay_amt6` | renamed, cast to double |
| `default_flag` | `default_payment_next_month` | renamed |

## Staging: `staging.stg_account_months`

The six monthly columns unpivoted to one row per account and month (180,000 rows).

| Column | Lineage | Rule |
| :--- | :--- | :--- |
| `account_id`, `credit_limit` | `stg_accounts` | |
| `month_index` | | 1 (September 2005) to 6 (April 2005) |
| `statement_month` | `month_index` | first day of the month |
| `pay_status` | `pay_status_m{k}` | |
| `months_delinquent` | `pay_status` | `greatest(pay_status, 0)`: codes -2, -1 and 0 count as not delinquent |
| `is_delinquent` | `pay_status` | `pay_status > 0` |
| `bill_amt`, `pay_amt` | `bill_amt_m{k}`, `pay_amt_m{k}` | |
| `utilization` | `bill_amt`, `credit_limit` | `bill_amt / credit_limit` (can be negative or above 1) |
| `prev_bill_amt` | `bill_amt` of month k+1 | the statement one month earlier; null for m6 |

## Features: `features.account_features`

One row per account. "In champion" refers to the current run; see `reports/tables/iv_and_selection.csv`.

| Feature | Definition | Lineage (raw columns) |
| :--- | :--- | :--- |
| `credit_limit` | Credit limit | `limit_bal` |
| `util_m1` | Latest bill / limit | `bill_amt1`, `limit_bal` |
| `avg_util_6m` | Mean of the six monthly utilisations | `bill_amt1-6`, `limit_bal` |
| `max_util_6m` | Highest monthly utilisation | `bill_amt1-6`, `limit_bal` |
| `util_change_6m` | Utilisation in m1 minus utilisation in m6 | `bill_amt1`, `bill_amt6`, `limit_bal` |
| `bill_growth_3m` | (bill m1 - bill m3) / limit | `bill_amt1`, `bill_amt3`, `limit_bal` |
| `n_months_over_limit_6m` | Months with bill above the limit | `bill_amt1-6`, `limit_bal` |
| `months_delinquent_m1` | Months past due in the latest month, `greatest(pay_status, 0)` | `pay_0` |
| `max_months_delinquent_6m` | Worst months past due over six months | `pay_0`, `pay_2-6` |
| `n_months_delinquent_6m` | Months with `pay_status > 0` | `pay_0`, `pay_2-6` |
| `current_delinq_streak` | Consecutive delinquent months counting back from m1 | `pay_0`, `pay_2-6` |
| `longest_delinq_streak_6m` | Longest run of consecutive delinquent months (gaps-and-islands) | `pay_0`, `pay_2-6` |
| `n_months_paid_duly_6m` | Months with status -1 (pay duly) | `pay_0`, `pay_2-6` |
| `n_months_status_neg2_6m` | Months with status -2 (undocumented) | `pay_0`, `pay_2-6` |
| `n_months_status_zero_6m` | Months with status 0 (undocumented) | `pay_0`, `pay_2-6` |
| `pay_ratio_m1` | Payment in m1 / bill in m2, capped at 1; 1 when the m2 bill is zero or negative | `pay_amt1`, `bill_amt2` |
| `avg_pay_ratio_5m` | Mean payment ratio over m1-m5 (same rule) | `pay_amt1-5`, `bill_amt2-6` |
| `min_pay_ratio_5m` | Lowest payment ratio over m1-m5 | `pay_amt1-5`, `bill_amt2-6` |
| `n_months_zero_payment_6m` | Months with a payment of zero | `pay_amt1-6` |
| `pay_to_limit_6m` | Total payments over six months / limit | `pay_amt1-6`, `limit_bal` |
| `default_flag` | Target | `default_payment_next_month` |

Assumption behind the payment ratios: the payment recorded in month m settles the statement issued one month earlier (m+1 in this index). The source does not state this.

Demographic fields (`sex`, `marriage`, `age`, `education`) stop at staging and are not features. See section 4.1 of `reports/model_validation.md`.

## Split: `features.model_split`

| Column | Rule |
| :--- | :--- |
| `account_id` | from `account_features` |
| `split` | within each `default_flag` class, accounts ordered by `md5(account_id || '-' || seed)`; first 60% train, next 20% validation, last 20% test |

## Model output: `model.scores`

| Column | Rule |
| :--- | :--- |
| `pd_champion` | WoE logistic regression PD |
| `pd_challenger` | Gradient boosting PD |
| `score_champion` | Scorecard score: 600 at 50:1 good:bad odds, 20 points to double the odds |
