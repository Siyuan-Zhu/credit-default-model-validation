-- Staging, one row per account and month (6 rows per account).
-- Unpivots the six monthly snapshots so that features can be computed with
-- ordinary aggregates and window functions.
--   pay_status: -2, -1, 0 (no delay; -2 and 0 are undocumented), 1..9 months late
--   months_delinquent: greatest(pay_status, 0)
--   prev_bill_amt: the statement one month earlier (m+1). The payment recorded
--   in month m is assumed to settle that earlier statement.
create schema if not exists staging;

create or replace table staging.stg_account_months as
with long as (
    select account_id, credit_limit, 1 as month_index, pay_status_m1 as pay_status, bill_amt_m1 as bill_amt, pay_amt_m1 as pay_amt from staging.stg_accounts
    union all
    select account_id, credit_limit, 2, pay_status_m2, bill_amt_m2, pay_amt_m2 from staging.stg_accounts
    union all
    select account_id, credit_limit, 3, pay_status_m3, bill_amt_m3, pay_amt_m3 from staging.stg_accounts
    union all
    select account_id, credit_limit, 4, pay_status_m4, bill_amt_m4, pay_amt_m4 from staging.stg_accounts
    union all
    select account_id, credit_limit, 5, pay_status_m5, bill_amt_m5, pay_amt_m5 from staging.stg_accounts
    union all
    select account_id, credit_limit, 6, pay_status_m6, bill_amt_m6, pay_amt_m6 from staging.stg_accounts
)
select
    account_id,
    month_index,
    -- statement month: m1 = 2005-09-01 ... m6 = 2005-04-01
    cast(date '2005-09-01' - to_months(month_index - 1) as date) as statement_month,
    pay_status,
    greatest(pay_status, 0)                     as months_delinquent,
    pay_status > 0                              as is_delinquent,
    bill_amt,
    pay_amt,
    credit_limit,
    bill_amt / credit_limit                     as utilization,
    lead(bill_amt) over (partition by account_id order by month_index) as prev_bill_amt
from long;
