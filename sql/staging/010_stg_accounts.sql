-- Staging, one row per account.
-- Renames source columns, keeps the raw category codes, and maps codes that the
-- UCI documentation does not define (EDUCATION 0/5/6, MARRIAGE 0) to the
-- documented "others" code, with a flag so the mapping stays visible.
-- Month index: m1 = September 2005 (most recent) ... m6 = April 2005.
create schema if not exists staging;

create or replace table staging.stg_accounts as
select
    id                                              as account_id,
    cast(limit_bal as double)                       as credit_limit,
    sex                                             as sex_code,
    age,
    education                                       as education_code_raw,
    case when education in (1, 2, 3, 4) then education else 4 end
                                                    as education_code,
    education not in (1, 2, 3, 4)                   as education_code_undocumented,
    marriage                                        as marriage_code_raw,
    case when marriage in (1, 2, 3) then marriage else 3 end
                                                    as marriage_code,
    marriage not in (1, 2, 3)                       as marriage_code_undocumented,
    pay_0 as pay_status_m1, pay_2 as pay_status_m2, pay_3 as pay_status_m3,
    pay_4 as pay_status_m4, pay_5 as pay_status_m5, pay_6 as pay_status_m6,
    cast(bill_amt1 as double) as bill_amt_m1, cast(bill_amt2 as double) as bill_amt_m2,
    cast(bill_amt3 as double) as bill_amt_m3, cast(bill_amt4 as double) as bill_amt_m4,
    cast(bill_amt5 as double) as bill_amt_m5, cast(bill_amt6 as double) as bill_amt_m6,
    cast(pay_amt1 as double) as pay_amt_m1, cast(pay_amt2 as double) as pay_amt_m2,
    cast(pay_amt3 as double) as pay_amt_m3, cast(pay_amt4 as double) as pay_amt_m4,
    cast(pay_amt5 as double) as pay_amt_m5, cast(pay_amt6 as double) as pay_amt_m6,
    default_payment_next_month                      as default_flag
from raw.credit_card_clients;
