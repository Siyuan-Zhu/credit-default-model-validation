-- Data-quality checks on the raw table. Each row is one check.
-- severity 'error' stops the pipeline when failing_rows > 0;
-- severity 'warn' records a known, documented issue that staging handles.
with r as (select * from raw.credit_card_clients),
n as (select count(*) as total from r)
select 'RAW_01' as check_id, 'raw' as layer,
       'Row count equals documented {{expected_rows}}' as description, 'error' as severity,
       abs((select total from n) - {{expected_rows}}) as failing_rows, (select total from n) as total_rows
union all
select 'RAW_02', 'raw', 'ID is unique', 'error',
       (select count(*) - count(distinct id) from r), (select total from n)
union all
select 'RAW_03', 'raw', 'No nulls in any of the 25 columns', 'error',
       (select count(*) from r where
            id is null or limit_bal is null or sex is null or education is null or marriage is null
         or age is null or pay_0 is null or pay_2 is null or pay_3 is null or pay_4 is null
         or pay_5 is null or pay_6 is null or bill_amt1 is null or bill_amt2 is null
         or bill_amt3 is null or bill_amt4 is null or bill_amt5 is null or bill_amt6 is null
         or pay_amt1 is null or pay_amt2 is null or pay_amt3 is null or pay_amt4 is null
         or pay_amt5 is null or pay_amt6 is null or default_payment_next_month is null),
       (select total from n)
union all
select 'RAW_04', 'raw', 'Target is 0 or 1', 'error',
       (select count(*) from r where default_payment_next_month not in (0, 1)), (select total from n)
union all
select 'RAW_05', 'raw', 'LIMIT_BAL is positive', 'error',
       (select count(*) from r where limit_bal <= 0), (select total from n)
union all
select 'RAW_06', 'raw', 'AGE between 18 and 100', 'error',
       (select count(*) from r where age not between 18 and 100), (select total from n)
union all
select 'RAW_07', 'raw', 'SEX in documented codes {1,2}', 'error',
       (select count(*) from r where sex not in (1, 2)), (select total from n)
union all
select 'RAW_08', 'raw', 'PAY_0..PAY_6 within observed range [-2, 9]', 'error',
       (select count(*) from r where
            pay_0 not between -2 and 9 or pay_2 not between -2 and 9 or pay_3 not between -2 and 9
         or pay_4 not between -2 and 9 or pay_5 not between -2 and 9 or pay_6 not between -2 and 9),
       (select total from n)
union all
select 'RAW_09', 'raw', 'PAY_AMT1..6 non-negative', 'error',
       (select count(*) from r where
            pay_amt1 < 0 or pay_amt2 < 0 or pay_amt3 < 0
         or pay_amt4 < 0 or pay_amt5 < 0 or pay_amt6 < 0),
       (select total from n)
union all
select 'RAW_10', 'raw', 'EDUCATION in documented codes {1,2,3,4} (0/5/6 mapped to 4 in staging)', 'warn',
       (select count(*) from r where education not in (1, 2, 3, 4)), (select total from n)
union all
select 'RAW_11', 'raw', 'MARRIAGE in documented codes {1,2,3} (0 mapped to 3 in staging)', 'warn',
       (select count(*) from r where marriage not in (1, 2, 3)), (select total from n)
union all
select 'RAW_12', 'raw', 'PAY_0..PAY_6 in documented codes {-1,1..9} (-2 and 0 are undocumented)', 'warn',
       (select count(*) from r where
            pay_0 in (-2, 0) or pay_2 in (-2, 0) or pay_3 in (-2, 0)
         or pay_4 in (-2, 0) or pay_5 in (-2, 0) or pay_6 in (-2, 0)),
       (select total from n)
union all
select 'RAW_13', 'raw', 'BILL_AMT1..6 non-negative (negative = credit balance)', 'warn',
       (select count(*) from r where
            bill_amt1 < 0 or bill_amt2 < 0 or bill_amt3 < 0
         or bill_amt4 < 0 or bill_amt5 < 0 or bill_amt6 < 0),
       (select total from n)
union all
select 'RAW_14', 'raw', 'BILL_AMT1..6 at or below LIMIT_BAL (above = over limit)', 'warn',
       (select count(*) from r where
            bill_amt1 > limit_bal or bill_amt2 > limit_bal or bill_amt3 > limit_bal
         or bill_amt4 > limit_bal or bill_amt5 > limit_bal or bill_amt6 > limit_bal),
       (select total from n)
