-- Reconciliation and referential checks between raw and staging.
with n as (select count(*) as total from staging.stg_accounts)
select 'STG_01' as check_id, 'staging' as layer,
       'stg_accounts row count equals raw row count' as description, 'error' as severity,
       abs((select count(*) from raw.credit_card_clients) - (select total from n)) as failing_rows,
       (select total from n) as total_rows
union all
select 'STG_02', 'staging', 'Every raw ID has exactly one staging row and vice versa', 'error',
       (select count(*) from raw.credit_card_clients r
          full outer join staging.stg_accounts s on r.id = s.account_id
         where r.id is null or s.account_id is null)
     + (select count(*) - count(distinct account_id) from staging.stg_accounts),
       (select total from n)
union all
select 'STG_03', 'staging', 'Mapped EDUCATION in {1..4} and MARRIAGE in {1..3}', 'error',
       (select count(*) from staging.stg_accounts
         where education_code not in (1, 2, 3, 4) or marriage_code not in (1, 2, 3)),
       (select total from n)
union all
select 'STG_04', 'staging', 'stg_account_months has exactly 6 months per account', 'error',
       (select count(*) from (
            select account_id from staging.stg_account_months
            group by account_id having count(*) <> 6 or count(distinct month_index) <> 6))
     + abs((select count(distinct account_id) from staging.stg_account_months) - (select total from n)),
       (select total from n)
union all
select 'STG_05', 'staging', 'Monthly bill and payment totals reconcile to raw', 'error',
       case when abs((select sum(bill_amt) + sum(pay_amt) from staging.stg_account_months)
               - (select sum(bill_amt1 + bill_amt2 + bill_amt3 + bill_amt4 + bill_amt5 + bill_amt6
                           + pay_amt1 + pay_amt2 + pay_amt3 + pay_amt4 + pay_amt5 + pay_amt6)
                    from raw.credit_card_clients)) < 0.5
            then 0 else 1 end,
       (select total from n)
