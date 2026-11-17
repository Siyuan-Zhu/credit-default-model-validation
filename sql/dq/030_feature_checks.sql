-- Checks on the model feature table and the train/validation/test split.
with f as (select * from features.account_features),
n as (select count(*) as total from f)
select 'FEA_01' as check_id, 'features' as layer,
       'account_features row count equals stg_accounts' as description, 'error' as severity,
       abs((select count(*) from staging.stg_accounts) - (select total from n)) as failing_rows,
       (select total from n) as total_rows
union all
select 'FEA_02', 'features', 'Every feature row joins to one staging account (unique key)', 'error',
       (select count(*) from f left join staging.stg_accounts s using (account_id) where s.account_id is null)
     + (select count(*) - count(distinct account_id) from f),
       (select total from n)
union all
select 'FEA_03', 'features', 'No null or non-finite model features', 'error',
       (select count(*) from f where
            not isfinite(credit_limit) or not isfinite(util_m1) or not isfinite(avg_util_6m)
         or not isfinite(max_util_6m) or not isfinite(util_change_6m) or not isfinite(bill_growth_3m)
         or not isfinite(pay_ratio_m1) or not isfinite(avg_pay_ratio_5m) or not isfinite(min_pay_ratio_5m)
         or not isfinite(pay_to_limit_6m)
         or credit_limit is null or util_m1 is null or avg_util_6m is null or max_util_6m is null
         or util_change_6m is null or bill_growth_3m is null or n_months_over_limit_6m is null
         or months_delinquent_m1 is null or max_months_delinquent_6m is null
         or n_months_delinquent_6m is null or current_delinq_streak is null
         or longest_delinq_streak_6m is null or n_months_paid_duly_6m is null
         or n_months_status_neg2_6m is null or n_months_status_zero_6m is null
         or pay_ratio_m1 is null or avg_pay_ratio_5m is null or min_pay_ratio_5m is null
         or n_months_zero_payment_6m is null or pay_to_limit_6m is null or default_flag is null),
       (select total from n)
union all
select 'FEA_04', 'features', 'Payment ratios within [0, 1]', 'error',
       (select count(*) from f where
            pay_ratio_m1 not between 0 and 1 or avg_pay_ratio_5m not between 0 and 1
         or min_pay_ratio_5m not between 0 and 1),
       (select total from n)
union all
select 'FEA_05', 'features', 'Month counts within [0, 6] and status counts sum to 6', 'error',
       (select count(*) from f where
            n_months_delinquent_6m not between 0 and 6 or n_months_zero_payment_6m not between 0 and 6
         or n_months_over_limit_6m not between 0 and 6
         or n_months_delinquent_6m + n_months_paid_duly_6m + n_months_status_neg2_6m
            + n_months_status_zero_6m <> 6),
       (select total from n)
union all
select 'FEA_06', 'features', 'Streak logic: current <= longest <= delinquent months', 'error',
       (select count(*) from f where
            current_delinq_streak > longest_delinq_streak_6m
         or longest_delinq_streak_6m > n_months_delinquent_6m),
       (select total from n)
union all
select 'FEA_07', 'features', 'Every account is in exactly one split', 'error',
       abs((select count(*) from features.model_split) - (select total from n))
     + (select count(*) from f left join features.model_split m using (account_id) where m.split is null)
     + (select count(*) from features.model_split where split not in ('train', 'validation', 'test')),
       (select total from n)
union all
select 'FEA_08', 'features', 'Split default rates within 0.5 pp of overall (stratification)', 'error',
       (select count(*) from (
            select m.split, avg(f.default_flag) as dr
            from f join features.model_split m using (account_id) group by m.split)
         where abs(dr - (select avg(default_flag) from f)) > 0.005),
       (select total from n)
