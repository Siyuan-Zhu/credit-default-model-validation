-- Model features, one row per account. Lineage for every column is in
-- docs/data_dictionary.md. Only behavioural fields (limit, bills, payments,
-- repayment status) are used; demographic fields stay in staging.
create schema if not exists features;

create or replace table features.account_features as
with months as (
    select
        *,
        -- payment ratio: share of the previous statement paid this month.
        -- No positive balance owed -> treated as fully paid (1.0). Capped at 1.
        case
            when prev_bill_amt is null then null
            when prev_bill_amt <= 0 then 1.0
            else least(pay_amt / prev_bill_amt, 1.0)
        end as pay_ratio
    from staging.stg_account_months
),
islands as (
    -- gaps-and-islands: consecutive delinquent months share the same grp value
    select
        account_id,
        month_index - row_number() over (partition by account_id order by month_index) as grp
    from months
    where is_delinquent
),
streaks as (
    select account_id, max(streak_len) as longest_delinq_streak_6m
    from (select account_id, grp, count(*) as streak_len from islands group by 1, 2)
    group by 1
),
agg as (
    select
        account_id,
        any_value(credit_limit)                                         as credit_limit,
        max(utilization) filter (where month_index = 1)                  as util_m1,
        avg(utilization)                                                 as avg_util_6m,
        max(utilization)                                                 as max_util_6m,
        max(utilization) filter (where month_index = 1)
          - max(utilization) filter (where month_index = 6)              as util_change_6m,
        (max(bill_amt) filter (where month_index = 1)
          - max(bill_amt) filter (where month_index = 3))
          / any_value(credit_limit)                                      as bill_growth_3m,
        count(*) filter (where bill_amt > credit_limit)                  as n_months_over_limit_6m,
        max(months_delinquent) filter (where month_index = 1)            as months_delinquent_m1,
        max(months_delinquent)                                           as max_months_delinquent_6m,
        count(*) filter (where is_delinquent)                            as n_months_delinquent_6m,
        coalesce(min(month_index) filter (where not is_delinquent), 7) - 1
                                                                         as current_delinq_streak,
        count(*) filter (where pay_status = -1)                          as n_months_paid_duly_6m,
        count(*) filter (where pay_status = -2)                          as n_months_status_neg2_6m,
        count(*) filter (where pay_status = 0)                           as n_months_status_zero_6m,
        max(pay_ratio) filter (where month_index = 1)                    as pay_ratio_m1,
        avg(pay_ratio)                                                   as avg_pay_ratio_5m,
        min(pay_ratio)                                                   as min_pay_ratio_5m,
        count(*) filter (where pay_amt = 0)                              as n_months_zero_payment_6m,
        sum(pay_amt) / any_value(credit_limit)                           as pay_to_limit_6m
    from months
    group by account_id
)
select
    a.account_id,
    a.credit_limit,
    a.util_m1,
    a.avg_util_6m,
    a.max_util_6m,
    a.util_change_6m,
    a.bill_growth_3m,
    a.n_months_over_limit_6m,
    a.months_delinquent_m1,
    a.max_months_delinquent_6m,
    a.n_months_delinquent_6m,
    a.current_delinq_streak,
    coalesce(s.longest_delinq_streak_6m, 0) as longest_delinq_streak_6m,
    a.n_months_paid_duly_6m,
    a.n_months_status_neg2_6m,
    a.n_months_status_zero_6m,
    a.pay_ratio_m1,
    a.avg_pay_ratio_5m,
    a.min_pay_ratio_5m,
    a.n_months_zero_payment_6m,
    a.pay_to_limit_6m,
    st.default_flag
from agg a
join staging.stg_accounts st using (account_id)
left join streaks s using (account_id)
order by a.account_id;
