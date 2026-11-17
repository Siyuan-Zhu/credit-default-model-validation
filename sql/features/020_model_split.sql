-- Stratified 60/20/20 train/validation/test split, done in SQL so it is
-- reproducible without Python's RNG. Accounts are ordered within each target
-- class by an md5 hash of the account id and a seed; md5 is stable across
-- DuckDB versions, unlike hash().
create schema if not exists features;

create or replace table features.model_split as
with ranked as (
    select
        account_id,
        default_flag,
        row_number() over (
            partition by default_flag
            order by md5(cast(account_id as varchar) || '-{{seed}}')
        ) as rn,
        count(*) over (partition by default_flag) as n
    from features.account_features
)
select
    account_id,
    case
        when rn <= round(0.6 * n) then 'train'
        when rn <= round(0.8 * n) then 'validation'
        else 'test'
    end as split
from ranked
order by account_id;
