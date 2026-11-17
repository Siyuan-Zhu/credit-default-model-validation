-- Documented versus observed category codes (UCI variable descriptions).
with codes as (
    select 'EDUCATION' as field, education as code from raw.credit_card_clients
    union all select 'MARRIAGE', marriage from raw.credit_card_clients
    union all select 'PAY_0..PAY_6', pay_0 from raw.credit_card_clients
    union all select 'PAY_0..PAY_6', pay_2 from raw.credit_card_clients
    union all select 'PAY_0..PAY_6', pay_3 from raw.credit_card_clients
    union all select 'PAY_0..PAY_6', pay_4 from raw.credit_card_clients
    union all select 'PAY_0..PAY_6', pay_5 from raw.credit_card_clients
    union all select 'PAY_0..PAY_6', pay_6 from raw.credit_card_clients
)
select
    field,
    code,
    count(*) as n_values,
    case
        when field = 'EDUCATION' and code = 1 then 'graduate school'
        when field = 'EDUCATION' and code = 2 then 'university'
        when field = 'EDUCATION' and code = 3 then 'high school'
        when field = 'EDUCATION' and code = 4 then 'others'
        when field = 'MARRIAGE' and code = 1 then 'married'
        when field = 'MARRIAGE' and code = 2 then 'single'
        when field = 'MARRIAGE' and code = 3 then 'others'
        when field = 'PAY_0..PAY_6' and code = -1 then 'pay duly'
        when field = 'PAY_0..PAY_6' and code between 1 and 8 then 'payment delay ' || code || ' month(s)'
        when field = 'PAY_0..PAY_6' and code = 9 then 'payment delay 9+ months'
        else 'undocumented'
    end as documented_meaning,
    case
        when field = 'EDUCATION' then code in (1, 2, 3, 4)
        when field = 'MARRIAGE' then code in (1, 2, 3)
        else code = -1 or code between 1 and 9
    end as documented
from codes
group by field, code
order by field, code
