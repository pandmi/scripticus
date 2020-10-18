select
	organization_id,
	campaign_id,
	campaign_name,
	country,
	zip_code,
	sum(impressions) as impressions,
	sum(clicks) as clicks,
	sum(pv_conversions) + sum(pc_conversions) as total_conversions,
	sum(mm_v1_revenue) as mm_v1_revenue,
	sum(total_spend) as total_spend
from
	(
	select
		impression_date as `date`,
		organization_id,
		campaign_name,
		country,
		campaign_id,
		zip_code,
		timestamp_gmt as `timestamp`,
		mm_uuid as mm_uuid,
		total_spend_cpm/1000 as total_spend,
		media_cost_cpm/1000 as media_cost,
		total_ad_cost_cpm/1000 as total_ad_cost,
		1 as impressions,
		0 as clicks,
		0 as pv_conversions,
		0 as pc_conversions,
		0 as mm_v1_revenue,
        0 as vst,
        0 as q1,
        0 as q2,
        0 as q3,
        0 as q4
	from
		mm_impressions
	where
		organization_id in (_organization_id_)
		and campaign_id in (_campaign_id_)
		and impression_date between '_start_date_' and '_end_date_'

	union all

	select
		event_date as `date`,
		organization_id,
		campaign_name,
		country,
		campaign_id,
		zip_code,
		event_timestamp_gmt as `timestamp`,
		null as mm_uuid,
		0 as total_spend,
		0 as media_cost,
		0 as total_ad_cost,
		0 as impressions,
		case event_type
			when 'click' then 1
			when 'conversion' then 0
		end as clicks,
		case event_type
			when 'click' then 0
			when 'conversion' then
			case pv_pc_flag
				when 'V' then 1
				when 'C' then 0
			end
		end as pv_conversions,
		case event_type
			when 'click' then 0
			when 'conversion' then
			case pv_pc_flag
				when 'V' then 0
				when 'C' then 1
			end
		end as pc_conversions,
		cast(regexp_extract(mm_v1,'^([0-9]+(\.[0-9]{2})?)',1) as double) as mm_v1_revenue,
        case event_type
            when 'click' then 0
            when 'conversion' then 0
            when 'video' then
            case event_subtype
                when 'vst' then 1
                when 'q1' then 0
                when 'q2' then 0
                when 'q3' then 0
                when 'q4' then 0
            end
        end as vst,
        case event_type
            when 'click' then 0
            when 'conversion' then 0
            when 'video' then
            case event_subtype
                when 'vst' then 0
                when 'q1' then 1
                when 'q2' then 0
                when 'q3' then 0
                when 'q4' then 0
            end
        end as q1,
        case event_type
            when 'click' then 0
            when 'conversion' then 0
            when 'video' then
            case event_subtype
                when 'vst' then 0
                when 'q1' then 0
                when 'q2' then 1
                when 'q3' then 0
                when 'q4' then 0
            end
        end as q2,
        case event_type
            when 'click' then 0
            when 'conversion' then 0
            when 'video' then
            case event_subtype
                when 'vst' then 0
                when 'q1' then 0
                when 'q2' then 0
                when 'q3' then 1
                when 'q4' then 0
            end
        end as q3,
        case event_type
            when 'click' then 0
            when 'conversion' then 0
            when 'video' then
            case event_subtype
                when 'vst' then 0
                when 'q1' then 0
                when 'q2' then 0
                when 'q3' then 0
                when 'q4' then 1
            end
        end as q4
	from
		mm_attributed_events
	where
		organization_id in (_organization_id_)
		and campaign_id in (_campaign_id_)
		and event_date between '_start_date_' and '_end_date_'
	) all_st
group by
	organization_id,
	campaign_id,
	campaign_name,
	country,
	zip_code
