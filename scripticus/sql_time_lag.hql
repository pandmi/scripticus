set hive.support.sql11.reserved.keywords=false;

select
	organization_id,
	organization_name,
	advertiser_name,
	campaign_id,
	campaign_name,
	time_lag,
	time_lag_bucket,
	sum(pv_conversions) as post_view_conversions,
	sum(pc_conversions) as post_click_conversions,
	sum(pv_conversions) + sum(pc_conversions) as total_conversions
from
	(
	select
		event_date as `date`,
		organization_id,
		advertiser_name,
		campaign_id,
		organization_name,
		campaign_name,
		event_timestamp_gmt as `timestamp`,
		-- if pv conversion, pc_time_lag is 0, if pc conversion, pv_time_lag is 0
		pv_time_lag + pc_time_lag as time_lag,
		case
			when (pv_time_lag + pc_time_lag)/60/60 < 1 then '(a) less than 1 hour'
			when (pv_time_lag + pc_time_lag)/60/60 between 1 and 3 then '(b) 1-3 hours'
			when (pv_time_lag + pc_time_lag)/60/60 between 3 and 12 then '(c) 3-12 hours'
			when (pv_time_lag + pc_time_lag)/60/60 between 12 and 24 then '(d) 12-24 hours'
			when (pv_time_lag + pc_time_lag)/60/60 between 24 and 48 then '(e) 1-2 days'
			when (pv_time_lag + pc_time_lag)/60/60 between 48 and 72 then '(f) 2-3 days'
			when (pv_time_lag + pc_time_lag)/60/60 between 72 and 96 then '(g) 3-4 days'
			when (pv_time_lag + pc_time_lag)/60/60 between 96 and 120 then '(h) 4-5 days'
			when (pv_time_lag + pc_time_lag)/60/60 between 120 and 168 then '(i) 5-7 days'
			when (pv_time_lag + pc_time_lag)/60/60 between 168 and 336 then '(j) 7-14 days'
			when (pv_time_lag + pc_time_lag)/60/60 >= 336 then '(k) over 2 weeks'
			else 'unknown'
		end as time_lag_bucket,
		case pv_pc_flag
			when 'V' then 1
			when 'C' then 0
		end as pv_conversions,
		case pv_pc_flag
			when 'V' then 0
			when 'C' then 1
		end as pc_conversions
	from
		mm_attributed_events
	where
		organization_id in (_organization_id_)
		and campaign_id in (_campaign_id_)
		and event_date between '_start_date_' and '_end_date_'
		and event_type = 'conversion'
	) hours
group by
	organization_id,
	organization_name,
	advertiser_name,
	campaign_id,
	campaign_name,
	time_lag,
	time_lag_bucket