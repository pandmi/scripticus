import requests  
import pandas as pd
import numpy as np
from io import StringIO
from io import BytesIO
import io as stringIOModule
import os
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import display
import json
from pandas import json_normalize
import http.client
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
import time
# import qds_sdk
import sqlalchemy
# from qds_sdk.commands import *
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.text import Text
import datetime
import matplotlib.dates as mdates
from pathlib import Path
from lxml import etree
import io
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

def t1_api_login(username,password,client_id,client_secret):
    response=requests.post('https://mediamath.auth0.com/oauth/token',
                                    data={'grant_type': 'password',
                                            'username': username,
                                            'password': password,
                                            'client_id': client_id,
                                            'client_secret': client_secret,
                                            'realm': 'MediaMathActiveDirectory'
                                            }) 
    resp=requests.get('https://api.mediamath.com/api/v2.0/session',
                                    headers={'Accept': 'application/vnd.mediamath.v1+json',
                                            'Authorization': 'Bearer {}'.format(response.json()['access_token'])
                                            })

    s = requests.session()
    x=response.json()['access_token']
    s.headers.update({'Accept': 'application/vnd.mediamath.v1+json','Authorization': 'Bearer {}'.format(response.json()['access_token'])})
    return resp, s, x


class T1_API():

    def __init__(self,username,password,client_id,client_secret):
        self.username=username
        self.password=password
        self.client_id=client_id
        self.client_secret=client_secret
        self.resp, self.session, self.token  = t1_api_login(self.username,self.password, self.client_id,self.client_secret)

    def t1_report(self,endpoint,*args,**kwargs): 
    # defining parameters
        params = {}
        for k in args:
            for v in k:
                if type(k[v]) is list:
                    params[v] = ','.join(k[v])
                else:
                    params[v] = k[v]
        for k in kwargs:
            if type(k) is list:
                params[k] = ','.join(kwargs[k])
            else:
                params[k] = kwargs[k]
    # creating a call
        picard = 'https://api.mediamath.com/reporting/v1/std/'
        data_url = picard + endpoint
        if endpoint == 'transparency': 
            endpoint = 'site_transparency'
        if endpoint == 'performance_uniques': 
            data_url = 'https://t1.mediamath.com/reporting/v1/std/performance' 
        if  endpoint in (
            'performance_usd',
            'performance_viewability',
            'site_transparency_viewability',
            'performance_aggregated',
            'performance_streaming',
            'performance'
            ):
            data_url = picard + endpoint
        elif endpoint == 'deals':
            data_url = 'https://api.mediamath.com/reporting-beta/v1/std/deals' 
        
        self.response = self.session.get(data_url, params=params, headers={'Accept-Encoding':'identity','Connection':'close'})
        data = self.response.content
        df = pd.read_csv(BytesIO(data))
        return df
     


    def campaign_meta_data(self, campaign_ids):
        camp_metadata = pd.DataFrame()
        for campaign_id in campaign_ids:
            version_data = []
            url_page = 'https://api.mediamath.com/api/v2.0/campaigns/{}'.format(campaign_id)
            strats_data = requests.get(url_page, cookies = self.resp.cookies).text
            st = etree.parse(io.StringIO(strats_data))
            ca_name = st.xpath('''.//prop[@name = 'name']/@value''')[0]
            ca_start_date = st.xpath('''.//prop[@name = 'start_date']/@value''')[0]
            ca_end_date = st.xpath('''.//prop[@name = 'end_date']/@value''')[0]
            ca_status = int(st.xpath('''.//prop[@name = 'status']/@value''')[0])
            ca_margin_pct = float(st.xpath('''.//prop[@name = 'margin_pct']/@value''')[0])
            ca_goal_type = st.xpath('''.//prop[@name = 'goal_type']/@value''')[0]
            ca_goal_value = float(st.xpath('''.//prop[@name = 'goal_value']/@value''')[0])
            ca_budget = float(st.xpath('''.//prop[@name = 'total_budget']/@value''')[0])
            ca_currency = st.xpath('''.//prop[@name = 'total_budget']/@currency_code''')[0]
            ca_impression_cap_type = st.xpath('''.//prop[@name = 'impression_cap_type']/@value''')[0]
            ca_impression_cap_automatic = st.xpath('''.//prop[@name = 'impression_cap_automatic']/@value''')[0]
            ca_spend_cap_automatic = st.xpath('''.//prop[@name = 'spend_cap_automatic']/@value''')[0]
            ca_spend_cap_type = st.xpath('''.//prop[@name = 'spend_cap_type']/@value''')[0]
            try:
                ca_spend_cap_amount = float(st.xpath('''.//prop[@name = 'spend_cap_amount']/@value''')[0])
            except:
                ca_spend_cap_amount = 0
            ca_frequency_type = st.xpath('''.//prop[@name = 'frequency_type']/@value''')[0]
            ca_frequency_interval = st.xpath('''.//prop[@name = 'frequency_interval']/@value''')[0]
            try:
                ca_frequency_amount = st.xpath('''.//prop[@name = 'frequency_amount']/@value''')[0]
            except:
                ca_frequency_amount = 0
            ca_frequency_optimization = int(st.xpath('''.//prop[@name = 'frequency_optimization']/@value''')[0])
            ca_suspicious_traffic_filter_level = int(st.xpath('''.//prop[@name = 'suspicious_traffic_filter_level']/@value''')[0])
            try:
                ca_merit_pixel_id = int(st.xpath('''.//prop[@name = 'merit_pixel_id']/@value''')[0])
            except:
                ca_merit_pixel_id = 0
            try:
                ca_pc_window_minutes = int(st.xpath('''.//prop[@name = 'pc_window_minutes']/@value''')[0])
            except:
                ca_pc_window_minutes = 0
            try:
                ca_pv_window_minutes = int(st.xpath('''.//prop[@name = 'pv_window_minutes']/@value''')[0])
            except:
                ca_pv_window_minutes = 0
     
            version_data_tmp = [campaign_id,
                                    ca_name,
                                    ca_start_date,
                                    ca_end_date,
                                    ca_status,
                                    ca_margin_pct,
                                    ca_goal_type,
                                    ca_goal_value,
                                    ca_budget,
                                    ca_currency,
                                    ca_impression_cap_type,
                                    ca_impression_cap_automatic,
                                    ca_spend_cap_automatic,
                                    ca_spend_cap_type,
                                    ca_spend_cap_amount,
                                    ca_frequency_type,
                                    ca_frequency_interval,
                                    ca_frequency_amount,
                                    ca_frequency_optimization,
                                    ca_suspicious_traffic_filter_level,
                                    ca_merit_pixel_id, 
                                    ca_pc_window_minutes,
                                    ca_pv_window_minutes]

            version_data.append(version_data_tmp)
            camp_metadata_tmp = pd.DataFrame(version_data, columns = ['campaign_id',
                                                                    'campaign_name',
                                                                    'campaign_start_date',
                                                                    'campaign_end_date',
                                                                    'campaign_status',
                                                                    'campaign_margin_pct',
                                                                    'campaign_goal_type',
                                                                    'campaign_goal_value',
                                                                    'campaign_budget',
                                                                    'campaign_currency',
                                                                    'campaign_impression_cap_type',
                                                                    'campaign_impression_cap_automatic',
                                                                    'campaign_spend_cap_automatic',
                                                                    'campaign_spend_cap_type',
                                                                    'campaign_spend_cap_amount',
                                                                    'campaign_frequency_type',
                                                                    'campaign_frequency_interval',
                                                                    'campaign_frequency_amount',
                                                                    'campaign_frequency_optimization',
                                                                    'campaign_suspicious_traffic_filter_level',
                                                                    'campaign_merit_pixel_id', 
                                                                    'campaign_pc_window_minutes',
                                                                    'camapign_pv_window_minutes'])
            
            if len(camp_metadata) == 0:
                camp_metadata = camp_metadata_tmp
            else:
                camp_metadata = pd.concat([camp_metadata, camp_metadata_tmp])

        camp_metadata_fin = camp_metadata[(camp_metadata['campaign_status'] == 1)]
        campaign_active_ids=camp_metadata_fin['campaign_id'].values        
        camp_pacing_medicine = camp_metadata_fin[['campaign_id','campaign_name','campaign_spend_cap_automatic', 'campaign_spend_cap_type',
                                            'campaign_spend_cap_amount','campaign_frequency_type','campaign_frequency_interval','campaign_frequency_amount']]
        return camp_pacing_medicine, campaign_active_ids
    
    def creative_meta_data(self, creative_ids):
        cr_metadata = pd.DataFrame()
        for creative_id in creative_ids:
            version_data = []
            url_page = 'https://api.mediamath.com/api/v2.0/atomic_creatives/{}'.format(creative_id)

            strats_data = requests.get(url_page, cookies = self.resp.cookies).text
            st = etree.parse(io.StringIO(strats_data))
            name = st.xpath('''.//prop[@name = 'name']/@value''')[0]
            created_on = st.xpath('''.//prop[@name = 'created_on']/@value''')[0]
            last_modified = st.xpath('''.//prop[@name = 'last_modified']/@value''')[0]
            status = int(st.xpath('''.//prop[@name = 'status']/@value''')[0])
            width = int(st.xpath('''.//prop[@name = 'width']/@value''')[0])
            height = int(st.xpath('''.//prop[@name = 'height']/@value''')[0])
            ad_format = st.xpath('''.//prop[@name = 'ad_format']/@value''')[0]
            is_https = st.xpath('''.//prop[@name = 'is_https']/@value''')[0]
            concept_id = int(st.xpath('''.//prop[@name = 'concept_id']/@value''')[0])
            rich_media = st.xpath('''.//prop[@name = 'rich_media']/@value''')[0]
            approval_status = st.xpath('''.//prop[@name = 'approval_status']/@value''')[0]
            rejected_reason = st.xpath('''.//prop[@name = 'rejected_reason']/@value''')[0]
            try:
                click_through_url = st.xpath('''.//prop[@name = 'click_through_url']/@value''')[0]
            except:
                click_through_url = 0

            is_mraid = float(st.xpath('''.//prop[@name = 'is_mraid']/@value''')[0])
            img_src=img_src=st.xpath('''.//prop[@name = 'tag']/@value''')


            version_data_tmp = [creative_id,
                                    name,
                                    created_on,
                                    last_modified,
                                    status,
                                    width,
                                    height,
                                    ad_format,
                                    is_https,
                                    concept_id,
                                    rich_media,
                                    approval_status,
                                    rejected_reason,
                                    click_through_url,
                                    is_mraid,
                                    img_src]

            version_data.append(version_data_tmp)
            cr_metadata_tmp = pd.DataFrame(version_data, columns = ['creative_id',
                                                                    'name',
                                                                    'created_on',
                                                                    'last_modified',
                                                                    'status',
                                                                    'width',
                                                                    'height',
                                                                    'ad_format',
                                                                    'is_https',
                                                                    'concept_id',
                                                                    'rich_media',
                                                                    'approval_status',
                                                                    'rejected_reason',
                                                                    'click_through_url',
                                                                    'is_mraid',
                                                                    'img_src'])

            if len(cr_metadata) == 0:
                cr_metadata = cr_metadata_tmp
            else:
                cr_metadata = pd.concat([cr_metadata, cr_metadata_tmp])

        creatives_meta = cr_metadata[['creative_id','width', 'height','ad_format','created_on','last_modified','click_through_url','img_src']]
        creatives_meta['img_src'] = creatives_meta['img_src'].astype(str)
        creatives_meta['img_src'] = creatives_meta['img_src'].str.extract('(https://creative.mathads.com.*?)\.jpg',expand=True)
        creatives_meta['img_src']= creatives_meta['img_src']+'.jpg'

        return creatives_meta
    
    
    
    def strategy_meta_data(self, campaign_ids):
        st_metadata = pd.DataFrame()
        for campaign_id in campaign_ids:
            offset = 0
            offset_total = 1
            version_data = []
            while offset < offset_total:
           
                url_page = 'https://api.mediamath.com/api/v2.0/strategies/limit/campaign={}?full=strategy&with=campaign&q=campaign.status%3D%3D1%26status%3D%3D1&page_limit=250&page_offset='.format(campaign_id) + str(offset)
                strats_data = requests.get(url_page, cookies = self.resp.cookies).text
                tree = etree.parse(io.StringIO(strats_data))
                offset_total = int(tree.xpath('''.//entities/@count''')[0])
               
                for st in tree.iterfind('entities/entity'):
                    ca_id = int(st.xpath('''.//entity[@type = 'campaign']/@id''')[0])
                    ca_name = st.xpath('''.//entity[@type = 'campaign']/@name''')[0]
                    st_id = int(st.attrib['id'])
                    st_name = st.attrib['name']
                    st_version = int(st.attrib['version'])
                    st_status = int(st.xpath('''.//prop[@name = 'status']/@value''')[0])
                    try:
                        st_budget = float(st.xpath('''.//prop[@name = 'budget']/@value''')[0])
                    except:
                        st_budget = 0
                    st_pacing_type = st.xpath('''.//prop[@name = 'pacing_type']/@value''')[0]
                    try:
                        st_freq_amt = int(st.xpath('''.//prop[@name = 'frequency_amount']/@value''')[0])
                    except:
                        st_freq_amt = 0
                    st_pacing_amt = float(st.xpath('''.//prop[@name = 'pacing_amount']/@value''')[0])
                    st_pacing_interval = st.xpath('''.//prop[@name = 'pacing_interval']/@value''')[0]
                    try:
                        st_goal_type = st.xpath('''.//prop[@name = 'goal_type']/@value''')[0]
                    except:
                        st_goal_type = 0
                    try:
                        st_goal_value = float(st.xpath('''.//prop[@name = 'goal_value']/@value''')[0])
                    except:
                        st_goal_value = 0
                    try:
                        st_bid_aggressiveness = float(st.xpath('''.//prop[@name = 'bid_aggressiveness']/@value''')[0])
                    except:
                        st_bid_aggressiveness = 0
                    try:
                        st_updated_on = st.xpath('''.//prop[@name = 'updated_on']/@value''')[0]
                    except:
                        st_updated_on = 0
                    try:
                        st_max_bid = float(st.xpath('''.//prop[@name = 'max_bid']/@value''')[0])
                    except:
                        st_max_bid = 0
                    try:
                        st_min_bid = float(st.xpath('''.//prop[@name = 'min_bid']/@value''')[0])
                    except:
                        st_min_bid  = 0
                    try:
                        st_effective_goal_value = float(st.xpath('''.//prop[@name = 'effective_goal_value']/@value''')[0])
                    except:
                        st_effective_goal_value = 0
                    try:
                        st_pixel_target_expr = st.xpath('''.//prop[@name = 'pixel_target_expr']/@value''')[0]
                    except:
                        st_pixel_target_expr = 0
                    try:
                        st_frequency_type= st.xpath('''.//prop[@name = 'frequency_type']/@value''')[0]
                    except:
                        st_frequency_type = 0
                    try:
                        st_frequency_interval= st.xpath('''.//prop[@name = 'frequency_interval']/@value''')[0]
                    except:
                        st_frequency_interval = 0
                    try:
                        st_description = st.xpath('''.//prop[@name = 'description']/@value''')[0]
                    except:
                        st_description = 0
                    try:
                        st_bid_price_is_media_only= int(st.xpath('''.//prop[@name = 'bid_price_is_media_only']/@value''')[0])
                    except:
                        st_bid_price_is_media_only = 0
                    try:
                        st_exchange_type_for_run_on_all = st.xpath('''.//prop[@name = 'exchange_type_for_run_on_all']/@value''')[0]
                    except:
                        st_exchange_type_for_run_on_all = 0
                    try:
                        st_site_restriction_transparent_urls = st.xpath('''.//prop[@name = 'site_restriction_transparent_urls']/@value''')[0]
                    except:
                        st_site_restriction_transparent_urls = 0           
                    try:
                        st_audience_segment_include_op = st.xpath('''.//prop[@name = 'audience_segment_include_op']/@value''')[0]
                    except:
                        st_audience_segment_include_op = 0
                    try:
                        st_targeting_segment_include_op = st.xpath('''.//prop[@name = 'targeting_segment_include_op']/@value''')[0]
                    except:
                        st_targeting_segment_include_op = 0
                    try:
                        st_pixel_target_expr = st.xpath('''.//prop[@name = 'pixel_target_expr']/@value''')[0]
                    except:
                        st_pixel_target_expr = 0
                    try:
                        st_run_on_all_pmp = int(st.xpath('''.//prop[@name = 'run_on_all_pmp']/@value''')[0])
                    except:
                        st_run_on_all_pmp  = 0
                    try:
                        st_run_on_display = int(st.xpath('''.//prop[@name = 'run_on_display']/@value''')[0])
                    except:
                        st_run_on_display  = 0
                    try:
                        st_run_on_mobile = int(st.xpath('''.//prop[@name = 'run_on_mobile']/@value''')[0])
                    except:
                        st_run_on_mobile = 0
                    try:
                        st_frequency_optimization= int(st.xpath('''.//prop[@name = 'frequency_optimization']/@value''')[0])
                    except:
                        st_frequency_optimization = 0
                    try:
                        st_use_dba_strategy_pacing= st.xpath('''.//prop[@name = 'use_dba_strategy_pacing']/@value''')[0]
                    except:
                        st_use_dba_strategy_pacing = 0

                    version_data_tmp = [ca_id,
                                        ca_name,
                                        st_id,
                                        st_name,
                                        st_version,
                                        st_status,
                                        st_budget,
                                        st_pacing_type,
                                        st_freq_amt,
                                        st_pacing_amt,
                                        st_pacing_interval,
                                        st_goal_type,
                                        st_goal_value,
                                        st_bid_aggressiveness,
                                        st_updated_on,
                                        st_max_bid,
                                        st_min_bid,
                                        st_effective_goal_value,
                                        st_pixel_target_expr,
                                        st_frequency_type,
                                        st_frequency_interval,
                                        st_description,
                                        st_bid_price_is_media_only,
                                        st_exchange_type_for_run_on_all, st_site_restriction_transparent_urls,st_audience_segment_include_op, 
                                        st_targeting_segment_include_op, st_pixel_target_expr, st_run_on_all_pmp, st_run_on_display,st_run_on_mobile,
                                        st_frequency_optimization,st_use_dba_strategy_pacing]
                    

                    version_data.append(version_data_tmp)

                offset = offset + 250
            
            st_metadata_tmp = pd.DataFrame(version_data, columns = ['campaign_id',
                                                                    'campaign_name',
                                                                    'strategy_id',
                                                                    'strategy_name',
                                                                    'strategy_version',
                                                                    'strategy_status',
                                                                    'strategy_budget',
                                                                    'pacing_type',
                                                                    'frequency_amount',
                                                                    'pacing_amount',
                                                                    'pacing_interval',
                                                                    'goal_type',
                                                                    'goal_value',
                                                                    'bid_aggressiveness',
                                                                    'updated_on',
                                                                    'max_bid',
                                                                    'min_bid',
                                                                    'effective_goal_value',
                                                                    'pixel_target_expr',
                                                                    'frequency_type',
                                                                    'frequency_interval',
                                                                    'description',
                                                                    'bid_price_is_media_only',
                                                                    'exchange_type_for_run_on_all', 'restriction_transparent_urls','audience_include_op', 'cntx_include_op',
                                                                    'pixel_target', 'on_all_pmp', 'on_display', 'on_mobile','frequency_optimization','dba_strategy_pacing'])
            
            if len(st_metadata) == 0:
                st_metadata = st_metadata_tmp
            else:
                st_metadata = pd.concat([st_metadata, st_metadata_tmp])
                      
        st_metadata_fin = st_metadata[(st_metadata['strategy_status'] == 1)]
        strategy_ids=st_metadata_fin['strategy_id'].values
        st_metadata_final = st_metadata_fin[['campaign_id', 'campaign_name', 'strategy_id', 'strategy_name','strategy_status','frequency_optimization','frequency_interval','frequency_type','frequency_amount', 'pacing_type', 'pacing_amount', 'min_bid', 'max_bid', 'goal_type',  'goal_value', 'bid_price_is_media_only','dba_strategy_pacing']]
        return strategy_ids, st_metadata_final

    def strategy_daypart(self, strategy_ids):
        st_daypart = pd.DataFrame()
        for strategy_id in strategy_ids:
            offset = 0
            offset_total = 1
            version_data = []
            while offset < offset_total:
                url_page = 'https://api.mediamath.com/api/v2.0/strategies/{}/day_parts?full=strategy&with=campaign&q=campaign.status%3D%3D1%26status%3D%3D1&page_limit=250&page_offset='.format(strategy_id) + str(offset)
                strats_data = requests.get(url_page, cookies = self.resp.cookies).text
                tree = etree.parse(io.StringIO(strats_data))
                offset_total = int(tree.xpath('''.//entities/@start''')[0])
                for st in tree.iterfind('entities/entity'):
                    st_id = int(st.xpath('''.//prop[@name = 'strategy_id']/@value''')[0])
                    ut = int(st.xpath('''.//prop[@name = 'user_time']/@value''')[0])
                    sh = int(st.xpath('''.//prop[@name = 'start_hour']/@value''')[0])
                    eh = int(st.xpath('''.//prop[@name = 'end_hour']/@value''')[0])
                    fd = st.xpath('''.//prop[@name = 'days']/@value''')
                    version_data_tmp = [st_id,
                                        ut,
                                        sh,
                                        eh,
                                        fd]
                    version_data.append(version_data_tmp)
                offset = offset + 250
            st_daypart_tmp = pd.DataFrame(version_data, columns = ['strategy_id',
                                                                    'user_time',
                                                                    'start_hour',
                                                                    'Scheduled End Hour',
                                                                    'days'
                                                                ])
            
            if len(st_daypart) == 0:
                st_daypart = st_daypart_tmp
            else:
                st_daypart = pd.concat([st_daypart, st_daypart_tmp])
        return st_daypart


    def last_two_days_performance(self, campaign_ids):
        dimensions='campaign_id,campaign_goal_type,campaign_goal_value'
        metrics='impressions,clicks,total_spend'
        dt_today = date.today()
        today=dt_today.strftime('%Y-%m-%d')
        dt = date.today() - timedelta(2)
        start_date = dt.strftime('%Y-%m-%d')
        end_date = dt.strftime('%Y-%m-%d')
        camp_goal_df = pd.DataFrame()      
        for campaign_id in campaign_ids:
            camp_goal_df_tmp= self.t1_report(endpoint='performance', dimensions=dimensions,
                         filter='campaign_id='+str(campaign_id),
                         metrics=metrics,
                         precision='4',time_rollup='by_day',order='date',start_date=start_date,end_date=end_date)
            if len(camp_goal_df) == 0:
                camp_goal_df = camp_goal_df_tmp
            else:
                camp_goal_df = pd.concat([camp_goal_df, camp_goal_df_tmp])
        camp_goal_id_df = camp_goal_df[['campaign_id','campaign_goal_type','campaign_goal_value']].rename(columns={"campaign_goal_type": "Goal"})
        return camp_goal_df, camp_goal_id_df


    def strategy_two_days_performance(self,campaign_ids):
        dimensions='campaign_id,strategy_id'
        metrics='impressions,clicks,total_conversions,total_spend'
        dt_today = date.today()
        dt = date.today() - timedelta(1)
        start_date = dt.strftime('%Y-%m-%d')
        end_date = dt.strftime('%Y-%m-%d')
        camp_perf_df = pd.DataFrame()
                
        for campaign_id in campaign_ids:
            
            camp_perf_df_tmp= self.t1_report(endpoint='performance', dimensions=dimensions,
                         filter='campaign_id='+str(campaign_id),
                         metrics=metrics,
                         precision='4',time_rollup='by_day',order='date',start_date=start_date,end_date=end_date)
    
            if len(camp_perf_df) == 0:
                camp_perf_df = camp_perf_df_tmp
            else:
                camp_perf_df = pd.concat([camp_perf_df, camp_perf_df_tmp])
        return camp_perf_df

    def winlos_report(self, campaign_ids):
        dimensions='organization_name,agency_name,advertiser_name,campaign_id,campaign_start_date,campaign_end_date,campaign_budget,strategy_id'
        metrics='average_bid_amount_cpm,average_win_amount_cpm,bid_rate,bids,matched_bid_opportunities,max_bid_amount_cpm,max_win_amount_cpm,min_bid_amount_cpm,min_win_amount_cpm,total_bid_amount_cpm,total_win_amount_cpm,win_rate,wins'
        dt = date.today() - timedelta(1)
        start_date = dt.strftime('%Y-%m-%d')
        end_date = dt.strftime('%Y-%m-%d')
        win_los_df = pd.DataFrame()
      
        for campaign_id in campaign_ids:
            win_los_df_tmp= self.t1_report(endpoint='win_loss', dimensions=dimensions,
                         filter='campaign_id='+str(campaign_id),
                         metrics=metrics,
                         precision='4',time_rollup='all',order='date',start_date=start_date,end_date=end_date)
            
            if len(win_los_df) == 0:
                win_los_df = win_los_df_tmp
            else:
                win_los_df = pd.concat([win_los_df, win_los_df_tmp])
        return win_los_df


    def winlos_report_mtd(self, campaign_ids):
        dt = date.today() - timedelta(1)
        sm=datetime.today().replace(day=1)
        start_date = sm.strftime('%Y-%m-%d')
        end_date = dt.strftime('%Y-%m-%d')
        win_los_df = pd.DataFrame()
        data = self.resp.json()
        sessionid=data['data']['session']['sessionid']
        conn = http.client.HTTPSConnection("api.mediamath.com")
        headers = { 'cookie': 'adama_session='+sessionid}
        
        for campaign_id in campaign_ids:
            url_winlos='https://api.mediamath.com/reporting/v1/std/win_loss?dimensions=organization_name,agency_name,advertiser_name,campaign_id,campaign_start_date,campaign_end_date,campaign_budget,strategy_id&filter=campaign_id={}&metrics=average_bid_amount_cpm,average_win_amount_cpm,bid_rate,bids,matched_bid_opportunities,max_bid_amount_cpm,max_win_amount_cpm,min_bid_amount_cpm,min_win_amount_cpm,total_bid_amount_cpm,total_win_amount_cpm,win_rate,wins&precision=2&time_rollup=all&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)
            conn.request("GET", url_winlos, headers=headers)
            win_los_df_tmp= pd.read_csv(conn.getresponse())
            if len(win_los_df) == 0:
                win_los_df = win_los_df_tmp
            else:
                win_los_df = pd.concat([win_los_df, win_los_df_tmp])
        return win_los_df
    

    def creative_performance_report(self, campaign_ids):
        dt_today = date.today()
        today=dt_today.strftime('%Y-%m-%d')
        start = date.today() - timedelta(1)
        end = date.today() - timedelta(2)
        start_date = end.strftime('%Y-%m-%d')
        end_date = start.strftime('%Y-%m-%d')
        creative_perf_df = pd.DataFrame()
        for campaign_id in campaign_ids:
            url_perf='https://api.mediamath.com/reporting/v1/std/performance?dimensions=campaign_id,campaign_name,concept_id,concept_name,creative_id&filter=campaign_id={}&metrics=impressions,total_spend&precision=4&time_rollup=all&order=date&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)
            data = self.resp.json()
            sessionid=data['data']['session']['sessionid']
            conn = http.client.HTTPSConnection("api.mediamath.com")
            headers = { 'cookie': 'adama_session='+sessionid}
            conn.request("GET", url_perf, headers=headers)
            creative_perf_df_tmp= pd.read_csv(conn.getresponse())
            if len(creative_perf_df) == 0:
                creative_perf_df = creative_perf_df_tmp
            else:
                creative_perf_df = pd.concat([creative_perf_df, creative_perf_df_tmp])
        
        creative_ids=creative_perf_df['creative_id'].unique()
       
        return creative_ids, creative_perf_df

    def creative_approval_report (self, creative_ids):
        creative_aprov_df = pd.DataFrame()
        for creative_id in creative_ids:
            url_perf='https://api.mediamath.com/api/v2.0/atomic_creatives/creative_approvals_report?atomic_creative_ids=({})'.format(creative_id)
            data = self.resp.json()
            sessionid=data['data']['session']['sessionid']
            conn = http.client.HTTPSConnection("api.mediamath.com")
            headers = { 'cookie': 'adama_session='+sessionid}
            conn.request("GET", url_perf, headers=headers)
            creative_aprov_df_tmp= pd.read_csv(conn.getresponse(), encoding='latin-1')
            if len(creative_aprov_df) == 0:
                creative_aprov_df = creative_aprov_df_tmp
            else:
                creative_aprov_df = pd.concat([creative_aprov_df, creative_aprov_df_tmp])
        
        return creative_aprov_df
 
    def creative_approval(self, campaign_ids):
        creative_ids, creative_perf_df = self.creative_performance_report(campaign_ids)
        creative_aprov_df = self.creative_approval_report(creative_ids)
        camp_aproval = pd.merge(creative_perf_df, creative_aprov_df ,  how='left', left_on=['creative_id'],right_on=['Creative ID'])
        camp_aproval_df = camp_aproval[(camp_aproval['Net Status'] == 'PENDING') | (camp_aproval['Net Status'] == 'REJECTED')]
        crap =  camp_aproval_df[['campaign_id', 'campaign_name', 'concept_name', 'Creative ID', 'Creative Name','Net Status', 'AdX Open Auction', 'AdX Deals', 'AppNexus',
            'Microsoft Ad Exchange', 'Right Media Exchange',
            'Brightroll for Display', 'MoPub']]
        crap.columns = ['Campaign ID', 'Campaign name', 'Concept Name', 'Creative ID', 'Creative Name','Net Status', 'AdX Open Auction', 'AdX Deals', 'AppNexus',
            'Microsoft Ad Exchange', 'Right Media Exchange',
            'Brightroll for Display', 'MoPub']
        
        return crap
       

    def get_deals(self, organization_id, strategy_ids):
            url = "https://api.mediamath.com/deals/v1.0/deals"
            headers = {"Content-Type": "application/json",'Authorization': 'Bearer {}'.format(self.token)}
            deals_df = pd.DataFrame()
            for strategy_id in strategy_ids:  
                offset = 0
                offset_total = 1
                while offset < offset_total:
                    querystring = {"owner.organization_id":organization_id,"strategy_id":strategy_id, "page_offset":offset}
                    deals_data  = requests.request("GET", url, headers=headers, params=querystring).text
                    deals_df_tmp = pd.io.json.json_normalize(json.loads(deals_data)['data'])
                    meta_df = pd.io.json.json_normalize(json.loads(deals_data)['meta'])
                    offset_total = meta_df['total_count'][0]
                    offset = offset + 100
                if len(deals_df) == 0:
                    deals_df = deals_df_tmp
                else:
                    deals_df = pd.concat([deals_df, deals_df_tmp])
            return deals_df

    def get_deal_groups(self, organization_id, strategy_ids):
            url = "https://api.mediamath.com/deals/v1.0/deal_groups"
            headers = {"Content-Type": "application/json",'Authorization': 'Bearer {}'.format(self.token)}
            df_dg = pd.DataFrame()
            for strategy_id in strategy_ids:  
                offset = 0
                offset_total = 1
                while offset < offset_total:
                    querystring = {"owner.organization_id":organization_id,"strategy_id":strategy_id, "page_offset":offset}
                    deals_data = requests.request("GET", url, headers=headers, params=querystring).text
                    df_dg_tmp = pd.io.json.json_normalize(json.loads(deals_data)['data'])
                    meta_df = pd.io.json.json_normalize(json.loads(deals_data)['meta'])
                    offset_total = meta_df['total_count'][0]
                    offset = offset + 100
                if len(df_dg) == 0:
                    df_dg = df_dg_tmp
                else:
                    df_dg = pd.concat([df_dg, df_dg_tmp])
            
            return df_dg

    def str_deal_metadata(self,strategy_ids):
            str_deal_metadata = pd.DataFrame()
            for strategy_id in strategy_ids:
                offset = 0
                offset_total = 1
                version_data = []  
                
                url_page = 'https://api.mediamath.com/api/v2.0/strategies/{}/deals'.format(strategy_id)
                strats_data = requests.get(url_page, cookies = self.resp.cookies).text
                tree = etree.parse(io.StringIO(strats_data))
                offset_total = len(tree.xpath("./entity/entity"))
                while offset < offset_total:
                
                    for st in tree.iterfind('./entity/entity'):
                        try:    
                            deal_id = int(st.attrib['id'])
                        except:
                            deal_id =0
                                
                        version_data_tmp = [strategy_id, deal_id]
                        version_data.append(version_data_tmp)
                    offset = offset + 250
                        
                str_deal_metadata_tmp = pd.DataFrame(version_data, columns = ['strategy_id',
                                                                        'deal_id'])
                if len(str_deal_metadata) == 0:
                    str_deal_metadata = str_deal_metadata_tmp
                else:
                    str_deal_metadata = pd.concat([str_deal_metadata, str_deal_metadata_tmp])
            return str_deal_metadata
    
    def targeting_segments_data(self, parent_targeting_segment_ids):
            st_metadata = pd.DataFrame()
            for parent_id in parent_targeting_segment_ids:
                offset = 0
                offset_total = 1
                version_data = []
                while offset < offset_total:
                
                    url_page = 'https://api.mediamath.com/api/v2.0/targeting_segments?full=*&parent={}'.format(parent_id)
                    strats_data = requests.get(url_page, cookies = self.resp.cookies).text
                    tree = etree.parse(io.StringIO(strats_data))
                    offset_total = int(tree.xpath('''.//entities/@count''')[0])
                
    
                    for st in tree.iterfind('entities/entity'):
                            segment_id = int(st.attrib['id'])

                            # segment_id = int(st.xpath('''.//entity[@type = 'targeting_segment']/@id''')[0])
                            targeting_vendor_id = int(st.xpath('''.//prop[@name = 'targeting_vendor_id']/@value''')[0])
                            parent_targeting_segment_id = int(st.xpath('''.//prop[@name = 'parent_targeting_segment_id']/@value''')[0])
                            try:            
                                external_code = int(st.xpath('''.//prop[@name = 'external_code']/@value''')[0])
                            except:
                                external_code = 0
                            try: 
                                name = st.xpath('''.//prop[@name = 'name']/@value''')[0]
                            except:
                                name = 0
                            try: 
                                full_path = st.xpath('''.//prop[@name = 'full_path']/@value''')[0]
                            except:
                                full_path = 0
                            version_data_tmp = [segment_id,
                                                targeting_vendor_id,
                                                parent_targeting_segment_id,
                                                external_code,
                                                name,
                                                full_path]
                            

                            version_data.append(version_data_tmp)

                    offset = offset + 1000
                
                st_metadata_tmp = pd.DataFrame(version_data, columns = ['segment_id',
                                            'targeting_vendor_id',
                                            'parent_targeting_segment_id',
                                            'external_code',
                                            'name',
                                            'full_path'])
                
                if len(st_metadata) == 0:
                    st_metadata = st_metadata_tmp
                else:
                    st_metadata = pd.concat([st_metadata, st_metadata_tmp])
                      
            return st_metadata

    def str_deal_group_metadata(self,strategy_ids):
            str_deal_group_metadata = pd.DataFrame()
            for strategy_id in strategy_ids:
                offset = 0
                offset_total = 1
                version_data = []  
                
                url_page = 'https://api.mediamath.com/api/v2.0/strategies/{}/deal_groups'.format(strategy_id)
                strats_data = requests.get(url_page, cookies = self.resp.cookies).text
                tree = etree.parse(io.StringIO(strats_data))
                offset_total = len(tree.xpath("./entity/entity"))
                while offset < offset_total:
                
                    for st in tree.iterfind('./entity/entity'):
                        try:    
                            deal_group_id = int(st.attrib['id'])
                        except:
                            deal_group_id =0
                                
                        version_data_tmp = [strategy_id, deal_group_id]
                        version_data.append(version_data_tmp)
                    offset = offset + 250
                       
                str_deal_group_metadata_tmp = pd.DataFrame(version_data, columns = ['strategy_id',
                                                                        'deal_group_id'])
                if len(str_deal_group_metadata) == 0:
                    str_deal_group_metadata = str_deal_group_metadata_tmp
                else:
                    str_deal_group_metadata = pd.concat([str_deal_group_metadata, str_deal_group_metadata_tmp])
            return str_deal_group_metadata

    def outpacing_campaigns(self, campaign_ids, campaigns_to_check):
        strategy_op_ids, st_metadata = self.strategy_meta_data(campaign_ids)
        st_daypart = self.strategy_daypart(strategy_op_ids)
        camp_goal_df, camp_goal_id_df = self.last_two_days_performance(campaign_ids)
        st_data = pd.merge(st_metadata, st_daypart, how='left', left_on='strategy_id', right_on='strategy_id')
        st_hour = st_data.groupby(['campaign_id'], sort=False)['Scheduled End Hour'].max()
        camp_outpacing = pd.merge(campaigns_to_check, st_hour, how='left', left_on='Campaign ID', right_on='campaign_id')
        camp_goal_id_df = camp_goal_df[['campaign_id','campaign_goal_type','campaign_goal_value']].rename(columns={"campaign_goal_type": "Goal"})
        camp_outpacing_fin = pd.merge(camp_outpacing, camp_goal_id_df, how='left', left_on='Campaign ID', right_on='campaign_id')
        camp_outpacing_fin['Review Date'] = date.today().strftime('%Y-%m-%d') 
        camp_outpacing_final =camp_outpacing_fin[['Review Date','Advertiser Name','Campaign ID','Campaign Name','Goal', 'Days Remaining', 'Spend To Pace', 
                                                'Latest Hour of Activity', 'Scheduled End Hour']].sort_values(by='Latest Hour of Activity', ascending=True)
        camp_outpacing_final['Scheduled End Hour'] =camp_outpacing_final['Scheduled End Hour'].fillna(23)
        camp_outpacing_final_n = camp_outpacing_final[(camp_outpacing_final['Latest Hour of Activity'] !=0 )&(camp_outpacing_final['Scheduled End Hour'] - camp_outpacing_final['Latest Hour of Activity'] > 1 )& (camp_outpacing_final['Days Remaining'] > 1)].sort_values(by='Latest Hour of Activity', ascending=False)
        return camp_outpacing_final_n


    def underpacing_campaigns(self, campaign_ids, campaigns_to_check):
        camp_pacing_medicine, campaign_active_ids = self.campaign_meta_data(campaign_ids)
        camp_underpacing = pd.merge(campaigns_to_check, camp_pacing_medicine, how='right', left_on='Campaign ID', right_on='campaign_id')
        camp_underpacing_final = camp_underpacing[['Start Date','Advertiser Name', 'Campaign ID','Campaign Name', 'Spend To Pace', 'Spend Yesterday','campaign_spend_cap_type', 'campaign_spend_cap_automatic','campaign_spend_cap_amount','campaign_frequency_type', 
                                                'campaign_frequency_interval', 'campaign_frequency_amount']].rename(columns={"Start Date": "start_date",'Advertiser Name':'advertiser_name',
                                                                                'Campaign ID':'campaign_id','Campaign Name':'campaign_name','campaign_spend_cap_type':'spend_cap_type','campaign_spend_cap_amount':'cap_amount','campaign_frequency_type':'frequency_type',
                                                                                'Spend To Pace':'spend_to_pace','Spend Yesterday':'spend_yesterday','campaign_frequency_interval':'frequency_interval', 'campaign_frequency_amount':'frequency_amount' }).sort_values(by=
                                                                                'spend_to_pace', ascending=False)
        camp_underpacing_final["frequency_amount"] = pd.to_numeric(camp_underpacing_final["frequency_amount"])
        camp_underpacing_final["cap_amount"] = pd.to_numeric(camp_underpacing_final["cap_amount"])
        camp_underpacing_final.drop(['campaign_spend_cap_automatic'], axis=1, inplace=True)
        return camp_underpacing_final, campaign_active_ids
    
    def underpacing_campaigns_t1(self, campaign_ids):
        camp_pacing_medicine, campaign_active_ids = self.campaign_meta_data(campaign_ids)
        camp_underpacing_final = camp_pacing_medicine[['campaign_id','campaign_name','campaign_spend_cap_automatic', 'campaign_spend_cap_type',
                                            'campaign_spend_cap_amount','campaign_frequency_type','campaign_frequency_interval','campaign_frequency_amount']].rename(columns={'campaign_spend_cap_type':'spend_cap_type','campaign_spend_cap_amount':'cap_amount','campaign_frequency_type':'frequency_type',
                                                                                'campaign_frequency_interval':'frequency_interval', 'campaign_frequency_amount':'frequency_amount' })                                                                               
        camp_underpacing_final["frequency_amount"] = pd.to_numeric(camp_underpacing_final["frequency_amount"])
        camp_underpacing_final["cap_amount"] = pd.to_numeric(camp_underpacing_final["cap_amount"])
        camp_underpacing_final.drop(['campaign_spend_cap_automatic'], axis=1, inplace=True)
        return camp_underpacing_final, campaign_active_ids

    def underpacing_strategies(self,organization_id, campaign_ids):
        strategy_ids, st_metadata_final = self.strategy_meta_data(campaign_ids)
        df_deals = self.get_deals(organization_id,strategy_ids)
        if len(df_deals) !=0:
            # df_deals = df_deals[['deal_id','deal_name','deal_identifier','deal_status','deal_floor_price','deal_creation_date']]
            df_deals = df_deals[['id','name','deal_identifier','status','price.value','created_on']]
            df_deals = df_deals.rename(columns = {'name' : 'deal_name',
                                                'deal_identifier' : 'deal_external_id',
                                                'id' : 'deal_id',
                                                'status' : 'deal_status',
                                                'price.value' : 'deal_floor_price',
                                                'created_on' : 'deal_creation_date'
                                                })

            df_deals = df_deals.sort_values('deal_name')
            df_deals_fin = df_deals[(df_deals['deal_status'] == True)]
        else:
            df_deals = pd.DataFrame(columns=['deal_name', 'deal_external_id', 'deal_id', 'deal_status','deal_floor_price','deal_creation_date'])
            df_deals_fin = df_deals
        df_dg_raw  = self.get_deal_groups(organization_id, strategy_ids)
        if len(df_dg_raw) !=0:
            df_dg_data = []
            df_dg_raw = df_dg_raw[['id','name','deal_ids', 'status']]
            for index, row in df_dg_raw.iterrows():
                for deal in row['deal_ids']:
                    r = [deal, row['id'], row['name'], row['status']]
                    df_dg_data.append(r)
            df_dg = pd.DataFrame(data=df_dg_data, columns=['deal_id','deal_group_id', 'deal_group_name', 'deal_group_status'])
            df_dg = df_dg[(df_dg['deal_group_status'] == True)]
        else:
            df_dg = pd.DataFrame(columns=['deal_id','deal_group_id', 'deal_group_name', 'deal_group_status'])
       

        str_deal_metadata = self.str_deal_metadata(strategy_ids)       
        str_deal_group_metadata = self.str_deal_group_metadata(strategy_ids)
        # 3.3. Combining all deals assigned to strategy
        str_dg_deal_metadata = pd.merge(str_deal_group_metadata, df_dg,  how='left', on=['deal_group_id'])
        str_dg_deal_metadata = str_dg_deal_metadata[['strategy_id','deal_id']]
        str_all_deals_metadata = pd.concat([str_dg_deal_metadata, str_deal_metadata])
        # 3.4. Getting deal price info
        str_deals = pd.merge(str_all_deals_metadata, df_deals_fin, how='left', left_on='deal_id', right_on='deal_id')
        str_deals['deal_floor_price'] = str_deals['deal_floor_price'].astype(float)
        str_deals_final = str_deals.groupby(['strategy_id']).agg({'deal_floor_price': ['mean', 'min', 'max']})
        str_deals_final.columns = ['deal_mean_price', 'deal_min_price', 'deal_max_price']
        str_deals_final = str_deals_final.reset_index()
        str_setup_overview = pd.merge(st_metadata_final, str_deals_final,  how='left', on=['strategy_id'])
        camp_perf_df = self.strategy_two_days_performance(campaign_ids)
        win_los_df = self.winlos_report(campaign_ids)
        # Final agregation
        str_correction = pd.merge(str_setup_overview, win_los_df ,  how='left', on=['campaign_id','strategy_id'])
        str_correction_perf = pd.merge(str_correction, camp_perf_df,  how='left', on=['campaign_id','strategy_id'])
        str_correction_final =  str_correction_perf[['campaign_id', 'campaign_name', 'strategy_id', 'strategy_name',
            'frequency_type', 'frequency_amount', 'pacing_type', 'pacing_amount','total_spend',
            'min_bid', 'max_bid', 'goal_type', 'goal_value',
            'deal_mean_price', 'deal_min_price',
            'deal_max_price', 'bid_rate', 'win_rate']]
        str_correction_final.columns = ['camp_id', 'campaign_name', 'strategy_id', 'strategy_name',
            'f_type', 'f_amount', 'pacing_type', 'pacing', 'daily_spend',
            'min_bid', 'max_bid', 'goal', 'goal_value',
            'deal_mean', 'deal_min',
            'deal_max', 'bid_rate', 'win_rate']
        # str_correction = pd.merge(str_outpacing_fin_30, win_los_df, how='left', left_on='strategy_id', right_on='strategy_id')
        strategy_underpacing = str_correction_final.replace(np.nan,0)
        strategy_underpacing["min_bid"] = pd.to_numeric(strategy_underpacing["min_bid"])
        strategy_underpacing["deal_max"] = pd.to_numeric(strategy_underpacing["deal_max"])
        strategy_underpacing["max_bid"] = pd.to_numeric(strategy_underpacing["max_bid"])
        strategy_underpacing['goal_value'] = pd.to_numeric(strategy_underpacing['goal_value'])
        strategy_underpacing['daily_spend'] = pd.to_numeric(strategy_underpacing['daily_spend'])
        strategy_underpacing['win_rate'] = pd.to_numeric(strategy_underpacing['win_rate'])
        strategy_troubleshooting = strategy_underpacing[(strategy_underpacing['daily_spend'] < 0.05)&(strategy_underpacing['min_bid'] >= strategy_underpacing['deal_max'] )]
        strategy_tr_ids=strategy_troubleshooting['strategy_id'].values
        return strategy_underpacing, strategy_tr_ids, strategy_ids

    def underpacing_strategies_mtd(self, organization_id,campaign_ids):
        strategy_ids, st_metadata_final = self.strategy_meta_data(campaign_ids)
        df_deals = self.get_deals(organization_id,strategy_ids)
        if len(df_deals) !=0:
            # df_deals = df_deals[['deal_id','deal_name','deal_identifier','deal_status','deal_floor_price','deal_creation_date']]
            df_deals = df_deals[['id','name','deal_identifier','status','price.value','created_on']]
            df_deals = df_deals.rename(columns = {'name' : 'deal_name',
                                                'deal_identifier' : 'deal_external_id',
                                                'id' : 'deal_id',
                                                'status' : 'deal_status',
                                                'price.value' : 'deal_floor_price',
                                                'created_on' : 'deal_creation_date'
                                                })

            df_deals = df_deals.sort_values('deal_name')
            df_deals_fin = df_deals[(df_deals['deal_status'] == True)]
        else:
            df_deals = pd.DataFrame(columns=['deal_name', 'deal_external_id', 'deal_id', 'deal_status','deal_floor_price','deal_creation_date'])
            df_deals_fin = df_deals
        df_dg_raw  = self.get_deal_groups(organization_id,strategy_ids)
        if len(df_dg_raw) !=0:
            df_dg_data = []
            df_dg_raw = df_dg_raw[['id','name','deal_ids', 'status']]
            for index, row in df_dg_raw.iterrows():
                for deal in row['deal_ids']:
                    r = [deal, row['id'], row['name'], row['status']]
                    df_dg_data.append(r)
            df_dg = pd.DataFrame(data=df_dg_data, columns=['deal_id','deal_group_id', 'deal_group_name', 'deal_group_status'])
            df_dg = df_dg[(df_dg['deal_group_status'] == True)]
        else:
            df_dg = pd.DataFrame(columns=['deal_id','deal_group_id', 'deal_group_name', 'deal_group_status'])
       

        str_deal_metadata = self.str_deal_metadata(strategy_ids)       
        str_deal_group_metadata = self.str_deal_group_metadata(strategy_ids)
        # 3.3. Combining all deals assigned to strategy
        str_dg_deal_metadata = pd.merge(str_deal_group_metadata, df_dg,  how='left', on=['deal_group_id'])
        str_dg_deal_metadata = str_dg_deal_metadata[['strategy_id','deal_id']]
        str_all_deals_metadata = pd.concat([str_dg_deal_metadata, str_deal_metadata])
        # 3.4. Getting deal price info
        str_deals = pd.merge(str_all_deals_metadata, df_deals_fin, how='left', left_on='deal_id', right_on='deal_id')
        str_deals['deal_floor_price'] = str_deals['deal_floor_price'].astype(float)
        str_deals_final = str_deals.groupby(['strategy_id']).agg({'deal_floor_price': ['mean', 'min', 'max']})
        str_deals_final.columns = ['deal_mean_price', 'deal_min_price', 'deal_max_price']
        str_deals_final = str_deals_final.reset_index()
        str_setup_overview = pd.merge(st_metadata_final, str_deals_final,  how='left', on=['strategy_id'])
        camp_perf_df = self.strategy_two_days_performance(campaign_ids)
        win_los_df = self.winlos_report_mtd(campaign_ids)
        # Final agregation
        str_correction = pd.merge(str_setup_overview, win_los_df ,  how='left', on=['campaign_id','strategy_id'])
        str_correction_perf = pd.merge(str_correction, camp_perf_df,  how='left', on=['campaign_name','campaign_id', 'strategy_name','strategy_id'])
        str_correction_final =  str_correction_perf[['campaign_id', 'campaign_name', 'strategy_id', 'strategy_name',
            'frequency_type', 'frequency_amount', 'pacing_type', 'pacing_amount','total_spend',
            'min_bid', 'max_bid', 'goal_type', 'goal_value',
            'deal_mean_price', 'deal_min_price',
            'deal_max_price', 'bid_rate', 'win_rate']]
        str_correction_final.columns = ['camp_id', 'campaign_name', 'strategy_id', 'strategy_name',
            'f_type', 'f_amount', 'pacing_type', 'pacing', 'daily_spend',
            'min_bid', 'max_bid', 'goal', 'goal_value',
            'deal_mean', 'deal_min',
            'deal_max', 'bid_rate', 'win_rate']
        # str_correction = pd.merge(str_outpacing_fin_30, win_los_df, how='left', left_on='strategy_id', right_on='strategy_id')
        strategy_underpacing = str_correction_final.replace(np.nan,0)
        strategy_underpacing["min_bid"] = pd.to_numeric(strategy_underpacing["min_bid"])
        strategy_underpacing["deal_max"] = pd.to_numeric(strategy_underpacing["deal_max"])
        strategy_underpacing["max_bid"] = pd.to_numeric(strategy_underpacing["max_bid"])
        strategy_underpacing['goal_value'] = pd.to_numeric(strategy_underpacing['goal_value'])
        strategy_underpacing['daily_spend'] = pd.to_numeric(strategy_underpacing['daily_spend'])
        strategy_underpacing['win_rate'] = pd.to_numeric(strategy_underpacing['win_rate'])
        strategy_troubleshooting = strategy_underpacing[(strategy_underpacing['daily_spend'] < 0.05)&(strategy_underpacing['min_bid'] >= strategy_underpacing['deal_max'] )]
        strategy_tr_ids=strategy_troubleshooting['strategy_id'].values
        return strategy_underpacing, strategy_tr_ids, strategy_ids

    def strategy_meta_data_id(self, strategy_ids):
        st_metadata = pd.DataFrame()
        for strategy_id in strategy_ids:
            version_data = []
            url_page = 'https://api.mediamath.com/api/v2.0/strategies/{}'.format(strategy_id)
            strats_data = requests.get(url_page, cookies = self.resp.cookies).text
            st = etree.parse(io.StringIO(strats_data))
            ca_id = int(st.xpath('''.//prop[@name = 'campaign_id']/@value''')[0])
            ca_name = st.xpath('''.//prop[@name = 'name']/@value''')[0]
            # st_id = int(st.attrib['id'])
            # st_name = st.attrib['name']

            st_id  = int(st.xpath('''.//entity[@type = 'strategy']/@id''')[0])
            st_name  = st.xpath('''.//entity[@type = 'strategy']/@name''')[0]


            st_version = int(st.xpath('''.//entity[@type = 'strategy']/@version''')[0])
            st_status = int(st.xpath('''.//prop[@name = 'status']/@value''')[0])
            try:
                st_budget = float(st.xpath('''.//prop[@name = 'budget']/@value''')[0])
            except:
                st_budget = 0
            st_pacing_type = st.xpath('''.//prop[@name = 'pacing_type']/@value''')[0]
            try:
                st_freq_amt = int(st.xpath('''.//prop[@name = 'frequency_amount']/@value''')[0])
            except:
                st_freq_amt = 0
            st_pacing_amt = float(st.xpath('''.//prop[@name = 'pacing_amount']/@value''')[0])
            st_pacing_interval = st.xpath('''.//prop[@name = 'pacing_interval']/@value''')[0]
            try:
                st_goal_type = st.xpath('''.//prop[@name = 'goal_type']/@value''')[0]
            except:
                st_goal_type = 0
            try:
                st_goal_value = float(st.xpath('''.//prop[@name = 'goal_value']/@value''')[0])
            except:
                st_goal_value = 0
            try:
                st_bid_aggressiveness = float(st.xpath('''.//prop[@name = 'bid_aggressiveness']/@value''')[0])
            except:
                st_bid_aggressiveness = 0
            try:
                st_updated_on = st.xpath('''.//prop[@name = 'updated_on']/@value''')[0]
            except:
                st_updated_on = 0
            try:
                st_max_bid = float(st.xpath('''.//prop[@name = 'max_bid']/@value''')[0])
            except:
                st_max_bid = 0
            try:
                st_min_bid = float(st.xpath('''.//prop[@name = 'min_bid']/@value''')[0])
            except:
                st_min_bid  = 0
            try:
                st_effective_goal_value = float(st.xpath('''.//prop[@name = 'effective_goal_value']/@value''')[0])
            except:
                st_effective_goal_value = 0
            try:
                st_pixel_target_expr = st.xpath('''.//prop[@name = 'pixel_target_expr']/@value''')[0]
            except:
                st_pixel_target_expr = 0
            try:
                st_frequency_type= st.xpath('''.//prop[@name = 'frequency_type']/@value''')[0]
            except:
                st_frequency_type = 0
            try:
                st_frequency_interval= st.xpath('''.//prop[@name = 'frequency_interval']/@value''')[0]
            except:
                st_frequency_interval = 0
            try:
                st_description = st.xpath('''.//prop[@name = 'description']/@value''')[0]
            except:
                st_description = 0
            try:
                st_bid_price_is_media_only= int(st.xpath('''.//prop[@name = 'bid_price_is_media_only']/@value''')[0])
            except:
                st_bid_price_is_media_only = 0
            try:
                st_exchange_type_for_run_on_all = st.xpath('''.//prop[@name = 'exchange_type_for_run_on_all']/@value''')[0]
            except:
                st_exchange_type_for_run_on_all = 0
            try:
                st_site_restriction_transparent_urls = st.xpath('''.//prop[@name = 'site_restriction_transparent_urls']/@value''')[0]
            except:
                st_site_restriction_transparent_urls = 0           
            try:
                st_audience_segment_include_op = st.xpath('''.//prop[@name = 'audience_segment_include_op']/@value''')[0]
            except:
                st_audience_segment_include_op = 0
            try:
                st_targeting_segment_include_op = st.xpath('''.//prop[@name = 'targeting_segment_include_op']/@value''')[0]
            except:
                st_targeting_segment_include_op = 0
            try:
                st_pixel_target_expr = st.xpath('''.//prop[@name = 'pixel_target_expr']/@value''')[0]
            except:
                st_pixel_target_expr = 0
            try:
                st_run_on_all_pmp = int(st.xpath('''.//prop[@name = 'run_on_all_pmp']/@value''')[0])
            except:
                st_run_on_all_pmp  = 0
            try:
                st_run_on_display = int(st.xpath('''.//prop[@name = 'run_on_display']/@value''')[0])
            except:
                st_run_on_display  = 0
            try:
                st_run_on_mobile = int(st.xpath('''.//prop[@name = 'run_on_mobile']/@value''')[0])
            except:
                st_run_on_mobile = 0

            version_data_tmp = [ca_id,
                                ca_name,
                                st_id,
                                st_name,
                                st_version,
                                st_status,
                                st_budget,
                                st_pacing_type,
                                st_freq_amt,
                                st_pacing_amt,
                                st_pacing_interval,
                                st_goal_type,
                                st_goal_value,
                                st_bid_aggressiveness,
                                st_updated_on,
                                st_max_bid,
                                st_min_bid,
                                st_effective_goal_value,
                                st_pixel_target_expr,
                                st_frequency_type,
                                st_frequency_interval,
                                st_description,
                                st_bid_price_is_media_only,
                                st_exchange_type_for_run_on_all, st_site_restriction_transparent_urls,st_audience_segment_include_op, 
                                st_targeting_segment_include_op, st_pixel_target_expr, st_run_on_all_pmp, st_run_on_display,st_run_on_mobile]
            

            version_data.append(version_data_tmp)  
            st_metadata_tmp = pd.DataFrame(version_data, columns = ['campaign_id',
                                                                    'campaign_name',
                                                                    'strategy_id',
                                                                    'strategy_name',
                                                                    'strategy_version',
                                                                    'strategy_status',
                                                                    'strategy_budget',
                                                                    'pacing_type',
                                                                    'frequency_amount',
                                                                    'pacing_amount',
                                                                    'pacing_interval',
                                                                    'goal_type',
                                                                    'goal_value',
                                                                    'bid_aggressiveness',
                                                                    'updated_on',
                                                                    'max_bid',
                                                                    'min_bid',
                                                                    'effective_goal_value',
                                                                    'pixel_target_expr',
                                                                    'frequency_type',
                                                                    'frequency_interval',
                                                                    'description',
                                                                    'bid_price_is_media_only',
                                                                    'exchange_type_for_run_on_all', 'restriction_transparent_urls','audience_include_op', 'cntx_include_op',
                                                                    'pixel_target', 'on_all_pmp', 'on_display', 'on_mobile'])
            
            if len(st_metadata) == 0:
                st_metadata = st_metadata_tmp
            else:
                st_metadata = pd.concat([st_metadata, st_metadata_tmp])
                      
        st_metadata_fin = st_metadata[(st_metadata['strategy_status'] == 1)]
        strategy_ids=st_metadata_fin['strategy_id'].values
        st_metadata_fin['campaign_name'] = st_metadata_fin['campaign_name'].str.split('_').str[0]
        st_metadata_final = st_metadata_fin[['campaign_id', 'campaign_name', 'strategy_id', 'strategy_name',  'frequency_type','frequency_amount', 'pacing_type', 'pacing_amount', 'min_bid', 'max_bid', 'goal_type',  'goal_value', 'bid_price_is_media_only']]
        return st_metadata_final


    def strategy_two_days_performance_id(self,strategy_ids):
        dimensions='campaign_id,strategy_id'
        metrics='impressions,clicks,total_conversions,total_spend'
        dt_today = date.today()
        dt = date.today() - timedelta(1)
        start_date = dt.strftime('%Y-%m-%d')
        end_date = dt.strftime('%Y-%m-%d')
        camp_perf_df = pd.DataFrame()
                
        for strategy_id in strategy_ids:
            
            camp_perf_df_tmp= self.t1_report(endpoint='performance', dimensions=dimensions,
                         filter='strategy_id='+str(strategy_id),
                         metrics=metrics,
                         precision='4',time_rollup='by_day',order='date',start_date=start_date,end_date=end_date)
    
            if len(camp_perf_df) == 0:
                camp_perf_df = camp_perf_df_tmp
            else:
                camp_perf_df = pd.concat([camp_perf_df, camp_perf_df_tmp])
        return camp_perf_df

    def winlos_report_id(self,strategy_ids):
        dimensions='organization_name,agency_name,advertiser_name,campaign_id,campaign_start_date,campaign_end_date,campaign_budget,strategy_id'
        metrics='average_bid_amount_cpm,average_win_amount_cpm,bid_rate,bids,matched_bid_opportunities,max_bid_amount_cpm,max_win_amount_cpm,min_bid_amount_cpm,min_win_amount_cpm,total_bid_amount_cpm,total_win_amount_cpm,win_rate,wins'
        dt = date.today() - timedelta(1)
        start_date = dt.strftime('%Y-%m-%d')
        end_date = dt.strftime('%Y-%m-%d')
        win_los_df = pd.DataFrame()
      
        for strategy_id in strategy_ids:
            win_los_df_tmp= self.t1_report(endpoint='win_loss', dimensions=dimensions,
                         filter='strategy_id='+str(strategy_id),
                         metrics=metrics,
                         precision='4',time_rollup='all',order='date',start_date=start_date,end_date=end_date)
            
            if len(win_los_df) == 0:
                win_los_df = win_los_df_tmp
            else:
                win_los_df = pd.concat([win_los_df, win_los_df_tmp])
        return win_los_df


    def underpacing_strategies_id(self,organization_id, strategy_ids):
        st_metadata_final = self.strategy_meta_data_id(strategy_ids)
        df_deals = self.get_deals(organization_id, strategy_ids)
        if len(df_deals) !=0:
            # df_deals = df_deals[['deal_id','deal_name','deal_identifier','deal_status','deal_floor_price','deal_creation_date']]
            df_deals = df_deals[['id','name','deal_identifier','status','price.value','created_on']]
            df_deals = df_deals.rename(columns = {'name' : 'deal_name',
                                                'deal_identifier' : 'deal_external_id',
                                                'id' : 'deal_id',
                                                'status' : 'deal_status',
                                                'price.value' : 'deal_floor_price',
                                                'created_on' : 'deal_creation_date'
                                                })

            df_deals = df_deals.sort_values('deal_name')
            df_deals_fin = df_deals[(df_deals['deal_status'] == True)]
        else:
            df_deals = pd.DataFrame(columns=['deal_name', 'deal_external_id', 'deal_id', 'deal_status','deal_floor_price','deal_creation_date'])
            df_deals_fin = df_deals
        df_dg_raw  = self.get_deal_groups(organization_id,strategy_ids)
        if len(df_dg_raw) !=0:
            df_dg_data = []
            df_dg_raw = df_dg_raw[['id','name','deal_ids', 'status']]
            for index, row in df_dg_raw.iterrows():
                for deal in row['deal_ids']:
                    r = [deal, row['id'], row['name'], row['status']]
                    df_dg_data.append(r)
            df_dg = pd.DataFrame(data=df_dg_data, columns=['deal_id','deal_group_id', 'deal_group_name', 'deal_group_status'])
            df_dg = df_dg[(df_dg['deal_group_status'] == True)]
        else:
            df_dg = pd.DataFrame(columns=['deal_id','deal_group_id', 'deal_group_name', 'deal_group_status'])
       

        str_deal_metadata = self.str_deal_metadata(strategy_ids)       
        str_deal_group_metadata = self.str_deal_group_metadata(strategy_ids)
        # 3.3. Combining all deals assigned to strategy
        str_dg_deal_metadata = pd.merge(str_deal_group_metadata, df_dg,  how='left', on=['deal_group_id'])
        str_dg_deal_metadata = str_dg_deal_metadata[['strategy_id','deal_id']]
        str_all_deals_metadata = pd.concat([str_dg_deal_metadata, str_deal_metadata])
        # 3.4. Getting deal price info
        str_deals = pd.merge(str_all_deals_metadata, df_deals_fin, how='left', left_on='deal_id', right_on='deal_id')
        str_deals['deal_floor_price'] = str_deals['deal_floor_price'].astype(float)
        str_deals_final = str_deals.groupby(['strategy_id']).agg({'deal_floor_price': ['mean', 'min', 'max']})
        str_deals_final.columns = ['deal_mean_price', 'deal_min_price', 'deal_max_price']
        str_deals_final = str_deals_final.reset_index()
        str_setup_overview = pd.merge(st_metadata_final, str_deals_final,  how='left', on=['strategy_id'])
        camp_perf_df = self.strategy_two_days_performance_id(strategy_ids)
        win_los_df = self.winlos_report_id(strategy_ids)
        # Final agregation
        str_correction = pd.merge(str_setup_overview, win_los_df ,  how='left', on=['campaign_id','strategy_id'])
        str_correction_perf = pd.merge(str_correction, camp_perf_df,  how='left', on=['campaign_id','strategy_id'])
        str_correction_final =  str_correction_perf[['campaign_id', 'campaign_name', 'strategy_id', 'strategy_name',
            'frequency_type', 'frequency_amount', 'pacing_type', 'pacing_amount','total_spend',
            'min_bid', 'max_bid', 'goal_type', 'goal_value',
            'deal_mean_price', 'deal_min_price',
            'deal_max_price', 'bid_rate', 'win_rate']]
        str_correction_final.columns = ['camp_id', 'geo', 'strategy_id', 'strategy_name',
            'f_type', 'f_amount', 'pacing_type', 'pacing', 'daily_spend',
            'min_bid', 'max_bid', 'goal', 'goal_value',
            'deal_mean', 'deal_min',
            'deal_max', 'bid_rate', 'win_rate']
        # str_correction = pd.merge(str_outpacing_fin_30, win_los_df, how='left', left_on='strategy_id', right_on='strategy_id')
        strategy_underpacing = str_correction_final.replace(np.nan,0)
        strategy_underpacing["min_bid"] = pd.to_numeric(strategy_underpacing["min_bid"])
        strategy_underpacing["deal_max"] = pd.to_numeric(strategy_underpacing["deal_max"])
        strategy_underpacing["max_bid"] = pd.to_numeric(strategy_underpacing["max_bid"])
        strategy_underpacing['goal_value'] = pd.to_numeric(strategy_underpacing['goal_value'])
        strategy_underpacing['daily_spend'] = pd.to_numeric(strategy_underpacing['daily_spend'])
        strategy_underpacing['win_rate'] = pd.to_numeric(strategy_underpacing['win_rate'])
        strategy_troubleshooting = strategy_underpacing[(strategy_underpacing['daily_spend'] < 0.05)&(strategy_underpacing['min_bid'] >= strategy_underpacing['deal_max'] )]
        strategy_tr_ids=strategy_troubleshooting['strategy_id'].values
        return strategy_underpacing






   


