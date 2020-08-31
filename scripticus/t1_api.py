import requests  
import pandas as pd
import numpy as np
from io import StringIO
import io as stringIOModule
import os
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import display
import json
import http.client
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
import time
import qds_sdk
import sqlalchemy
from qds_sdk.commands import *
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

# test test test

def t1_api_login(username,password,client_id,client_secret):
    response=requests.post('http://auth.mediamath.com/oauth/token',
                                    data={'grant_type': 'http://auth0.com/oauth/grant-type/password-realm',
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
    return resp


class T1_API():

    def __init__(self,username,password,client_id,client_secret):
        self.username=username
        self.password=password
        self.client_id=client_id
        self.client_secret=client_secret
        self.resp = t1_api_login(self.username,self.password, self.client_id,self.client_secret)


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
                
        camp_pacing_medicine = camp_metadata[['campaign_id','campaign_name','campaign_spend_cap_automatic', 'campaign_spend_cap_type',
                                            'campaign_spend_cap_amount','campaign_frequency_type','campaign_frequency_interval','campaign_frequency_amount']]
        return camp_pacing_medicine
    

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
                                                                    'pixel_target', 'on_all_pmp', 'on_display', 'on_mobile'])
            
            if len(st_metadata) == 0:
                st_metadata = st_metadata_tmp
            else:
                st_metadata = pd.concat([st_metadata, st_metadata_tmp])
                      
        st_metadata_fin = st_metadata[(st_metadata['strategy_status'] == 1)]
        strategy_ids=st_metadata_fin['strategy_id'].values
        st_metadata_final = st_metadata_fin[['campaign_id', 'campaign_name', 'strategy_id', 'strategy_name',  'frequency_type','frequency_amount', 'pacing_type', 'pacing_amount', 'min_bid', 'max_bid', 'goal_type',  'goal_value', 'bid_price_is_media_only']]
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
        dt_today = date.today()
        today=dt_today.strftime('%Y-%m-%d')
        dt = date.today() - timedelta(2)
        start_date = dt.strftime('%Y-%m-%d')
        end_date = dt.strftime('%Y-%m-%d')
        camp_goal_df = pd.DataFrame()
        for campaign_id in campaign_ids:
            url_devtech_org='https://api.mediamath.com/reporting/v1/std/performance?dimensions=campaign_id,campaign_goal_type,campaign_goal_value&filter=campaign_id={}&metrics=impressions,clicks,total_spend&precision=4&time_rollup=by_day&order=date&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)
            data = self.resp.json()
            sessionid=data['data']['session']['sessionid']
            conn = http.client.HTTPSConnection("api.mediamath.com")
            headers = { 'cookie': 'adama_session='+sessionid}
            conn.request("GET", url_devtech_org, headers=headers)
            camp_goal_df_tmp= pd.read_csv(conn.getresponse())
            if len(camp_goal_df) == 0:
                camp_goal_df = camp_goal_df_tmp
            else:
                camp_goal_df = pd.concat([camp_goal_df, camp_goal_df_tmp])
        camp_goal_id_df = camp_goal_df[['campaign_id','campaign_goal_type','campaign_goal_value']].rename(columns={"campaign_goal_type": "Goal"})
        return camp_goal_df, camp_goal_id_df


    def strategy_two_days_performance(self, campaign_ids):
        dt_today = date.today()
        dt = date.today() - timedelta(1)
        start_date = dt.strftime('%Y-%m-%d')
        end_date = dt.strftime('%Y-%m-%d')
        camp_perf_df = pd.DataFrame()
        for campaign_id in campaign_ids:
            url_perf='https://api.mediamath.com/reporting/v1/std/performance?dimensions=campaign_id,campaign_name,strategy_id,strategy_name&filter=campaign_id={}&metrics=impressions,clicks,total_conversions,total_spend&precision=4&time_rollup=by_day&order=date&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)

            data = self.resp.json()
            sessionid=data['data']['session']['sessionid']
            conn = http.client.HTTPSConnection("api.mediamath.com")
            headers = { 'cookie': 'adama_session='+sessionid}
            conn.request("GET", url_perf, headers=headers)
            camp_perf_df_tmp= pd.read_csv(conn.getresponse())
        

            if len(camp_perf_df) == 0:
                camp_perf_df = camp_perf_df_tmp
            else:
                camp_perf_df = pd.concat([camp_perf_df, camp_perf_df_tmp])
        return camp_perf_df

    def winlos_report(self, campaign_ids):
        dt = date.today() - timedelta(1)
        start_date = dt.strftime('%Y-%m-%d')
        end_date = dt.strftime('%Y-%m-%d')
        win_los_df = pd.DataFrame()
        for campaign_id in campaign_ids:
            url_winlos='https://api.mediamath.com/reporting/v1/std/win_loss?dimensions=organization_name,agency_name,advertiser_name,campaign_id,campaign_start_date,campaign_end_date,campaign_budget,strategy_id&filter=campaign_id={}&metrics=average_bid_amount_cpm,average_win_amount_cpm,bid_rate,bids,matched_bid_opportunities,max_bid_amount_cpm,max_win_amount_cpm,min_bid_amount_cpm,min_win_amount_cpm,total_bid_amount_cpm,total_win_amount_cpm,win_rate,wins&precision=2&time_rollup=all&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)

            data = self.resp.json()
            sessionid=data['data']['session']['sessionid']
            conn = http.client.HTTPSConnection("api.mediamath.com")
            headers = { 'cookie': 'adama_session='+sessionid}
            conn.request("GET", url_winlos, headers=headers)
            win_los_df_tmp= pd.read_csv(conn.getresponse())
            if len(win_los_df) == 0:
                win_los_df = win_los_df_tmp
            else:
                win_los_df = pd.concat([win_los_df, win_los_df_tmp])
        return win_los_df

    def get_deals(self, strategy_ids):
            data = self.resp.json()
            sessionid=data['data']['session']['sessionid']
            conn = http.client.HTTPSConnection("api.mediamath.com")
            headers = { 'cookie': 'adama_session='+sessionid}
            deals_df = pd.DataFrame()
            for strategy_id in strategy_ids:  
                offset = 0
                offset_total = 1
                while offset < offset_total:
                    url_page = 'https://api.mediamath.com/media/v1.0/deals?strategy_id={}&page_limit=1000&page_offset='.format(str(strategy_id)) + str(offset)
                    deals_data = requests.get(url_page, headers=headers).text
                    deals_df_tmp = pd.io.json.json_normalize(json.loads(deals_data)['data'])
                    meta_df = pd.io.json.json_normalize(json.loads(deals_data)['meta'])
                    offset_total = meta_df['total_count'][0]
                    offset = offset + 100
                if len(deals_df) == 0:
                    deals_df = deals_df_tmp
                else:
                    deals_df = pd.concat([deals_df, deals_df_tmp])
            return deals_df

    def get_deal_groups(self, strategy_ids):
            data = self.resp.json()
            sessionid=data['data']['session']['sessionid']
            conn = http.client.HTTPSConnection("api.mediamath.com")
            headers = { 'cookie': 'adama_session='+sessionid}
            df_dg = pd.DataFrame()
            for strategy_id in strategy_ids:  
                offset = 0
                offset_total = 1
                while offset < offset_total:
                    url_page = 'https://api.mediamath.com/media/v1.0/deal_groups?strategy_id={}&page_limit=100&page_offset='.format(str(strategy_id)) + str(offset)
                    deals_data = requests.get(url_page, headers=headers).text
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
        camp_pacing_medicine = self.campaign_meta_data(campaign_ids)
        camp_underpacing = pd.merge(campaigns_to_check, camp_pacing_medicine, how='left', left_on='Campaign ID', right_on='campaign_id')
        camp_underpacing_final = camp_underpacing[['Start Date','Advertiser Name', 'Campaign ID','Campaign Name', 'Spend To Pace', 'Spend Yesterday','campaign_spend_cap_type', 'campaign_spend_cap_automatic','campaign_spend_cap_amount','campaign_frequency_type', 
                                                'campaign_frequency_interval', 'campaign_frequency_amount']].rename(columns={"Start Date": "start_date",'Advertiser Name':'advertiser_name',
                                                                                'Campaign ID':'campaign_id','Campaign Name':'campaign_name','campaign_spend_cap_type':'spend_cap_type','campaign_spend_cap_amount':'cap_amount','campaign_frequency_type':'frequency_type',
                                                                                'Spend To Pace':'spend_to_pace','Spend Yesterday':'spend_yesterday','campaign_frequency_interval':'frequency_interval', 'campaign_frequency_amount':'frequency_amount' }).sort_values(by=
                                                                                'spend_to_pace', ascending=False)
        camp_underpacing_final["frequency_amount"] = pd.to_numeric(camp_underpacing_final["frequency_amount"])
        camp_underpacing_final["cap_amount"] = pd.to_numeric(camp_underpacing_final["cap_amount"])
        camp_underpacing_final.drop(['campaign_spend_cap_automatic'], axis=1, inplace=True)
        return camp_underpacing_final

    def underpacing_strategies(self, campaign_ids, organization_id):
        strategy_ids, st_metadata_final = self.strategy_meta_data(campaign_ids)
        df_deals = self.get_deals(strategy_ids)
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
        
        df_dg_raw  = self.get_deal_groups(strategy_ids)
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
        return strategy_underpacing, strategy_tr_ids

      

        # video = '/reporting/v1/std/video?dimensions=campaign_name,strategy_name,exchange_name,creative_name,concept_name,creative_size&filter=campaign_id={}&metrics=impressions,clicks,post_click_conversions,post_view_conversions,total_conversions,total_spend,media_cost,video_start,video_complete,viewability_rate_100_percent,viewability_rate&precision=4&time_rollup=by_day&order=date&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)
# geo = '/reporting/v1/std/geo?dimensions=agency_name,advertiser_name,country_name,region_name,metro_name,campaign_id,campaign_name,campaign_start_date,campaign_end_date,campaign_budget&filter=campaign_id={}&metrics=impressions,clicks,post_click_conversions,post_view_conversions,total_conversions,media_cost,total_ad_cost,total_spend,video_start,video_complete&time_rollup=all&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)
# device_technology='/reporting/v1/std/device_technology?dimensions=organization_name,advertiser_name,campaign_id,campaign_name,campaign_start_date,campaign_end_date,campaign_currency_code,campaign_timezone,connection_type,device_type,os_version,inventory_type,browser,exchange_name,strategy_id,strategy_name&filter=campaign_id={}&metrics=impressions,clicks,post_click_conversions,post_view_conversions,total_conversions,total_spend,video_start,video_complete&precision=4&time_rollup=by_day&order=date&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)
# reach_frequency='/reporting/v1/std/reach_frequency?dimensions=organization_id,organization_name,agency_name,advertiser_name,campaign_id,campaign_name,frequency,frequency_bin&filter=campaign_id={}&metrics=impressions,uniques,clicks,post_click_conversions,post_view_conversions,total_conversions&time_window=last_30_days&time_rollup=all'.format(campaign_id)
# site_transparency='/reporting/v1/std/site_transparency?dimensions=site_domain,organization_name,agency_name,advertiser_name,campaign_id,campaign_name,campaign_start_date,campaign_end_date,campaign_budget,exchange_name,strategy_id,strategy_name&filter=campaign_id={}&metrics=impressions,clicks,post_click_conversions,post_view_conversions,total_conversions,media_cost,total_ad_cost,total_spend,video_start,video_complete&time_window=campaign_to_date&time_rollup=all'.format(campaign_id)
# contextual_insights='/reporting/v1/std/contextual_insights?dimensions=organization_name,agency_name,advertiser_name,campaign_id,campaign_name,campaign_start_date,campaign_end_date,campaign_budget,exchange_name,path,target_id,target_name,vendor_id,vendor_name&filter=campaign_id={}&metrics=impressions,clicks,post_click_conversions,post_view_conversions,total_conversions,media_cost,total_ad_cost,total_spend&precision=2&time_rollup=all&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)	
# audience_index_pixel='/reporting/v1/std/audience_index_pixel?dimensions=advertiser_id,pixel_external_id,pixel_id,pixel_name,pixel_tag_type,pixel_type,audience_id,audience_name,audience_path&filter=pixel_id={}&metrics=matched_users,audience_index&time_window=last_14_days&time_rollup=all'.format(pixel_id)
# day_part= '/reporting/v1/std/day_part?dimensions=campaign_name,strategy_name,exchange_name,day_part_name,weekday_name&filter=campaign_id={}&metrics=impressions,clicks,post_click_conversions,post_view_conversions,total_conversions,total_spend,media_cost,video_start,video_complete&precision=4&time_window=campaign_to_date&time_rollup=all'.format(campaign_id) 
# win_loss='/reporting/v1/std/win_loss?dimensions=organization_name,agency_name,advertiser_name,campaign_id,campaign_name,campaign_start_date,campaign_end_date,campaign_budget,strategy_name&filter=campaign_id={}&metrics=average_bid_amount_cpm,average_win_amount_cpm,bid_rate,bids,matched_bid_opportunities,max_bid_amount_cpm,max_win_amount_cpm,min_bid_amount_cpm,min_win_amount_cpm,total_bid_amount_cpm,total_win_amount_cpm,win_rate,wins&precision=2&time_rollup=all&start_date={}&end_date={}'.format(campaign_id, start_date, end_date)	
# postal_code='/reporting/v1/std/postal_code?dimensions=organization_id,organization_name,agency_name,advertiser_name,campaign_id,campaign_name,strategy_id,strategy_name,postal_code&filter=campaign_id={}&metrics=impressions,clicks,post_click_conversions,post_view_conversions,total_conversions,video_start,video_complete&time_window=last_30_days&time_rollup=all'.format(campaign_id)
# site_transparency='/reporting/v1/std/site_transparency?dimensions=site_domain,organization_name,agency_name,advertiser_name,campaign_id,campaign_name,campaign_start_date,campaign_end_date,campaign_budget&filter=campaign_id={}&metrics=impressions,clicks,post_click_conversions,post_view_conversions,total_conversions,media_cost,total_ad_cost,total_spend,video_start,video_complete,viewability_rate_100_percent,viewability_rate&time_window=campaign_to_date&time_rollup=all'.format(campaign_id)
# brain_feature_value = 'https://t1.mediamath.com/reporting/v1/std/brain_feature_value?dimensions=campaign_id,feature,feature_report_type,feature_value,index,mean,model_goal,position&filter=campaign_id={}&precision=4&time_rollup=all&order=date&time_windows=last_30_days'.format(campaign_id)
# watermark = '/reporting/v1/std/watermark?dimensions=organization_name,agency_name,advertiser_name,campaign_id,campaign_name,strategy_name,campaign_start_date,campaign_end_date,campaign_budget&filter=campaign_id={}&metrics=non_watermark_impressions,watermark_impressions,watermark_spend,non_watermark_spend&precision=2&time_rollup=all&time_window=last_3_days'.format(campaign_id)	
# brain_feature_summary = '/reporting/v1/std/brain_feature_summary?dimensions=campaign_id,feature,index,model_goal,position&filter=campaign_id={}&precision=4&time_rollup=by_day&time_window=last_30_days'.format(campaign_id)
# day_part= '/reporting/v1/std/day_part?dimensions=campaign_name,strategy_name,exchange_name,day_part_name,weekday_name&filter=campaign_id={}&metrics=impressions,clicks,post_click_conversions,post_view_conversions,total_conversions,total_spend,media_cost,video_start,video_complete&precision=4&time_window=campaign_to_date&time_rollup=all'.format(campaign_id) 


# def t1_report(uri, dimensions,metrics, sortby, ascending):
#     data = resp.json()
#     sessionid=data['data']['session']['sessionid']
#     conn = http.client.HTTPSConnection("api.mediamath.com")
#     headers = { 'cookie': 'adama_session='+sessionid}
#     conn.request("GET", uri, headers=headers)
#     df = pd.read_csv(conn.getresponse())
#     if dimensions != None:
#         if (uri != day_part)& (uri != site_transparency):
#             df['start_date'] = pd.to_datetime(df['start_date'].astype(str), format='%Y/%m/%d')
#             df['week_number'] = df['start_date'].dt.week
#         columns=dimensions+metrics
#         df = df[columns].groupby(dimensions, as_index=False).sum()
#         df['CPM'] = (df.total_spend*1000)/df.impressions
#         df['CTR'] = df.clicks/df.impressions
#         df['CPC'] = df.total_spend/df.clicks
#         df['CPA'] = df.total_spend/df.total_conversions
#         df['RR'] = df.total_conversions/(df.impressions/1000)
#         df=df.sort_values(by=sortby, ascending=ascending)
#         df.replace([np.inf, -np.inf], np.nan)
    
#     return df

# def table_style(df,color,kpi):
#     cm = sns.light_palette(color, as_cmap=True)
#     format_dict = {'total_spend':'{0:,.1f}', 'total_revenue':'{0:,.1f}','NDC':'{0:,.2f}','LP':'{0:,.0f}', 'CPA_LP':'{0:,.2f}','CPC':'{0:,.2f}','CPA_Signup':'{0:,.2f}','CTR': '{:.2%}', 'CPA_NDC':'{0:,.1f}','CPA_DC':'{0:,.1f}','ROI': '{:.2f}','CPM': '{0:,.1f}', 'vCPM': '{0:,.1f}','CPC': '{0:,.1f}','CPA': '{0:,.1f}'}
#     stdf = df.style.background_gradient(cmap=cm, subset=kpi).format(format_dict).hide_index()
#     return stdf


