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
from pandas.io.json import json_normalize
import http.client
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
import time
import qds_sdk
import sqlalchemy
import qds_sdk
from qds_sdk.qubole import Qubole
from qds_sdk.commands import Command
from qds_sdk.commands import HiveCommand
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.text import Text
import datetime
import matplotlib.dates as mdates
from pathlib import Path
from lxml import etree
import io
from datetime import datetime, date, timedelta
import re
# import warnings
# warnings.filterwarnings('ignore')

def t1_api_login(username,password,client_id,client_secret):
    response=requests.post('http://auth.mediamath.com/oauth/token',
                                    data={'grant_type': 'http://auth0.com/oauth/grant-type/password-realm',
                                            'username': username,
                                            'password': password,
                                            'client_id': client_id,
                                            'client_secret': client_secret,
                                            'realm': 'MediaMathActiveDirectory'
                                            }) 
    s = requests.session()
    s.headers.update({'Accept': 'application/vnd.mediamath.v1+json','Authorization': 'Bearer {}'.format(response.json()['access_token'])})
    resp = s.get('https://api.mediamath.com/api/v2.0/session')
    return s

class T1_API():

    def __init__(self,username,password,client_id,client_secret):
        self.username=username
        self.password=password
        self.client_id=client_id
        self.client_secret=client_secret
        self.session = t1_api_login(self.username,self.password, self.client_id,self.client_secret)
        # picard = 'https://api.mediamath.com/reporting/v1/std/'
        # r = self.session.get(picard + 'meta')
        # self.available_endpoints = list(json.loads(r.content)['reports'].keys())
        # self.available_endpoints.extend([
        #     'performance_usd',
        #     'performance_viewability',
        #     'site_transparency_viewability',
        #     'deals',
        #     'performance_aggregated',
        #     'performance_streaming'
        #     ])
        # try:
        #     assert endpoint in self.available_endpoints
        #     self.endpoint = endpoint
        #     if self.endpoint == 'deals': # This is while 'deals' is in beta
        #         r = self.session.get('https://api.mediamath.com/reporting-beta/v1/std/deals/meta')
        #     else:
        #         r = self.session.get(picard + endpoint + '/meta')
        #     self.info = json.loads(r.content)
        # except AssertionError:
        #     raise AssertionError('{} is not a supported endpoint.  Endpoint must be one of the following: {}'.format(endpoint,self.available_endpoints))
        # except Exception as err:
        #     print(r.content)
        #     raise Exception(err)

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

        if  endpoint in (
            'performance_usd',
            'performance_viewability',
            'site_transparency_viewability',
            'performance_aggregated',
            'performance_streaming'
            ):
            data_url = picard + endpoint
        elif endpoint == 'deals':
            data_url = 'https://api.mediamath.com/reporting-beta/v1/std/deals' 
        
        self.response = self.session.get(data_url, params=params, headers={'Accept-Encoding':'identity','Connection':'close'})
        data = self.response.content
        df = pd.read_csv(BytesIO(data))
        return df

    def pivot(self, df, dimensions, metrics, kpi, sortby, ascending):
        if 'week' in dimensions:
            df['start_date'] = pd.to_datetime(df['start_date'].astype(str), format='%Y/%m/%d')
            df['week'] = df['start_date'].dt.week
        if 'start_date' in dimensions:    
            df['start_date'] = pd.to_datetime(df['start_date'].astype(str), format='%Y/%m/%d')

            # data_performance['month'] = data_performance['start_date'].dt.to_period('M')
        columns=dimensions+metrics
        df = df[columns].groupby(dimensions).sum().reset_index()

        if 'CPM' in kpi:    
            df['CPM'] = (df.total_spend*1000)/df.impressions
        if 'CTR' in kpi:  
            df['CTR'] = df.clicks/df.impressions
        if 'CPC' in kpi:  
            df['CPC'] = df.total_spend/df.clicks
        if 'CPA' in kpi:  
            df['CPA'] = df.total_spend/df.total_conversions
        if 'CPA_pc' in kpi:  
            df['CPA_pc'] = df.total_spend/df.post_click_conversions
        if 'RR' in kpi:  
            df['RR'] = df.total_conversions/(df.impressions/1000)
        if 'VR' in kpi:  
            df['VR'] = df.in_view/df.measurable
        if 'ROI' in kpi:  
            df['ROI'] = df.total_revenue/df.total_spend
        if 'VCR' in kpi:
            df['VCR'] = df.video_complete/df.video_start
        df=df.sort_values(by=sortby, ascending=ascending)
        df.replace([np.inf, -np.inf], np.nan)
        return df 
   
    def create_site_list(self,organization_id,new_list_name,content,sltype):
        url = "https://api.mediamath.com/api/v2.0/site_lists/upload"
        payload = "organization_id={organization_id}&name={new_list_name}&status=on&content={content}&restriction={sltype}".format(
        organization_id=organization_id, new_list_name=new_list_name,content=content,sltype=sltype)
        headers = {'content-type': 'application/x-www-form-urlencoded','Accept': 'application/vnd.mediamath.v1+json'}
        self.response = self.session.post(url,data=payload,headers=headers)
        r = json.loads(self.response.content)
        new_list_id = r['data']['id']
        return new_list_id

    def assign_list_to_strategy(self,list_id,strategy_id):
        url = 'https://api.mediamath.com/api/v2.0/strategies/{}/site_lists'.format(strategy_id)
        payload = 'site_lists.1.id={}&site_lists.1.assigned={}'.format(list_id,'strategy')
        headers = {
        'accept': "application/vnd.mediamath.v1+json",
        'content-type': "application/x-www-form-urlencoded"
        }
        self.response = self.session.post(url,data=payload,headers=headers)
        check_list_assignment(list_id)
        return response
    
    def check_list_assignment(self,list_id):
        url = "https://api.mediamath.com/api/v2.0/site_lists/{}/assignments".format(list_id)
        payload = "full=*"
        headers = {'content-type': 'application/x-www-form-urlencoded',
                'Accept': 'application/vnd.mediamath.v1+json'}
        self.response = self.session.get(url, data=payload, headers=headers)
        print(json.loads(self.response.content))

    def all_creative_concepts(self,advertiser_id):
        concept_metadata = pd.DataFrame()
        for campaign_id in campaign_ids:
            url = "https://api.mediamath.com/api/v2.0/concepts/limit/advertiser={}".format(advertiser_id)
            payload = "full=*"
            headers = {'content-type': 'application/x-www-form-urlencoded','Accept': 'application/vnd.mediamath.v1+json'}
            self.response = self.session.get(url,data=payload,headers=headers)
            r = json.loads(self.response.content)
            concept_metadata_tmp = json_normalize(r['data'])
            if len(camp_goal_df) == 0:
                concept_metadata = concept_metadata_tmp
            else:
                concept_metadata = pd.concat([concept_metadata, concept_metadata_tmp])
        return concept_metadata


def filter_strategy_site_lists(strategy_site_lists,keyword):
    autoblacklists = [i for i in strategy_site_lists if keyword in i['name']]
    if len(autoblacklists) == 0:
        return None
    else:
        return autoblacklists


def find_replace_multi(string, dictionary):
    for item in dictionary.keys():
        string = re.sub(item, dictionary[item], string)
    return string

def qubole(api_token,sql,replacements,filename):
    Qubole.configure(api_token=api_token)
    with open(sql,'r') as f:
        query = f.read()

    query = find_replace_multi(query,replacements)
    hc = HiveCommand.run(query=query)
    cmd = Command.find(hc.id)
    out_file = filename + '.csv'
    
    with open(out_file, 'wb') as writer:
        cmd.get_results(writer, delim='\t', inline=False)

    df = pd.read_csv(out_file, delimiter='\t')

    return df

