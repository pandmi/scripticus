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
from databricks import sql


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
        if 'CPM_usd' in kpi:    
            df['CPM_usd'] = (df.total_spend_usd*1000)/df.impressions
        if 'CTR' in kpi:  
            df['CTR'] = df.clicks/df.impressions
        if 'CPC' in kpi:  
            df['CPC'] = df.total_spend/df.clicks
        if 'CPC_usd' in kpi:  
            df['CPC_usd'] = df.total_spend_usd/df.clicks
        if 'CPA' in kpi:  
            df['CPA'] = df.total_spend/df.total_conversions
        if 'CPA_usd' in kpi:  
            df['CPA_usd'] = df.total_spend_usd/df.total_conversions
        if 'CPA_pc' in kpi:  
            df['CPA_pc'] = df.total_spend/df.post_click_conversions
        if 'RR' in kpi:  
            df['RR'] = df.total_conversions/(df.impressions/1000)
        if 'VR' in kpi:  
            df['VR'] = df.in_view/df.measurable
        if 'ROI' in kpi:  
            df['ROI'] = df.total_revenue/df.total_spend
        if 'ROI_usd' in kpi:  
            df['ROI_usd'] = df.total_revenue_usd/df.total_spend_usd
        if 'SSP_fee_pct' in kpi:  
            df['SSP_fee_pct'] = df.ssp_technology_fee_usd/df.media_cost_usd
        if 'VCR' in kpi:
            df['VCR'] = df.video_complete/df.video_start
        if 'CPCV' in kpi:
            df['CPCV'] = df.total_spend/df.video_complete
        if 'mCPM' in kpi:
            df['mCPM'] = (df.media_cost*1000)/df.impressions
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

    # def assign_list_to_strategy(self,list_id,strategy_id):
    #     url = 'https://api.mediamath.com/api/v2.0/strategies/{}/site_lists'.format(strategy_id)
    #     payload = 'site_lists.1.id={}&site_lists.1.assigned={}'.format(list_id,'strategy')
    #     headers = {
    #     'accept': "application/vnd.mediamath.v1+json",
    #     'content-type': "application/x-www-form-urlencoded"
    #     }
    #     self.response = self.session.post(url,data=payload,headers=headers)
    #     check_list_assignment(list_id)
    #     return response
    
    # def check_list_assignment(self,list_id):
    #     url = "https://api.mediamath.com/api/v2.0/site_lists/{}/assignments".format(list_id)
    #     payload = "full=*"
    #     headers = {'content-type': 'application/x-www-form-urlencoded',
    #             'Accept': 'application/vnd.mediamath.v1+json'}
    #     self.response = self.session.get(url, data=payload, headers=headers)
    #     print(json.loads(self.response.content))

    def all_creative_concepts(self,advertiser_ids):
        concept_metadata = pd.DataFrame()
        for advertiser_id in advertiser_ids:
            url = "https://api.mediamath.com/api/v2.0/concepts/limit/advertiser={}".format(advertiser_id)
            payload = "full=*"
            headers = {'content-type': 'application/x-www-form-urlencoded','Accept': 'application/vnd.mediamath.v1+json'}
            self.response = self.session.get(url,data=payload,headers=headers)
            r = json.loads(self.response.content)
            concept_metadata_tmp = json_normalize(r['data'])
            if len(concept_metadata) == 0:
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

def databricks(db_token,sqlfile,replacements):
    connection = sql.connect(
      server_hostname='mediamath-analytics-datascience.cloud.databricks.com',
      http_path='/sql/1.0/warehouses/b3c3e75a3b658371',
      access_token=db_token)

    cursor = connection.cursor()

    with open(sqlfile,'r') as f:
        query = f.read()

    query = find_replace_multi(query,replacements)


    cursor.execute(query)
    result = cursor.fetchall()
    df = pd.DataFrame.from_records(result)
    df.columns=[x[0] for x in cursor.description ]

    cursor.close()
    connection.close()
    return df

def qubole(api_token,sql,replacements,filename):
    Qubole.configure(api_token=api_token)
    with open(sql,'r') as f:
        query = f.read()
        
    label='Trading-spark'
    query = find_replace_multi(query,replacements)
    hc = HiveCommand.run(query=query, label=label)
    cmd = Command.find(hc.id)
    out_file = filename + '.csv'
    
    with open(out_file, 'wb') as writer:
        cmd.get_results(writer)

    df = pd.read_csv(out_file, delimiter='\t')

    return df

def find_replace_multi(string, dictionary):
    for item in dictionary.keys():
        string = re.sub(item, dictionary[item], string)
    return string
    
def qubole_by_id(api_token,hcid,filename):
    Qubole.configure(api_token=api_token)
    cmd = Command.find(hcid)
    out_file = filename + '.csv'
    with open(out_file, 'wb') as writer:
        cmd.get_results(writer)

    df = pd.read_csv(out_file, delimiter='\t')

    return df

def qubole_by_id_nd(api_token,hcid,filename):
    Qubole.configure(api_token=api_token)
    cmd = Command.find(hcid)
    out_file = filename + '.csv'
    with open(out_file, 'wb') as writer:
        cmd.get_results(writer)

    df = pd.read_csv(out_file)

    return df

def qubole_by_id_raw(api_token,hcid,filename):
    Qubole.configure(api_token=api_token)
    cmd = Command.find(hcid)
    out_file = filename + '.csv'
    with open(out_file, 'wb') as writer:
        cmd.get_results(writer)

    return out_file



def creative_classifier(df,creative,name_in_strategy):
    df2=df[df['name'].str.contains(creative)]
    dft = df2[['name', 'id']]
    df_nc =dft.drop_duplicates(subset=None, keep='first', inplace=False)
    df_nc['Language'] = df_nc['name'].str.split('_').str[3].str.upper()
    df_nc['Country'] = df_nc['name'].str.split('_').str[1]
    df_nc['Vertical'] = df_nc['name'].str.split('_').str[4].replace('DIS', "DESK") 
    df_nc['Creative'] = name_in_strategy
    df_nc['Identifier'] = df_nc['name'].str.split('_').str[4]
    x=df_nc['Country']+'_'+df_nc['Vertical']+'_'+df_nc['Creative'] +'_'+df_nc['Language']
    y=df_nc['Country']+'_'+df_nc['Vertical']+'_'+df_nc['Creative'] +'_'
    df_nc['Creative_identifyer']  = np.where(df_nc['Country']=='CH', x, y)
    df_nc_creatives=df_nc.groupby('Creative_identifyer').agg({'id': list})
    df_nc_creatives['id'] = ('; '.join(map(str,df_nc_creatives.id))).replace('[', '').replace(']', '')
    return df_nc_creatives

