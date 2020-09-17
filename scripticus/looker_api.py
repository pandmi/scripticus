import logging
from io import StringIO
import io as stringIOModule
import requests
import pandas as pd
import numpy as np
import os
import json
import http.client
import time


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(str(time.strftime("%d_%m_%Y")) +"_looker_API_Calls" + ".log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Testing Logs')

class LookerAPI(object):
    """Class that contains methods and variables related to looker API authentication"""
    def __init__(self, api_info):
        self.api_endpoint = api_info['api_endpoint']
        self.client_secret = api_info['client_secret']
        self.client_id = api_info['client_id']
        self.login_endpoint = api_info['login_url']
        # print(self.login_endpoint)

    def login(self):
        "login to looker API"
        try:
            auth_data = {'client_id':self.client_id, 'client_secret':self.client_secret}
            r = requests.post( self.login_endpoint,data=auth_data) # error handle here
            json_auth = json.loads(r.text)['access_token']
            return json_auth
        except requests.exceptions.RequestException as e:
            logger.error(e)

    def run_look(self, look_id, json_auth,return_format='csv'):
        "run looks"
        try:
            look_run_url = self.api_endpoint + '/looks/{0}/run/{1}'.format(look_id,return_format)
            #r = requests.get(look_run_url, headers={'Authorization': "token " + json_auth})
            r = requests.get(look_run_url + '?' + 'access_token=' + json_auth)
            return r.text
        except requests.exceptions.RequestException as e:
            logger.error(e)

    def run_query(self, query_id, json_auth, return_format='csv'):
        "run query"
        try:
            query_run_url = self.api_endpoint + '/queries/{0}/run/{1}'.format(query_id,return_format)
            #r = requests.get(query_run_url, headers={'Authorization': "token " + json_auth})
            r = requests.get(query_run_url + '?' + 'access_token=' + json_auth)
            return r.text
        except requests.exceptions.RequestException as e:
            logger.error(e)

    def run_query_slug(self, query_slug, json_auth):
        "run query"
        try:
            query_slug_run_url = self.api_endpoint + '/queries/slug/{0}'.format(query_slug)
            #r = requests.get(query_run_url, headers={'Authorization': "token " + json_auth})
            r = requests.get(query_slug_run_url + '?' + 'access_token=' + json_auth)
            qid=json.loads(r.text)["id"]
            # print("Query_id: " + str(qid))
            return LookerAPI.run_query(self, qid, json_auth)
        except requests.exceptions.RequestException as e:
            logger.error(e)


def looker_df(slug, organization_id, organisation_name, advertiser_filter, pct_treshold, up_treshold, creds):
        demo = LookerAPI(creds) 
        json_auth = demo.login()
        query_response = demo.run_query_slug(slug, json_auth)
        records = pd.read_csv(StringIO(query_response), parse_dates=True )
        records.columns = ['Organization',
            'Agency',
            'Advertiser Name',
            'Campaign ID',
            'Campaign Name',
            'Managed Service Flag',
            'Start Date',
            'End Date',
            'Days Remaining',
            'Currency',
            'Total Budget',
            'Budget Remaining',
            'Spend Yesterday',
            'Pacing Ratio',
            'Spend To Pace',
            'Projected Spend - real time',
            'Missed spend - Yesterday',
            'Latest Hour of Activity',
            'Pacing exclusion']
        records['Value Underpacing'] =records['Spend To Pace'] -records['Spend Yesterday']
        records_pacing=records[['Start Date','End Date','Organization', 'Advertiser Name','Campaign ID','Campaign Name','Days Remaining', 'Spend To Pace','Spend Yesterday','Value Underpacing','Pacing Ratio', 'Latest Hour of Activity']]
        if pct_treshold is None:
            up_campaigns_to_check = records_pacing[(records_pacing['Organization'] == organisation_name)&(records_pacing['Days Remaining'] > 1)&(records_pacing['Advertiser Name'].str.contains(advertiser_filter))].sort_values(by='Spend To Pace', ascending=False)
        if advertiser_filter and pct_treshold:
            up_campaigns_to_check = records_pacing[(records_pacing['Pacing Ratio'] < pct_treshold)&(records_pacing['Value Underpacing'] > up_treshold)&(records_pacing['Organization'] == organisation_name)&(records_pacing['Days Remaining'] > 1)&(records_pacing['Advertiser Name'].str.contains(advertiser_filter))].sort_values(by='Value Underpacing', ascending=False)
        else: 
            up_campaigns_to_check = records_pacing[(records_pacing['Pacing Ratio'] < pct_treshold)&(records_pacing['Value Underpacing'] > up_treshold)&(records_pacing['Organization'] == organisation_name)&(records_pacing['Days Remaining'] > 1)&(records_pacing['Advertiser Name'].str.contains(advertiser_filter))].sort_values(by='Value Underpacing', ascending=False)
        up_campaigns_ids=up_campaigns_to_check['Campaign ID'].values
        if advertiser_filter:
             op_campaigns_to_check = records_pacing[(records_pacing['Organization'] == organisation_name) & (records_pacing['Advertiser Name'].str.contains(advertiser_filter))].sort_values(by='Latest Hour of Activity', ascending=True)
        else: 
            op_campaigns_to_check = records_pacing[(records_pacing['Organization'] == organisation_name)].sort_values(by='Latest Hour of Activity', ascending=True)
        op_campaigns_ids=op_campaigns_to_check['Campaign ID'].values
        return up_campaigns_to_check, op_campaigns_to_check, up_campaigns_ids,op_campaigns_ids

    def looker_df_lv(slug, organization_id, organisation_name, advertiser_filter,creds):
        demo = LookerAPI(creds) 
        json_auth = demo.login()
        query_response = demo.run_query_slug(slug, json_auth)
        records = pd.read_csv(StringIO(query_response), parse_dates=True )
        records.columns = ['Organization',
            'Agency',
            'Advertiser Name',
            'Campaign ID',
            'Campaign Name',
            'Managed Service Flag',
            'Start Date',
            'End Date',
            'Days Remaining',
            'Currency',
            'Total Budget',
            'Budget Remaining',
            'Spend Yesterday',
            'Pacing Ratio',
            'Spend To Pace',
            'Projected Spend - real time',
            'Missed spend - Yesterday',
            'Latest Hour of Activity',
            'Pacing exclusion']
        records['Value Underpacing'] =records['Spend To Pace'] -records['Spend Yesterday']
        records_pacing=records[['Start Date','End Date','Organization', 'Advertiser Name','Campaign ID','Campaign Name','Days Remaining', 'Spend To Pace','Spend Yesterday','Value Underpacing','Pacing Ratio', 'Latest Hour of Activity']]
        if advertiser_filter:
            up_campaigns_to_check = records_pacing[(records_pacing['Organization'] == organisation_name)&(records_pacing['Days Remaining'] > 1)&(records_pacing['Advertiser Name'].str.contains(advertiser_filter))].sort_values(by='Value Underpacing', ascending=False)
        else: 
            up_campaigns_to_check = records_pacing[(records_pacing['Organization'] == organisation_name)&(records_pacing['Days Remaining'] > 1)&(records_pacing['Advertiser Name'].str.contains(advertiser_filter))].sort_values(by='Value Underpacing', ascending=False)
        up_campaigns_ids=up_campaigns_to_check['Campaign ID'].values
        if advertiser_filter:
             op_campaigns_to_check = records_pacing[(records_pacing['Organization'] == organisation_name) & (records_pacing['Advertiser Name'].str.contains(advertiser_filter))].sort_values(by='Latest Hour of Activity', ascending=True)
        else: 
            op_campaigns_to_check = records_pacing[(records_pacing['Organization'] == organisation_name)].sort_values(by='Latest Hour of Activity', ascending=True)
        op_campaigns_ids=op_campaigns_to_check['Campaign ID'].values
        return up_campaigns_to_check, op_campaigns_to_check, up_campaigns_ids,op_campaigns_ids