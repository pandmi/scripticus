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
        data = response.content
        df = pd.read_csv(BytesIO(data))
        return df

