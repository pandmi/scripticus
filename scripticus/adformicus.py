from datetime import date
from datetime import datetime, timedelta
from google_auth_oauthlib.flow import InstalledAppFlow
from google.api_core import retry
from google.auth import credentials as google_auth_credentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient import discovery
from googleapiclient import http
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pathlib import Path
from requests.auth import HTTPBasicAuth
from six.moves.urllib.request import urlopen
import base64
import csv
import datetime
import datetime as dt
import hashlib
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
import requests
import seaborn as sns
import time
import ast

# Singular

# Singular

def sng_create_async_report(dimensions, metrics, cohort_metrics,  start_date, end_date, api_key, time_breakdown="day", report_format="csv", cohort_periods='ltv'):
    payload = {
        "dimensions": dimensions,
        "metrics": metrics,
        "cohort_metrics": cohort_metrics,

        "start_date": start_date,
        "end_date": end_date,
        "format": report_format,
        "time_breakdown": time_breakdown,
        "cohort_periods": cohort_periods
    }
    params = {"api_key": api_key}
    response = requests.post("https://api.singular.net/api/v2.0/create_async_report", params=params, json=payload)
    response.raise_for_status()
    return response.json()["value"]["report_id"]



# Function to check report status and get the download URL
def sng_get_report_download_url(report_id, api_key):
    params = {"report_id": report_id, "api_key": api_key}
    while True:
        response = requests.get("https://api.singular.net/api/v2.0/get_report_status", params=params)
        response.raise_for_status()
        status_data = response.json()["value"]
        if status_data["status"] == "DONE":
            return status_data["download_url"]
        elif status_data["status"] == "FAILED":
            raise Exception("Report generation failed.")
        time.sleep(10)  # Wait before retrying


# Singular!!!
def sng_source2networks(value):
    mappings = {
        'adform': 'Real Time Bidding (Media)',
        'applesearchads': 'Apple Search Ads',
        'addressable': 'Addressable (Media)',
        'coingecko': 'Coingecko (Media)',
        'geckoterminal': 'Geckoterminal (Media)',
         'dailymotion':'DailyMotion (Media)',
        'match2one':'Match2One (Media)',
        'pubfuture':'PubFuture (Media)', 
        'realtimebidding':'Real Time Bidding (Media)',
        'wallstmemes':'wallstmemes (Media)',    
        'coinmarketcap': 'CoinMarketCap (Media)',
        'coinzilla': 'Coinzilla (Dextools)', 
        'dexscreener': 'DexScreener (Media)',
        'etherscan': 'Etherscan (Media)',
        'ethscan/bscscan': 'Etherscan (Media)',
        'exploreads': 'Explorads (Media)',
        'hueads': 'Hue Ads (Media)',
        'mediamath': 'MediaMath (Media)',
        'twitter': 'Twitter (Media)',
        'moloco': 'Moloco',
        'kayzen': 'Kayzen',
        'persona.lyrtb':'Personaly',
        'meta(custom)': 'Meta (Media)',
        'meta': 'Meta (Media)',
        'coincarp':'CoinCarp (Media)',
        'datawrkz':'Datawrkz (Media)',
        'coincarp(custom)':'CoinCarp (Media)',
        'adwords':'Google Ads',
        'exoclick': 'Exoclick',
         'exoclick(custom)': 'Exoclick'


    }
    
    value_str = str(value)  # Convert value to string for substring checks

    # Check for main mapping matches
    for key, replacement in mappings.items():
        if key in value_str:
            return replacement
    
    return value


def sng_get_brand_conversion_stats(api_key, dimensions, metrics, cohort_metrics, start_date, end_date):
    report_id_sing = sng_create_async_report(dimensions, metrics, cohort_metrics,  start_date, end_date, api_key, cohort_periods='ltv')
    download_url_sing = sng_get_report_download_url(report_id_sing, api_key)
    df_sing = pd.read_csv(download_url_sing)
    df_sing['Brand']=df_sing['app'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
    df_sing['network']=df_sing['source'].str.replace(' ', '').str.lower().apply(sng_source2networks)
    df_sing['date']=df_sing['start_date']
    df_sing['2760dafd981b4ae8988469327363bfd8'] = df_sing['2760dafd981b4ae8988469327363bfd8'].apply(ast.literal_eval)
    df_sing['custom_signups'] = df_sing['2760dafd981b4ae8988469327363bfd8'].apply(lambda x: int(x['ltv']) if x['ltv'] is not None else 0)
    df_sing['revenue'] = df_sing['revenue'].apply(ast.literal_eval)
    df_sing['custom_revenue'] = df_sing['revenue'].apply(lambda x: int(x['ltv']) if x['ltv'] is not None else 0)
    df_sing=df_sing[['date','network','Brand','custom_installs', 'custom_signups']].groupby(['date','network','Brand']).sum().reset_index()
    df_sing=df_sing[df_sing['Brand']=='bestwalletapp']
    
    report_id = sng_create_async_report(dimensions, metrics, cohort_metrics,  start_date, end_date, api_key, cohort_periods='actual')
    download_url = sng_get_report_download_url(report_id, api_key)
    df_sing_act = pd.read_csv(download_url)
    df_sing_act['Brand']=df_sing_act['app'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
    df_sing_act['network']=df_sing_act['source'].str.replace(' ', '').str.lower().apply(sng_source2networks)
    df_sing_act['date']=df_sing_act['start_date']
    df_sing_act['2760dafd981b4ae8988469327363bfd8'] = df_sing_act['2760dafd981b4ae8988469327363bfd8'].apply(ast.literal_eval)
    df_sing_act['custom_signups'] = df_sing_act['2760dafd981b4ae8988469327363bfd8'].apply(lambda x: int(x['actual']) if x['actual'] is not None else 0)
    df_sing_act['revenue'] = df_sing_act['revenue'].apply(ast.literal_eval)
    df_sing_act['custom_revenue'] = df_sing_act['revenue'].apply(lambda x: int(x['actual']) if x['actual'] is not None else 0)
    df_sing_act=df_sing_act[['date','network','Brand','custom_revenue']].groupby(['date','network','Brand']).sum().reset_index()
    df_sing_act=df_sing_act[df_sing_act['Brand']=='bestwalletapp']
    df_sing=pd.merge(df_sing, df_sing_act,  how='left', left_on=['date','Brand','network'], right_on=['date','Brand','network'])
    return df_sing



# Spend


def calculate_cpm_spend(df):
    cpm_values = {'777Score.it (Media)': 1.85,'Forklog (Media)': 5,'Etherscan (Media)': 5, 'Coingecko (Media)': 4.55, 'Techopedia (Media)':0, 'BuySellAds (Media)': 3.75, 'Geckoterminal (Media)': 4.55, 'PubFuture (Media)': 0.2, 'Footboom (Media)': 3.429347885}
    
    # Apply the spend calculation only to networks A, B, and C
    def compute_spend(row):
        if row['network'] in cpm_values:
            return (row['impressions'] / 1000) * cpm_values[row['network']]
        return row['total_spend']  # Keep the original spend for other networks
    
    # Apply the calculation function to the spend column
    df['total_spend'] = df.apply(compute_spend, axis=1)
    
    return df


# Randomizer

import random
import string

def generate_query_name(base_name="campaign_spend"):
    """
    Generates a unique query name by appending a random string.
    
    Parameters:
        base_name (str): Base name for the query (default is "campaign_spend").
        
    Returns:
        str: A unique query name.
    """
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    query_name = f"{base_name}_{random_suffix}"
    return query_name



# Vizible

def vzb_get_access_token(email, password):
    """Authenticate and obtain an access token and advertiser ID."""
    url = f"https://live-uk-api.vizibl.ai/api/v2/token/"
    payload = {"email": email, "password": password}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["access"], data.get("advertiser_id")



def vzb_create_specific_report_query(api_endpoint, access_token, advertiser_id, start_date, end_date,dimensions, metrics, interval, query_name):
    """Create a specific report query based on given dimensions, filters, and limits."""
    if 'bdsp' in api_endpoint:
        url = f"{api_endpoint}/v2/query/?advertiser_id={advertiser_id}"
        headers = {"Authorization": f"Bearer {access_token}"}

        payload = {
            "name": query_name,
            "report_type": "analytics",
            "time_range": "custom",
            "custom_start_date": start_date,
            "custom_end_date": end_date,
            "dimensions": dimensions,
            "interval": interval,
            "limits": [
                {
                    "field": "imps",
                    "operator": "greater_than",
                    "value": 100
                }
            ],
            "metrics": metrics
        }
    else:
        url = f"{api_endpoint}/v2/query/?advertiser_id={advertiser_id}"
        headers = {"Authorization": f"Bearer {access_token}"}

        payload = {
            "name": query_name,
            "report_type": "analytics",
            "time_range": "custom",
            "start_date": start_date,
            "end_date": end_date,
            "dimensions": dimensions,
            "interval": interval,
            "limits": [
                {
                    "field": "imps",
                    "operator": "greater_than",
                    "value": 100
                }
            ],
            "metrics": metrics
        }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        print("Error creating specific report query:", response.json())
    response.raise_for_status()
    return response.json()["query"]["id"]



def vzb_create_report(api_endpoint, access_token, advertiser_id, query_id):
    """Generate a report using the query ID."""
    url = f"{api_endpoint}/v2/report/?advertiser_id={advertiser_id}&query_id={query_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.post(url, headers=headers)
    response.raise_for_status()
    return response.json()["report"]["id"]


def vzb_check_report_status(api_endpoint, access_token, advertiser_id, report_id):
    """Check the status of the report."""
    url = f"{api_endpoint}/v2/report/{report_id}/?advertiser_id={advertiser_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    while True:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        status = response.json()["report"]["status"]

        if status == "completed":
            return response.json()["report"]["url"]
        elif status == "failed":
            raise Exception("Report generation failed.")
        
        time.sleep(5)  # Wait before checking again

def vzb_get_campaign_stats(email, password, api_endpoint, start_date, end_date, dimensions,metrics,interval):
    access_token, advertiser_id= vzb_get_access_token(email, password)
    query_name=generate_query_name()
    query_id=vzb_create_specific_report_query(api_endpoint, access_token, advertiser_id,start_date,end_date, dimensions,metrics,interval, query_name)
    report_id= vzb_create_report(api_endpoint, access_token, advertiser_id, query_id)
    report_link=vzb_check_report_status(api_endpoint, access_token, advertiser_id, report_id)
    df_vz = pd.read_csv(report_link)

    df_vz['network']='Datawrkz (Media)'
    df_vz['Brand'] = df_vz['insertion_order'].str.split('_').str[1]
    df_vz['Brand']=df_vz['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
    df_vz['Brand']=df_vz['Brand'].apply(brand_clean_polish)
    df_vz = add_presale_to_brand(df_vz, external_column='insertion_order')
    df_vz=df_columns_rename(df_vz)
    df_vz['total_spend']=df_vz['total_cost'].astype(float)
    df_vz['total_spend_campaign_currency']=df_vz['total_spend'].astype(float)
    df_vz=df_vz[['date','network','Brand','total_spend','total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    return df_vz


# Plattform 161


# Authenticate and get tokens
def p161_authenticate(login, password):
    url = f"https://ui.platform161.com/api/v3/m2o/tokens"
    payload = {
        "user": {
            "login": login,
            "password": password
        }
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["access_token"], data["csrf_token"]
    else:
        raise Exception(f"Authentication failed: {response.text}")

# Create a general report definition
def p161_create_report(access_token, csrf_token, dimensions,metrics,timeframe):
    url = f"https://ui.platform161.com/api/v3/m2o/general_reports"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-CSRF-Token": csrf_token
    }
    payload = {
        "data": {
            "type": "general_report",
            "attributes": {
                "measures": metrics,
                "interval": "daily",
                "groupings": dimensions,
                 "period": timeframe,

                "sort": "campaign_id.desc date.asc"
            }
        }
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        return response.json()["data"]["id"]
    else:
        raise Exception(f"Failed to create report: {response.text}")

# Request report results and get total pages
def p161_request_report_results(access_token, csrf_token, report_id):
    url = f"https://ui.platform161.com/api/v3/m2o/general_reports/{report_id}/relationships/results"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-CSRF-Token": csrf_token
    }
    response = requests.post(url, headers=headers)
    if response.status_code == 202:
        data = response.json()
        return data["data"]["id"]
    else:
        raise Exception(f"Failed to request report results: {response.text}")

# Retrieve report results and save as DataFrame
# Use total_pages from the request_report_results function
def p161_get_report_results_df(access_token, report_id, result_id):
    base_url = f"https://ui.platform161.com/api/v3/m2o/general_reports/{report_id}/relationships/results/{result_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(base_url, headers=headers)
    total_pages = response.json()["meta"]["report_pages"].get("total_pages", 1)

    
    all_data_rows = []

    for current_page in range(1, total_pages + 1):
        paginated_url = f"{base_url}?report_page[number]={current_page}"
        response = requests.get(paginated_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data["data"]["attributes"]["generation_status"] == "completed":
                data_rows = data["data"]["attributes"].get("data_rows", [])
                all_data_rows.extend(data_rows)
            else:
                print("Report generation in progress, retrying in 10 seconds...")
                time.sleep(10)
        else:
            raise Exception(f"Failed to get report results: {response.text}")

    return pd.DataFrame(all_data_rows)

def get_p161_report(login, password, dimensions, metrics):
    timeframe="last_30_days"
    access_token, csrf_token = p161_authenticate(login, password)
    report_id = p161_create_report(access_token, csrf_token, dimensions, metrics,timeframe)
    result_id = p161_request_report_results(access_token, csrf_token, report_id)
    time.sleep(10)
    df_p161 = p161_get_report_results_df(access_token, report_id, result_id)
    df_p161['network']='Match2One (Media)'
    df_p161['Brand'] = df_p161['campaign_name'].str.split('_').str[0]
    df_p161['Brand']=df_p161['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
    df_p161['Brand']=df_p161['Brand'].apply(brand_clean_polish)
    df_p161 = add_presale_to_brand(df_p161)
    df_p161=df_columns_rename(df_p161)
    df_p161['total_spend']=df_p161['total_cost'].astype(float)
    df_p161['total_spend_campaign_currency']=df_p161['total_spend'].astype(float)
    df_p161=df_p161[['date','network','Brand','total_spend','total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    df_p161['date'] = pd.to_datetime(df_p161['date'])
    return df_p161


# Match2One


def authenticate_match2one(email: str, password: str) -> str:
    """
    Authenticates with Match2One API and returns the authentication token.

    :param email: The user's email for authentication.
    :param password: The user's password for authentication.
    :return: The authentication token if successful, otherwise an error message.
    """
    # Authentication data
    auth_data = {
        "email": email,
        "password": password
    }

    # Define the login URL
    login_url = "https://back.app.match2one.com/v1/login"

    try:
        # Send the POST request
        response = requests.post(login_url, headers={"Content-Type": "application/json"}, data=json.dumps(auth_data))

        # Check if the request was successful
        if response.status_code == 200:
            token_data = response.json()
            token = token_data.get('token')
            return token
        else:
            return f"Failed to authenticate. Status code: {response.status_code}. Response: {response.text}"

    except requests.RequestException as e:
        return f"An error occurred: {e}"



def get_m2o_data(name: str, token: str, request_data: dict) -> pd.DataFrame:
    base_url = f"https://api.app.match2one.com/v2/reports/{name}/execute/csv"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # Make the POST request
    response = requests.post(base_url, headers=headers, data=json.dumps(request_data))
    
    if response.status_code == 200:
        # Convert CSV response to a DataFrame
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    else:
        raise Exception(f"Failed to fetch report data. Status code: {response.status_code}. Response: {response.text}")


        
def get_m2o_report(email, password, start_date, end_date):
    token = authenticate_match2one(email, password)
    token = token.replace('Bearer ', '')
    request_data = {
    "parameters": [
        {
            "key": "start_date",
            "condition": "ge",
            "value": start_date
        },
        {
            "key": "end_date",
            "condition": "lt",
            "value": end_date
        }
    ],
    "zoneId": "UTC"}

    # name = 'accounts_daily_performance'
    report_data = get_m2o_data('accounts_daily_performance', token, request_data)


    request_data_lib = {
        "parameters": [

        ]
    }
    # name='accounts_dictionary'
    ac_dict = get_m2o_data('accounts_dictionary', token, request_data_lib)
    ac_dict.columns=['account_id', 'account_name', 'account_status', 'created_at']

    ac_dict=ac_dict[['account_id', 'account_name', 'account_status']]

    df_m2o_=pd.merge(report_data, ac_dict,  how='left', left_on=['account_id'], right_on=['account_id'])
    
    # name='campaigns_dictionary'
    cmp_dict = get_m2o_data('campaigns_dictionary', token, request_data_lib)

    cmp_dict.columns=['campaign_id', 'campaign_name', 'campaign_status', 'campaign_created_at', 'account_id']
    df_m2o=pd.merge(df_m2o_, cmp_dict,  how='left', left_on=['account_id', 'campaign_id'], right_on=['account_id', 'campaign_id'])
    df_m2o['network']='Match2One (Media)'
    df_m2o['Brand'] = df_m2o['campaign_name'].str.split('_').str[0]
    df_m2o['Brand']=df_m2o['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
    df_m2o['Brand']=df_m2o['Brand'].apply(brand_clean_polish)
    df_m2o = add_presale_to_brand(df_m2o,  external_column='account_name')
    df_m2o=df_columns_rename(df_m2o)
    df_m2o['total_spend_campaign_currency']=df_m2o['total_spend'].astype(float)
    df_m2o['total_spend']=df_m2o['total_spend'].astype(float)
    df_m2o=df_m2o[['date','network','Brand','total_spend','total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    return df_m2o
    


# Coinzilla



def cz_create_token(command, access_key, secret_key, body=None):
    timestamp = int(time.time())  # Current Unix timestamp in seconds
    body = ''  # Empty body if not provided

    # Concatenate the required elements
    signature_string = access_key + str(timestamp) + command + body + secret_key
    signature = hashlib.sha256(signature_string.encode()).hexdigest()

    # Create the token by base64 encoding
    token_payload = {
        "accessKey": access_key,
        "timestamp": timestamp,
        "signature": signature
    }
    token = base64.b64encode(json.dumps(token_payload).encode()).decode()
    return token



def cz_get_campaigns(start_date, end_date, token, cz_api_url, command='campaigns'):
    """
    Fetch campaign performance data from the Coinzilla API.

    Parameters:
        start_date (str): The start date for the data in YYYY-MM-DD format.
        end_date (str): The end date for the data in YYYY-MM-DD format.
        token (str): Authentication token for the API.
        command (str): The API command to fetch specific data (default is 'statistics').

    Returns:
        pd.DataFrame: DataFrame containing the campaign performance data.
        str: Error message if the API request fails.
    """
    # Define the request headers with the token
    headers = {
        "Content-Type": "application/json",
        "CZILLA-AUTHENTICATION": token
    }

    # Build the URL for performance or statistics data
    url = f"{cz_api_url}{command}"

    # Add date parameters if specified
    if start_date and end_date:
        url += f"?startDate={start_date}&endDate={end_date}"
    
    try:
        # Send the request
        status_response = requests.get(url, headers=headers)
        
        # Check for a successful response
        if status_response.status_code == 200:
            report_data = status_response.json()
            # Ensure the 'response' key exists before converting to a DataFrame
            if 'response' in report_data:
                return pd.DataFrame(report_data['response'])
            else:
                return "Error: 'response' key not found in API response."
        else:
            # Return detailed error message
            return f"Error: {status_response.status_code}, {status_response.text}"
    except requests.exceptions.RequestException as e:
        # Handle any network-related errors
        return f"Request failed: {str(e)}"



def cz_get_campaign_performance(command, start_date, end_date, cz_api_url, token, uid=None, group_by=None):
    # Define the request headers with the token
    headers = {
        "Content-Type": "application/json",
        "CZILLA-AUTHENTICATION": token
    }
    if uid:
        command=command+'/'+ f"{uid}"
    else: 
        command=command
    
    # Build the base URL for the statistics or performance endpoint
    url = f"{cz_api_url}{command}"


    # Build query parameters
    query_params = []
    if start_date and end_date:
        query_params.append(f"startDate={start_date}&endDate={end_date}")
    if group_by:
        query_params.append(f"group={group_by}")
    
    # Append query parameters to the URL
    if query_params:
        url += "?" + "&".join(query_params)

    
    # Send the request
    status_response = requests.get(url, headers=headers)
    
    # Check for a successful response
    if status_response.status_code == 200:
        report_data = status_response.json()
        flattened_data = {date: metrics for entry in report_data['response'] for date, metrics in entry.items()}
        df = pd.DataFrame(flattened_data).T 
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # return status_response.json()
        return df
        # return url

    else:
        return f"Error: {status_response.status_code}, {status_response.text}"



def get_cz_data(df, start_date, end_date, cz_api_url, token, group_by,command):
    results = []

    for uid, name in zip(df['uid'], df['name']):
        try:
            # Call the function for each uuid
            data = cz_get_campaign_performance(command, start_date, end_date, cz_api_url, token, uid=uid, group_by=group_by)

            # Add the name column to the fetched data
            if isinstance(data, pd.DataFrame):
                data['name'] = name

            results.append(data)
        except Exception as e:
            print(f"Error fetching data for UUID {uid}: {e}")

    # Optionally concatenate results into a single DataFrame if the results are DataFrames
    if results and isinstance(results[0], pd.DataFrame):
        final_result = pd.concat(results, ignore_index=True)
        return final_result

    return results 

def get_cz_campaign_stats(access_key, secret_key, api_url, start_date, end_date):
    token = cz_create_token(command='campaigns', access_key=access_key, secret_key=secret_key, body=None) 
    df_cz_ps_ids=cz_get_campaigns(start_date, end_date, token, api_url, command='campaigns')
    
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    group_by = "date"  
    command='statistics'
    # Initialize an empty DataFrame to store results
    all_data = pd.DataFrame()

    # Loop through each day in the date range
    current_date = start_date_dt
    while current_date <= end_date_dt:
        single_day = current_date.strftime("%Y-%m-%d")
        token = cz_create_token(command=command, access_key=access_key, secret_key=secret_key, body=None)  # Test with 'performance' or 'statistics'


        # Call the function for the specific day
        df_cz_ps = get_cz_data(df_cz_ps_ids, single_day, single_day, api_url, token, group_by, command)

        # Add a column with the current date
        df_cz_ps['date'] = single_day

        # Append the results to the all_data DataFrame
        all_data = pd.concat([all_data, df_cz_ps], ignore_index=True)

        # Move to the next day
        current_date += timedelta(days=1)


    df_cz_ps=all_data

    df_cz_ps['network']='Coinzilla (Dextools)'
    df_cz_ps['Brand'] = df_cz_ps['name'].str.split('-').str[0]

    df_cz_ps['Brand']=df_cz_ps['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)

    df_cz_ps = add_presale_to_brand(df_cz_ps, external_column='name')


    df_cz_ps=df_columns_rename(df_cz_ps)
    
    return df_cz_ps


# Personally

def pers_get_campaign_stats(api_key, adv_id, start_date, end_date):
    f_start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%m-%Y")
    f_end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%Y")
    
    base_url = "http://reporting.personaly.bid/rtb/singular"
    params = {
        'startDate': f_start_date,  # replace with actual start date in dd-mm-yyyy format
        'endDate': f_end_date,       # replace with actual end date in dd-mm-yyyy format
        'advertiserId': adv_id,   # replace with your advertiser ID
        'apiKey': api_key,         # replace with your API key
        'groupBy': 'date'              # grouping by date to get daily stats
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    df_pers = pd.DataFrame(data)
    df_pers['network']='Personaly'

    df_pers['Brand'] = df_pers['campaign_name'].str.split('_').str[0]

    df_pers['Brand']=df_pers['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
    df_pers['Brand']=df_pers['Brand'].apply(brand_clean_polish)
    df_pers = add_presale_to_brand(df_pers, external_column='campaign_name')
    df_pers=df_columns_rename(df_pers)
    df_pers['total_spend']=df_pers['total_spend'].astype(float)
    df_pers['total_spend_campaign_currency']=df_pers['total_spend'].astype(float)
    df_pers['adv_impressions']=df_pers['impressions']

    df_pers['adv_clicks']=df_pers['clicks']
    df_pers['adv_installs']=df_pers['installs']
    df_pers=df_pers[['date','network','Brand','adv_impressions','adv_clicks','adv_installs','total_spend','total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    df_pers['date'] = pd.to_datetime(df_pers['date'], format="%d-%m-%Y")
    return df_pers


# Hueads


import requests
import pandas as pd

def ha_get_token(base_url, username, password):
    # Define the base URL and the endpoint
    auth_endpoint = "auth"

    # Construct the URL with parameters
    url = f"{base_url}{auth_endpoint}?login={username}&password={password}"

    try:
        # Send the GET request and get the response
        response = requests.get(url)
        if response.status_code == 200:
            return response.text  # The token is expected to be plain text in the response body
        else:
            print("Failed to get the token. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred while getting the token:", str(e))



def ha_get_campaign_meta(base_url, username,password):
    token=ha_get_token(base_url,username, password)
   
    campaign_controller = "Campaign"
    version = 4
    range_param = "0-10"

    base_url=base_url+'api/'
    
    # Construct the URL
    url = f"{base_url}{campaign_controller}/?version={version}&token={token}"
    
    try:
        # Make the GET request
        response = requests.get(url)
    
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON data from the response
            data = json.loads(response.text)
            rows = data['response']['rows']
            df = pd.DataFrame.from_dict(rows, orient='index')
            df=df[['id', 'created', 'name', 'pricing_model',
            'description', 'start_date', 'end_date', 'is_active',
           'budget_total', 'cost_total', 'budget_daily', 'cost_today']]
            # df=df[df['is_active']==True]
            return df
    
        else:
            print("Failed to retrieve data. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred:", str(e))



def ha_get_campaign_reports(df_input, start_date, end_date, username, password, base_url):
    
    report_controller = "AdvertiserReports"
    dimension = "campaign"
    date_filter = f"date:{start_date}_{end_date}"
    token=ha_get_token(base_url,username, password)
    base_url=base_url+'api/'
    df = pd.DataFrame()

    # Loop through the input DataFrame rows
    for index, row in df_input.iterrows():
        drill_down_value = row['id']
        campaign_name = row['name']
        
        # Construct the URL
        url = f"{base_url}{report_controller}/{dimension}={drill_down_value}/date?filters={date_filter}&version=4&token={token}"
        
        # Make the request
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the API response
            data = response.json().get('response', {}).get('total', {})
            df_tmp = pd.DataFrame([data])
            
            # Add metadata columns
            df_tmp['start_date'] = start_date
            df_tmp['end_date'] = end_date
            df_tmp['campaign_id'] = drill_down_value
            df_tmp['campaign_name'] = campaign_name
            
            # Append to the main DataFrame
            df = pd.concat([df, df_tmp], ignore_index=True)
        else:
            print(f"Failed to fetch data for campaign ID {drill_down_value}. Status code: {response.status_code}")
            continue
    df=df[['start_date','end_date', 'campaign_id', 'campaign_name', 'adv_impressions','adv_clicks','adv_cost']]

    return df


def ha_get_campaign_stats(username, password, base_url, start_date, end_date):
    df = ha_get_campaign_meta(base_url, username, password)

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = pd.DataFrame()
    current_date = start_date_dt

    while current_date <= end_date_dt:
        single_day = current_date.strftime("%Y-%m-%d")
        df_hueads = ha_get_campaign_reports(df, single_day, single_day, username, password, base_url)
        df_hueads['date'] = single_day
        all_data = pd.concat([all_data, df_hueads], ignore_index=True)
        current_date += timedelta(days=1)

    df_hueads = all_data
    df_hueads['network'] = 'Hue Ads (Media)'

    if not df_hueads.empty and 'campaign_name' in df_hueads.columns:
        df_hueads['Brand'] = df_hueads['campaign_name'].str.split('_').str[1]
        df_hueads['Brand'] = (
            df_hueads['Brand']
            .str.replace(' ', '', regex=False)
            .str.lower()
            .apply(brand_cleanup)
            .apply(brand_clean_polish)
        )

        df_hueads = add_presale_to_brand(df_hueads)
        df_hueads = df_columns_rename(df_hueads)
        df_hueads=df_hueads[['date','network','campaign_id', 'campaign_name','Brand','adv_impressions', 'adv_clicks', 'total_spend']]

    return df_hueads

  
# Cointtraffic

def ctrf_get_campaign_stats(token, url, start_date, end_date):
    headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": token}
    payload = {
    "date_from": start_date,
    "date_to": end_date,
    "group_by": ["DATE", "CAMPAIGN"]}
    # Make the POST request
    response = requests.post(url, headers=headers, json=payload)
    # Check the response status code and print the content of the response
    if response.status_code == 200:
        # Convert JSON object into DataFrame and extract "id" and "name" fields from "campaign" column
        df_ctrf = pd.json_normalize(response.json(), sep='_')
    else:
        return('Error:', response.status_code, response.text)
    df_ctrf['network']='Cointraffic (Media)'
    df_ctrf['Brand'] = df_ctrf['campaign_name'].str.split('_').str[1]
    df_ctrf['Brand']=df_ctrf['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
    df_ctrf['Brand']=df_ctrf['Brand'].apply(brand_clean_polish)
    df_ctrf = add_presale_to_brand(df_ctrf)
    df_ctrf=df_columns_rename(df_ctrf)
    df_ctrf['total_spend_campaign_currency']=df_ctrf['spent_budget'].astype(float)
    df_ctrf['total_spend']=df_ctrf['spent_budget'].astype(float)
    df_ctrf=df_ctrf[['date','network','Brand','total_spend','total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    df_ctrf['date'] = pd.to_datetime(df_ctrf['date'])
    return df_ctrf


# Help Functions


def client2vertical(df):
    casinos = ['sambaslots','nokyc','tgc', 'wsm', 'instantcasino', 'luckyblock', 'megadice','coinpoker','goldenpanda', 'thehighroller', 'bovada','highroller', 'bovadaus', 'coincasino']
    cryptos=['pepeunchained','mindofpepe', 'memebet', 'freedumfighters', 'flockerz', 'cryptoallstars', 'catslap', 'bestwalletapp', 'bestwalletapp-presale', 'wepe', 'memeindex', 'solaxy']
    if 'client' in df.columns:
        df['Vertical'] = df['client'].apply(
            lambda x: "Casino" if (
                ("Casino" in x or "Casinos" in x)) 
            else "Crypto" if (
                "Presale" in x or "Crypto" in x
            ) 
            else "Unknown"
        )
           
        df['Vertical']= np.where(df['Brand'].isin(casinos), 'Casino', df['Vertical'])
        df['Vertical']= np.where(df['Brand'].isin(cryptos), 'Crypto', df['Vertical'])
    else:
        df['Vertical'] = np.where(df['Brand'].isin(casinos), 'Casino', 'Crypto')
        df['Vertical'] = np.where(df['Brand'].isin(cryptos), 'Crypto', 'Casino')

    return df



def adform_brand_cleanup(df):
    df['Brand']= df['Brand'].str.replace(' ','')
    df['Brand']= df['Brand'].str.lower()
    df['Brand']= df['Brand'].str.replace(' ','')
    df['Brand']= df['Brand'].str.lower()
    df['Brand']= df['Brand'].str.replace('-casinocampaign','')
    df['Brand']= df['Brand'].str.replace('wsmcasino','wsm')
    df['Brand']= df['Brand'].str.replace('duplicate-','')
    df['Brand']= df['Brand'].str.replace('-global','')
    df['Brand']= df['Brand'].str.replace('-americas','')
    df['Brand']= df['Brand'].str.replace('-america','')
    df['Brand']= df['Brand'].str.replace('-casinos','')
    df['Brand']= df['Brand'].str.replace('-casino','')
    df['Brand']= df['Brand'].str.replace('-apac','')
    df['Brand']= df['Brand'].str.replace('americas','')
    df['Brand']= df['Brand'].str.replace('america','')
    df['Brand']= df['Brand'].str.replace('americactv','')
    df['Brand']= df['Brand'].str.replace('ctv','')
    df['Brand']= df['Brand'].str.replace('coinpoker(old)','coinpoker')
    df['Brand']=df['Brand'].str.replace('allluckyblock','luckyblock')
    df['Brand']=df['Brand'].str.replace('tgcasino','tgc')
    df['Brand']= df['Brand'].str.replace('coinpoker(old)','coinpoker')
    df['Brand']= df['Brand'].str.replace('wallstmemes','wsm')
    df['Brand']= df['Brand'].str.replace('memebet-presale','memebet')
    df['Brand']= df['Brand'].str.replace('memebet-presaleapac','memebet')
    df['Brand']= df['Brand'].str.replace('-300x250','')
    df['Brand']= df['Brand'].str.replace('-300x600','')
    df['Brand']= df['Brand'].str.replace('-320x50','')
    df['Brand']= df['Brand'].str.replace('-728x90','')
    df['Brand']= df['Brand'].str.replace('_320x100','')
    df['Brand']= df['Brand'].str.replace('-728x90','')
    df['Brand']= df['Brand'].str.replace('-[archive]','')
    df['Brand']= df['Brand'].str.replace('Ciprian','ciprian')
    df['Brand']= df['Brand'].str.replace('-old','')
    df['Brand']= df['Brand'].str.replace('-presaleapac','')
    df['Brand']= df['Brand'].str.replace('-presale','')
    df['Brand']= df['Brand'].str.replace('wallstmemes','wsm')
    df['Brand']= df['Brand'].str.replace('cryptopolitan-tgc-970x90','tgc')
    df['Brand']= df['Brand'].str.replace('tg','tgc')
    df['Brand']= df['Brand'].str.replace('tgcc','tgc')
    df['Brand']= df['Brand'].str.replace('-presaleapac','')
    df['Brand']= df['Brand'].str.replace('-presale','')
    df['Brand']= df['Brand'].str.replace('wallstmemes','wsm')
    df['Brand']= df['Brand'].str.replace('memebetapac','memebet')
    df['Brand']= df['Brand'].str.replace('99bitcoinsapac','99bitcoins')
    df['Brand']= df['Brand'].str.replace('coinpokercasino','coinpoker')
    df['Brand']= df['Brand'].str.replace('megadicetg','megadice')
    df['Brand']= df['Brand'].str.replace('megadice-latam','megadice')
    df['Brand']= df['Brand'].str.replace('spongetokenv2','sponge')
    df['Brand']= df['Brand'].str.replace('spongetokenv2','sponge')
    df['Brand']= df['Brand'].str.replace('apac','')
    df['Brand']= df['Brand'].str.replace('slothanacoin','slothana')
    df['Brand']= df['Brand'].str.replace('luckyblock-latam','luckyblock')
    df['Brand']= df['Brand'].str.replace('snorters','snorter') 
    df['Brand']= df['Brand'].str.replace('snortertoken','snorter') 
    df['Brand']= df['Brand'].str.replace('btchyper','bitcoinhyper')    
    df['Brand']= df['Brand'].str.replace('-presaleapac','')
    df['Brand']= df['Brand'].str.replace('-presale','')
    df['Brand']= df['Brand'].str.replace('wallstmemes','wsm')
    df['Brand']= df['Brand'].str.replace('memebetapac','memebet')
    df['Brand']= df['Brand'].str.replace('99bitcoinsapac','99bitcoins')
    df['Brand']= df['Brand'].str.replace('coinpokercasino','coinpoker')
    df['Brand']= df['Brand'].str.replace('megadicetg','megadice')
    df['Brand']= df['Brand'].str.replace('megadicepresale','megadice')
    df['Brand']= df['Brand'].str.replace('megadicec','megadice')
    df['Brand']= df['Brand'].str.replace('luckyblock-latam','luckyblock')


    return df


def brand_cleanup(value, add_presale=False):
    value_str = str(value)
    if value_str == 'cas':
            return 'cryptoallstars'
    if value_str == 'lb':
            return 'luckyblock'
    if value_str == 'cp1':
            return 'coinpoker'
    if value_str == 'mindofpepe-presale':
            return 'mindofpepe'
    if value_str == 'tonaldtrump-cryptopresale':
            return 'tonaldtrump'
    if value_str == 'infinaeon-presale':
            return 'infinaeon'
    if value_str == 'wallstreetmemecasino':
            return 'wsm'
    if value_str == 'btchyper':
            return 'bitcoinhyper'
    if value_str == 'bitcoinhyper':
            return 'bitcoinhyper'

 
    mappings = {
    'bestwallet': 'bestwalletapp',
    'snoertertoken': 'snorter',
    'tonaldtokendextoolsus': 'tonaldtrump',
    'bwapp': 'bestwalletapp',
    'allstar': 'cryptoallstars',
    'goldenpanda': 'goldenpanda',
    'highroller': 'highroller',
    'sambaslot': 'sambaslots',
    'btchyper':'bitcoinhyper',
    'doge20': 'dogecoin',
    'dogecoin20': 'dogecoin',
    'snorter': 'snorter',
    'snorters': 'snorter',
    'snortertoken': 'snorter',
    'bitcoinhyper': 'bitcoinhyper',
    'bovada': 'bovada',
    'fepe': 'fepe',
    'tonaldtrumps': 'tonaldtrump',
    'no-kyc': 'nokyc',
    'nokyc': 'nokyc',
    'tgcasino' : 'tgc',
    'tgcasinoapac': 'tgc',
    'flocker': 'flockerz',
    'wallstmemes' : 'wsm',
    'wsmcasino': 'wsm', 
    'wsm': 'wsm', 
    'memebet' : 'memebet',
    'flokerz' : 'flockerz',
    'wallstmemes' : 'wsm',
    'memeindex' : 'memeindex',
    'solaxy' : 'solaxy',
    'instantcasino' : 'instantcasino',
    'dogeverse' : 'dogeverse',
    'wallstreetmemes': 'wsm',
    'tgc' : 'tgc',
    'tg' : 'tgc',
    'wallstmemes' : 'wsm',
    '99bitcoins' : '99bitcoins',
    'coinpoker' : 'coinpoker',
    'megadice' : 'megadice',
    'md' : 'megadice',
    'spongetoken' : 'sponge',
    'slothanacoin' : 'slothana',
    'luckyblock' : 'luckyblock',
    'bitcoinmetrix' : 'bitcoinminetrix',
    'bitcoinetf' : 'btcetf-token', 
    'pepeunch':'pepeunchained',
    'wepe-wallstreetpepe':'wepe',
    'wepe':'wepe',
    'coincasino':'coincasino',
    'peencrypto':'peencrypto',
    'catslap':'catslap',
    'mindofpepe':'mindofpepe',
    'btc-bull':'btcbull',
    'broccoli':'broccoli-cryptotoken'
        
    }
    
    additional_mapping = ['presale']

    value_str = str(value)  # Convert value to string for substring checks
    main_match = None
    additional_match = None

    # Check for main mapping matches
    for key, replacement in mappings.items():
        if key in value_str:
            main_match = replacement
    
    # Check for additional substrings
    if add_presale:
        for extra in additional_mapping:
            if extra in value_str:
                additional_match = 'presale'  # Always map to 'pp'
        
        # Combine main match and 'pp' if both exist
        if main_match and additional_match:
            return f"{main_match}-presale"
        elif main_match:
            return main_match
    elif main_match:
        return main_match
    
    return value  # Keep the original value if no match



def add_presale_to_brand(df, external_column='campaign_name', specific_brand='bestwallet'):
    # Apply the logic to modify the 'Brand' column
    df['Brand'] = df.apply(
        lambda row: (
            f"{row['Brand']}-presale"
            if (
                'presale' in str(row[external_column]).lower() and 
                'presale' not in str(row['Brand']).lower() and
                 specific_brand in str(row['Brand']).lower()  # Check if Brand is 'bestw'
            )
            else row['Brand']
        ),
        axis=1
    )

    return df



def brand_clean_polish(value):
        value_str = str(value).lower()   
        for unwanted in ['-cryptopresales','-global','-presale','-america','-americas','-casinos','-casino','-apac','americas','america','americactv','ctv','(old)','-300x250','-300x600','-320x50','-728x90',
                         '_320x100','-728x90','-[archive]','-old','apac', '-cryptopresale']:
                value_str = value_str.replace(unwanted, '')
        # if 'presale' in value_str:
        #     # Insert a hyphen before 'presale' only if it doesn't already exist
        #     return value_str.replace('presale', '-presale') if '-presale' not in value_str else value_str
        return value_str





# Adform
import requests
import json
import time
import pandas as pd

list_scopes='https://api.adform.com/scope/eapi https://api.adform.com/scope/api.publishers.readonly https://api.adform.com/scope/api.placements.read https://api.adform.com/scope/buyer.reportingstats https://api.adform.com/scope/buyer.masterdata https://api.adform.com/scope/buyer.stats https://api.adform.com/scope/buyer.advertisers https://api.adform.com/scope/agencies.readonly https://api.adform.com/scope/buyer.campaigns.api'


def adf_get_access_token(client_id, client_secret):
    TOKEN_URL = 'https://id.adform.com/sts/connect/token'
    payload = {
        'grant_type': 'client_credentials',
        'scope': list_scopes
    }

    response = requests.post(TOKEN_URL, auth=HTTPBasicAuth(client_id, client_secret), data=payload)
    
    if response.status_code == 200:
        token_info = response.json()
        return token_info['access_token']
    else:
        raise Exception(f"Failed to get access token: {response.status_code} {response.text}")

    return response



def map_conversions(df):
    # Initialize new columns with None
    df['Registration'] = None
    df['WalletConnected'] = None
    df['Deposit'] = None
    df['FTD'] = None  # Added in case of Casino

    # Apply mapping rules based on Vertical
    for index, row in df.iterrows():
        vertical = row['Vertical']
        
        if vertical == 'Crypto':
            df.at[index, 'WalletConnected'] = row['conversions_2']
            df.at[index, 'Deposit'] = row['conversions_3']
            # Conversion point 1 is 'no info', so no assignment

        elif vertical == 'Casino':
            df.at[index, 'Registration'] = row['conversions_1']
            df.at[index, 'FTD'] = row['conversions_2']
            df.at[index, 'Deposit'] = row['conversions_3']
    
    return df


def fetch_adform_data(dimensions, metrics, custom_filter, date_range, access_token):
    """
    Fetches data from the Adform API and returns it as a DataFrame.

    Parameters:
    - dimensions (list): A list of dimensions to include in the report.
    - date_range (tuple): A tuple containing the start and end dates in the format ("YYYY-MM-DD", "YYYY-MM-DD").
    - access_token (str): The access token for authenticating with the Adform API.

    Returns:
    - pd.DataFrame: DataFrame containing the report data.
    - str: Error message if the request fails.
    """
    # Define the base URL for the Adform API
    base_url = "https://api.adform.com"

    # Define the endpoint for the Adform API
    url = f"{base_url}/v1/buyer/stats/data"

    # Define the headers, including the Authorization header with your access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Define the body of the request
    full_filter = {
        "date": {
            "from": f"{date_range[0]}T00:00:00.000Z",
            "to": f"{date_range[1]}T23:59:59.999Z"
        },
        "campaign": { "active": "all" }
    }
    full_filter.update(custom_filter or {})  # Merge custom filters, if any

    body = {
        "dimensions": dimensions,
        "metrics": metrics,
        "filter": full_filter,
        "includeRowCount": True,
        "includeTotals": False
    }

    try:
        # Make the POST request to initiate the report generation
        response = requests.post(url, headers=headers, data=json.dumps(body))

        # Check the response status code
        if response.status_code == 202:
            # Get the location URL to poll for report status
            location_path = response.headers.get("Location")
            if not location_path:
                return "No Location header found in the response."
            else:
                # Ensure the location URL is complete
                location_url = location_path if location_path.startswith("http") else base_url + location_path
                time.sleep(50)  # Wait for the report to be generated

                # Polling loop
                while True:
                    status_response = requests.get(location_url, headers=headers)
                    if status_response.status_code == 200:
                        # The report is ready
                        report_data = status_response.json()
                        # df = pd.json_normalize(report_data['reportData'])
                        columns = report_data['reportData']['columnHeaders']
                        rows = report_data['reportData']['rows']
                        
                        # Create DataFrame
                        df = pd.DataFrame(rows, columns=columns)
                        return df
                    elif status_response.status_code == 202:
                        # The report is still being processed
                        time.sleep(10)  # Wait for 10 seconds before retrying
                    else:
                        return f"Failed to retrieve report data. Status code: {status_response.status_code}\n{status_response.text}"
        else:
            return f"Failed to initiate report generation. Status code: {response.status_code}\n{response.text}"
    
    except Exception as e:
        return str(e)




# Currency conversions - Functions




import pandas as pd
from datetime import datetime, timedelta

import requests
import pandas as pd



def get_conversion_rates(from_symbol,to_symbol,apikey, days):
    function = 'FX_DAILY'
    outputsize = 'compact'
    
    # Request data from Alpha Vantage
    params = {
        'function': function,
        'from_symbol': from_symbol,
        'to_symbol': to_symbol,
        'apikey': apikey,
    }
    response = requests.get('https://www.alphavantage.co/query', params=params)
    data = response.json()
    
    if 'Time Series FX (Daily)' not in data:
        raise ValueError("Error fetching data from Alpha Vantage: " + str(data))
    
    # Convert response to DataFrame
    df = pd.DataFrame.from_dict(data['Time Series FX (Daily)'], orient='index')
    df = df[['4. close']].astype(float).reset_index()
    df.columns = ['date', 'rate']
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Get date range for the last 60 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    # Filter the last 60 days
    df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
    
    # Ensure all dates are present
    all_dates = pd.date_range(start=start_date, end=end_date - timedelta(days=1)).date
    df_full = pd.DataFrame(all_dates, columns=['date']).merge(df, on='date', how='left').fillna(method='ffill')
    
    return df_full




def get_eur_to_usd_rates():
    function = 'FX_DAILY'
    from_symbol = 'EUR'
    to_symbol = 'USD'
    outputsize = 'compact'
    
    # Request data from Alpha Vantage
    params = {
        'function': function,
        'from_symbol': from_symbol,
        'to_symbol': to_symbol,
        'apikey': 'GP58HS6L18FD4R4H',
    }
    response = requests.get('https://www.alphavantage.co/query', params=params)
    data = response.json()
    
    if 'Time Series FX (Daily)' not in data:
        raise ValueError("Error fetching data from Alpha Vantage: " + str(data))
    
    # Convert response to DataFrame
    df = pd.DataFrame.from_dict(data['Time Series FX (Daily)'], orient='index')
    df = df[['4. close']].astype(float).reset_index()
    df.columns = ['date', 'rate']
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Get date range for the last 60 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=60)
    
    # Filter the last 60 days
    df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
    
    # Ensure all dates are present
    all_dates = pd.date_range(start=start_date, end=end_date - timedelta(days=1)).date
    df_full = pd.DataFrame(all_dates, columns=['date']).merge(df, on='date', how='left').fillna(method='ffill')
    
    return df_full




def df_convert_gbp(df,df_gbp2usd):
    df['date'] = pd.to_datetime(df['date'])
    df_gbp2usd['date'] = pd.to_datetime(df_gbp2usd['date'])
    df = pd.merge(df, df_gbp2usd, on='date', how='left')
    df=df[df['campaignCurrency']!= '-']
    if 'total_spend' in df.columns:
        df['spend_usdt']=df['total_spend']*df['rate'] 
        df['total_spend'] = np.where(df['campaignCurrency'] == 'GBP', df['spend_usdt'], df['total_spend'])
        # columns_to_drop = ['spend_usdt']
        # df = df.drop(columns=columns_to_drop)
    if 'media_cost' in df.columns:
        df['media_cost_usdt']=df['media_cost']*df['rate'] 
        df['media_cost'] = np.where(df['campaignCurrency'] == 'GBP', df['media_cost_usdt'], df['media_cost'])
    if 'sales' in df.columns:
        df['sales_usdt']=df['sales']*df['rate'] 
        df['sales'] = np.where(df['campaignCurrency'] == 'GBP', df['sales_usdt'], df['sales'])
    # if 'total_revenue' in df.columns:
    #     df['total_revenue_usdt']=df['total_revenue']*df['rate'] 
    #     df['total_revenue'] = np.where(df['campaignCurrency'] == 'GBP', df['total_revenue_usdt'], df['total_revenue'])


    return df





def df_convert_eur(df,df_eur2usd):
    df['date'] = pd.to_datetime(df['date'])
    df_eur2usd['date'] = pd.to_datetime(df_eur2usd['date'])
    df = pd.merge(df, df_eur2usd, on='date', how='left')
    df['total_spend']=df['total_spend_campaign_currency']*df['EUR_to_USD'] 
    return df







def get_campaign_dict(campaign_ids, access_token):
    """
    Retrieves tracking mappings for a list of campaign IDs, normalizes the JSON response into DataFrames,
    and returns a dictionary mapping each campaign ID to a list of tracking IDs.

    Parameters:
    - campaign_ids (list): List of campaign IDs.
    - access_token (str): Access token required for API authentication.
    - af (object): Object that contains the method `get_tracking_ids` to retrieve tracking mappings.

    Returns:
    - campaign_dict (dict): A dictionary where keys are campaign IDs and values are lists of tracking IDs.
    """
    dfti = []

    # Iterate over the campaign IDs
    for campaign_id in campaign_ids:
        # Call the function to get tracking mappings
        tracking_mappings = get_tracking_ids(access_token, campaign_id)

        # Normalize the JSON and create a DataFrame
        df = pd.json_normalize(tracking_mappings)

        # Append the DataFrame to the list
        dfti.append(df)

    # Combine all the individual DataFrames into one DataFrame
    final_df = pd.concat(dfti, ignore_index=True)

    # Group by 'campaignId' and convert to dictionary
    campaign_dict = final_df.groupby('campaignId')['id'].apply(list).to_dict()

    return campaign_dict

def find_order_of_substring(row):
    parts = row['line_item'].split('_')  # Split the string by '_'
    substring = row['PRO Type']  # Get the value from the PRO column
    if substring in parts:
        return parts.index(substring) + 1  # +1 to make the order 1-based
    else:
        return None  # Return None if the substring is not found


def get_target(row):
    parts = row['line_item'].split('_')
    order = row['Target_order']
    order_2=row['Target_order']+1
    if 0 <= order < len(parts) and 0 <= order_2 < len(parts):  # Check if the order is within the range of parts
        return parts[order]+'_'+parts[order_2]
    elif 0 <= order < len(parts):  # Check if the order is within the range of parts
        return parts[order]
    else:
        return None  # Return None if the order is out of range





def df_convert_currency(df, df_rates):
    df['date'] = pd.to_datetime(df['date'])
    df_rates['date'] = pd.to_datetime(df_rates['date'])
    
    # Merge the dataframes on date
    df = pd.merge(df, df_rates, on='date', how='left')
    
    # Apply conversion based on the 'currency' column
    df['total_spend'] = df.apply(
        lambda row: row['total_spend_campaign_currency'] * row['GBP_to_USD']
        if row['currency_code'] == 'GBP'
        else row['total_spend_campaign_currency'] * row['EUR_to_USD'], 
        axis=1
    )
    
    return df




# Function to find brand and city in the Name column
def df_format_li(row, ssp_providers, ad_formats, pro_types):
    ssp_provider_found = [ssp_provider for ssp_provider in ssp_providers if ssp_provider in row['line_item']]
    ad_format_found = [ad_format for ad_format in ad_formats if ad_format in row['line_item']]
    pro_type_found = [pro_type for pro_type in pro_types if pro_type in row['line_item']]
    
    row['SSP_Provider'] = ssp_provider_found[0] if ssp_provider_found else None
    row['Ad_Format'] = ad_format_found[0] if ad_format_found else None
    row['PRO_Type'] = pro_type_found[0] if pro_type_found else None
    
    return row

def line_item_clickout(nc, df):
    brands = nc.Brand.dropna().unique()
    campaign_types = nc['Campaign_Type'].dropna().unique()
    ssp_providers = nc['SSP_Provider'].dropna().unique()
    ad_formats = nc['Ad_Format'].dropna().unique()
    pro_types = nc['PRO_Type'].dropna().unique()
    targets = nc.Target.dropna().unique()
    
    # Apply the df_format_li function to each row, passing necessary parameters
    df = df.apply(df_format_li, axis=1, ssp_providers=ssp_providers, ad_formats=ad_formats, pro_types=pro_types)
    
    # Extract Brand and Campaign Type from the line_item column
    df['Brand'] = df['line_item'].str.split('_').str[0]
    df['Campaign_Type'] = df['line_item'].str.split('_').str[1]
    
    return df


## Beutifilization


def table_style(df,color,kpi):
        cm = sns.light_palette(color, as_cmap=True)
        format_dict = {'eCPM':'{0:,.2f}','Media eCPM':'{0:,.2f}','eCPCV':'{0:,.4f}','Total Spend':'{0:,.2f}','clickers_mu':'{0:,.0f}','total_conversions':'{0:,.0f}','pc_conversions':'{0:,.0f}','measurable':'{0:,.0f}','in_view':'{0:,.0f}',\
                       'VR': '{:.2%}','audience_index_impression':'{0:,.0f}', 'Spend To Pace': '{:.1f}',  'Days Remaining': '{:.0f}', \
                       'target_accuracy':'{0:,.0%}','audience_index_clicks':'{0:,.0f}','total_spend':'{0:,.1f}', 'bid_rate': '{:,.1f}',\
                       'win_rate': '{:,.1f}','min_bid_amount_cpm': '{:,.1f}', 'max_bid_amount_cpm': '{:,.1f}','watermark_spend': '{:,.1f}',\
                       'Latest Hour of Activity':'{0:,.0f}','Scheduled End Hour':'{0:,.0f}', 'total_revenue':'{0:,.1f}','NDC':'{0:,.2f}',\
                       'LP':'{0:,.0f}', 'CPA_LP':'{0:,.2f}','CPC':'${0:,.2f}','CPA_Signup':'{0:,.2f}','CTR': '{:.2%}','RR': '{:.3%}','spend_share': '{:.2%}', 'CPA_NDC':'{0:,.1f}','CPA_pc':'{0:,.1f}','CPA(lead_club)':'{0:,.1f}','CPA(product_registration)':'{0:,.1f}',\
                       'CPA_DC':'{0:,.1f}','ROI': '{:.2f}','impressions':'{0:,.0f}','clicks':'{0:,.0f}','ROI_segment': '{:.2f}','CPM': '${0:,.2f}', 'FTD Sales': '${0:,.2f}', 'vCPM': '{0:,.1f}','CPA': '{0:,.1f}','SSP_fee_pct': '{:.2%}','ROI_usd': '{:.2f}', 'avg_frequency':'{0:,.2f}','total_reach':'{0:,.0f}',\
                       'CPA_usd':'{0:,.2f}','CPC_usd':'{0:,.2f}','CPM_usd': '{0:,.1f}', 'CPA(eng45)':'{0:,.1f}','CPA(dealer)':'{0:,.1f}','CPA(shopnow)':'{0:,.1f}','CPA(purchase)': '{0:,.0f}', 'total_spend_usd':'{0:,.1f}','total_revenue_usd':'{0:,.1f}','WalletConnected CPA': '${0:,.2f}','FTD CPA': '${0:,.2f}',\
                       'media_cost_usd':'{0:,.1f}','ssp_technology_fee_usd':'{0:,.2f}', 'Total Cost': '${0:,.2f}','total_spend_campaign_currency': '{0:,.2f}','Media Cost': '${0:,.2f}','CPCV':'{0:,.3f}','video_complete':'{0:,.0f}','disc_rev_uplift':'{0:,.0f}','consideration_shopnow':'{0:,.0f}','consideration_dealer':'{0:,.0f}','lead_club':'{0:,.0f}','product_registration':'{0:,.0f}','engagement_45s':'{0:,.0f}','purchase_confirm':'{0:,.0f}','Deposit CPA':'${0:,.2f}','Registration CPA':'${0:,.2f}','Sales':'${0:,.2f}','Deposit_Sales':'${0:,.2f}','Deposit_Sales_dw':'${0:,.2f}','FTD_Sales':'${0:,.2f}','FTD_Sales_dw':'${0:,.2f}','Cost':'${0:,.2f}','ROI (35%)':'${0:,.2f}','ROAS (35%)':'{0:,.2f}' ,'ROAS':'{0:,.2f}','Cost Comparison (pm)': '{:.2%}','Sales Comparison (pm)': '{:.2%}','Registrations': '{:.0f}','FTD': '{:.0f}','Registration': '{:.0f}','WalletConnected': '{:.0f}','Deposit': '{:.0f}','Deposit': '{:.0f}','FTD_dw': '{:.0f}','Deposit_dw': '{:.0f}','ROAS (35%) Comparison (pm)': '{:.2%}' }
        stdf = df.style.background_gradient(cmap=cm, subset=kpi).format(format_dict).hide()
        return stdf



def pivot( df, dimensions, metrics, kpi, sortby, ascending):
        # if 'week' in dimensions:
        #     df['start_date'] = pd.to_datetime(df['start_date'].astype(str), format='%Y-%m-%d')
        #     df['week'] = df['start_date'].dt.week
        # if 'start_date' in dimensions:    
        #     df['start_date'] = pd.to_datetime(df['start_date'].astype(str), format='%Y-%m-%d')

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
            df['CPA_pc'] = df.total_spend/df.pc_conversions
        if 'RR' in kpi:  
            df['RR'] = df.total_conversions/(df.impressions)
        if 'VR' in kpi:  
            df['VR'] = df.in_view/df.measurable      
        if 'SSP_fee_pct' in kpi:  
            df['SSP_fee_pct'] = df.ssp_technology_fee_usd/df.media_cost_usd
        if 'VCR' in kpi:
            df['VCR'] = df.video_complete/df.video_start
        if 'CPCV' in kpi:
            df['CPCV'] = df.total_spend/df.video_complete
        if 'mCPM' in kpi:
            df['mCPM'] = (df.media_cost*1000)/df.impressions
        if 'CPA(eng45)' in kpi:
            df['CPA(eng45)'] = df.total_spend/df.engagement_45s
        if 'CPA(dealer)' in kpi:
            df['CPA(dealer)'] = df.total_spend/df.consideration_dealer    
        if 'CPA(shopnow)' in kpi:
            df['CPA(shopnow)'] = df.total_spend/df.consideration_shopnow
        if 'CPA(purchase)' in kpi:
            df['CPA(purchase)'] = df.total_spend/df.purchase_confirm
        if 'CPA(lead_club)' in kpi:
            df['CPA(lead_club)'] = df.total_spend/df.lead_club
        if 'CPA(product_registration)' in kpi:
            df['CPA(product_registration)'] = df.total_spend/df.product_registration
        if 'Registration CPA' in kpi:
            df['Registration CPA'] = df.total_spend/df.Registration.replace(0, np.nan)
            df['Registration CPA'] = df['Registration CPA'].fillna(0)
        if 'FTD CPA' in kpi:
            df['FTD CPA'] = df.total_spend/df['FTD'].replace(0, np.nan)   
            df['FTD CPA'] = df['FTD CPA'].fillna(0)
        if 'FTD CPA_pc' in kpi:
            df['FTD CPA_pc'] = df.total_spend/df['FTD_pc'].replace(0, np.nan)   
            df['FTD CPA_pc'] = df['FTD CPA_pc'].fillna(0)
        if 'FTD CPA_pv' in kpi:
            df['FTD CPA_pv'] = df.total_spend/df['FTD_pv'].replace(0, np.nan)   
            df['FTD CPA_pv'] = df['FTD CPA_pv'].fillna(0)
        if 'Deposit CPA' in kpi:
            df['Deposit CPA'] = df.total_spend/df['Deposit'].replace(0, np.nan)  
            df['Deposit CPA'] = df['Deposit CPA'].fillna(0)
        if 'Deposit CPA_pc' in kpi:
            df['Deposit CPA_pc'] = df.total_spend/df['Deposit_pc'].replace(0, np.nan)  
            df['Deposit CPA_pc'] = df['Deposit CPA_pc'].fillna(0)
        if 'Deposit CPA_pv' in kpi:
            df['Deposit CPA_pv'] = df.total_spend/df['Deposit_pv'].replace(0, np.nan)  
            df['Deposit CPA_pv'] = df['Deposit CPA_pv'].fillna(0)
        if 'WalletConnected CPA' in kpi:
            df['WalletConnected CPA'] = df.total_spend/df['WalletConnected'].replace(0, np.nan)  
            df['WalletConnected CPA'] = df['WalletConnected CPA'].fillna(0)

        if 'ROI' in kpi:  
            df['ROI'] = df.total_revenue-df.total_spend
        if 'ROAS' in kpi:  
            df['ROAS'] = df.total_revenue/df.total_spend
        if 'ROAS (35%)' in kpi:  
            df['ROAS (35%)'] = (df.total_revenue*0.35)/df.total_spend
        if 'ROI (35%)' in kpi:  
            df['ROI (35%)'] = (df.total_revenue*0.35)-df.total_spend
        if 'ROI_usd' in kpi:  
            df['ROI_usd'] = df.total_revenue_usd/df.total_spend_usd
        df=df.sort_values(by=sortby, ascending=ascending)
        df.replace([np.inf, -np.inf], np.nan)
        # df=df_cct_rename(df)
    
        return df 




def df_columns_rename(df):
    df = df.rename(columns={'Date': 'date', 'Advertiser': 'advertiser',
                        'Advertiser Currency': 'advertiser_currency', 'lineItem': 'line_item',
                        'lineItemID': 'line_item_id', 'campaign': 'campaign_name',
                        'campaignID': 'campaign_id', 'Exchange ID': 'exchange_id','Impr.':'impressions',
                        'Impressions': 'impressions', 'Billable Impression': 'billable_impressions','cost':'total_spend',
                        'Click Rate (CTR)': 'ctr','conversionsAll':'total_conversions',
                        'Total Conversions': 'total_conversions', 'Post-Click Conversions': 'pc_conversions',
                        'Post-View Conversions': 'pv_conversions','salesAll': 'total_revenue', 'sales': 'total_revenue', 
                         'Exchange': 'exchange', 'DV360 Site':'app/url','App/URL':'app/url','Day of Week':'weekday_name','Time of Day':'time_of_day',
                        'Engagements': 'engagements','Browser':'browser', 'Device Make':'device_make',
                        'Device Model':'device_model', 'Device Type':'device_type', 'Environment':'environment', 'Operating System':'operating_system',
                        'Inventory Source':'inventory_source','Complete Views (Video)': 'complete_views', 'Total Media Cost (Advertiser Currency)': 'total_spend','DV360 Line Item': 'line_item',  'DV360 Line Item ID': 'line_item_id','DV360 Campaign ID': 'campaign_id','DV360 Campaign': 'campaign_name','DV360 Insertion Order': 'Insertion Order','DV360 Cost (Account Currency)': 'total_spend','Total Revenue': 'total_revenue','Conversion 1': 'conversions_1','Conversion 2': 'conversions_2','Conversion 3': 'conversions_3','campaignId': 'campaign_id', 'Total Spent': 'total_spend','adv_cost': 'total_spend', 'total_costs': 'total_spend',  'money': 'total_spend'}) 
    return df

def df_cct_rename(df):
    df = df.rename(columns={'campaign_name': 'Campaign Name', 
                            'impressions': 'Impressions',
                            'clicks': 'Clicks', 
                            'month': 'Month', 
                            'media_cost': 'Media Cost',
                            'total_spend': 'Total Cost',
                            'total_revenue': 'Total Sales',
                             'ROI': 'ROAS',
                           'Registration': 'Registrations' ,'Deposit': 'Deposits' ,'Deposit Conversion': 'Deposits' ,'Walletconnected': 'WalletConnect' }) 
    return df



def df_format_campaign(df):
    df['Buyer'] = df['campaign_name'].str.split('_').str[1]
    df['Country'] = df['campaign_name'].str.split('_').str[2]
    df['Network'] = df['campaign_name'].str.split('_').str[3]
    df['Campaign Type'] = df['campaign_name'].str.split('_').str[4]
    df['Product'] = df['campaign_name'].str.split('_').str[5]
    df['Year_start'] = df['campaign_name'].str.split('_').str[7]
    df['Month_start'] = df['campaign_name'].str.split('_').str[8]
    df['Strategy'] = df['campaign_name'].str.split('_').str[6] 
    return df


# Moloco

import requests
import time
import pandas as pd
from io import StringIO

def fetch_moloko_report(api_key, ad_account_id, start_date, end_date, token_url, report_url, timeout=120):
    """
    Fetches a MOL report as a DataFrame.

    Parameters:
    - api_key (str): API key for authentication
    - ad_account_id (str): Ad account ID
    - start_date (str): Report start date in "YYYY-MM-DD"
    - end_date (str): Report end date in "YYYY-MM-DD"
    - token_url (str): URL to retrieve access token
    - report_url (str): URL to request the report
    - timeout (int): Max time to wait for report readiness (seconds)

    Returns:
    - pd.DataFrame: The final report as a DataFrame
    """

    # === STEP 1: Get Access Token ===
    token_payload = { "api_key": api_key }
    token_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    token_response = requests.post(token_url, headers=token_headers, json=token_payload)
    if token_response.status_code != 200:
        raise Exception(f"❌ Failed to get token: {token_response.text}")
    access_token = token_response.json().get("token")

    # === STEP 2: Request Report ===
    report_payload = {
        "date_range": {
            "start": start_date,
            "end": end_date
        },
        "ad_account_id": ad_account_id,
        "dimensions": ["DATE", "CAMPAIGN"]
    }

    report_headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    report_response = requests.post(report_url, json=report_payload, headers=report_headers)
    if report_response.status_code != 202:
        raise Exception(f"❌ Report request failed: {report_response.text}")

    report_status_url = report_response.json().get("status")

    # === STEP 3: Poll Until Report is Ready ===
    start_time = time.time()
    while True:
        status_response = requests.get(report_status_url, headers=report_headers)
        status_json = status_response.json()

        if "location_csv" in status_json:
            report_download_url = status_json["location_csv"]
            break
        elif time.time() - start_time > timeout:
            raise Exception("⏳ Timed out waiting for report.")
        else:
            time.sleep(5)

    # === STEP 4: Download CSV Report ===
    download_response = requests.get(report_download_url)
    if download_response.status_code != 200:
        raise Exception(f"❌ Failed to download report: {download_response.text}")

    csv_data = download_response.content.decode("utf-8")
    df = pd.read_csv(StringIO(csv_data))
    
    return df

def mol_get_brand_stats(API_KEY, AD_ACCOUNT_ID, start_date, end_date):
    # # === CONFIGURATION ===
    TOKEN_URL = "https://api.moloco.cloud/cm/v1/auth/tokens"

    REPORT_URL = f"https://api.moloco.cloud/cm/v1/reports?ad_account_id={AD_ACCOUNT_ID}"

    # === STEP 1: GET ACCESS TOKEN ===

    token_payload = {
        "api_key": API_KEY
    }
    token_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    token_response = requests.post(TOKEN_URL, headers=token_headers, json=token_payload)

    if token_response.status_code != 200:
        raise Exception(f"❌ Failed to get token: {token_response.text}")

    access_token = token_response.json()["token"]

    # === STEP 2: REQUEST REPORT ===
    report_payload = {
        "date_range": {
            "start": start_date,
            "end": end_date
        },
        "ad_account_id": AD_ACCOUNT_ID,
        "dimensions": ["DATE","CAMPAIGN"]
    }

    report_headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    report_response = requests.post(REPORT_URL, json=report_payload, headers=report_headers)


    if report_response.status_code != 202:
        raise Exception(f"❌ Report request failed: {report_response.text}")

    # === STEP 3: POLL UNTIL REPORT IS READY ===
    report_status_url = report_response.json()["status"]

    timeout = 120  # seconds
    start_time = time.time()

    while True:
        status_response = requests.get(report_status_url, headers=report_headers)
        status_json = status_response.json()

        if "location_csv" in status_json:
            report_download_url = status_json["location_csv"]
            break

        elif time.time() - start_time > timeout:
            raise Exception("⏳ Timed out waiting for report.")
        else:
            time.sleep(5)


    # === STEP 4: DOWNLOAD REPORT WITHOUT AUTH HEADER ===
    download_response = requests.get(report_download_url)  # No headers here!

    if download_response.status_code == 200:
        csv_data = download_response.content.decode("utf-8")
        df_mol = pd.read_csv(StringIO(csv_data))  
        df_mol=df_columns_rename(df_mol)
        df_mol['date']=df_mol['date']
        df_mol['Brand'] = df_mol['Campaign_Title']
        df_mol['Brand']=df_mol['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
        df_mol['Brand']=df_mol['Brand'].apply(brand_clean_polish)
        df_mol = add_presale_to_brand(df_mol,  external_column='Brand')
        df_mol=df_mol.fillna(0)
        df_mol['impressions'] = df_mol['impressions']
        df_mol['impressions'] = pd.to_numeric(df_mol['impressions'], errors='coerce')  # Convert to float, replacing errors with NaN
        df_mol['impressions'] = df_mol['impressions'].fillna(0).astype(int)  # Fill NaNs with 0 and convert to int
        df_mol['adv_impressions'] = df_mol['impressions']
        df_mol['adv_impressions']=df_mol['impressions'].astype(int)
        df_mol['clicks'] = df_mol['Clicks']
        df_mol['clicks'] = df_mol['clicks'].replace('', '0')  # Replace empty strings with '0'
        df_mol['clicks'] = df_mol['clicks'].fillna(0)  # Replace NaNs with 0
        df_mol['adv_clicks'] = df_mol['clicks'].astype(int)  # Convert to int
        df_mol['total_spend']=df_mol['Spend']
        df_mol['total_spend'] = df_mol['total_spend'].replace('n/a', 0)
        df_mol['total_spend']=df_mol['total_spend'].replace('-', np.nan)
        df_mol['total_spend']=df_mol['total_spend'].replace('N/A', 0)
        df_mol['total_spend']=df_mol['total_spend'].astype(float)
        df_mol['total_spend_campaign_currency']=df_mol['total_spend']
        df_mol['network']='Moloco'
        df_mol=df_mol[['date','network','Brand','adv_clicks','adv_impressions','total_spend', 'total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    
      
    else:
        return(f"❌ Failed to download report: {download_response.status_code}. Response: {download_response.text}")
    return df_mol




# Exoclick

import requests
import json
from datetime import datetime
from functools import lru_cache

# Set your token here
API_URL = "https://api.exoclick.com/v2"
TOKEN_CACHE_DURATION = 1500  # 25 minutes in seconds

class ExoClickAPI:
    def __init__(self, api_token):
        self.api_token = api_token
        self.token_data = self.login()

    def login(self):
        url = f"{API_URL}/login"
        payload = {"api_token": self.api_token}
        headers = {"Content-Type": "application/json"}

        for _ in range(25):
            res = requests.post(url, data=json.dumps(payload), headers=headers)
            if res.status_code == 200:
                return res.json()
            else:
                print(f"Login attempt failed: {res.status_code} - {res.text}")
        raise Exception("Failed to authenticate with API")

    def get_auth_header(self):
        return {"Authorization": f"{self.token_data['type']} {self.token_data['token']}"}

    def options_data(self, method="GET", payload=None):
        headers = self.get_auth_header()
        if payload:
            return {
                "headers": headers,
                "json": payload
            }
        else:
            return {"headers": headers}

    def api_get_main_data(self, from_date=None, to_date=None, order="cost", limit=10000):
        if not from_date:
            from_date = datetime.today().replace(day=1).strftime("%Y%m%d")
        if not to_date:
            to_date = datetime.today().strftime("%Y%m%d")

        url = (f"{API_URL}/statistics/advertiser/date"
               f"?date-to={to_date}&date-from={from_date}"
               f"&limit={limit}"
               f"&include=totals&additional_group_by=campaign&exclude_deleted=false")

        for _ in range(25):
            res = requests.get(url, headers=self.get_auth_header())
            if res.status_code == 200:
                return res.json().get("result", [])
        print(f"Error fetching main data: {res.status_code} - {res.text}")
        return []

    def get_campaign_info(self, campaign_id):
        url = f"{API_URL}/campaigns/{campaign_id}"
        for _ in range(25):
            res = requests.get(url, headers=self.get_auth_header())
            if res.status_code == 200:
                return res.json().get("result", {})
        print(f"Error fetching campaign info: {res.status_code} - {res.text}")
        return {}

    def get_pricing_model(self, model):
        pricing_model = {"cpc": "1", "cpm": "2", "cpa": "3", "smart": "4", "cpv": "5"}
        return pricing_model.get(model, model)

    def get_network(self, value):
        return {"All": "0", "RON": "1", "Premium": "2", "Members Area": "3"}.get(value)

    def get_partner(self, value):
        return {"Enabled": "1", "Disabled": "0"}.get(value)

# Working with GADS, Apple and Twitter

def df_convert_currency(df, df_rates):
    df['date'] = pd.to_datetime(df['date'])
    df_rates['date'] = pd.to_datetime(df_rates['date'])
    
    # Merge the dataframes on date
    df = pd.merge(df, df_rates, on='date', how='left')
    
    # Apply conversion based on the 'currency' column
    df['total_spend'] = df.apply(
        lambda row: row['total_spend_campaign_currency'] * row['GBP_to_USD']
        if row['currency_code'] == 'GBP'
        else row['total_spend_campaign_currency'] * row['EUR_to_USD'], 
        axis=1
    )
    
    return df

   
def fetch_googleads_report(client, start_date, end_date):
    query= f"SELECT * FROM `sunny-hope-447708-q0.dbt_rhopp_google_ads.google_ads__account_report`"
    df_gads = client.query(query).result().to_dataframe()
    df_gads['network']='Google Ads'
    df_gads['date_day'] = pd.to_datetime(df_gads['date_day'])
    df_gads['date'] = df_gads['date_day'].dt.strftime('%Y-%m-%d')
    df_gads['Brand'] = df_gads['source_relation'].str.split('_').str[2]
    df_gads['Brand']=df_gads['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
    df_gads['Brand']=df_gads['Brand'].apply(brand_clean_polish)
    df_gads = add_presale_to_brand(df_gads, external_column='source_relation')
    df_gads=df_columns_rename(df_gads)
    df_gads['total_spend']=df_gads['spend'].astype(float)
    df_gads['total_spend_campaign_currency']=df_gads['total_spend'].astype(float)
    df_gads['adv_impressions']=df_gads['impressions']
    df_gads['adv_clicks']=df_gads['clicks']
    
    query = f"""SELECT *FROM `dwh-landing-v1.exchange_rates.currency_api_usd_daily` WHERE Date >= '{start_date}' AND Date <= '{end_date}'"""
    currency_rates = client.query(query).result().to_dataframe()
    filtered_rates = currency_rates[currency_rates['To_Currency'].isin(['EUR', 'GBP'])]
    df_rates = filtered_rates.pivot(index='Date', columns='To_Currency', values='Rate').reset_index()
    df_rates.columns.name = None  # remove the pivot column name
    df_rates.rename(columns={
        'Date': 'date',
        'EUR': 'EUR_to_USD',
        'GBP': 'GBP_to_USD'}, inplace=True)
    df_gads=df_convert_currency(df_gads, df_rates)
    df_gads=df_gads[['date','network','Brand','adv_impressions','adv_clicks','total_spend','total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    df_gads=df_gads[(df_gads['Brand']=='bestwalletapp')|(df_gads['Brand']=='jemlit')]
    df_gads=df_gads[(df_gads['date']>=start_date)&(df_gads['date']<=end_date)]
    df_gads['date'] = pd.to_datetime(df_gads['date'])
    return df_gads


def fetch_appleads_report(client, start_date, end_date):
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_appleads_apple_search_ads.apple_search_ads__organization_report`WHERE date_day >= '{start_date}' and date_day <='{end_date}'"
    df_apple = client.query(query).result().to_dataframe()
    df_apple['Brand']='bestwalletapp'
    df_apple['date_day'] = pd.to_datetime(df_apple['date_day'])
    df_apple['date'] = df_apple['date_day'].dt.date
    df_apple['network']='Apple Search Ads'
    df_apple['adv_impressions']=df_apple['impressions']
    df_apple['adv_clicks']=0
    df_apple=df_apple.fillna(0)
    df_apple['total_spend']=df_apple['spend']
    df_apple['total_spend'] = df_apple['spend'].replace('n/a', 0)
    df_apple['total_spend']=df_apple['spend'].replace('-', np.nan)
    df_apple['total_spend']=df_apple['spend'].replace('N/A', 0)
    df_apple['total_spend']=df_apple['spend'].astype(float)
    df_apple['total_spend_campaign_currency']=df_apple['total_spend']
    df_apple=df_apple[['date','network','Brand','adv_clicks','adv_impressions','total_spend', 'total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    df_apple=df_apple[['date', 'network', 'Brand', 'total_spend', 'total_spend_campaign_currency','adv_clicks', 'adv_impressions']]
    df_apple['date'] = pd.to_datetime(df_apple['date'])
    return df_apple

def fetch_twitter_report(client, start_date, end_date):
    query = f"""SELECT *FROM `dwh-landing-v1.paid_media_twitter_jemlit_twitter_ads.twitter_ads__account_report`WHERE date_day >= TIMESTAMP('{start_date}') AND date_day <= TIMESTAMP('{end_date}')"""
    df_tw_jemlit = client.query(query).result().to_dataframe()
    df_tw_jemlit['Brand']='jemlit'
    df_tw_jemlit['date_day'] = pd.to_datetime(df_tw_jemlit['date_day'])
    df_tw_jemlit['date'] = df_tw_jemlit['date_day'].dt.date
    df_tw_jemlit['network']='Twitter (Media)'
    df_tw_jemlit['adv_impressions']=df_tw_jemlit['impressions']
    df_tw_jemlit['adv_clicks']=df_tw_jemlit['clicks']
          
    query = f"""SELECT *FROM `dwh-landing-v1.paid_media_twitter_bestwallet_twitter_ads.twitter_ads__account_report`WHERE date_day >= TIMESTAMP('{start_date}') AND date_day <= TIMESTAMP('{end_date}')"""
    df_tw_betswallet = client.query(query).result().to_dataframe()
    df_tw_betswallet = client.query(query).result().to_dataframe()
    df_tw_betswallet['Brand']='bestwallet'
    df_tw_betswallet['date_day'] = pd.to_datetime(df_tw_betswallet['date_day'])
    df_tw_betswallet['date'] = df_tw_betswallet['date_day'].dt.date
    df_tw_betswallet['network']='Twitter (Media)'
    df_tw_betswallet['adv_impressions']=df_tw_betswallet['impressions']
    df_tw_betswallet['adv_clicks']=df_tw_betswallet['clicks']
    
    query = f"""SELECT *FROM `dwh-landing-v1.paid_media_twitter_coinpoker_twitter_ads.twitter_ads__account_report`WHERE date_day >= TIMESTAMP('{start_date}') AND date_day <= TIMESTAMP('{end_date}')"""
    df_tw_cp = client.query(query).result().to_dataframe()
    df_tw_cp['Brand']='coinpoker'
    df_tw_cp['date_day'] = pd.to_datetime(df_tw_cp['date_day'])
    df_tw_cp['date'] = df_tw_cp['date_day'].dt.date
    df_tw_cp['network']='Twitter (Media)'
    df_tw_cp['adv_impressions']=df_tw_cp['impressions']
    df_tw_cp['adv_clicks']=df_tw_cp['clicks']
    
    df_fbtw = pd.concat([df_tw_betswallet, df_tw_cp, df_tw_jemlit], ignore_index=True)
    df_fbtw['date']=df_fbtw['date'].astype(str)
    df_fbtw['date'] = pd.to_datetime(df_fbtw['date'])
    df_fbtw['date'] = df_fbtw['date'].dt.strftime('%Y-%m-%d')
    df_fbtw=df_fbtw[(df_fbtw['date']>=start_date)&(df_fbtw['date']<=end_date)]
    df_fbtw=df_fbtw.fillna(0)
    df_fbtw['total_spend']=df_fbtw['spend']
    df_fbtw['total_spend'] = df_fbtw['spend'].replace('n/a', 0)
    df_fbtw['total_spend']=df_fbtw['spend'].replace('-', np.nan)
    df_fbtw['total_spend']=df_fbtw['spend'].replace('N/A', 0)
    df_fbtw['total_spend']=df_fbtw['spend'].astype(float)
    df_fbtw['total_spend_campaign_currency']=df_fbtw['total_spend']
    df_fbtw=df_fbtw[['date','network','Brand','adv_clicks','adv_impressions','total_spend', 'total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    return df_fbtw


# Facebook
def fetch_meta_report(client, start_date, end_date):
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.meta_campaigns_raw`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_fb = client.query(query).result().to_dataframe()
    df_fb=df_columns_rename(df_fb)
    df_fb['date'] = pd.to_datetime(df_fb['date'])
    df_fb['Brand'] = df_fb['brand']
    df_fb['Brand']=df_fb['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
    df_fb['Brand']=df_fb['Brand'].apply(brand_clean_polish)
    df_fb = add_presale_to_brand(df_fb,  external_column='Brand')
    df_fb=df_fb.fillna(0)
    df_fb['impressions'] = df_fb['impressions'].fillna(0).astype(int)  # Fill NaNs with 0 and convert to int
    df_fb['adv_impressions'] = df_fb['impressions']
    df_fb['adv_impressions']=df_fb['impressions'].astype(int)
    df_fb['clicks'] = df_fb['clicks'].fillna(0)  # Replace NaNs with 0
    df_fb['adv_clicks'] = df_fb['clicks'].astype(int)  # Convert to int
    df_fb['total_spend']=df_fb['total_spent']
    df_fb['total_spend'] = df_fb['total_spend'].replace('n/a', 0)
    df_fb['total_spend']=df_fb['total_spend'].replace('-', np.nan)
    df_fb['total_spend']=df_fb['total_spend'].replace('N/A', 0)
    df_fb['total_spend']=df_fb['total_spend'].astype(float)
    df_fb['total_spend_campaign_currency']=df_fb['total_spend']
    df_fb['network']='Meta (Media)'
    df_fb=df_fb[['date','network','Brand','adv_clicks','adv_impressions','total_spend', 'total_spend_campaign_currency']].groupby(['date','network','Brand']).sum().reset_index()
    return df_fb




# Facebook
def fetch_etherscan_report(client, start_date, end_date):
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.etherscan_placements_raw`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_etherscan = client.query(query).result().to_dataframe()
    df_etherscan=df_columns_rename(df_etherscan)
    df_etherscan['date'] = pd.to_datetime(df_etherscan['date'])
    df_etherscan['date']=df_etherscan['date'].dt.strftime('%Y-%m-%d')
    df_etherscan['network']='Etherscan (Media)'
    df_etherscan['Brand'] = df_etherscan['Brand'].str.split('_').str[0]
    df_etherscan['Brand']=df_etherscan['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
    df_etherscan['Brand']=df_etherscan['Brand'].apply(brand_clean_polish)
    df_etherscan = add_presale_to_brand(df_etherscan,  external_column='Brand')
    df_etherscan['Brand'] = df_etherscan['Brand'].replace({
            'mipepe':'mindofpepe',
            'mi': 'memeindex',
        })
    df_etherscan['impressions']=df_etherscan['impressions'].str.replace(',', '')
    df_etherscan['impressions'] = df_etherscan['impressions'].replace('n/a', 0)
    df_etherscan['impressions'] = df_etherscan['impressions'].replace('N/A', 0)
    df_etherscan=df_etherscan.fillna(0)
    df_etherscan['eth_impressions'] = df_etherscan['impressions'].replace('-', np.nan)
    df_etherscan['eth_impressions'] = df_etherscan['eth_impressions'].astype(float).astype('Int64')  # Keeps NaNs
    df_etherscan["impressions"] = pd.to_numeric(df_etherscan["impressions"], errors="coerce")
    df_etherscan['Clicks'] = pd.to_numeric(df_etherscan["Clicks"], errors="coerce")
    df_etherscan['total_spend_eth']=df_etherscan['Total_Spent']
    df_etherscan['total_spend_eth']=df_etherscan['total_spend_eth'].str.replace(',', '')
    df_etherscan['total_spend_eth'] = df_etherscan['total_spend_eth'].replace('n/a', 0)
    df_etherscan['total_spend_eth']=df_etherscan['total_spend_eth'].replace('-', np.nan)
    df_etherscan['total_spend_eth']=df_etherscan['total_spend_eth'].replace('N/A', 0)
    df_etherscan['total_spend_eth']=df_etherscan['total_spend_eth'].astype(float)
    df_etherscan=df_etherscan[['date', 'Brand', 'Site', 'Placement','Clicks','network', 'eth_impressions', 'total_spend_eth']]
    df_etherscan=df_etherscan[['date','network','Brand','total_spend_eth', 'eth_impressions']].groupby(['date','network','Brand']).sum().reset_index()

    return df_etherscan

# FIX Spend


import calendar

def days_in_month(date):
    return calendar.monthrange(date.year, date.month)[1]

def fetch_fix_spend(client, start_date, end_date):
    query = f"SELECT Network as network , Monthly_budget as monthly_budget FROM `dwh-landing-v1.paid_media_network_raw.fix_budgets`WHERE Start_Date <= '{start_date}' and End_Date >='{end_date}'"
    df_fix_budget_rest = client.query(query).result().to_dataframe()
    df_fix_budget_rest['network']=df_fix_budget_rest['network'].str.replace('Cynes.com', 'cnyes.com (Media)')
    start_date_dt = pd.to_datetime(start_date)
    df_fix_budget_rest['days_in_month'] = start_date_dt.days_in_month
    df_fix_budget_rest['daily_budget'] = df_fix_budget_rest['monthly_budget'] / df_fix_budget_rest['days_in_month']
    fix_networks = df_fix_budget_rest.network.unique()
    fix_networks_sql = ", ".join(f"'{n}'" for n in fix_networks)
    
    query = f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.adform_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}' and network IN ({fix_networks_sql}) and  network != 'DexScreener (Media)'"
    df_fs_corr_dsp = client.query(query).result().to_dataframe()
    df_fs_corr_fix_dict=pd.merge(df_fs_corr_dsp, df_fix_budget_rest,  how='left', left_on=['network'], right_on=['network'])
    df_fs_corr_fix_dict=df_fs_corr_fix_dict[(df_fs_corr_fix_dict['monthly_budget']>0)&(df_fs_corr_fix_dict['impressions']>10)]
    df_fs_corr_fix_dict['total_impressions'] = df_fs_corr_fix_dict.groupby([ 'date','network'])['impressions'].transform('sum')
    df_fs_corr_fix_dict['fix_allocated_budget'] = (df_fs_corr_fix_dict['impressions'] / df_fs_corr_fix_dict['total_impressions']) * df_fs_corr_fix_dict['daily_budget']
    df_fs_corr_fix_dict=df_fs_corr_fix_dict[['date','network', 'Brand', 'fix_allocated_budget']]
    df_fs_corr_fix_dict.columns=['date','network','Brand', 'fix_budget']
    df_fs_corr_fix_dict['date'] = pd.to_datetime(df_fs_corr_fix_dict['date'], errors='coerce')  # Convert to datetime
    df_fs_corr_fix_dict['date']=df_fs_corr_fix_dict['date'].dt.strftime('%Y-%m-%d')    
    
    stop_clients="'Casinos - APAC','Casinos - Americas','Crypto Presales - APAC','Crypto Presales - Shared Tracking - Global','Casinos - Shared Tracking'"
    query = f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.adform_banners_dexscreener`WHERE date >= '{start_date}' and date <='{end_date}' and client NOT IN ({stop_clients})"
    final_df = client.query(query).result().to_dataframe()
    final_df['Brand']=final_df['campaign'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
    df_dex_excl=final_df[final_df['banner'].str.contains('hotbar', case=False, na=False)]
    df_dex_fix=final_df[final_df['banner'].str.contains('search|homepage|heavyusers|serach', case=False, na=False)]
    df_dex_fix['date'] = pd.to_datetime(df_dex_fix['date'], errors='coerce')  # Convert to datetime
    df_dex_fix['date']=df_dex_fix['date'].dt.strftime('%Y-%m-%d')
   
    df_dex_fix_ds=df_dex_fix[['date','network','Brand','impressions']].groupby(['date','network','Brand']).sum().reset_index()
    df_fs_corr_fix_dict_dex=pd.merge(df_dex_fix_ds, df_fix_budget_rest,  how='left', left_on=['network'], right_on=['network'])
    df_fs_corr_fix_dict_dex=df_fs_corr_fix_dict_dex[(df_fs_corr_fix_dict_dex['monthly_budget']>0)&(df_fs_corr_fix_dict_dex['impressions']>10000)]
    df_fs_corr_fix_dict_dex['total_impressions'] = df_fs_corr_fix_dict_dex.groupby([ 'date','network'])['impressions'].transform('sum')
    df_fs_corr_fix_dict_dex['fix_allocated_budget'] = (df_fs_corr_fix_dict_dex['impressions'] / df_fs_corr_fix_dict_dex['total_impressions']) * df_fs_corr_fix_dict_dex['daily_budget']
    df_fs_corr_fix_dict_dex=df_fs_corr_fix_dict_dex[['date','network', 'Brand', 'fix_allocated_budget','impressions']]
    df_fs_corr_fix_dict_dex.columns=['date','network','Brand','fix_budget', 'impressions']
    df_fs_corr_fix_dict_final = pd.concat([df_fs_corr_fix_dict, df_fs_corr_fix_dict_dex], ignore_index=True)

    final_df=final_df[~final_df['banner'].str.contains('hotbar|search|homepage|heavyusers|serach', case=False, na=False)]
    final_df['banner_spend']=(final_df['impressions']/1000)*5
    final_df=df_columns_rename(final_df)
    final_df['Brand']=final_df['campaign_name'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
    final_df = add_presale_to_brand(final_df)
    final_df['Brand']=final_df['Brand'].str.replace('tonaldtrumps', 'tonaldtrump')
    final_df_ds=final_df[['date','network','Brand','banner_spend']].groupby(['date','network', 'Brand']).sum().reset_index()
    final_df_ds['date'] = pd.to_datetime(final_df_ds['date'], errors='coerce')
    final_df_ds['date'] = final_df_ds['date'].dt.strftime('%Y-%m-%d')

    df_dex_excl=df_columns_rename(df_dex_excl)
    df_dex_excl['Brand']=df_dex_excl['campaign_name'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
    df_dex_excl = add_presale_to_brand(df_dex_excl)
    df_dex_excl['Brand']=df_dex_excl['Brand'].str.replace('tonaldtrumps', 'tonaldtrump')
    df_dex_excl=df_dex_excl[['date','network','Brand','impressions','clicks' ]].groupby(['date','network', 'Brand']).sum().reset_index()
    df_dex_excl.columns=['date', 'network', 'Brand', 'del_impressions', 'del_clicks']
    df_dex_excl['date'] = pd.to_datetime(df_dex_excl['date'], errors='coerce')
    df_dex_excl['date'] = df_dex_excl['date'].dt.strftime('%Y-%m-%d')

    final_df_ds_fin=pd.merge(df_dex_excl, final_df_ds,  how='left', left_on=['date','network', 'Brand'], right_on=['date','network', 'Brand'])
    diff_df_gt = final_df_ds.merge(df_dex_excl, on=['date','Brand', 'network'], how='left', indicator=True).query('_merge == "left_only"')
    diff_df_gt = diff_df_gt.drop('_merge', axis=1)
    final_df_ds_fin = pd.concat([final_df_ds_fin, diff_df_gt], ignore_index=True)

    df_fs_corr_fix_dictgt=pd.merge(df_fs_corr_fix_dict_final, final_df_ds_fin,  how='left', left_on=['date','network', 'Brand'], right_on=['date','network', 'Brand'])
    diff_df_gt = final_df_ds_fin.merge(df_fs_corr_fix_dict_final, on=['date','Brand', 'network'], how='left', indicator=True).query('_merge == "left_only"')
    diff_df_gt = diff_df_gt.drop('_merge', axis=1)
    df_fs_corr_fix_dictgt = pd.concat([df_fs_corr_fix_dictgt, diff_df_gt], ignore_index=True)
    df_fs_corr_fix_dictgt = df_fs_corr_fix_dictgt.fillna(0)
    df_fs_corr_fix_dictgt['fix_budget']=df_fs_corr_fix_dictgt['fix_budget']+df_fs_corr_fix_dictgt['banner_spend']
    df_fs_corr_fix_dictgt=df_fs_corr_fix_dictgt[['date', 'network', 'Brand', 'fix_budget', 'del_impressions','del_clicks']]
    return df_fs_corr_fix_dictgt



# Mail operation

def connect_to_gmail(EMAIL_USER, EMAIL_PASS):
    IMAP_SERVER = "imap.gmail.com"
    IMAP_PORT = 993
    
    """Connect to Gmail using IMAP."""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_USER, EMAIL_PASS)
        return mail
    except imaplib.IMAP4.error as e:
        return str(e)

def fetch_csv_attachments(mail,SENDER_EMAIL, SUBJECT):
    """Fetch the most recent CSV attachment from emails with the specified subject from the sender."""
    mail.select('"[Gmail]/All Mail"')  # Searches both inbox and archived emails
    status, messages = mail.search(None, f'FROM "{SENDER_EMAIL}" SUBJECT "{SUBJECT}"')

    if status != "OK" or not messages[0]:
        return "No emails found from the sender with the specified subject."

    message_ids = messages[0].split()
    latest_email_id = message_ids[-1]  # Get the ID of the last email

    status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
    if status != "OK":
        return "Error fetching email."

    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])

            # Extract attachments
            for part in msg.walk():
                filename = part.get_filename()
                if filename and filename.endswith(".csv"):
                    attachment_data = part.get_payload(decode=True)
                    return filename, attachment_data  # Return the first CSV attachment found

    return "No CSV attachments found."


def detect_delimiter(data):
    """Detect the delimiter of a CSV file."""
    sample = data.decode(errors="ignore").splitlines()[0]  # Read first line
    sniffer = csv.Sniffer()
    try:
        return sniffer.sniff(sample).delimiter  # Auto-detect delimiter
    except csv.Error:
        return ","  # Default to comma

def load_csv_dataframe(data,first_cell):
    """Load CSV data into a Pandas DataFrame, skipping metadata before 'Date'."""
    try:
        delimiter = detect_delimiter(data)
        # Read raw CSV data into a list of lines
        raw_lines = data.decode(errors="ignore").splitlines()

        # Find the first row where the first column is exactly "Date"
        start_idx = next((i for i, line in enumerate(raw_lines) if line.strip().split(delimiter)[0] == first_cell), None)

        if start_idx is None:
            return "Error: 'Date' not found in the CSV."

        # Read the CSV from the detected row onward
        df = pd.read_csv(BytesIO(data), encoding="utf-8", engine="python", sep=delimiter, skiprows=start_idx)

        return df

    except Exception as e:
        return str(e)

    
def gmail_get_csv_report(EMAIL_USER, EMAIL_PASS,SENDER_EMAIL,SUBJECT,first_cell):
    mail = connect_to_gmail(EMAIL_USER, EMAIL_PASS)
    csv_attachment = fetch_csv_attachments(mail,SENDER_EMAIL, SUBJECT)
    if csv_attachment:
        filename, data = csv_attachment
        df_cmc = load_csv_dataframe(data,first_cell)
        if df_cmc is not None:
            return df_cmc
        else:
            return "Failed to parse CSV."
    else:
        return "No CSV attachments found."
    
def parse_mixed_dates(date_str):
    try:
        return pd.to_datetime(date_str, format="%m/%d/%Y")  # Try four-digit year
    except ValueError:
        return pd.to_datetime(date_str, format="%m/%d/%y")  # Try two-digit year    
    
def gmail_get_cmc_report(EMAIL_USER, EMAIL_PASS,SENDER_EMAIL,SUBJECT, first_cell):
    mail = connect_to_gmail(EMAIL_USER, EMAIL_PASS)
    csv_attachment = fetch_csv_attachments(mail,SENDER_EMAIL, SUBJECT)
    if csv_attachment:
        filename, data = csv_attachment
        df_cmc = load_csv_dataframe(data, first_cell)
        if df_cmc is not None:
            df_cmc=df_cmc[df_cmc['Date']!='Total']
            df_cmc['network']='CoinMarketCap (Media)'
            df_cmc['Brand']=df_cmc['Creative'].str.replace(' ', '').str.lower().apply(brand_cleanup)
            df_cmc['Brand']=df_cmc['Brand'].apply(brand_clean_polish)
            df_cmc = add_presale_to_brand(df_cmc, external_column='Creative')
            df_cmc=df_columns_rename(df_cmc)
            df_cmc['Total impressions']=df_cmc['Total impressions'].str.replace(',', '')
            df_cmc['Total impressions'] = df_cmc['Total impressions'].replace('n/a', 0)
            df_cmc['Total clicks']=df_cmc['Total clicks'].str.replace(',', '')
            df_cmc['Total clicks'] = df_cmc['Total clicks'].replace('n/a', 0)
            df_cmc['impressions_cmc']=df_cmc['Total impressions'].astype(int)
            df_cmc['clicks_cmc']=df_cmc['Total clicks'].astype(int)
            df_cmc= df_cmc[(~df_cmc['Line item'].str.contains('Coin.Network'))]
            df_cmc['total_spend_cmc']= (df_cmc['impressions_cmc']/1000)*5
            df_cmc['total_spend_cmc']= np.where(df_cmc['Line item'].str.contains('US_Incremental'),  (df_cmc['impressions_cmc']/1000)*3.5, df_cmc['total_spend_cmc'])
            df_cmc['total_spend_cmc']= np.where(df_cmc['Creative'].str.contains('CoinPoker|InstantCasino|LuckyBlock|TG-Casino'),  (df_cmc['impressions_cmc']/1000)*5, df_cmc['total_spend_cmc'])
            df_cmc['date'] = df_cmc['date'].astype(str).apply(parse_mixed_dates)
            df_cmc=df_cmc[(~df_cmc['Line item'].str.contains('Direct'))]
            df_cmc=df_cmc[['date','network','Brand','impressions_cmc', 'clicks_cmc', 'total_spend_cmc']].groupby(['date','network','Brand']).sum().reset_index()
            return df_cmc
        else:
            return "Failed to parse CSV."
    else:
        return "No CSV attachments found."


    
import pandas as pd
import requests
from io import StringIO
import imaplib
import email
from datetime import datetime
import pytz
import base64

def download_csv_to_dataframe(url):
    try:
        # Send a request to fetch the CSV content
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses (e.g., 404, 500)
        
        # Convert the response content to a DataFrame
        csv_data = StringIO(response.text)
        lines = csv_data.getvalue().strip().split('\n')

        # Find the header line starting with 'Date'
        header_line_index = next((i for i, line in enumerate(lines) if line.startswith('Date')), 0)
        header = lines[header_line_index].split(',')

        # Read the DataFrame using only the relevant data
        data_lines = '\n'.join(lines[header_line_index + 1:])
        df = pd.read_csv(StringIO(data_lines), names=header, parse_dates=['Date'])

        # Ensure the 'Date' column is sorted (if not already)
        df = df.sort_values(by='Date')
        
        # Drop the last row instead of all rows with the most recent date
        df = df.iloc[:-1]

        return df

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error for {url}: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error for {url}: {req_err}")
    except pd.errors.EmptyDataError:
        print(f"No data found at {url}")
    except KeyError:
        print(f"Missing 'Date' column in {url}")
    return None




def fetch_todays_report_imap(EMAIL_USER, EMAIL_PASS, SENDER_EMAIL, link_starts_with):
    try:
        today_str = datetime.now(pytz.UTC).strftime("%d-%b-%Y")
        mail = connect_to_gmail(EMAIL_USER, EMAIL_PASS)
        mail.select('"[Gmail]/All Mail"')

        # Search for emails from the sender since today
        status, data = mail.search(None, f'(FROM "{SENDER_EMAIL}" SINCE "{today_str}")')
        if status != "OK":
            return None

        for num in reversed(data[0].split()):
            status, msg_data = mail.fetch(num, '(RFC822)')
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            email_date = email.utils.parsedate_to_datetime(msg["Date"]).astimezone(pytz.UTC).date()
            today = datetime.now(pytz.UTC).date()
            if email_date != today:
                continue

            # Walk through message parts
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type in ["text/plain", "text/html"]:
                    try:
                        body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                        for line in body.splitlines():
                            if line.strip().startswith(link_starts_with):
                                return line.strip()
                    except Exception:
                        continue
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None
    

def fetch_latest_report_imap(EMAIL_USER, EMAIL_PASS, SENDER_EMAIL, link_starts_with):
    try:
        mail = connect_to_gmail(EMAIL_USER, EMAIL_PASS)
        mail.select('"[Gmail]/All Mail"')

        # Search for all emails from the sender (no date filtering)
        status, data = mail.search(None, f'(FROM "{SENDER_EMAIL}")')
        if status != "OK" or not data or not data[0]:
            return None

        message_ids = data[0].split()

        # Process newest messages first
        for msg_id in reversed(message_ids):
            status, msg_data = mail.fetch(msg_id, '(RFC822)')
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])

            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type in ["text/plain", "text/html"]:
                    try:
                        body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                        for line in body.splitlines():
                            if line.strip().startswith(link_starts_with):
                                return line.strip()
                    except Exception:
                        continue

        return None

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def gmail_get_linked_report(EMAIL_USER, EMAIL_PASS, SENDER_EMAIL, link_starts_with):
    report_link = fetch_todays_report_imap(EMAIL_USER, EMAIL_PASS, SENDER_EMAIL, link_starts_with)
    if report_link:
        df_imps_cgk = download_csv_to_dataframe(report_link)
        if df_imps_cgk is not None:
            df_imps_cgk['Brand'] = df_imps_cgk.apply(lambda row: "coinpoker" if row['Advertiser'] == "Finixio - CoinPoker" 
                else brand_clean_polish(brand_cleanup(row['Campaign'].replace(' ', '').lower())), axis=1)

            df_imps_cgk['Brand_creative']=df_imps_cgk['Creative'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
            df_imps_cgk['Brand'] = np.where(~df_imps_cgk['Brand'].str.contains('gamingbutton|sponsoredsearch'),  df_imps_cgk['Brand'], df_imps_cgk['Brand_creative'])

            df_imps_cgk['network'] = np.where(df_imps_cgk['Campaign'].str.contains('Coingecko|CoinGecko'),  'Coingecko (Media)', 'Geckoterminal (Media)')
            df_imps_cgk['impressions']=df_imps_cgk['Impressions']
            df_imps_cgk['date']=df_imps_cgk['Date']
            df_imps_cgk=calculate_cpm_spend(df_imps_cgk)
            df_imps_cgk['impressions_cmc']=df_imps_cgk['impressions']
            df_imps_cgk['clicks_cmc']=df_imps_cgk['Clicks']

            df_imps_cgk['total_spend_cmc']=df_imps_cgk['total_spend']
            df_imps_cgk=df_imps_cgk[['date','network','Brand','impressions_cmc', 'clicks_cmc','total_spend_cmc']].groupby(['date','network','Brand']).sum().reset_index()
            
            return df_imps_cgk
        else:
            return "Failed to parse report link."
    else:
        return "No report link found."


def gmail_get_cgkgam_report(EMAIL_USER, EMAIL_PASS,SENDER_EMAIL,SUBJECT, first_cell):
    mail = connect_to_gmail(EMAIL_USER, EMAIL_PASS)
    csv_attachment = fetch_csv_attachments(mail,SENDER_EMAIL, SUBJECT)
    if csv_attachment:
        filename, data = csv_attachment
        df_cmc = load_csv_dataframe(data, first_cell)
        if df_cmc is not None:
            df_cmc['Brand']=df_cmc['Line item'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
            df_cmc['Brand_creative']=df_cmc['Creative'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
            df_cmc['Brand'] = np.where(~df_cmc['Brand'].str.contains('gamingbutton|sponsoredsearch'),  df_cmc['Brand'], df_cmc['Brand_creative'])
            df_cmc['network'] = np.where(df_cmc['Line item'].str.contains('Coingecko|CoinGecko'),  'Coingecko (Media)', 'Geckoterminal (Media)')
            df_cmc['impressions']=df_cmc['Ad server impressions']
            df_cmc['date']= pd.to_datetime(df_cmc['Date'], format="%m/%d/%y", errors="coerce").dt.strftime("%Y-%m-%d")
            df_cmc['impressions']=df_cmc['impressions'].str.replace(',', '')
            df_cmc['impressions'] = df_cmc['impressions'].replace('n/a', 0)
            df_cmc['clicks']=df_cmc['Ad server clicks'].str.replace(',', '')
            df_cmc['clicks'] = df_cmc['clicks'].replace('n/a', 0)
            df_cmc['impressions']=df_cmc['impressions'].astype(int)
            df_cmc['clicks']=df_cmc['clicks'].astype(int)
            df_cmc['impressions_cmc']=df_cmc['impressions'].astype(int)
            df_cmc['clicks_cmc']=df_cmc['clicks'].astype(int)
            df_cmc=calculate_cpm_spend(df_cmc)
            df_cmc['impressions_cmc']=df_cmc['impressions']
            df_cmc['clicks_cmc']=df_cmc['clicks']
            df_cmc['total_spend_cmc']=df_cmc['total_spend']
            df_cmc=df_cmc[['date','network','Brand','impressions_cmc', 'clicks_cmc','total_spend_cmc']].groupby(['date','network','Brand']).sum().reset_index()
            return df_cmc
        else:
            return "Failed to parse CSV."
    else:
        return "No CSV attachments found."


# Report Daily Performance

def fill_vertical(row, vertical_mapping):
    if pd.isna(row['Vertical']) or row['Vertical'] == 0:
        return vertical_mapping.get(row['Brand'], None)
    return row['Vertical']

def cl_brand_report(client, start_date, end_date):
    # 1. Adform  Data
    query = f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.adform_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_fs_corr = client.query(query).result().to_dataframe()
   
    # 2. DSP spend data 
    # Coinzilla
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.coinzilla_juhamegadice_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_cz_ps = client.query(query).result().to_dataframe()
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.coinzilla_mattgascoigne_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_cz = client.query(query).result().to_dataframe()
    query = f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.coinzilla_paidmedia_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_cz_pm = client.query(query).result().to_dataframe()
    query = f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.coinzilla_icpaidmedia_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_cz_ic = client.query(query).result().to_dataframe()
    df_cz_fin = pd.concat([df_cz_ps,df_cz,df_cz_pm, df_cz_ic], ignore_index=True)
    df_cz_fin['date'] = pd.to_datetime(df_cz_fin['date'])
    df_cz_psf=df_cz_fin[['date','network','Brand','total_spend']].groupby(['date','network','Brand']).sum().reset_index()
    df_cz_psf['total_spend_campaign_currency']=df_cz_psf['total_spend']
    query = f"SELECT * FROM `dwh-landing-v1.exchange_rates.currency_api_usd_daily`WHERE Date >= '{start_date}' and Date <='{end_date}'"
    currency_rates = client.query(query).result().to_dataframe()
    eur_usd_today=currency_rates[currency_rates['To_Currency']=='EUR']
    eur_usd_today=eur_usd_today[['Date','Rate']]
    eur_usd_today.columns=['date', 'EUR_to_USD']
    df_cz_all=df_convert_eur(df_cz_psf,eur_usd_today)
    df_cz_all=df_cz_all[['date', 'network','Brand','total_spend', 'total_spend_campaign_currency']]
    
    # EX/Hue
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.hueads_campaign_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_hueads = client.query(query).result().to_dataframe()
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.pressize_campaign_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_pr = client.query(query).result().to_dataframe()
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.explorads_campaign_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_exp = client.query(query).result().to_dataframe()
    df_hpe = pd.concat([df_hueads, df_pr, df_exp], ignore_index=True)
    df_hpef = df_hpe[['date','network','Brand','total_spend', 'adv_clicks','adv_impressions']].groupby(['date','network','Brand']).sum().reset_index()
    df_hpef['total_spend_campaign_currency']=df_hpef['total_spend']
    # Cointraffic
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.cointraffic_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_ctrf = client.query(query).result().to_dataframe()
    # Match2One
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.m2o_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_m2o = client.query(query).result().to_dataframe()
    # P161 - Match2One
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.p161_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_p161 = client.query(query).result().to_dataframe()

    # Vizibl
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.vizibl_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_vz = client.query(query).result().to_dataframe()
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.bdsp_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_bsw = client.query(query).result().to_dataframe()
        
    # Personaly
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.personally_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_pers = client.query(query).result().to_dataframe()

    # Google Ads & Twitter & Apple
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.googleads_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_gads = client.query(query).result().to_dataframe()
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.appleads_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_apple = client.query(query).result().to_dataframe()
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.twitter_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_fbtw = client.query(query).result().to_dataframe()

    # Facebook
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.meta_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_fb = client.query(query).result().to_dataframe()
    # Moloco
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.moloco_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_mol = client.query(query).result().to_dataframe()
    # Exoclick
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.exoclick_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_excl = client.query(query).result().to_dataframe()
    # MediaMath
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.mediamath_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_mmf = client.query(query).result().to_dataframe()
    
    # 3.  Adform + DSP Data
    df_dsps_api_ = pd.concat([df_cz_all, df_mmf, df_hpef, df_m2o, df_vz,df_bsw, df_p161, df_fbtw, df_fb, df_gads, df_pers, df_excl,df_mol,df_apple,df_ctrf], ignore_index=True)
    df_dsps_api=df_dsps_api_[['date', 'network','Brand','total_spend','total_spend_campaign_currency', 'adv_clicks', 'adv_impressions','adv_installs', 'mm_installs']].groupby(['date','network','Brand']).sum().reset_index()
    df_dsps_api = df_dsps_api.rename(columns={'total_spend': 'total_spend_dsp'})
    df_dsps_api = df_dsps_api.drop('total_spend_campaign_currency', axis=1)
    df_dsps_api['Brand']= df_dsps_api['Brand'].str.replace('mindofpepe-cryptopresale','mindofpepe')
    df_fs_corr_dsp=pd.merge(df_fs_corr, df_dsps_api,  how='left', left_on=['date','Brand','network'], right_on=['date','Brand','network'])
    diff_df = df_dsps_api.merge(df_fs_corr, on=['date','Brand', 'network'], how='left', indicator=True).query('_merge == "left_only"')
    diff_df = diff_df.drop('_merge', axis=1)
    diff_df['Brand']= diff_df['Brand'].str.replace('cryptopresales','presale')
    diff_df['Brand']= diff_df['Brand'].str.replace('cryptopresale','presale')
    diff_df['Brand']=diff_df['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
    vertical_mapping = df_fs_corr.dropna(subset=['Vertical']).set_index('Brand')['Vertical'].to_dict()
    diff_df['Vertical'] = diff_df.apply(lambda row: fill_vertical(row, vertical_mapping), axis=1)
    df_fs_corr_dsp = pd.concat([df_fs_corr_dsp, diff_df], ignore_index=True)
    
    
    # 4. Additiona Sources
    # Singular
    query= f"SELECT date,network, Brand, custom_installs, custom_signups, custom_revenue  FROM `dwh-landing-v1.paid_media_network_raw.singular_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_sing = client.query(query).result().to_dataframe()
    df_sing.fillna(0, inplace=True)
    df_fs_corr_dsp_f=pd.merge(df_fs_corr_dsp, df_sing,  how='left', left_on=['date','Brand','network'], right_on=['date','Brand','network'])
    df_fs_corr_dsp_f['Install'] = df_fs_corr_dsp_f['custom_installs'].where(((df_fs_corr_dsp_f['Brand'] == 'bestwalletapp')), 0)
    df_fs_corr_dsp_f['SignUp'] = df_fs_corr_dsp_f['custom_signups'].where(((df_fs_corr_dsp_f['Brand'] == 'bestwalletapp')),0)
    df_fs_corr_dsp_f['total_revenue'] = df_fs_corr_dsp_f['custom_revenue'].where(((df_fs_corr_dsp_f['Brand'] == 'bestwalletapp')), df_fs_corr_dsp_f['total_revenue'])


    # Etherscan
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.etherscan_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_etherscan = client.query(query).result().to_dataframe()
    df_etherscan.fillna(0, inplace=True)
    df_fs_corr_dsp_fe=pd.merge(df_fs_corr_dsp_f, df_etherscan,  how='left', left_on=['date','network', 'Brand'], right_on=['date','network', 'Brand'])
    diff_df_eth = df_etherscan.merge(df_fs_corr_dsp_f, on=['date','Brand', 'network'], how='left', indicator=True).query('_merge == "left_only"')
    diff_df_eth = diff_df_eth.drop('_merge', axis=1)
    vertical_mapping = df_fs_corr_dsp_fe.dropna(subset=['Vertical']).set_index('Brand')['Vertical'].to_dict()
    diff_df_eth['Vertical'] = diff_df_eth.apply(lambda row: fill_vertical(row, vertical_mapping), axis=1)
    df_fs_corr_dsp_fe = pd.concat([df_fs_corr_dsp_fe, diff_df_eth], ignore_index=True)

    # CoinMarketCap
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.coinmarketcap_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_cmc = client.query(query).result().to_dataframe()
    df_cmc.fillna(0, inplace=True)

    # Geckoterminal
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.cgk_adzerk_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_imps_cgk = client.query(query).result().to_dataframe()
    df_imps_cgk.fillna(0, inplace=True)
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.cgk_gam_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_imps_gam = client.query(query).result().to_dataframe()
    df_imps_gam.fillna(0, inplace=True)
    df_cggam = pd.concat([df_imps_cgk, df_imps_gam], ignore_index=True)
    df_cggam['Brand']=df_cggam['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup).apply(brand_clean_polish)
    df_cggt = pd.concat([df_cggam, df_cmc], ignore_index=True)
    df_cggt=df_cggt[['date','network','Brand','impressions_cmc', 'clicks_cmc','total_spend_cmc']].groupby(['date','network','Brand']).sum().reset_index()
    df_cggt=df_cggt[df_cggt['Brand']!='geckoterminal_trending_bar']
    df_cggt['Brand']=df_cggt['Brand'].str.replace(' ', '').str.lower().apply(brand_cleanup)
    df_cggt['Brand']=df_cggt['Brand'].apply(brand_clean_polish)
    df_cggt = df_cggt[(df_cggt['date'] >= start_date) & (df_cggt['date'] <= end_date)]
    df_fs_corr_dsp_fecmc=pd.merge(df_fs_corr_dsp_fe, df_cggt,  how='left', left_on=['date','network', 'Brand'], right_on=['date','network', 'Brand'])
    diff_df_cggt = df_cggt.merge(df_fs_corr_dsp_fe, on=['date','Brand', 'network'], how='left', indicator=True).query('_merge == "left_only"')
    diff_df_cggt = diff_df_cggt.drop('_merge', axis=1)
    vertical_mapping = df_fs_corr_dsp_fecmc.dropna(subset=['Vertical']).set_index('Brand')['Vertical'].to_dict()
    diff_df_cggt['Vertical'] = diff_df_cggt.apply(lambda row: fill_vertical(row, vertical_mapping), axis=1)
    df_fs_corr_dsp_fecmc = pd.concat([df_fs_corr_dsp_fecmc, diff_df_cggt], ignore_index=True)
    
    # 5. Spend - fix budget allocation
    query= f"SELECT * FROM `dwh-landing-v1.paid_media_network_raw.fixnetworks_brand_daily`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_fs_corr_fix_dictgt = client.query(query).result().to_dataframe()
    df_fs_corr_fix_dictgt.fillna(0, inplace=True)

    # 5. Final cost
    df_w_cost=pd.merge(df_fs_corr_dsp_fecmc, df_fs_corr_fix_dictgt,  how='left', left_on=['date','network', 'Brand'], right_on=['date','network', 'Brand'])
    diff_df_gt = df_fs_corr_fix_dictgt.merge(df_fs_corr_dsp_fecmc, on=['date','Brand', 'network'], how='left', indicator=True).query('_merge == "left_only"')
    diff_df_gt = diff_df_gt.drop('_merge', axis=1)
    vertical_mapping = df_w_cost.dropna(subset=['Vertical']).set_index('Brand')['Vertical'].to_dict()
    diff_df_gt['Vertical'] = diff_df_gt.apply(lambda row: fill_vertical(row, vertical_mapping), axis=1)
    df_w_cost = pd.concat([df_w_cost, diff_df_gt], ignore_index=True)
    df_w_cost['budget_to_allocate'] = df_w_cost[['total_spend_dsp', 'fix_budget','total_spend', 'total_spend_cmc','total_spend_eth']].max(axis=1)
    df_w_cost['impressions'] = np.where((df_w_cost['network']=='Etherscan (Media)'), df_w_cost['eth_impressions'], df_w_cost['impressions'])
    df_w_cost['total_spend'] = np.where(df_w_cost['network']=='Real Time Bidding (Media)', df_w_cost['total_spend'], df_w_cost['budget_to_allocate'])
    df_w_cost['impressions'] = np.where(df_w_cost['network']=='Explorads (Media)', df_w_cost['adv_clicks'], df_w_cost['impressions'])
    df_w_cost['impressions'] = np.where(df_w_cost['network']=='Hue Ads (Media)', df_w_cost['adv_clicks'], df_w_cost['impressions'])
    df_w_cost['impressions'] = np.where(df_w_cost['network'].str.contains('Personaly|Moloco|Kayzen|Exoclick|Apple'), df_w_cost['adv_impressions'], df_w_cost['impressions'])
    df_w_cost['clicks'] = np.where(df_w_cost['network'].str.contains('Personaly|Moloco|Kayzen|Exoclick|Apple'), df_w_cost['adv_clicks'], df_w_cost['clicks'])
    df_w_cost['impressions'] = np.where(df_w_cost['network']=='DexScreener (Media)', (df_w_cost['impressions']-df_w_cost['del_impressions']), df_w_cost['impressions'])
    df_w_cost['clicks'] = np.where(df_w_cost['network']=='DexScreener (Media)', (df_w_cost['clicks']-df_w_cost['del_clicks']), df_w_cost['clicks'])
    df_w_cost['impressions'] = np.where(((df_w_cost['network']=='Meta (Media)')|(df_w_cost['network']=='Twitter (Media)')), df_w_cost['adv_impressions'], df_w_cost['impressions'])
    df_w_cost['impressions'] = np.where(df_w_cost['network']=='Google Ads', df_w_cost['adv_impressions'], df_w_cost['impressions'])
    df_w_cost['clicks'] = np.where(df_w_cost['network']=='Google Ads', df_w_cost['adv_clicks'], df_w_cost['clicks'])
    df_w_cost['impressions'] = np.where(((df_w_cost['network']=='CoinMarketCap (Media)')|(df_w_cost['network']=='Coingecko (Media)')|(df_w_cost['network']=='Geckoterminal (Media)')), df_w_cost['impressions_cmc'], df_w_cost['impressions'])
    df_w_cost['clicks'] = np.where(((df_w_cost['network']=='CoinMarketCap (Media)')|(df_w_cost['network']=='Coingecko (Media)')|(df_w_cost['network']=='Geckoterminal (Media)')), df_w_cost['clicks_cmc'], df_w_cost['clicks'])
    df_w_cost['total_spend'] = np.where(((df_w_cost['network']=='CoinMarketCap (Media)')|(df_w_cost['network']=='Coingecko (Media)')|(df_w_cost['network']=='Geckoterminal (Media)')), df_w_cost['total_spend_cmc'], df_w_cost['total_spend'])
    df_w_cost['impressions'] = np.where(((df_w_cost['network']=='DexScreener (Media)')&(df_w_cost['Brand']=='basedawgz')), 0 , df_w_cost['impressions'])
    df_w_cost['impressions'] = np.where(((df_w_cost['network']=='DexScreener (Media)')&(df_w_cost['Brand']=='playdoge')), 0 ,df_w_cost['impressions'] )
    df_w_cost['clicks'] = np.where(((df_w_cost['network']=='DexScreener (Media)')&(df_w_cost['Brand']=='basedawgz')), 0 ,df_w_cost['clicks'] )
    df_w_cost['clicks'] = np.where(((df_w_cost['network']=='DexScreener (Media)')&(df_w_cost['Brand']=='playdoge')), 0 ,df_w_cost['clicks'] )
    df_w_cost['impressions'] = np.where(((df_w_cost['network']=='CoinCarp (Media)')&(df_w_cost['Brand']=='basedawgz')), 0 ,df_w_cost['impressions'] )
    df_w_cost['impressions'] = np.where(((df_w_cost['network']=='CoinCarp (Media)')&(df_w_cost['Brand']=='playdoge')), 0 ,df_w_cost['impressions'] )
    df_w_cost['clicks'] = np.where(((df_w_cost['network']=='CoinCarp (Media)')&(df_w_cost['Brand']=='basedawgz')), 0 ,df_w_cost['clicks'] )
    df_w_cost['clicks'] = np.where(((df_w_cost['network']=='CoinCarp (Media)')&(df_w_cost['Brand']=='playdoge')), 0 ,df_w_cost['clicks'] )
    df_w_cost['total_spend'] = np.where(((df_w_cost['network']=='cardplayer.om (Media)')), 0, df_w_cost['total_spend'])
    df_w_cost['impressions']=df_w_cost['impressions'].fillna(0).astype(int)
    df_w_cost['clicks']=df_w_cost['clicks'].fillna(0).astype(int)
    df_w_cost = df_w_cost.fillna(0)
    df_w_cost['total_revenue'] = np.where(((df_w_cost['Brand'] == 'bestwalletapp')), df_w_cost['custom_revenue'], (df_w_cost['Deposit_Sales']+df_w_cost['FTD_Sales']))
    format_kpi=['Registration CPA', 'ROAS', 'ROAS (35%)','ROI (35%)']
    rename_brands = {
        'coinpoker': 'CoinPoker',
        'coinpokercasino': 'CoinPoker',
        'megadicetg': 'MegaDice',
        'instantcasino': 'Instant Casino',
        'luckyblock': 'LuckyBlock',
        'lb': 'LuckyBlock',
        'megadice': 'MegaDice',
        'tgc': 'TG Casino',
        'wsm': 'Wallstreetmemes',
        'cryptoallstars': 'Crypto All Stars',
        'flockerz': 'Flockerz',
        'memebet': 'Memebet',
        'basedawgz': 'Base Dawgz',
        'shibashootout': 'Shiba Shootout',
        'memegames': 'Meme Games',
        'playdoge': 'PlayDoge',
        'megadicepresale': 'Megadice-presale',
        'pepeunchained': 'Pepe Unchained',
        'slothana': 'Slothana',
        'goldenpanda': 'Golden Panda',    
        'bestwalletapp': 'BestWalletApp-Promo',
        'bovada': 'Bovada',
        'freedumfighters': 'Freedum Fighters',
        'catslap': 'CatSlap', 
        'thehighroller':'TheHighRoller',
        'highroller':'TheHighRoller',
        'wepe':'Wepe',
        'sambaslots': 'Samba Slots',
        'solaxy': 'Solaxy', 
        'tonaldtokendextoolsus': 'tonaldtrump',
        'nokyc':'NO KYC',
        'bestwalletapp-presale':'BestWalletApp-Presale',
        'memeindex':'Meme Index',
        'coincasino':'Coin.Casino'}

    df_w_cost['Vertical']= np.where(df_w_cost['Brand']=='thehighroller', 'Casino', df_w_cost['Vertical'])
    df_w_cost['Vertical']= np.where(df_w_cost['Brand']=='pepeunchained', 'Crypto', df_w_cost['Vertical'])
    df_w_cost=df_w_cost[(df_w_cost['network']!='Techopedia (Media)')&(df_w_cost['network']!='A-Ads (Media)')&(df_w_cost['network']!='Opera (Media)')&(df_w_cost['network']!='Cryptopolitan (Media)')]
    df_w_cost=df_w_cost[(df_w_cost['date']>=start_date)&(df_w_cost['date']<=end_date)]
    report_metrics=['impressions', 'clicks', 'total_spend','total_spend_campaign_currency','Registration','WalletConnected','Deposit', 'Deposit_Sales','total_revenue','FTD','FTD_Sales', 'Install', 'SignUp']
    report_kpi=['CPM','CPC', 'CTR', 'Registration CPA', 'FTD CPA', 'Deposit CPA', 'ROAS', 'ROAS (35%)','ROI (35%)']
    df=pivot(df_w_cost, dimensions=['date','Vertical','Brand','network'],\
                                   metrics=report_metrics,  kpi = report_kpi, sortby = ['impressions'], ascending = [False])
    vertical_mapping = df_w_cost.dropna(subset=['Vertical']).set_index('Brand')['Vertical'].to_dict()
    df['Vertical'] = df.apply(lambda row: fill_vertical(row, vertical_mapping), axis=1)
    df['brand_id']=df['Brand']
    df['Brand']= df['Brand'].replace(rename_brands)
    df['Brand']= df['Brand'].apply(lambda x: x.capitalize() if isinstance(x, str) else x)
    df=df[(df['Vertical']!=0)]
    report_metrics=['impressions', 'clicks', 'total_spend','Registration','WalletConnected','Deposit', 'Deposit_Sales', 'FTD','FTD_Sales', 'Install', 'SignUp', 'total_revenue']
    df_looker=pivot(df, dimensions=['date','network', 'brand_id','Brand', 'Vertical'],\
                                   metrics=report_metrics,  kpi = report_kpi, sortby = ['impressions'], ascending = [False])
    df_looker.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_looker['week_number'] = df_looker['date'].dt.isocalendar().week
    df_looker['week_year'] = df_looker['date'].dt.isocalendar().year
    df_looker['week_start'] = df_looker['date'] - pd.to_timedelta(df_looker['date'].dt.weekday, unit='D')
    df_looker=df_looker.fillna(0)
    df_looker =df_looker[['date', 'week_start', 'Vertical',  'brand_id', 'Brand', 'network', 'impressions', 'clicks', 'total_spend', 'Registration', 'WalletConnected', 'Deposit', 'FTD',  'total_revenue','Deposit_Sales', 'FTD_Sales',  'Install', 'SignUp']]
    df_looker["month"] = df_looker["date"].dt.strftime("%Y_%m")
    df_looker['week_start'] = df_looker['week_start'].dt.strftime('%Y-%m-%d')
    return df_looker

# Pivoted report
import pandas as pd
import numpy as np
from datetime import timedelta
from functools import reduce
from datetime import date, timedelta


# Aggregation function
def agg_df(df, group_keys, metric_columns, all_metrics, date_filter, label):
    filtered = df[date_filter]   
    if filtered.empty:
        return pd.DataFrame(columns=group_keys + ['metric', label])
    
    agg = filtered.groupby(group_keys)[metric_columns].sum().reset_index()

    # Safe KPI calculations
    impressions = agg['impressions'].replace(0, np.nan).astype(float)
    clicks = agg['clicks'].replace(0, np.nan).astype(float)
    total_spend = agg['total_spend'].astype(float)
    revenue = agg['total_revenue'].astype(float)

    agg['CPM'] = (total_spend * 1000) / impressions
    agg['CTR'] = clicks / impressions
    agg['CPC'] = total_spend / clicks
    agg['Registration_CPA'] = total_spend / agg['Registration'].replace(0, np.nan)
    agg['FTD_CPA'] = total_spend / agg['FTD'].replace(0, np.nan)
    agg['Deposit_CPA'] = total_spend / agg['Deposit'].replace(0, np.nan)
    agg['ROAS'] = revenue / total_spend
    agg['ROAS_35'] = (revenue * 0.35) / total_spend
    agg['ROI_35'] = (revenue * 0.35) - total_spend
    


    agg = agg.replace([np.inf, -np.inf], np.nan)

    long_df = agg.melt(id_vars=group_keys, value_vars=all_metrics,
                       var_name='metric', value_name=label)
    return long_df



def pivoted_brand_daily_report(client):
    yesterday = date.today() - timedelta(days=1)
    current_date = pd.Timestamp(yesterday)
    # Define date ranges
    start_of_month = pd.Timestamp(current_date).replace(day=1)
    prev_month = (start_of_month - pd.DateOffset(months=1)).replace(day=1)
    prev_month_end = start_of_month - timedelta(days=1)

    # Handle months of different lengths
    mtd_days = (current_date - start_of_month).days
    prev_month_mtd_end = prev_month + timedelta(days=min(mtd_days, (prev_month_end - prev_month).days))
    
    start_date = prev_month.strftime("%Y-%m-%d")
    end_date = yesterday.strftime("%Y-%m-%d")
     
    query = f"SELECT * FROM `dwh-landing-v1.paid_media_reports_eu_west_2.looker_daily_brand_performance`WHERE date >= '{start_date}' and date <='{end_date}'"
    df_looker_upd = client.query(query).result().to_dataframe()
    group_keys = ['date','Vertical', 'brand_id', 'Brand']
    metric_columns = ['impressions', 'clicks', 'total_spend', 'Registration',
                      'WalletConnected', 'Deposit', 'FTD', 'total_revenue',
                      'Deposit_Sales', 'FTD_Sales', 'Install', 'SignUp']
    df = df_looker_upd.groupby(group_keys)[metric_columns].sum().reset_index()
    df['date'] = pd.to_datetime(df['date'])
    # Define keys and metrics
    group_keys = ['Vertical', 'brand_id', 'Brand']
    metric_columns = ['impressions', 'clicks', 'total_spend', 'Registration',
                      'WalletConnected', 'Deposit', 'FTD', 'total_revenue',
                      'Deposit_Sales', 'FTD_Sales', 'Install', 'SignUp']
    kpi_columns = ['CPM', 'CTR', 'CPC', 'Registration_CPA', 'FTD_CPA', 'Deposit_CPA',
                   'ROAS', 'ROAS_35', 'ROI_35']
    all_metrics = metric_columns + kpi_columns
    # Get unique sorted dates
    all_dates = df['date'].sort_values().unique()

    results = []
  
    # Define filters
    yest_filter = df['date'] == current_date
    mtd_filter = (df['date'] >= start_of_month) & (df['date'] <= current_date)
    prev_mtd_filter = (df['date'] >= prev_month) & (df['date'] <= prev_month_mtd_end)
    prev_total_filter = (df['date'] >= prev_month) & (df['date'] <= prev_month_end)

    # Aggregate
    df_yest = agg_df(df,group_keys, metric_columns,all_metrics, yest_filter, 'Yesterday')
    df_mtd = agg_df(df,group_keys, metric_columns,all_metrics, mtd_filter, 'MTD')
    df_prev_mtd = agg_df(df, group_keys, metric_columns,all_metrics,prev_mtd_filter, 'Previous Month MTD')
    df_prev_total = agg_df(df,group_keys, metric_columns, all_metrics,prev_total_filter, 'Previous Month Total')

    # Base frame with valid group/metric combinations
    base_frame = pd.DataFrame([
        (v, b, br, m)
        for (v, b, br) in df[group_keys].drop_duplicates().itertuples(index=False)
        for m in all_metrics
    ], columns=group_keys + ['metric'])

    merged = base_frame.copy()

    for d in [df_yest, df_mtd, df_prev_mtd, df_prev_total]:
        if not d.empty and 'metric' in d.columns:
            merged = pd.merge(merged, d, on=group_keys + ['metric'], how='left')

    # Fill missing columns just in case
    for col in ['Yesterday', 'MTD', 'Previous Month MTD', 'Previous Month Total']:
        if col not in merged.columns:
            merged[col] = 0

    merged['Deviation'] = merged['MTD'] - merged['Previous Month MTD']
    merged['date'] = current_date
    merged.fillna(0, inplace=True)

    results.append(merged)

    # Final concatenated results
    final_df = pd.concat(results, ignore_index=True)

    # Define your custom metric order list
    metric_order_list = [
        'impressions', 'clicks', 'total_spend', 'total_revenue',
        'Registration', 'WalletConnected', 'Deposit', 'Deposit_Sales', 'FTD',
        'FTD_Sales', 'Install', 'SignUp', 'CPM', 'CTR', 'CPC',
        'Registration_CPA', 'FTD_CPA', 'Deposit_CPA', 'ROAS', 'ROAS_35',
        'ROI_35'
    ]

    # Create a mapping from metric to order index
    metric_order_map = {metric: i for i, metric in enumerate(metric_order_list, start=1)}

    # Add the column to the DataFrame
    final_df['metric_order'] = final_df['metric'].map(metric_order_map)

    # Rename metric names
    metric_rename_map = {
        'impressions': 'Impressions',
        'clicks': 'Clicks',
        'total_spend': 'Spend',
        'Registration': 'Registration',
        'WalletConnected': 'WalletConnected',
        'Deposit': 'Reccuring Deposits',
        'Deposit_Sales': 'Reccuring Deposits Sales',
        'FTD': 'FTD',
        'FTD_Sales': 'FTD Sales',
        'Install': 'Install',
        'SignUp': 'SignUp',
        'total_revenue': 'Revenue'
    }

    final_df['metric'] = final_df['metric'].replace(metric_rename_map)

    final_df=final_df[['date','Vertical', 'brand_id', 'Brand','metric_order', 'metric', 'Yesterday', 'MTD',
           'Previous Month MTD', 'Previous Month Total', 'Deviation']]

    final_df.columns=['date', 'Vertical', 'brand_id', 'Brand', 'metric_order', 'metric','Yesterday', 'MTD',
           'Previous_Month_MTD', 'Previous_Month_Total', 'Deviation']


    # Step 1: Identify brand groups where any metric is non-zero
    non_zero_brands = final_df[
        (final_df['Yesterday'] > 0) |
        (final_df['MTD'] > 0) |
        (final_df['Previous_Month_MTD'] > 0) |
        (final_df['Previous_Month_Total'] > 0)
    ][['Vertical', 'brand_id', 'Brand']].drop_duplicates()

    # Step 2: Merge to retain all rows for those brands
    final_df = final_df.merge(non_zero_brands, on=['Vertical', 'brand_id', 'Brand'], how='inner')

    final_df['date'] = pd.to_datetime(final_df['date'])
    return final_df
