import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


#   campaigns - underpacing 

def frequency_min_up(x):
    r = 'red'
    m1 = x['f_amount'] > 0
    m2 = x['f_amount'] < 4
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['f_amount'] = np.where(((m1 & m2)), 'background-color: {}'.format(r), df1['f_amount'])
    return df1
   
def frequency_min(x):
    r = 'red'
    m1 = x['frequency_amount'] > 0
    m2 = x['frequency_amount'] < 4
    m3 = x['frequency_interval'] == 'day'
    m4 = x['frequency_interval'] == 'month'
    m5 = x['frequency_interval'] == 'week'
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['frequency_amount'] = np.where(((m1 & m2 & m3)| m4 | m5), 'background-color: {}'.format(r), df1['frequency_amount'])
    return df1


def even(data, color='red'):

    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == 'even'
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def highlight_greater(x):
    r = 'red'
    m1 = x['cap_amount'] < x['spend_to_pace']
    m2 = x['cap_amount'] > 0
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['cap_amount'] = np.where((m1&m2), 'background-color: {}'.format(r), df1['cap_amount'])

    return df1

def min_deal_bid(x):
    r = 'red'
    m1 = x['min_bid'] < x['deal_max']
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['min_bid'] = np.where(m1, 'background-color: {}'.format(r), df1['min_bid'])
    df1['deal_max'] = np.where(m1, 'background-color: {}'.format(r), df1['deal_max'])
    return df1

def reach_cpm(x):
    r = 'yellow'
    m0 = x['goal'] =='reach'
    m1 = x['goal_value'] > x['max_bid'] 
    m2 = x['goal_value'] < x['min_bid']
    m3 = x['goal_value'] == x['max_bid'] 
    m4 = x['goal_value'] == x['min_bid']
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['goal_value'] = np.where(((m1|m2|m3|m4)&m0), 'background-color: {}'.format(r), df1['goal_value'])
    return df1


def pacing_up(x):
    r = 'lightgreen'
    m1 = x['pacing'] == x['daily_spend'] 
    m2 = x['pacing']*0.85 < x['daily_spend'] 
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['pacing'] = np.where(((m1|m2)), 'background-color: {}'.format(r), df1['pacing'])
    return df1

def troubleshooting(x):
    r = 'gray'
    m1 = x['daily_spend'] == 0
    m2 = x['daily_spend'] <1
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['daily_spend'] = np.where(((m1|m2)), 'background-color: {}'.format(r), df1['daily_spend'])
    return df1


def vcr(x):
    r = 'skyblue'
    m0 = x['goal'] =='vcr'
    m1 = x['goal_value'] > 70

    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['goal_value'] = np.where(((m1)&m0), 'background-color: {}'.format(r), df1['goal_value'])
    return df1


def winrate(x):
    r = 'orange'
    m0 = x['win_rate'] < 30 
    m1 = x['win_rate'] >0 
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['max_bid'] = np.where(((m0&m1)), 'background-color: {}'.format(r), df1['max_bid'])
    df1['goal_value'] = np.where(((m0&m1)), 'background-color: {}'.format(r), df1['goal_value'])
    df1['win_rate'] = np.where(((m0&m1)), 'background-color: {}'.format(r), df1['win_rate'])
    return df1


def t1_optimized(x):
    r = 'slateblue'
    m0 = x['f_type'] == 'no-limit'
    m1 = x['f_amount'] == 0 
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['f_type'] = np.where(((m0|m1)), 'background-color: {}'.format(r), df1['f_type'])
    df1['f_amount'] = np.where(((m0|m1)), 'background-color: {}'.format(r), df1['f_amount'])

    return df1


def bid_pass(x):
    r = 'salmon'
    m0 = x['goal'] =='ctr'
    m1 = x['goal'] == 'vcr'
    m2 = x['goal'] == 'viewability_rate'
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['goal'] = np.where(((m0|m1|m2)), 'background-color: {}'.format(r), df1['goal'])
    return df1


def reach(data, color='pink'):
    
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == 'reach'
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)  

def pending(x):
    r = 'orange'
    m1 = x['Net Status'] =='PENDING'
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['Net Status'] = np.where(m1, 'background-color: {}'.format(r), df1['Net Status'])
 
    return df1


def rejected(x):
    r = 'red'
    m1 = x['Net Status'] =='REJECTED'
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['Net Status'] = np.where(m1, 'background-color: {}'.format(r), df1['Net Status'])
 
    return df1



def table_style(df,color,kpi):
        cm = sns.light_palette(color, as_cmap=True)
        format_dict = {'clickers_mu':'{0:,.0f}','total_conversions':'{0:,.0f}','measurable':'{0:,.0f}','in_view':'{0:,.0f}',\
                       'VR': '{:.2%}','audience_index_impression':'{0:,.0f}', 'Spend To Pace': '{:.1f}',  'Days Remaining': '{:.0f}', \
                       'target_accuracy':'{0:,.0%}','audience_index_clicks':'{0:,.0f}','total_spend':'{0:,.1f}', 'bid_rate': '{:,.1f}',\
                       'win_rate': '{:,.1f}','min_bid_amount_cpm': '{:,.1f}', 'max_bid_amount_cpm': '{:,.1f}','watermark_spend': '{:,.1f}',\
                       'Latest Hour of Activity':'{0:,.0f}','Scheduled End Hour':'{0:,.0f}', 'total_revenue':'{0:,.1f}','NDC':'{0:,.2f}',\
                       'LP':'{0:,.0f}', 'CPA_LP':'{0:,.2f}','CPC':'{0:,.2f}','CPA_Signup':'{0:,.2f}','CTR': '{:.2%}','VCR': '{:.0%}','spend_share': '{:.2%}', 'CPA_NDC':'{0:,.1f}',\
                       'CPA_DC':'{0:,.1f}','ROI': '{:.2f}','ROI_segment': '{:.2f}','CPM': '{0:,.1f}', 'vCPM': '{0:,.1f}','CPA': '{0:,.1f}','SSP_fee_pct': '{:.2%}','ROI_usd': '{:.2f}',\
                       'CPA_usd':'{0:,.2f}','CPC_usd':'{0:,.2f}','CPM_usd': '{0:,.1f}', 'total_spend_usd':'{0:,.1f}','total_revenue_usd':'{0:,.1f}',\
                       'media_cost_usd':'{0:,.1f}','ssp_technology_fee_usd':'{0:,.2f}', 'disc_conv_uplift':'{0:,.0%}','disc_order_uplift':'{0:,.0f}','disc_rev_uplift':'{0:,.0f}'}
        stdf = df.style.background_gradient(cmap=cm, subset=kpi).format(format_dict).hide_index()
        return stdf

    
    
def table_style_ftd(df,color,kpi):
        cm = sns.light_palette(color, as_cmap=True)
        format_dict = {'clickers_mu':'{0:,.0f}','total_conversions':'{0:,.0f}','measurable':'{0:,.0f}','in_view':'{0:,.0f}',\
                       'VR': '{:.2%}','audience_index_impression':'{0:,.0f}', 'Spend To Pace': '${:.1f}',  'Days Remaining': '{:.0f}', \
                       'target_accuracy':'{0:,.0%}','audience_index_clicks':'{0:,.0f}','total_spend':'${0:,.1f}', 'bid_rate': '{:,.1f}',\
                       'win_rate': '{:,.1f}','min_bid_amount_cpm': '${:,.1f}', 'max_bid_amount_cpm': '${:,.1f}','watermark_spend': '{:,.1f}',\
                       'Latest Hour of Activity':'{0:,.0f}','Scheduled End Hour':'{0:,.0f}', 'total_revenue':'${0:,.1f}','NDC':'{0:,.2f}',\
                       'LP':'{0:,.0f}', 'CPA_LP':'${0:,.2f}','CPC':'${0:,.2f}','CPA_Signup':'${0:,.2f}','CTR': '{:.2%}', 'CPA_NDC':'${0:,.1f}',\
                       'CPA_DC':'${0:,.1f}','ROI': '{:.2f}','CPM': '${0:,.1f}', 'vCPM': '${0:,.1f}','CPA': '${0:,.1f}','SSP_fee_pct': '{:.2%}','ROI_usd': '{:.2f}',\
                       'CPA_usd':'${0:,.2f}','CPC_usd':'${0:,.2f}','CPM_usd': '${0:,.1f}', 'total_spend_usd':'${0:,.1f}','total_revenue_usd':'${0:,.1f}',\
                       'media_cost_usd':'${0:,.1f}','ssp_technology_fee_usd':'{0:,.2f}','FTD CPA':'${0:,.2f}','Registrations CPA':'${0:,.2f}'}
        stdf = df.style.background_gradient(cmap=cm, subset=kpi).format(format_dict).hide_index()
        return stdf
    
    
def up_campaign_table(data_upc):
    format_dict = {'spend_to_pace':'{0:,.1f}','measurable':'{0:,.0f}','in_view':'{0:,.0f}','VR': '{:.2%}','spend_yesterday':'{0:,.1f}','cap_amount':'{0:,.1f}','pacing':'{0:,.2f}','daily_spend':'{0:,.2f}','min_bid':'{0:,.2f}','max_bid':'{0:,.2f}','goal_value':'{0:,.2f}','deal_min':'{0:,.1f}','deal_max':'{0:,.1f}','bid_rate': '{:,.1f}','win_rate': '{:,.1f}','Spend To Pace':'{0:,.1f}', 'latest_hour_of_delivery':'{0:,.0f}','end_hour':'{0:,.0f}', 'CPA_LP':'{0:,.2f}','CPA_Signup':'{0:,.2f}','CTR': '{:.2%}', 'CPA_NDC':'{0:,.1f}','CPA_DC':'{0:,.1f}','ROI': '{:.2f}','CPM': '{0:,.1f}', 'vCPM': '{0:,.1f}','CPC': '{0:,.1f}','CPA': '{0:,.1f}'}
    tpsps = data_upc.style.format(format_dict).hide_index().\
    apply(highlight_greater, axis=None).\
    apply(frequency_min,  axis=None).\
    apply(even,  subset=['spend_cap_type', 'frequency_type'])
    return tpsps

    
def up_campaign_table_t1(data_upc):
    format_dict = {'spend_to_pace':'{0:,.1f}','measurable':'{0:,.0f}','in_view':'{0:,.0f}','VR': '{:.2%}','spend_yesterday':'{0:,.1f}','cap_amount':'{0:,.1f}','pacing':'{0:,.2f}','daily_spend':'{0:,.2f}','min_bid':'{0:,.2f}','max_bid':'{0:,.2f}','goal_value':'{0:,.2f}','deal_min':'{0:,.1f}','deal_max':'{0:,.1f}','bid_rate': '{:,.1f}','win_rate': '{:,.1f}','Spend To Pace':'{0:,.1f}', 'latest_hour_of_delivery':'{0:,.0f}','end_hour':'{0:,.0f}', 'CPA_LP':'{0:,.2f}','CPA_Signup':'{0:,.2f}','CTR': '{:.2%}', 'CPA_NDC':'{0:,.1f}','CPA_DC':'{0:,.1f}','ROI': '{:.2f}','CPM': '{0:,.1f}', 'vCPM': '{0:,.1f}','CPC': '{0:,.1f}','CPA': '{0:,.1f}'}
    tpsps = data_upc.style.format(format_dict).hide_index().\
    apply(frequency_min,  axis=None).\
    apply(even,  subset=['spend_cap_type', 'frequency_type'])
    return tpsps

def up_strategy_table(data_ups):
    format_dict = {'spend_to_pace':'{0:,.1f}','spend_yesterday':'{0:,.1f}','measurable':'{0:,.0f}','in_view':'{0:,.0f}','VR': '{:.2%}','cap_amount':'{0:,.1f}','pacing':'{0:,.2f}','min_bid':'{0:,.2f}','max_bid':'{0:,.2f}','goal_value':'{0:,.2f}','deal_min':'{0:,.1f}','deal_max':'{0:,.1f}','bid_rate': '{:,.1f}','win_rate': '{:,.1f}','Spend To Pace':'{0:,.1f}', 'latest_hour_of_delivery':'{0:,.0f}','end_hour':'{0:,.0f}','deal_mean':'{0:,.2f}', 'daily_spend':'{0:,.2f}', 'CPA_LP':'{0:,.2f}','CPC':'{0:,.2f}','CPA_Signup':'{0:,.2f}','CTR': '{:.2%}', 'CPA_NDC':'{0:,.1f}','CPA_DC':'{0:,.1f}','ROI': '{:.2f}','CPM': '{0:,.1f}', 'vCPM': '{0:,.1f}','CPA': '{0:,.1f}'}
    strundst = data_ups.style.format(format_dict).hide_index().\
    apply(reach,  subset=['goal']).\
    apply(vcr, axis=None).\
    apply(winrate, axis=None).\
    apply(min_deal_bid, axis=None).\
    apply(pacing_up, axis=None).\
    apply(troubleshooting, axis=None).\
    apply(t1_optimized, axis=None).\
    apply(bid_pass, axis=None).\
    apply(frequency_min_up, axis=None).\
    apply(even,  subset=['pacing_type', 'f_type'])
    return strundst

def bidder_heatmap(df, width, height, color, linewdths, lineclr, filter):
    plt.figure(figsize=(width, height))
    if filter:
        sns.heatmap(df, cmap=color, mask = df > filter, cbar=None, linewidths=.5, linecolor='lightgrey')
    else: 
        sns.heatmap(df, cmap=color, linewidths=linewdths, linecolor=lineclr, cbar=True, cbar_kws={'label': 'matches'})

    plt.xlabel("targeting dimensions")
    return plt

def color_max_white(val):
    color = 'white' if pd.isna(val) else 'black'
    return 'color: %s' % color

def bidder_table(df, color):
    dfs=df.style.format("{:.0f}").background_gradient(cmap=color, axis=None).applymap(color_max_white).highlight_null(null_color='white')
    return dfs

def up_creative_table(crap):
    format_dict = {'spend_to_pace':'{0:,.1f}','spend_yesterday':'{0:,.1f}','cap_amount':'{0:,.1f}','pacing':'{0:,.2f}','daily_spend':'{0:,.2f}','min_bid':'{0:,.2f}','max_bid':'{0:,.2f}','goal_value':'{0:,.2f}','deal_min':'{0:,.1f}','deal_max':'{0:,.1f}','bid_rate': '{:,.1f}','win_rate': '{:,.1f}','Spend To Pace':'{0:,.1f}', 'latest_hour_of_delivery':'{0:,.0f}','end_hour':'{0:,.0f}', 'CPA_LP':'{0:,.2f}','CPA_Signup':'{0:,.2f}','CTR': '{:.2%}', 'CPA_NDC':'{0:,.1f}','CPA_DC':'{0:,.1f}','ROI': '{:.2f}','CPM': '{0:,.1f}', 'vCPM': '{0:,.1f}','CPC': '{0:,.1f}','CPA': '{0:,.1f}'}
    craps = crap.style.format(format_dict).hide_index().\
    apply(pending, axis=None).\
    apply(rejected, axis=None)
    return craps

def genSankey(df,cat_cols=[],value_cols='',title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['#4B8BBE','#306998','#FFE873','#FFD43B','#646464']
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp =  list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
        
    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))
    
    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum
        
    # transform df into a source-target pair
    for i in range(len(cat_cols)-1):
        if i==0:
            sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            sourceTargetDf.columns = ['source','target','count']
        else:
            tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            tempDf.columns = ['source','target','count']
            sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()
        
    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
    
    # creating the sankey diagram
    data = dict(
        type='sankey',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = labelList,
          color = colorList
        ),
        link = dict(
          source = sourceTargetDf['sourceID'],
          target = sourceTargetDf['targetID'],
          value = sourceTargetDf['count']
        )
      )
    
    layout =  dict(
        title = title,
        font = dict(
          size = 10
        )
    )
       
    fig = dict(data=[data], layout=layout)
    return fig
