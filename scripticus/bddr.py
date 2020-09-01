import json,urllib.request
from pandas import json_normalize
import time
import pandas as pd
import numpy as np

def bidder_strategy_target(strategy_ids, bidder_number, bidder_location, sec):
    df_final = pd.DataFrame()
     
    for  count, strategy_id in enumerate(strategy_ids,1):
        if count % 10 == 0: 
            bidder_number += 1
                         
        url_bidder_sts = 'http://{}-bidder-x{}.mediamath.com:8081/strategy_target?op=create&strategy={}&max_recs=&max_runtime=&stop=&fmt=json'.format(bidder_location,bidder_number, strategy_id)
        data_start = urllib.request.urlopen(url_bidder_sts).read()
        output_start = json.loads(data_start)
      
             
       
        try:   
            test_id = output_start['test_id']
            time.sleep(sec)
            url_bidder_stf = 'http://{}-bidder-x{}.mediamath.com:8081/strategy_target?op=report&fmt=json&test_id={}'.format(bidder_location,bidder_number, test_id)
            data_final = urllib.request.urlopen(url_bidder_stf).read()
            output_final = json.loads(data_final)            
            df_target = json_normalize(output_final['result'],record_path=['by_target'])
            df_overall = json_normalize(output_final['result']['overall'])     
        
     
        
        except KeyError:

            df_target = json_normalize(output_start['result'],record_path=['by_target'])
            df_overall = json_normalize(output_start['result']['overall'])

        df_target['include'] = df_target['include'].apply({True:'I', False:'E'}.get)
        df_target['targeting_dimension'] =  df_target['dimension'] + ' (' + df_target['include'] + ')' 
        df_overall['targeting_dimension'] = 'Overall'       
        df = pd.concat([df_target, df_overall], axis=0)
        df = df[['targeting_dimension','matched']]
        df = df.set_index('targeting_dimension')
        df = df.T
        df['strategy_id'] = strategy_id
        df = df.loc[:,~df.columns.duplicated()]
                    
        if len(df_final) == 0:
            df_final = df
        else:
            df_final = pd.concat([df_final, df], sort=False,ignore_index=True)
        df_final = df_final[ ['Overall'] + [ col for col in df_final.columns if col != 'Overall' ] ]
        df_final = df_final[ ['strategy_id'] + [ col for col in df_final.columns if col != 'strategy_id' ] ]
        df_final.set_index('strategy_id', inplace=True)
        # df_final = df_final.apply(pd.to_numeric, errors='ignore')
 
    return df_final







