
### Scripticus

A package for rapid reporting, campaign monitoring and optimization automation within the T1 MediaMath ecosystem. 

This package uses the following publicly open APIs as data sources: [MediaMath](https://apidocs.mediamath.com/), [Qubole](https://pypi.org/project/qds_sdk/) and [Looker](https://developers.looker.com/api/overview/).

### Features

- Access to all T1 platform reporting data and log-level marketing data from Qubole
- Get bidding insights
- Find under- and outpacing- strategies for further troubleshooting
- Get recommendations for campaign optimization
- Optimize campaign budgets, creatives, supply paths and more
- Visualize all results via heatmaps, bar and line charts, etc.
- Create and send a mail with findings and recommendations formatted as html or js - presentation file


### Usage example

```
from scripticus import looker_api, bddr, t1_api as mm, beautifulization as bfz, mailicus as ms

# Get recommendations for underpacing campaigns
df_upc=t1_report.underpacing_campaigns(up_campaigns_ids, up_campaigns_to_check)  # returns a table with underpacing campaigns, their KPIs, and the most important settings. 
bfz.up_campaign_table(df_upc) # returns a formatted table in which the settings to be changed are marked in correspomnding color.

# Get recommendations for underpacing startegies
df_ups=t1_report.underpacing_strategies(organization_id, campaign_active_up_ids)  # returns a table with underpacing startegies, their KPIs, and the most important settings. 
bfz.up_strategy_table(df_ups)  # returns a formatted table in which the settings to be changed are marked in correspomnding color.
```
