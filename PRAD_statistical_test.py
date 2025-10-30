import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

df_PRAD_Resistant = pd.read_csv("./pred_ctPRAD_Resistant.csv", index_col=0)
df_PRAD_Sensitive = pd.read_csv("./pred_ctPRAD_Sensitive.csv", index_col=0)

group_Resistant = pd.read_csv('metadata_ctProstate_Resistant.tsv',sep='\t',index_col=0)
group_Sensitive = pd.read_csv('metadata_ctProstate_Sensitive.tsv',sep='\t',index_col=0)

group_Resistant = pd.concat([group_Resistant,df_PRAD_Resistant],axis=1)
group_Sensitive = pd.concat([group_Sensitive,df_PRAD_Sensitive],axis=1)

group_Sensitive = group_Sensitive[group_Sensitive['batch'].str.strip() != 'Pilot']
group_Resistant = group_Resistant[group_Resistant['batch'].str.strip() != 'Pilot']

def calculate_timepoint_means(group):
    sorted_group = group.sort_values('timepoint')
    
    baseline_data = sorted_group[sorted_group['timepoint'] == 0]
    baseline_mean = baseline_data['tumor_fraction'].mean() if len(baseline_data) > 0 else 0
    
    followup_data = sorted_group[sorted_group['timepoint'] > 0]
    if len(followup_data) > 0:
        first_followup_time = followup_data['timepoint'].min()
        first_followup_data = followup_data[followup_data['timepoint'] == first_followup_time]
        first_followup_mean = first_followup_data['tumor_fraction'].mean()
    else:
        first_followup_mean = 0
    
    return pd.Series({
        'baseline_mean': baseline_mean,
        'first_followup_mean': first_followup_mean
    })
    
y_Sensitive = group_Sensitive.groupby('ID').apply(calculate_timepoint_means).dropna(how='all')
y_Resistant = group_Resistant.groupby('ID').apply(calculate_timepoint_means).dropna(how='all')


y_baseline = y_Sensitive['baseline_mean'].values
y_first_followup = y_Sensitive['first_followup_mean'].values
x_baseline = 0.25
x_first_followup = 0.75
_, p_value = stats.ttest_rel(y_baseline,y_first_followup)

plt.figure(figsize=(8, 6))

for i in range(len(y_first_followup)):
    plt.scatter([x_baseline, x_first_followup], [y_baseline[i], y_first_followup[i]], label=f'Patient {i+1}',color='#009dff')
    plt.plot([x_baseline, x_first_followup], [y_baseline[i], y_first_followup[i]], color='#009dff')

plt.ylabel('Predicted tumor fraction',fontsize=16)
plt.text(0.5, 0.9, f'$p$ = {p_value:.4f}', fontsize=15, color='r', ha='center')
plt.ylim(0,1)
plt.xlim(0.1,0.9)
plt.yticks(fontsize=14)
plt.xticks([x_baseline, x_first_followup], ['baseline', 'first_followup'],fontsize=16)
plt.title("Treatment Sensitive",fontsize=16)
plt.savefig("Treatment_Sensitive.jpg", dpi=300, bbox_inches='tight')


y_baseline = y_Resistant['baseline_mean'].values
y_first_followup = y_Resistant['first_followup_mean'].values
 
x_baseline = 0.25  
x_first_followup = 0.75
_, p_value = stats.ttest_rel(y_baseline,y_first_followup)

plt.figure(figsize=(8, 6))

for i in range(len(y_first_followup)):
    plt.scatter([x_baseline, x_first_followup], [y_baseline[i], y_first_followup[i]], label=f'Patient {i+1}',color='#009dff')
    plt.plot([x_baseline, x_first_followup], [y_baseline[i], y_first_followup[i]], color='#009dff')

plt.ylabel('Predicted tumor fraction',fontsize=16)
plt.text(0.5, 0.9, f'$p$ = {p_value:.4f}', fontsize=15, color='r', ha='center')
plt.ylim(0,1)
plt.xlim(0.1,0.9)
plt.yticks(fontsize=14)
plt.xticks([x_baseline, x_first_followup], ['baseline', 'first_followup'],fontsize=16)
plt.title("Treatment Resistant",fontsize=16)
plt.savefig("Treatment_Resistant.jpg", dpi=300, bbox_inches='tight')

