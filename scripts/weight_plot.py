from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


exp_files1 = ['de_1k_dev', 'de_2k_dev', 'de_4k_dev', 'de_full_dev']
# exp_files = ['de_1k_gsd-tw_sharemlp_dev', 'de_2k_gsd-tw_sharemlp_dev', 'de_4k_gsd-tw_sharemlp_dev', 'de_full_gsd-tw_sharemlp_dev']
# exp_files2 = ['it_1k_po-tw_dev']
exp_files3 = ['it_1k_isdt-tw_dev', 'it_2k_isdt-tw_dev', 'it_4k_isdt-tw_dev', 'it_full_isdt-tw_dev']
exp_files4 = ['it_1k_isdt-po_dev150', 'it_2k_isdt-po_dev150', 'it_4k_isdt-po_dev150', 'it_full_isdt-po_dev150']
exp_files5 = ['de_1k_gsd-tw_nosharemlp_dev', 'de_full_gsd-tw_nosharemlp_dev']
exp_files = exp_files1 + exp_files3 + exp_files4 
# exp_files = ['de_full_none']
# exp_files = ['it_4k_isdt-tw_dev']


weight_dir = '0_1'

for exp_file in exp_files:
    data = pd.read_csv(f'results/domain_weights/{weight_dir}/{exp_file}.csv', index_col=0)
    data = data.iloc[1:-1]
    column_names = [f"{c.split('-')[0]}" for c in data.columns]
    column_names[0] = f"{column_names[0]} ({exp_file.split('_')[1]})"
    data.columns = column_names
    # df = data[['ud-regular', 'tw-regular']].reset_index().melt('index', var_name='cols',  value_name='vals')
    df = data.reset_index().melt('index', var_name='cols',  value_name='vals')
    g = sns.catplot(x="index", y="vals", hue='cols', data=df, kind='point', aspect=1.7, legend=False)
    a = data.values.reshape(-1)
    b = np.sort(a[a.nonzero()])
    min_val = b[0]
    max_val = b[-1]
    g.set(ylim=(min_val - 0.5, max_val + 0.5))
    g.set(xlim=(-0.1, 18.5))

    t1, t2 = list(data.loc[0.5].values)
    plt.axvline(9, 0, max_val, color='crimson', ls=':')
    plt.axhline(t1, 0, 20, color='blue', ls=':')
    plt.axhline(t2, 0, 20, color='orange', ls=':')
    plt.scatter(data.iloc[:,0].argmax(), data.iloc[:,0].max(), color='#316AA3', marker='D', linewidths=5)
    plt.scatter(data.iloc[:,1].argmax(), data.iloc[:,1].max(), color='orange', marker='D', linewidths=5)

    plt.legend(fontsize='medium', title_fontsize='25', loc='best')
    plt.xlabel(f"Weights for {data.columns[0]}", fontsize=12)
    plt.ylabel(f"LAS", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/domain_weights/{weight_dir}/{exp_file}.png')