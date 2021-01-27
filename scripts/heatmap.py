from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

exp_root = Path('exp')

results = exp_root / 'results.txt'
results_data = {}

with results.open('r') as results:
    lines = results.read().strip()

file_data = lines.split('\n\n')
for data_item in file_data:
    data_item = data_item.strip().split('\n')
    lang = data_item[0].strip().title()
    lang_data = []
    for row in data_item[1:]:
        row = row.strip().split(' & ')
        row = [float(i) for i in row]
        lang_data.append(row)

    results_data[lang] = np.array(lang_data)

heatmap_data = {}
for lang in results_data:
    lang_data = results_data[lang]
    lang_base = lang_data[0,:]
    lang_mean = np.mean(lang_data[1:,:], axis=0)
    heatmap_data[lang] = lang_mean - lang_base

heatmap_frame = pd.DataFrame.from_dict(heatmap_data, orient='index')
heatmap = sns.heatmap(heatmap_frame)
plt.savefig('tmp/heatmap.png')
