from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# exp_root = Path('exp')

# results = exp_root / 'results.txt'
# results_data = {}

# with results.open('r') as results:
#     lines = results.read().strip()

# file_data = lines.split('\n\n')
# for data_item in file_data:
#     data_item = data_item.strip().split('\n')
#     lang = data_item[0].strip().title()
#     lang_data = []
#     for row in data_item[1:]:
#         row = row.strip().split(' & ')
#         row = [float(i) for i in row]
#         lang_data.append(row)

#     results_data[lang] = np.array(lang_data)

# heatmap_data = {}
# for lang in results_data:
#     lang_data = results_data[lang]
#     lang_base = lang_data[0,:]
#     lang_mean = np.mean(lang_data[1:,:], axis=0)
#     heatmap_data[lang] = lang_mean - lang_base

# heatmap_frame = pd.DataFrame.from_dict(heatmap_data, orient='index')
# heatmap = sns.heatmap(heatmap_frame)
# plt.savefig('tmp/heatmap.png')

exp_root = Path('exp')
results_dir = Path('results')

# lang_codes = {
#     'arabic': 'ar',
#     'chinese': 'zh',
#     'english': 'en',
#     'finnish': 'fi',
#     'french': 'fr',
#     'german': 'de',
#     'greek': 'el',
#     'hungarian': 'hu',
#     'korean': 'ko',
#     'russian': 'ru',
#     'turkish': 'tr',
#     'vietnamese': 'vi'
# }

lang_codes = {
    'ar': 'arabic',
    'de': 'german',
    'el': 'greek',
    'en': 'english',
    'fi': 'finnish',
    'fr': 'french',
    'hu': 'hungarian',
    'ko': 'korean',
    'ru': 'russian',
    'tr': 'turkish',
    'vi': 'vietnamese',
    'zh': 'chinese'
}


features = ['none', 'none-ft', 'char', 'char-ft', 'tag', 'tag-char', 'tag-ft', 'tag-char-ft', 'tag-char-ft-bert']
names = ['word', 'FT', 'word+char', 'FT+char', 'POS', 'POS+char', 'FT+POS', 'FT+POS+Char', 'FT+POS+Char+B']

lang = 'arabic'

def read_lang_results(lang):
    lang_result_file = results_dir / f'{lang}.results'
    lang_results = {}
    with open(lang_result_file, 'r') as lang_result_file:
        lang_result_lines = lang_result_file.read()
    lang_result_lines = lang_result_lines.strip().split('\n')
    current_lang = lang_result_lines[0]
    for line in lang_result_lines:
        line = line.strip().split(' & ')
        if len(line) == 1:
            current_lang = line[0]
            lang_results[current_lang] = []
        else:
            line = [float(score) for score in line]
            lang_results[current_lang].append(line)

    combined_lang_results = None
    chosen_results = []
    for feat in features:
        chosen_results.append(np.array(lang_results[feat]))    
    
    combined_lang_results = np.concatenate(chosen_results, axis=1)
    return combined_lang_results


def process_lang_results(lang_data, target):
    if target == 'joint':
        row_indices = [1, 2, 3, 4]
        row_indices = [3, 4]
    elif target == 'alternating':
        row_indices = [1, 2]
    elif target == 'shared':
        row_indices = [2, 4]
    elif target == 'unshared':
        row_indices = [1, 3]
    else:
        row_indices = [1, 2, 3, 4]
    lang_mean = np.mean(lang_data[row_indices,:], axis=0)
    lang_base = lang_data[0,:]
    return lang_mean - lang_base


def create_dataframe(target='regular'):
    sns.set_context("paper")
    # sns.set(font_scale=0.5)
    data = {}
    for lang_code, lang_name in lang_codes.items():
        lang_data = read_lang_results(lang_name)
        data[lang_code] = process_lang_results(lang_data, target)
    # tasks = ['UD', 'SUD']
    # scores = ['UAS', 'LAS']
    # columns = pd.MultiIndex.from_product([features, tasks, scores], names=['features', 'tasks', 'scores'])
    columns = [f for f in names for j in range(4)]
    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    # grid_kws = {"height_ratios": (.05, .9), "hspace": .3}
    # f, (cbar_ax, ax) = plt.subplots(2, gridspec_kw=grid_kws)
    fig, ax = plt.subplots() # figsize=(12, 11)
    # ax = sns.heatmap(df, cbar_kws = dict(use_gridspec=False,location="top"))
    heatmap = sns.heatmap(df, xticklabels=4, vmin=-4, vmax=4, square=True, cbar_kws={"use_gridspec": False, "location": "top"})
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=8)

    plt.savefig(f'results/{target}.png')


create_dataframe('regular')
create_dataframe('alternating')
create_dataframe('shared')
create_dataframe('joint')
create_dataframe('unshared')