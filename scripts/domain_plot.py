from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

results_dir = Path('results/domain')
result_file = 'de_word.csv'

#############
# Common for all plots
#############
linestyles = ['-', '-', '-', '--', '--', '--']
color = ['red', 'blue', '#00FF00', 'red', 'blue', '#00FF00']
plt.rcParams['font.size'] = '13'

# # ###############
# # # German Plots
# # ###############
# # figure, axes = plt.subplots(1, 3, figsize=(20, 5))
# # full_data = pd.read_csv(results_dir / result_file, index_col=0)
# # data = full_data.iloc[0:5].astype('float')
# # column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# # df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# # g = sns.pointplot(ax=axes[0], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# # g.set(xlim=(-0.1, 4.1))
# # g.get_legend().remove()
# # g.title.set_text('Word')
# # g.set_xlabel('GSD Training Size')
# # g.set_ylabel('LAS', fontsize=12)

# # data = full_data.iloc[6:11].astype('float')
# # column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# # data.columns = column_names
# # df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# # g = sns.pointplot(ax=axes[1], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# # g.set(xlim=(-0.1, 4.1))
# # g.get_legend().remove()
# # g.title.set_text('Word + Tag')
# # g.set_xlabel('GSD Training Size')
# # g.set_ylabel('LAS', fontsize=12)

# # data = full_data.iloc[12:17].astype('float')
# # column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# # data.columns = column_names
# # df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# # g = sns.pointplot(ax=axes[2], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# # g.set(xlim=(-0.1, 4.1))
# # g.get_legend().remove()
# # g.title.set_text('Word + Tag + Bert')
# # g.set_xlabel('GSD Training Size')
# # g.set_ylabel('LAS', fontsize=12)

# # plt.tight_layout()
# # plt.savefig(f'results/domain/german.png')

# ##############
# #Italian plots (Twittiro)
# ##############
# figure, axes = plt.subplots(1, 3, figsize=(20, 5))
# full_data = pd.read_csv(results_dir / result_file, index_col=0)

# data = full_data.iloc[18:23].astype('float')
# column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# g = sns.pointplot(ax=axes[0], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# g.set(xlim=(-0.1, 4.1))
# g.get_legend().remove()
# g.title.set_text('Word')
# g.set_xlabel('ISDT Training Size')
# g.set_ylabel('LAS', fontsize=12)

# data = full_data.iloc[24:29].astype('float')
# column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# data.columns = column_names
# df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# g = sns.pointplot(ax=axes[1], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# g.set(xlim=(-0.1, 4.1))
# g.get_legend().remove()
# g.title.set_text('Word+POS')
# g.set_xlabel('ISDT Training Size')
# g.set_ylabel('LAS', fontsize=12)

# data = full_data.iloc[30:35].astype('float')
# column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# data.columns = column_names
# df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# g = sns.pointplot(ax=axes[2], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# g.set(xlim=(-0.1, 4.1))
# g.get_legend().remove()
# g.title.set_text('Word+POS+BERT')
# g.set_xlabel('ISDT Training Size')
# g.set_ylabel('LAS', fontsize=12)

# plt.tight_layout()
# plt.savefig(f'results/domain/italian_tw.png')

# # ##############
# # #Italian plots (Postwita)
# # ##############
# figure, axes = plt.subplots(1, 3, figsize=(20, 5))
# full_data = pd.read_csv(results_dir / result_file, index_col=0)


# data = full_data.iloc[36:41].astype('float')
# column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# g = sns.pointplot(ax=axes[0], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# g.set(xlim=(-0.1, 4.1))
# g.get_legend().remove()
# g.title.set_text('Word')
# g.set_xlabel('ISDT Training Size')
# g.set_ylabel('LAS', fontsize=12)

# data = full_data.iloc[42:47].astype('float')
# column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# data.columns = column_names
# df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# g = sns.pointplot(ax=axes[1], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# g.set(xlim=(-0.1, 4.1))
# g.get_legend().remove()
# g.title.set_text('Word+POS')
# g.set_xlabel('ISDT Training Size')
# g.set_ylabel('LAS', fontsize=12)

# data = full_data.iloc[48:53].astype('float')
# column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# data.columns = column_names
# df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# g = sns.pointplot(ax=axes[2], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# g.set(xlim=(-0.1, 4.1))
# g.get_legend().remove()
# g.title.set_text('Word+POS+BERT')
# g.set_xlabel('ISDT Training Size')
# g.set_ylabel('LAS', fontsize=12)

# plt.tight_layout()
# plt.savefig(f'results/domain/italian_po.png')

# # #############
# # # Legend
# # #############
# # from matplotlib.lines import Line2D

# # legend_elements = [
# #     Line2D([0], [0], color='r', lw=2, linestyle='-',  label='Single Task Baseline (ISDT/GSD)'),
# #     Line2D([0], [0], color='#00FF00', lw=2, linestyle='-',  label='Shared MTL (ISDT/GSD)'),
# #     Line2D([0], [0], color='b', lw=2, linestyle='-',  label='Unshared MTL (ISDT/GSD)'),
# #     Line2D([0], [0], color='r', lw=2, linestyle='--', label='Single Task Baseline (Twitter)'),
# #     Line2D([0], [0], color='#00FF00', lw=2, linestyle='--', label='Shared MTL (Twitter)'),
# #     Line2D([0], [0], color='b', lw=2, linestyle='--', label='Unshared MTL (Twitter)'),
# # ]
# # fig, ax = plt.subplots()
# # ax.legend(handles=legend_elements, loc='center', handlelength=3)
# # ax.set_axis_off()
# # plt.tight_layout()
# # plt.savefig('results/domain/legend.png', bbox_inches='tight', pad_inches=0)


# ###############
# # German Plots with Legend
# ###############
# figure, axes = plt.subplots(2, 2, figsize=(20, 12))
# full_data = pd.read_csv(results_dir / result_file, index_col=0)
# data = full_data.iloc[0:5].astype('float')
# column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# g = sns.pointplot(ax=axes[0,0], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# g.set(xlim=(-0.1, 4.1))
# g.get_legend().remove()
# g.title.set_text('Word')
# g.set_xlabel('GSD Training Size')
# g.set_ylabel('LAS', fontsize=12)

# data = full_data.iloc[6:11].astype('float')
# column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# data.columns = column_names
# df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# g = sns.pointplot(ax=axes[0,1], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# g.set(xlim=(-0.1, 4.1))
# g.get_legend().remove()
# g.title.set_text('Word+POS')
# g.set_xlabel('GSD Training Size')
# g.set_ylabel('LAS', fontsize=12)

# data = full_data.iloc[12:17].astype('float')
# column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
# data.columns = column_names
# df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
# g = sns.pointplot(ax=axes[1,0], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
# g.set(xlim=(-0.1, 4.1))
# g.get_legend().remove()
# g.title.set_text('Word+POS+BERT')
# g.set_xlabel('GSD Training Size')
# g.set_ylabel('LAS', fontsize=12)

# legend_elements = [
#     Line2D([0], [0], color='r', lw=2, linestyle='-',  label='Single Task Baseline (ISDT/GSD)'),
#     Line2D([0], [0], color='#00FF00', lw=2, linestyle='-',  label='Shared MTL (ISDT/GSD)'),
#     Line2D([0], [0], color='b', lw=2, linestyle='-',  label='Unshared MTL (ISDT/GSD)'),
#     Line2D([0], [0], color='r', lw=2, linestyle='--', label='Single Task Baseline (Twitter)'),
#     Line2D([0], [0], color='#00FF00', lw=2, linestyle='--', label='Shared MTL (Twitter)'),
#     Line2D([0], [0], color='b', lw=2, linestyle='--', label='Unshared MTL (Twitter)'),
# ]
# plt.rcParams['font.size'] = '25'
# axes[1,1].legend(handles=legend_elements, loc='center', handlelength=3)
# axes[1,1].set_axis_off()

# plt.tight_layout()
# plt.savefig(f'results/domain/german_legend.png')


# ##############
# #Italian plots (Postwita)
# ##############
figure, axes = plt.subplots(1, 2, figsize=(15, 5))
full_data = pd.read_csv(results_dir / result_file, index_col=0)
plt.rcParams['font.size'] = '13'

data = full_data.iloc[54:58].astype('float')
column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
g = sns.pointplot(ax=axes[0], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
g.set(xlim=(-0.1, 3.1))
g.get_legend().remove()
g.title.set_text('Word+POS')
g.set_xlabel('PoSTWITA Training Size')
g.set_ylabel('LAS', fontsize=12)

data = full_data.iloc[59:64].astype('float')
column_names = ['stl', 'mtl-noshare', 'mtl-share', 'tw-stl', 'tw-mtl-noshare', 'tw-mtl-share']
data.columns = column_names
df = data.iloc[0:5].reset_index().melt('index', var_name='cols',  value_name='vals')
g = sns.pointplot(ax=axes[1], x="index", y="vals", hue='cols', data=df, kind='point', linestyles=linestyles, palette=color)
g.set(xlim=(-0.1, 4.1))
g.get_legend().remove()
g.title.set_text('Word+POS')
g.set_xlabel('ISDT Training Size')
g.set_ylabel('LAS', fontsize=12)

plt.tight_layout()
plt.savefig(f'results/domain/domain_diff.png')