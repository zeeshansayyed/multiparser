from os import read
from pathlib import Path
import pudb
import re
import time
import datetime
import numpy as np

exp_root = Path('exp')

train_log = exp_root / 'arabic/ud-sud/padt-char/alternating-sharemlp/10/train-2021-01-15--05:45:40.log'

def test_log_completenes(train_log):
    with train_log.open('r') as train_log:
        lines = train_log.read()
    lines = lines.strip().split('\n\n')
    for line in lines[-5:]:
        if 'training completed' in line:
            return True
    return False

def read_train_log(train_log):
    with train_log.open('r') as train_log:
        lines = train_log.read()
    lines = lines.strip().split('\n\n')
    epochs = []
    t = 0
    for line in lines[-7:]:
        epoch_re = re.search(r"INFO Epoch (.*?) ", line)
        if epoch_re:
            epochs.append(epoch_re.group(1))

    time_re = re.search(r"INFO (.*?)s elapsed", line)
    if time_re:
        t = time_re.group(1)
        x = time.strptime(t.split('.')[0],'%H:%M:%S')
        t = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    
    return max(epochs), t

def test_dir_completeness(dir_path):
    for train_log in dir_path.glob('train-*.log'):
        if test_log_completenes(train_log):
            return True
    return False

exp_root = Path('exp')
languages = ['arabic', 'chinese', 'english', 'german', 'greek', 'hungarian', 'korean', 'russian', 'turkish', 'vietnamese']
treebanks = ['padt', 'gsd', 'ewt', 'gsd', 'gdt', 'szeged', 'gsd', 'gsd', 'imst', 'vtb']
exp_types = ['baseline', 'ud-sud']
exp_subtypes = ['tag', 'tag-ft', 'char', 'char-ft', 'tag-char-ft']
tasks = ['ud', 'sud']
# languages = ['arabic'] 
losses = ['alternating', 'joint']
mlps = ['nosharemlp', 'sharemlp']

alt_epochs, joint_epochs, alt_secs, joint_secs = [], [], [], []

for lang, treebank in zip(languages, treebanks):
    for exp_type in exp_types:
        expdir = exp_root / lang / exp_type
        if exp_type != 'baseline':
            for exp_subtype in exp_subtypes:
                for loss in losses:
                    for mlp in mlps:
                        subexpdir = expdir / f'{treebank}-{exp_subtype}' / f'{loss}-{mlp}' / '10'
                        if subexpdir.is_dir():
                            for train_log in subexpdir.glob('train-*.log'):
                                if test_log_completenes(train_log):
                                    epochs, t = read_train_log(train_log)
                                    epochs = float(epochs)
                                    if loss == 'alternating':
                                        alt_epochs.append(epochs)
                                        alt_secs.append(t)
                                    elif loss == 'joint':
                                        joint_epochs.append(epochs)
                                        joint_secs.append(t)

print(f"Mean Epochs (Alternatiing): {np.mean(alt_epochs)}")
print(f"Mean Time in secs (Alternatiing): {np.mean(alt_secs)}")
print(f"Mean Epochs (Joint): {np.mean(joint_epochs)}")
print(f"Mean Time in secs (Joint): {np.mean(joint_secs)}")


# Code for checking whether all experiments have completed successfully
# for lang, treebank in zip(languages, treebanks):
#     for exp_type in exp_types:
#         expdir = exp_root / lang / exp_type
#         if exp_type != 'baseline':
#             for exp_subtype in exp_subtypes:
#                 for loss in losses:
#                     for mlp in mlps:
#                         subexpdir = expdir / f'{treebank}-{exp_subtype}' / f'{loss}-{mlp}' / '10'
#                         if subexpdir.is_dir():
#                             if not test_dir_completeness(subexpdir):
#                                 print(subexpdir)

