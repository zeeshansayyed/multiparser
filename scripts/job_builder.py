from pathlib import Path
import pudb

job_dir = "jobs"

current_cmd = "python -m supar.cmds.multi_parser train -b -d 0 -c config.ini --tree "

lang = 'arabic'
lang_code = 'ar'
treebank = 'padt'
tasks = ['ud', 'sud']
# feats = ['char', 'tag', 'bert']
# fastText = False

task_data = {
    'ud': {
        'train': f'data/ud/ud-treebanks-v2.6/UD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-ud-train.conllu',
        'dev': f'data/ud/ud-treebanks-v2.6/UD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-ud-dev.conllu',
        'test': f'data/ud/ud-treebanks-v2.6/UD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-ud-test.conllu',
    },
    'sud': {
        'train': f'data/sud/sud-treebanks-v2.6/SUD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-sud-train.conllu',
        'dev': f'data/sud/sud-treebanks-v2.6/SUD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-sud-dev.conllu',
        'test': f'data/sud/sud-treebanks-v2.6/SUD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-sud-test.conllu',
    }
}


common_flags = {
    '-d': 0,
    '-c':  'config.ini',
    '--tree': None
}

train_flags = {
    '-b': None,
    '--n-embed': 300,
    '--batch-size': 20000,
    '--patience': 50,
    '--punct': None
}

eval_flags = {
    '--punct': None
}

def _append_flags(flags):
    cmd = ""
    for k, v in flags.items():
        if v != None:
            cmd += f"{k} {v} "
        else:
            cmd += f"{k} "
    return cmd

def build_starter_cmd(cmd_type, baseline=False):
    if baseline:
        cmd = "python -m supar.cmds.biaffine_dependency "
    else:
        cmd = "python -m supar.cmds.multi_parser "
    cmd += f"{cmd_type} "

    cmd += _append_flags(common_flags)
    if cmd_type == 'train':
        cmd += _append_flags(train_flags)
    elif cmd_type == 'evaluate':
        cmd += _append_flags(eval_flags)
    return cmd

def build_data_cmd(cmd_type):
    cmd = ""
    if cmd_type == 'train':
        cmd += f"--task-names {' '.join(tasks)} "
        for split in ['train', 'dev', 'test']:
            cmd += f"--{split} "
            for t in tasks:
                cmd += f"{task_data[t][split]} "
    else:
        for t in tasks:
            cmd += f"--task {t} "
            cmd += "--data "
            for split in ['dev', 'test']:
                cmd += f"{task_data[t][split] } "
    return cmd

def build_baseline_data_cmd(cmd_type, task_name, split=None):
    cmd = ""
    if cmd_type == 'train':
        for split in ['train', 'dev', 'test']:
            cmd += f"--{split} "
            cmd += f"{task_data[task_name][split]} "
    else:
        cmd += f"--data {task_data[task_name][split]} "
    return cmd

def build_cmd(cmd_type, feats, fastText, seeds=[1], baseline=False):
    for seed in seeds:
        for loss in ['joint', 'alternating']:
            for mlp in ['sharemlp', 'nosharemlp']:
                current_cmd = build_starter_cmd(cmd_type, baseline=baseline)
                if cmd_type == 'train':
                    if loss == 'joint':
                        current_cmd += '--joint-loss '
                    if mlp == 'sharemlp':
                        current_cmd += '--share-mlp '
                    current_cmd += f"-f {','.join(feats)} "
                    if fastText:
                        current_cmd += f"--embed data/embeddings/cc.{lang_code}.300.vec.{treebank} "
                    current_cmd += f"--seed {seed} "

                exp_name = f"{treebank}-{'-'.join(feats)}"
                if fastText:
                    exp_name += '-ft'
                exp_subname = f"{'-'.join([loss, mlp])}"
                if seed == 1:
                    exp_dir = f"exp/{lang}/{'-'.join(tasks)}/{exp_name}/{exp_subname}"
                else:
                    exp_dir = f"exp/{lang}/{'-'.join(tasks)}/{exp_name}/{exp_subname}/{seed}"

                current_cmd += f"-p {exp_dir} "
                current_cmd += build_data_cmd(cmd_type)
                print(current_cmd)

def build_baseline_cmd(cmd_type, feats, fastText, seeds=[1], split='dev'):
    for seed in seeds:
        for task in tasks:
            current_cmd = build_starter_cmd(cmd_type, baseline=True)
            if cmd_type == 'train':
                current_cmd += f"-f {','.join(feats)} "
                if fastText:
                    current_cmd += f"--embed data/embeddings/cc.{lang_code}.300.vec.{treebank} "
                current_cmd += f"--seed {seed} "
            
            exp_name = f"{task}-{treebank}-{'-'.join(feats)}"
            if fastText:
                exp_name += '-ft'
            if seed == 1:
                exp_dir = f"exp/{lang}/baseline/{exp_name}"
            else:
                exp_dir = f"exp/{lang}/baseline/{exp_name}/{seed}"
            
            current_cmd += f"-p {exp_dir}/{task}.model "
            current_cmd += build_baseline_data_cmd(cmd_type, task_name=task, split=split)
            if cmd_type == 'predict':
                current_cmd += f"--pred {exp_dir}/{exp_name}-{split}.conllu"
            print(current_cmd)

    

seeds = [10, 20, 30, 40, 50]

# padt.jobs -> train
# for feats in [['tag'], ['char'], ['tag', 'char']]:
#     for fastText in [True, False]:
#         print(f"# Train jobs for {feats} and fastText = {fastText}")
#         build_cmd('train', feats, fastText, seeds=seeds)
# # padt.jobs -> evaluate
# for feats in [['tag'], ['char'], ['tag', 'char']]:
#     for fastText in [True, False]:
#         print(f"Evaluate jobs for {feats} and fastText = {fastText}")
#         build_cmd('evaluate', feats, fastText, seeds=seeds)

# for feats in [['tag'], ['char'], ['tag', 'char']]:
#     for fastText in [True, False]:
#         print(f"Predict jobs for {feats} and fastText = {fastText}")
#         build_cmd('train', feats, True)

# PADT-Baseline.jobs
for feats in [['tag'], ['char'], ['tag', 'char']]:
    for fastText in [True, False]:
        print(f"# Train jobs for {feats} and fastText = {fastText}")
        build_baseline_cmd('train', feats, fastText, seeds=seeds)

for feats in [['tag'], ['char'], ['tag', 'char']]:
    for fastText in [True, False]:
        print(f"Evaluate jobs for {feats} and fastText = {fastText}")
        build_baseline_cmd('evaluate', feats, fastText, seeds=seeds)