from pathlib import Path
import pudb
import argparse

# job_dir = "jobs"

# current_cmd = "python -m supar.cmds.multi_parser train -b -d 0 -c config.ini --tree "

# lang = 'arabic'
# lang_code = 'ar'
# treebank = 'padt'
# res_treebank = 'padtspmrl'
# tasks = ['ud', 'sud', 'spmrl']
# feats = ['char', 'tag', 'bert']
# fastText = False

# task_data = {
#     'ud': {
#         'train': f'data/ud/ud-treebanks-v2.6/UD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-ud-train.conllu',
#         'dev': f'data/ud/ud-treebanks-v2.6/UD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-ud-dev.conllu',
#         'test': f'data/ud/ud-treebanks-v2.6/UD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-ud-test.conllu',
#     },
#     'sud': {
#         'train': f'data/sud/sud-treebanks-v2.6/SUD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-sud-train.conllu',
#         'dev': f'data/sud/sud-treebanks-v2.6/SUD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-sud-dev.conllu',
#         'test': f'data/sud/sud-treebanks-v2.6/SUD_{lang.title()}-{treebank.upper()}/{lang_code}_{treebank}-sud-test.conllu',
#     },
#     'spmrl': {
#         'train': f'data/spmrl/{lang}/train.{lang}.gold.conll',
#         'dev': f'data/spmrl/{lang}/dev.{lang}.gold.conll',
#         'test': f'data/spmrl/{lang}/test.{lang}.gold.conll',
#     }
# }


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

def build_data_cmd(cmd_type, tasks, task_data):
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

def build_baseline_data_cmd(cmd_type, task_name, task_data, split=None):
    cmd = ""
    if cmd_type == 'train':
        for split in ['train', 'dev', 'test']:
            cmd += f"--{split} "
            cmd += f"{task_data[task_name][split]} "
    else:
        cmd += f"--data {task_data[task_name][split]} "
    return cmd

def build_cmd(cmd_type, lang_name, lang_code, tasks, task_data, feats, fastText, embed_extension, seeds, losses, mlps):
    for seed in seeds:
        for loss in losses:
            for mlp in mlps:
                current_cmd = build_starter_cmd(cmd_type, baseline=False)
                if cmd_type == 'train':
                    if loss == 'joint':
                        current_cmd += '--joint-loss '
                    if mlp == 'sharemlp':
                        current_cmd += '--share-mlp '
                    current_cmd += f"-f {','.join(feats)} "
                    if fastText:
                        current_cmd += f"--embed data/embeddings/cc.{lang_code}.300.vec.{embed_extension} "
                    if 'bert' in feats:
                        current_cmd += f"--bert bert-base-multilingual-cased "
                    current_cmd += f"--seed {seed} "

                exp_name = f"{embed_extension}-{'-'.join(feats)}"
                if fastText:
                    exp_name += '-ft'
                exp_subname = f"{'-'.join([loss, mlp])}"
                if seed == 1:
                    exp_dir = f"exp/{lang_name}/{'-'.join(tasks)}/{exp_name}/{exp_subname}"
                else:
                    exp_dir = f"exp/{lang_name}/{'-'.join(tasks)}/{exp_name}/{exp_subname}/{seed}"

                current_cmd += f"-p {exp_dir} "
                current_cmd += build_data_cmd(cmd_type, tasks, task_data)
                print(current_cmd)

def build_baseline_cmd(cmd_type, lang_name, lang_code, tasks, task_data, feats, fastText, embed_extension, seeds=[1], split='dev'):
    for seed in seeds:
        for task in tasks:
            current_cmd = build_starter_cmd(cmd_type, baseline=True)
            if cmd_type == 'train':
                current_cmd += f"-f {','.join(feats)} "
                if fastText:
                    current_cmd += f"--embed data/embeddings/cc.{lang_code}.300.vec.{embed_extension} "
                if 'bert' in feats:
                    current_cmd += f"--bert bert-base-multilingual-cased "
                current_cmd += f"--seed {seed} "

            exp_name = f"{task}-{embed_extension}-{'-'.join(feats)}"
            if fastText:
                exp_name += '-ft'
            if seed == 1:
                exp_dir = f"exp/{lang_name}/baseline/{exp_name}"
            else:
                exp_dir = f"exp/{lang_name}/baseline/{exp_name}/{seed}"

            current_cmd += f"-p {exp_dir}/{task}.model "
            current_cmd += build_baseline_data_cmd(cmd_type, task, task_data, split=split)
            if cmd_type == 'predict':
                current_cmd += f"--pred {exp_dir}/{exp_name}-{split}.conllu"
            print(current_cmd)



# seeds = [10, 20, 30, 40, 50]

# # padt.jobs -> train
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
# for feats in [['tag'], ['char'], ['tag', 'char']]:
#     for fastText in [True, False]:
#         print(f"# Train jobs for {feats} and fastText = {fastText}")
#         build_baseline_cmd('train', feats, fastText, seeds=seeds)

# for feats in [['tag'], ['char'], ['tag', 'char']]:
#     for fastText in [True, False]:
#         print(f"Evaluate jobs for {feats} and fastText = {fastText}")
#         build_baseline_cmd('evaluate', feats, fastText, seeds=seeds)


def build(lang_name, lang_code, treebank, tasks, features, use_fasttext, seeds,
          additional_tasks, additional_train, additional_dev, additional_test,
          embed_extension, losses, mlps, jobs, verbose):
    if lang_name == 'hungarian':
        treebank = treebank.title()
    else:
        treebank = treebank.upper()

    task_data = {
        'ud': {
            'train': f'data/ud/ud-treebanks-v2.7/UD_{lang_name.title()}-{treebank}/{lang_code}_{treebank.lower()}-ud-train.conllu',
            'dev': f'data/ud/ud-treebanks-v2.7/UD_{lang_name.title()}-{treebank}/{lang_code}_{treebank.lower()}-ud-dev.conllu',
            'test': f'data/ud/ud-treebanks-v2.7/UD_{lang_name.title()}-{treebank}/{lang_code}_{treebank.lower()}-ud-test.conllu',
        },
        'sud': {
            'train': f'data/sud/sud-treebanks-v2.7/SUD_{lang_name.title()}-{treebank}/{lang_code}_{treebank.lower()}-sud-train.conllu',
            'dev': f'data/sud/sud-treebanks-v2.7/SUD_{lang_name.title()}-{treebank}/{lang_code}_{treebank.lower()}-sud-dev.conllu',
            'test': f'data/sud/sud-treebanks-v2.7/SUD_{lang_name.title()}-{treebank}/{lang_code}_{treebank.lower()}-sud-test.conllu',
        }
    }

    if use_fasttext == 'both':
        all_fasttext = [False, True]
    elif use_fasttext == 'true':
        all_fasttext = [True]
    else:
        all_fasttext = [False]

    if additional_tasks:
        assert len(additional_tasks) == len(additional_train) == len(additional_dev) == len(additional_test)
        for task, train, dev, test in zip(additional_tasks, additional_train, additional_dev, additional_test):
            task_data[task] = {'train': train, 'dev': dev, 'test': test}
            tasks.append(task)

    features = [f.split('-') for f in features]
    if not embed_extension:
        embed_extension = treebank.lower()

    # Multitask Train
    def mt():
        if 'mt' in jobs:
            for feats in features:
                for fastText in all_fasttext:
                    if verbose:
                        print(f"\n# Multitask Train jobs for {feats} and fastText = {fastText} and tasks {'-'.join(tasks)} and lang={lang_name}")
                    build_cmd('train', lang_name, lang_code, tasks, task_data, feats, fastText, embed_extension, seeds, losses, mlps)

    # Baseline Train
    def bt():
        if 'bt' in jobs:
            for feats in features:
                for fastText in all_fasttext:
                    if verbose:
                        print(f"\n# Baseline Train jobs for {feats} and fastText = {fastText} and tasks {'-'.join(tasks)} and lang={lang_name}")
                    build_baseline_cmd('train', lang_name, lang_code, tasks, task_data, feats, fastText, embed_extension, seeds)

    # Multitask Evaluate
    def me():
        if 'me' in jobs:
            for feats in features:
                for fastText in all_fasttext:
                    if verbose:
                        print(f"\nMultitask Evaluate jobs for {feats} and fastText = {fastText} and tasks {'-'.join(tasks)} and lang={lang_name}")
                    build_cmd('evaluate', lang_name, lang_code, tasks, task_data, feats, fastText, embed_extension, seeds, losses, mlps)

    #Multitask predict
    def mp():
        if 'mp' in jobs:
            for feats in features:
                for fastText in all_fasttext:
                    if verbose:
                        print(f"\nMultitask Predict jobs for {feats} and fastText = {fastText} and tasks {'-'.join(tasks)} and lang={lang_name}")
                    build_cmd('predict', lang_name, lang_code, tasks, task_data, feats, fastText, embed_extension, seeds, losses, mlps)

    # Baseline Evaluate
    def be():
        if 'be' in jobs:
            for feats in features:
                for fastText in all_fasttext:
                    if verbose:
                        print(f"\nBaseline Evaluate jobs for {feats} and fastText = {fastText} and tasks {'-'.join(tasks)} and lang={lang_name}")
                    build_baseline_cmd('evaluate', lang_name, lang_code, tasks, task_data, feats, fastText, embed_extension, seeds)

    # Baseline Predict
    def bp():
        if 'bp' in jobs:
            for feats in features:
                for fastText in all_fasttext:
                    if verbose:
                        print(f"\nBaseline Predict jobs for {feats} and fastText = {fastText} and tasks {'-'.join(tasks)} and lang={lang_name}")
                    build_baseline_cmd('predict', lang_name, lang_code, tasks, task_data, feats, fastText, embed_extension, seeds)

    job_commands = {
        'mt': mt, 'bt': bt, 'me': me, 'mp': mp, 'be': be, 'bp': bp
    }
    for job in jobs:
        job_commands[job]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shrink Fasttext embeddings")
    parser.add_argument('--lang-name', '-ln', required=True, help="Language Name")
    parser.add_argument('--lang-code', '-lc', required=True,  help="Language Code")
    parser.add_argument('--treebank', '-t', required=True, help="Name of the treebank")
    parser.add_argument('--tasks', nargs='+', default=['ud', 'sud'])
    parser.add_argument('--features', '-f', nargs='+', help="Random seeds to be used")
    parser.add_argument('--use-fasttext', '-ft', default='both', choices=['true', 'false', 'both'], help="Should use fastText embeddings?")
    parser.add_argument('--seeds', '-s', type=int, nargs='+', help="Random seeds to be used")
    parser.add_argument('--losses', nargs='+', default=['joint', 'alternating'])
    parser.add_argument('--mlps', nargs='+', default=['sharemlp', 'nosharemlp'])
    parser.add_argument('--additional-tasks', '-d', nargs='+', help="Additional datafiles to consider")
    parser.add_argument('--additional-train', nargs='+', help='paths to additional train files')
    parser.add_argument('--additional-dev', nargs='+', help='paths to additional dev files')
    parser.add_argument('--additional-test', nargs='+', help='paths to additional test files')
    parser.add_argument('--embed-extension', default=None, help="Extension of FastText embeddings")
    parser.add_argument('--jobs', nargs='+', default=['mt', 'me', 'mp', 'bt', 'be', 'bp'], help="Which jobs to build")
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    args = parser.parse_args()
    args = vars(args)
    build(**args)















# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Gather experiment results")
#     parser.add_argument('--lang', '-l', default='arabic', help="Language")
#     parser.add_argument('--lang-code', '-lc', default='ar', help="Language Code")
#     parser.add_argument('--treebank', '-t', default='padt')
#     parser.add_argument('--tasks', nargs='+', default=['ud', 'sud'])
#     parser.add_argument('')

#     args = parser.parse_args()
