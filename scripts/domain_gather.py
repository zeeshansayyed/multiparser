from pathlib import Path
import argparse
from eval.conll18_ud_eval import load_conllu_file, evaluate as conll_evaluate
from collections import defaultdict
import pudb
import numpy as np
import pandas as pd

exp_root = Path('exp')
data_root = Path('data')

lang_codes = {
    'arabic': 'ar',
    'chinese': 'zh',
    'english': 'en',
    'finnish': 'fi',
    'french': 'fr',
    'german': 'de',
    'greek': 'el',
    'hungarian': 'hu',
    'korean': 'ko',
    'russian': 'ru',
    'turkish': 'tr',
    'vietnamese': 'vi',
    'italian': 'it',
}


def get_gold_file(lang, treebank, split):
    if lang == 'german' and treebank == 'tweeDe':
        gold_file = data_root / 'tweeDe' / f'tweeDe_{split}.conllu'
    else:
        if treebank.lower() == 'postwita':
            treebank = 'PoSTWITA'
            split += '150'
        else:
            treebank = treebank.upper()
        task = 'ud'
        data_dir_name = f"{task.upper()}_{lang.title()}-{treebank}"
        data_file_name = f"{lang_codes[lang]}_{treebank.lower()}-{task}-{split}.conllu"
        gold_file = data_root / task / f'{task}-treebanks-v2.7' / data_dir_name / data_file_name

    return gold_file


def evaluate(lang, conll_file, treebank, split):
    gold_file = get_gold_file(lang, treebank, split)
    # print(f"Gold file is {gold_file}")
    gold_ud = load_conllu_file(str(gold_file))
    pred_ud = load_conllu_file(str(conll_file))
    evaluation = conll_evaluate(gold_ud, pred_ud)
    UAS = "{:.2f}".format(100 * evaluation["UAS"].f1)
    LAS = "{:.2f}".format(100 * evaluation["LAS"].f1)
    return UAS, LAS


def choose_best(list_a, list_b):
    print(list_a)
    print(list_b)
    list_c = []
    for i, j in zip(list_a, list_b):
        if float(i) > float(j):
            list_c.append(i)
        else:
            list_c.append(j)
    return list_c


def gather_multitask(lang, tasks, treebanks, embed_extension, exp_type, loss, mlp,
                     seed, split, finetune):
    multitask_result = []
    for task, treebank in zip(tasks, treebanks):
        exp_name = f"{embed_extension}-{exp_type}"
        exp_subname = f"{'-'.join([loss, mlp])}"
        exp_dir = exp_root / lang / '-'.join(
            tasks) / exp_name / exp_subname / seed
        conll_name = f"{task}-{lang_codes[lang]}_{embed_extension}-{task}-{split}.conllu"
        conll_file = exp_dir / conll_name
        UAS, LAS = evaluate(lang, conll_file, task=task, treebank=treebank)
        multitask_result += [UAS, LAS]
    if finetune == 'standard':
        finetune_result = []
        for task, treebank in zip(tasks, treebanks):
            exp_name = f"{embed_extension}-{exp_type}"
            exp_subname = f"{'-'.join([loss, mlp])}"
            exp_dir = exp_root / lang / '-'.join(
                tasks) / exp_name / exp_subname / seed
            conll_name = f"whole-{task}-{task}-{lang_codes[lang]}_{embed_extension}-{task}-{split}.conllu"
            conll_file = exp_dir / conll_name
            UAS, LAS = evaluate(lang, conll_file, task=task, treebank=treebank)
            finetune_result += [UAS, LAS]
        multitask_result = choose_best(multitask_result, finetune_result)
    return multitask_result


def print_results(baseline_results, multitask_results, debug=False, sep=' & '):
    if debug:
        for exp_type in baseline_results:
            print(exp_type)
            print(sep.join(baseline_results[exp_type]))
            for multi_res in multitask_results[exp_type]:
                print(sep.join(multi_res))

    print(f"Order: {' -> '.join(list(baseline_results.keys()))}")
    base_out = []
    for base_res in baseline_results.values():
        base_out += base_res
    print(sep.join(base_out))

    for i in range(4):
        multi_out = []
        for multi_res in multitask_results.values():
            multi_out += multi_res[i]
        print(sep.join(multi_out))


def gather(lang, tasks, treebanks, embed_extension, exp_types, losses, mlps, seed,
           debug=False, finetune=False, gather_test=False, std_dev=False,
           baseline=False):
    if gather_test:
        split = 'test'
    else:
        split = 'dev'

    if not embed_extension:
        embed_extension = treebanks[0]

    if len(treebanks) == 1 and len(tasks) > 1:
        treebanks = treebanks * len(tasks)

    assert len(tasks) == len(treebanks)

    baseline_results = {}
    multitask_results = defaultdict(list)

    for exp_type in exp_types:
        # res = gather_baseline(lang, tasks, treebanks, embed_extension, exp_type,
        #                       seed, split)
        # baseline_results[exp_type] = res

        for loss in losses:
            for mlp in mlps:
                res = gather_multitask(lang, tasks, treebanks, embed_extension,
                                       exp_type, loss, mlp, seed, split, finetune)
                multitask_results[exp_type].append(res)

    print_results(baseline_results, multitask_results, debug=debug)


tasks = ['gsd', 'tw']
treebanks = ['gsd', 'tweeDe']
lang = 'german'

# tasks = ['isdt', 'po']
# treebanks = ['isdt', 'postwita']
# lang = 'italian'

# tasks = ['isdt', 'tw']
# treebanks = ['isdt', 'twittiro']
# lang = 'italian'

# tasks = ['po', 'tw']
# treebanks = ['postwita', 'twittiro']
# lang = 'italian'

og_split = 'dev'
feature = 'none'
include_finetune = False
regular_results = []
partial_results = []
whole_results = []
mtl_setting = 'sharemlp'
# exp_sizes = ['full']
exp_sizes = ['1k', '2k', '4k', 'full']
ratios = [f"{0.05+i*0.05:.2f}-{1-(0.05+i*0.05):.2f}" for i in range(19)]
# ratios = ['0.85-0.15']
ratios = [f"{0.1+i*0.1:.1f}-{2-(0.1+i*0.1):.1f}" for i in range(19)]
for exp_size in exp_sizes:
    results = defaultdict(lambda: defaultdict(list))
    for task, treebank in zip(tasks, treebanks):
        if treebank == 'postwita':
            split = og_split + '150'
        else:
            split = og_split
        for ratio in ratios:
            # exp_dir = exp_root / 'domain' / lang / 'gsd-tw' / 'tag' / exp_size / f'sharemlp-{ratio}' # Only for old german
            exp_dir = exp_root / 'domain' / 'gsd-tweeDe' / 'tag' / exp_size / f'sharemlp-{ratio}' # Weird German 1-1
            # exp_dir = exp_root / 'domain' / lang / f"{'-'.join(tasks)}" / 'tag' / exp_size / f'sharemlp-{ratio}' # Regular
            # exp_dir = exp_root / 'domain' / lang / f"{'-'.join(tasks)}" / feature / exp_size / mtl_setting / ratio # New
            if treebank == 'tweeDe':
                regular_conll_name = f"{task}-{treebank}_{split}.conllu"
            else:
                regular_conll_name = f"{task}-{lang_codes[lang]}_{treebank}-ud-{split}.conllu"

            regular_res, partial_res, whole_res = [], [], []

            for seed in ['10', '20', '30']:
                curr_exp_dir = exp_dir / seed
                regular_conll_file = curr_exp_dir / regular_conll_name
                partial_conll_file = curr_exp_dir / f"partial-{task}-{regular_conll_name}"
                whole_conll_file = curr_exp_dir / f"whole-{task}-{regular_conll_name}"
                # try:
                UAS, LAS = evaluate(lang, regular_conll_file, treebank, og_split)
                # except:
                #     print(f"Error in {regular_conll_file}")
                #     UAS, LAS = 0.0, 0.0
                regular_res += [float(LAS)]
                if include_finetune:
                    UAS, LAS = evaluate(lang, partial_conll_file, treebank, split)
                    partial_res += [float(LAS)]
                    UAS, LAS = evaluate(lang, whole_conll_file, treebank, split)
                    whole_res += [float(LAS)]

            results['regular'][task].append(regular_res)
            if include_finetune:
                results['partial'][task].append(partial_res)
                results['whole'][task].append(whole_res)

    data = pd.DataFrame(index=[r.split('-')[0] for r in ratios])
    for finetune in results:
        for task in results[finetune]:
            data[f'{task}-{finetune}'] = np.array(results[finetune][task]).mean(axis=1)

    # data.to_csv(f"results/domain/{lang_codes[lang]}_{exp_size}_{'-'.join(tasks)}_{split}.csv")
    data.to_csv(f"results/domain_weights/0_2/{lang_codes[lang]}_{exp_size}_{'-'.join(tasks)}_{mtl_setting}_{split}.csv")





















# for seed in ['10', '20']:
#     regular_res, partial_res, whole_res = [], [], []
#     for task, treebank in zip(tasks, treebanks):
#         exp_dir = exp_root / 'domain' / 'german' / 'gsdt-tw' / 'tag' / '1k' / 'sharemlp-0.05-0.95' / seed
#         if treebank == 'tweeDe':
#             regular_conll_name = f"{task}-{treebank}_{split}.conllu"
#         else:
#             regular_conll_name = f"{task}-{lang_codes[lang]}_{treebank}-{task}-{split}.conllu"
#         regular_conll_file = exp_dir / regular_conll_name
#         partial_conll_file = exp_dir / f"partial-{task}-{regular_conll_name}"
#         whole_conll_file = exp_dir / f"whole-{task}-{regular_conll_name}"
#         # print(conll_name)
#         UAS, LAS = evaluate(lang, regular_conll_file, treebank, split)
#         regular_res += [float(LAS)] # [float(UAS), float(LAS)]
#         if finetune:
#             UAS, LAS = evaluate(lang, partial_conll_file, treebank, split)
#             partial_res += [float(LAS)] # [float(UAS), float(LAS)]
#             UAS, LAS = evaluate(lang, whole_conll_file, treebank, split)
#             whole_res += [float(LAS)] # [float(UAS), float(LAS)]
        
#     regular_results.append(regular_res)
#     whole_results.append(whole_res)
#     partial_results.append(partial_res)
# print(np.array(regular_results).mean(axis=0))
# print(np.array(partial_results).mean(axis=0))
# print(np.array(whole_results).mean(axis=0))