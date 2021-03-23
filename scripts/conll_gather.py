from pathlib import Path
import argparse
from eval.conll18_ud_eval import load_conllu_file, evaluate as conll_evaluate
from collections import defaultdict
import pudb

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
    'vietnamese': 'vi'
}

def get_gold_file(lang, task, treebank, split):
    if lang == 'hungarian':
        treebank = treebank.title()
    else:
        treebank = treebank.upper()
    if task in ('ud', 'sud'):
        data_dir_name = f"{task.upper()}_{lang.title()}-{treebank}"
        data_file_name = f"{lang_codes[lang]}_{treebank.lower()}-{task}-{split}.conllu"
        gold_file = data_root / task / f'{task}-treebanks-v2.7' / data_dir_name / data_file_name
    elif task == 'spmrl':
        if lang == 'arabic':
            lang = lang.lower()
        else:
            lang = lang.title()
        gold_file = data_root / 'spmrl' / lang.lower() / f'{split}.{lang}.gold.conll_909'
    else:
        raise Exception("Task not supported")
    return gold_file


def evaluate(lang, conll_file, task=None, treebank=None, split=None):
    conll_stem = conll_file.stem.split('-')
    if not task:
        task = conll_stem[0]
    if not treebank:
        treebank = conll_stem[1]
    if not split:
        split = conll_stem[-1]
    gold_file = get_gold_file(lang, task, treebank, split)
    gold_ud = load_conllu_file(str(gold_file))
    pred_ud = load_conllu_file(str(conll_file))
    evaluation = conll_evaluate(gold_ud, pred_ud)
    UAS = "{:.2f}".format(100 * evaluation["UAS"].f1)
    LAS = "{:.2f}".format(100 * evaluation["LAS"].f1)
    return UAS, LAS


def gather_baseline(lang, tasks, treebanks, embed_extension, exp_type, seed, split):
    baseline_result = []
    for task, treebank in zip(tasks, treebanks):
        exp_name = f"{task}-{embed_extension}-{exp_type}"
        exp_dir = exp_root / lang / 'baseline' / exp_name / seed
        conll_file = exp_dir / f"{exp_dir.parent.name}-{split}.conllu"
        UAS, LAS = evaluate(lang, conll_file, task=task, treebank=treebank)
        baseline_result += [UAS, LAS]
    return baseline_result


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


def gather_multitask(lang, tasks, treebanks, embed_extension, exp_type, loss, mlp, seed, split, finetune):
    multitask_result = []
    for task, treebank in zip(tasks, treebanks):
        exp_name = f"{embed_extension}-{exp_type}"
        exp_subname = f"{'-'.join([loss, mlp])}"
        exp_dir = exp_root / lang / '-'.join(tasks) / exp_name / exp_subname / seed
        conll_name = f"{task}-{lang_codes[lang]}_{embed_extension}-{task}-{split}.conllu"
        conll_file = exp_dir / conll_name
        UAS, LAS = evaluate(lang, conll_file, task=task, treebank=treebank)
        multitask_result += [UAS, LAS]
    if finetune == 'standard':
        finetune_result = []
        for task, treebank in zip(tasks, treebanks):
            exp_name = f"{embed_extension}-{exp_type}"
            exp_subname = f"{'-'.join([loss, mlp])}"
            exp_dir = exp_root / lang / '-'.join(tasks) / exp_name / exp_subname / seed
            conll_name = f"whole-{task}-{task}-{lang_codes[lang]}_{embed_extension}-{task}-{split}.conllu"
            conll_file = exp_dir / conll_name
            UAS, LAS = evaluate(lang, conll_file, task=task, treebank=treebank)
            finetune_result += [UAS, LAS]
        multitask_result = choose_best(multitask_result, finetune_result)
    return multitask_result


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
        res = gather_baseline(lang, tasks, treebanks, embed_extension, exp_type, seed, split)
        baseline_results[exp_type] = res

        for loss in losses:
            for mlp in mlps:
                res = gather_multitask(lang, tasks, treebanks, embed_extension, exp_type, loss, mlp, seed, split, finetune)
                multitask_results[exp_type].append(res)
    
    print_results(baseline_results, multitask_results, debug=debug)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather experiment results")
    parser.add_argument('--lang', '-l', default='arabic', help="Language")
    parser.add_argument('--tasks',  nargs='+', default=['ud', 'sud'], help="Tasks in experiment (in correct order)")
    parser.add_argument('--treebanks', nargs='+', required='True', help="Name of treebank")
    parser.add_argument('--embed-extension', default=None, help="Name of treebank if ud/sud. If mixed tasks, then extension given to ft embedding file")
    parser.add_argument('--exp-types', '-et', nargs='+', required=True, help="Experiment Directory. For eg. tag, char, tag-char")
    parser.add_argument('--finetune', '-f', default=None, choices=['partial', 'whole', 'standard'])
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--gather-test', '-t', action='store_true')
    parser.add_argument('--baseline', '-b', action='store_true')
    parser.add_argument('--std-dev', '-s', action='store_true')
    parser.add_argument('--losses', nargs='+', default=['alternating', 'joint'])
    parser.add_argument('--mlps', nargs='+', default=['nosharemlp', 'sharemlp'])
    parser.add_argument('--seed', type=str, default='10', help="Which seeds to gather from?")
    args = parser.parse_args()
    args = vars(args)
    gather(**args)



    # if args.baseline:
    #     gather_baseline_dir(exp_root / args.lang / 'baseline' / args.exp_name,
    #                         args.seeds, baseline=args.baseline, debug=args.debug,
    #                         std_dev=args.std_dev)
    # else:
    #     gather_multitask(args.lang, args.exp_type, args.exp_name, args.losses,
    #                      args.seeds, args.debug, args.finetune, args.gather_test,
    #                      std_dev=args.std_dev)

# print(gather_baseline('arabic', ['ud'], ['padt'], 'spmrlpadt', 'tag', '10', 'dev'))
# print(gather_baseline('arabic', ['sud'], ['padt'], 'spmrlpadt', 'tag', '10', 'dev'))
# print(gather_baseline('arabic', ['ud'], ['padt'], 'spmrlpadt', 'char-ft', '10', 'dev'))
# print(gather_baseline('arabic', ['sud'], ['padt'], 'spmrlpadt', 'char-ft', '10', 'dev'))

# print(gather_multitask('arabic', ['ud', 'sud'], ['padt'], 'spmrlpadt', 'tag', 'alternating', 'nosharemlp', '10', 'dev'))
# print(gather_multitask('arabic', ['sud'], ['padt'], 'spmrlpadt', 'tag', 'alternating', 'sharemlp','10', 'dev'))
# print(evaluate('arabic', exp_root / 'arabic/ud-spmrl/spmrlpadt-char-ft/alternating-nosharemlp/10/ud-ar_padt-ud-dev.conllu', 'ud', 'padt', 'dev'))
# print(evaluate('arabic', exp_root / 'arabic/sud-spmrl/spmrlpadt-char-ft/alternating-nosharemlp/10/sud-ar_padt-sud-dev.conllu', 'sud', 'padt', 'dev'))
# print(evaluate('arabic', exp_root / 'arabic/ud-spmrl/spmrlpadt-char-ft/alternating-sharemlp/10/ud-ar_padt-ud-dev.conllu', 'ud', 'padt', 'dev'))
# print(evaluate('arabic', exp_root / 'arabic/sud-spmrl/spmrlpadt-char-ft/alternating-sharemlp/10/sud-ar_padt-sud-dev.conllu', 'sud', 'padt', 'dev'))