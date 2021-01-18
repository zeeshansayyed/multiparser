from pathlib import Path
import pudb
from collections import defaultdict
import numpy as np
import re
import argparse

exp_root = Path('exp')

def process_eval_line(line, baseline=False):

    if 'dev' in line:
        data_type = 'dev'
    else:
        data_type = 'test'

    # line = line.split('\t')
    # print(line)
    if baseline:
        model = 'base'
    else:
        model = re.search(r"Model=(.*?)\.model", line).group(1)
        model = '-'.join(model.split('-')[-2:])
    # uas_regex = re.search(r"UAS:(.*?)%", line)
    # UAS, LAS = None, None
    # if uas_regex:
    #     UAS = uas_regex.group(1)
    # las_regex = re.search(r"LAS:(.*?)%", line)
    # if las_regex:
    #     LAS = uas_regex.group(1)
    UAS = re.search(r"UAS:(.*?)%", line).group(1)
    LAS = re.search(r"LAS:(.*?)%", line).group(1)
    return data_type, model, (UAS, LAS)

def read_results(exp, baseline=False):
    eval_file = exp / 'evaluate.log'
    with eval_file.open('r') as eval_file:
        lines = eval_file.read()
    lines = lines.split('\n\n')[1].strip().split('\n')
    # print(len(lines))
    results = {'dev': {}, 'test': {}}
    if baseline:
        line = lines[-2]
        UAS = re.search(r"UAS:(.*?)%", line).group(1)
        LAS = re.search(r"LAS:(.*?)%", line).group(1)
        results = None, None, (UAS, LAS)
    else:
        for line in lines:
            data_type, model, metric = process_eval_line(line, baseline=baseline)
            # print(data_type, model, metric)
            results[data_type][model] = metric
    return results

def write_result(results, tasks=['ud', 'sud'], splits=['dev'], finetune=False, file=None, debug=False):
    """Write the results of an exp_dir in either one line (regular) or multiple lines (finetuning)"""
    ret_res = []
    if isinstance(splits, str):
        splits = list(splits)
    for split in splits:
        curr_res = results[split]
        line = []
        linetitle = f"regular ({split}):"
        if debug:
            print(f"{linetitle:<20}", end='', file=file)
        for task in tasks:
            line += [score.strip() for score in curr_res[task]]
        if debug:
            print(' & '.join(line), file=file)
        ret_res.append(line)
        if finetune:
            for from_task in tasks + ['total']:
                line = []
                if debug:
                    print(f"{from_task + ':':<20}", end='', file=file)
                for to_task in tasks:
                    line += [score.strip() for score in curr_res[f"{from_task}-{to_task}"]]
                if debug:
                    print(' & '.join(line), file=file)
                ret_res.append(line)
    return ret_res

def write_baseline_result(results, debug=False):
    _, _, results = results
    linetitle = f"baseline:"
    if debug:
        print(f"{linetitle:<20}", end='')
    line = [score.strip() for score in results]
    if debug:
        print(' & '.join(line))
    return line



def write_all_results(base_dir, tasks=['ud', 'sud'], finetune=False, gather_test=False):
    """Recursively calls write results on all subdirectories. Creates a results.txt for each 
        exp_dir
    """
    splits = ['dev']
    if gather_test:
        splits.append('test')

    for eval_file in base_dir.glob('**/evaluate.log'):
        sub_dir = eval_file.parent
        out_file = sub_dir / 'results.txt'
        results = read_results(sub_dir)
        with out_file.open('w') as out_file:
            write_result(results, file=out_file, finetune=finetune, splits=splits, tasks=tasks)



def collate_multiple_runs(lang, exp_type, exp_name, exp_sub_name):
    exp_dir = exp_root / lang / exp_type / exp_name / exp_sub_name
    write_all_results(exp_dir, gather_test=True)
    results = exp_dir / 'results.txt'
    with results.open('w') as results:
        for sub_dir in exp_dir.glob('*'):
            if sub_dir.is_dir():
                sub_res = sub_dir / 'results.txt'
                print(sub_dir.name, file=results)
                with sub_res.open('r') as sub_res:
                    for line in sub_res:
                        print(line.strip(), file=results)

    all_results = defaultdict(list)
    results = exp_dir / 'results.txt'
    with results.open('r') as results:
        for line in results:
            if ':' in line:
                key, values = line.split(':')
                key = key.strip()
                values = values.strip().split(' & ')
                values = [float(val) for val in values]
                all_results[key].append(values)

    # for key in all_results:
    #     all_results[key] = list(np.array(all_results[key]).mean(axis=0))

    results = exp_dir / 'results.txt'
    with results.open('a') as results:
        print("Mean of Results", file=results)
        for k, v in all_results.items():
            v = list(np.array(v).mean(axis=0))
            v = [f"{i:.2f}" for i in v]
            print(f"{k + ':':<20}{' & '.join(v)}", file=results)
        print("Std Dev Results", file=results)
        for k, v in all_results.items():
            v = list(np.array(v).std(axis=0))
            v = [f"{i:.2f}" for i in v]
            print(f"{k + ':':<20}{' & '.join(v)}", file=results)

lang = 'arabic'
exp_type = 'ud-sud'
exp_name = 'padt-tag'

# for exp_name in ['padt-tag', 'padt-tag-ft', 'padt-char', 'padt-char-ft', 'padt-tag-char', 'padt-tag-char-ft']:
#     print(exp_name)
#     exp_dir = exp_root / lang / exp_type / exp_name
#     for sub_dir in exp_dir.glob('*'):
#         if sub_dir.is_dir():
#             collate_multiple_runs(lang, exp_type, exp_name, sub_dir.name)

def gather_baseline_dir(exp_dir, baseline=False, debug=False, std_dev=False):
    if debug:
        print(exp_dir)
    exp_results = []
    for run in exp_dir.glob('*'):
        if run.is_dir():
            if debug:
                print(run)
            run_results = read_results(run, baseline=baseline)
            if debug:
                print(run_results)
            run_results = write_baseline_result(run_results, debug=debug)
            run_results = np.array(run_results, dtype=float)
            exp_results.append(run_results)
    exp_results = np.stack(exp_results, axis=1)

    means = [f"{i:.2f}" for i in np.mean(exp_results, axis=1).tolist()]
    std = [f"{i:.2f}" for i in np.std(exp_results, axis=1).tolist()]

    if not std_dev:
        print(' & '.join(means))
    else:
        means_stds = [f"{m} ({s})" for m, s in zip(means, std)]

        print(' & '.join(means_stds))
    # sub_exp_results_mean = ' & '.join([f"{i:.2f}" for i in np.mean(exp_results, axis=1).tolist()])
    # sub_exp_results_std = ' & '.join([f"{i:.2f}" for i in np.std(exp_results, axis=1).tolist()])
    # print(sub_exp_results_mean)
    # print(sub_exp_results_std)

def gather_multitask(lang, exp_type, exp_name, debug=False, file_print=False, gather_test=False, std_dev=False):

    exp_dir = exp_root / lang / exp_type / exp_name
    
    for loss in ['alternating', 'joint']:
        for mlp in ['nosharemlp', 'sharemlp']:
            sub_exp = exp_dir / f'{loss}-{mlp}'
            sub_exp_results = []
            for run in sub_exp.glob('*'):
                if run.is_dir():
                    if debug:
                        print(run)
                    run_results = read_results(run)
                    run_results = write_result(run_results, debug=debug)
                    run_results = np.array(run_results, dtype=float)
                    sub_exp_results.append(run_results)

            sub_exp_results = np.stack(sub_exp_results, axis=2)
            means = [f"{i:.2f}" for i in np.mean(sub_exp_results, axis=2).tolist()[0]]
            stds = [f"{i:.2f}" for i in np.std(sub_exp_results, axis=2).tolist()[0]]
            if not std_dev:
                print(' & '.join(means))
            else:
                means_stds = [f"{m} ({s})" for m, s in zip(means, stds)]
                print(' & '.join(means_stds))
            # sub_exp_results_mean = ' & '.join([f"{i:.2f}" for i in np.mean(sub_exp_results, axis=2).tolist()[0]])
            # sub_exp_results_std = ' & '.join([f"{i:.2f}" for i in np.std(sub_exp_results, axis=2).tolist()[0]])
            # print(sub_exp_results_mean)
            # print(sub_exp_results_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather experiment results")
    parser.add_argument('--lang', '-l', default='arabic', help="Language")
    parser.add_argument('--exp-type', '-et', default='ud-sud',
                        help="Type of experiment: (single/ud-sud)")
    parser.add_argument('--exp-name', '-en', required=True,
                        help="Experiment Directory")
    parser.add_argument('--file-print', '-f', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--gather-test', '-t', action='store_true')
    parser.add_argument('--baseline', '-b', action='store_true')
    parser.add_argument('--std-dev', '-s', action='store_true')
    args = parser.parse_args()
    if args.baseline:
        gather_baseline_dir(exp_root / args.lang / 'baseline' / args.exp_name, baseline=args.baseline, debug=args.debug, std_dev=args.std_dev)
    else:
        # args = vars(args)
        gather_multitask(args.lang, args.exp_type, args.exp_name, args.debug, args.file_print, args.gather_test, std_dev=args.std_dev)