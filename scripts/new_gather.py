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
        # model = '-'.join(model.split('-')[-2:])

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
            if finetune == 'standard':
                ftype = 'whole'
                if debug:
                    print(f"standard{':':<20}", end='', file=file)
                for task in tasks:
                    line = []
                    line += [score.strip() for score in curr_res[f"{ftype}-{task}-{task}"]]
                if debug:
                    print(' & '.join(line), file=file)
                ret_res.append(line)
            else:
                for from_task in tasks + ['total']:
                    line = []
                    if debug:
                        print(f"{from_task + ':':<20}", end='', file=file)
                    for to_task in tasks:
                        line += [score.strip() for score in curr_res[f"{finetune}-{from_task}-{to_task}"]]
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


def gather_baseline_dir(exp_dir, seeds, baseline=False, debug=False, std_dev=False):
    if len(seeds) == 0:
        all_seeds = True
        print("No seeds provided, will average all seeds present")
    else:
        all_seeds = False
        print(f"Will averate seeds {' '.join(seeds)}")

    if debug:
        print(exp_dir)
    exp_results = []
    for run in exp_dir.glob('*'):
        if run.is_dir() and (run.name in seeds or all_seeds):
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


def gather_multitask(lang, exp_type, exp_name, losses, seeds, debug=False, finetune=False, gather_test=False, std_dev=False):

    exp_dir = exp_root / lang / exp_type / exp_name
    if len(seeds) == 0:
        all_seeds = True
        print("No seeds provided, will average all seeds present")
    else:
        all_seeds = False
        print(f"Will averate seeds {' '.join(seeds)}")
    
    for loss in losses:
        for mlp in ['nosharemlp', 'sharemlp']:
            sub_exp = exp_dir / f'{loss}-{mlp}'
            sub_exp_results = []
            for run in sub_exp.glob('*'):
                if run.is_dir() and (run.name in seeds or all_seeds):
                    if debug:
                        print(run)
                    run_results = read_results(run)
                    run_results = write_result(run_results, finetune=finetune, debug=debug)
                    run_results = np.array(run_results, dtype=float)
                    sub_exp_results.append(run_results)

            sub_exp_results = np.stack(sub_exp_results, axis=2)
            all_means = [[f"{i:.2f}" for i in row] for row in np.mean(sub_exp_results, axis=2).tolist()]
            all_stds = [[f"{i:.2f}" for i in row] for row in np.std(sub_exp_results, axis=2).tolist()]

            if not std_dev:
                for means in all_means:
                    print(' & '.join(means))
            else:
                for means, stds in zip(all_means, all_stds):
                    means_stds = [f"{m} ({s})" for m, s in zip(means, stds)]
                    print(' & '.join(means_stds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather experiment results")
    parser.add_argument('--lang', '-l', default='arabic', help="Language")
    parser.add_argument('--exp-type', '-et', default='ud-sud',
                        help="Type of experiment: (single/ud-sud)")
    parser.add_argument('--exp-name', '-en', required=True,
                        help="Experiment Directory")
    parser.add_argument('--finetune', '-f', default=None, choices=['partial', 'whole', 'standard'])
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--gather-test', '-t', action='store_true')
    parser.add_argument('--baseline', '-b', action='store_true')
    parser.add_argument('--std-dev', '-s', action='store_true')
    parser.add_argument('--losses', nargs='+', default=['alternating', 'joint'])
    parser.add_argument('--seeds', nargs='+', default=[], help="Which seeds to gather from?")
    args = parser.parse_args()
    if args.baseline:
        gather_baseline_dir(exp_root / args.lang / 'baseline' / args.exp_name, args.seeds, baseline=args.baseline, debug=args.debug, std_dev=args.std_dev)
    else:
        gather_multitask(args.lang, args.exp_type, args.exp_name, args.losses, args.seeds, args.debug, args.finetune, args.gather_test, std_dev=args.std_dev)