from pathlib import Path
import re
import argparse

exp_root = Path('exp')
exp_dir = exp_root / 'arabic' / 'ud-sud' / 'padt-tree-proj-tag-ft'
result_file = exp_dir / 'results.tab'

task_names = ['ud', 'sud']

def process_eval_line(line):
    if 'dev' in line:
        data_type = 'dev'
    else:
        data_type = 'test'

    # line = line.split('\t')
    # print(line)
    model = re.search(r"Model=(.*?)\.model", line).group(1)
    model = '-'.join(model.split('-')[-2:])
    UAS = re.search(r"UAS:(.*?)%", line).group(1)
    LAS = re.search(r"LAS:(.*?)%", line).group(1)
    return data_type, model, (UAS, LAS)

def read_dir_results(exp):
    eval_file = exp / 'evaluate.log'
    with eval_file.open('r') as eval_file:
        lines = eval_file.read()
    lines = lines.split('\n\n')[1].strip().split('\n')
    # print(len(lines))
    results = {'dev': {}, 'test': {}}
    for line in lines:
        data_type, model, metric = process_eval_line(line)
        # print(data_type, model, metric)
        results[data_type][model] = metric
    return results

def write_dir_results(results, file, split='dev'):
    vanilla = [results[split][tname][i] for tname in task_names for i in range(2)]
    if file != None:
        # print("Water")
        print(' & '.join(vanilla), file=file)
    else:
        print(' & '.join(vanilla))
    models = [f'{i}-{j}' for i in task_names for j in task_names]
    m = 0
    for i in range(len(task_names)):
        finetune = []
        for j in range(len(task_names)):
            finetune.append(results[split][models[m]][0])
            finetune.append(results[split][models[m]][1])
            m += 1
        if file:
            print(' & '.join(finetune), file=file)
        else:
            print(' & '.join(finetune))
    total = [results[split][f'total-{tname}'][i] for tname in task_names for i in range(2)]
    if file:
        print(' & '.join(total), file=file)
    else:
        print(' & '.join(total))

# for exp in exp_dir.glob('*'):
#     if exp.is_dir():
#         print(exp.name)
#         r = read_dir_results(exp)
#         write_dir_results(r)

def gather(lang, exp_name, file_print=False, debug=False, gather_test=False):
    exp_dir = exp_root / lang / 'ud-sud' / exp_name
    with open(exp_dir / 'results.txt', 'w') as rfile:
        if file_print:
            print("Dev Results", file=rfile)
        else:
            print("Dev Results")
        for loss_type in ['alternating', 'joint']:
            for mlp in ('sharemlp', 'nosharemlp'):
                for opt_type in ('single', 'multiple'):
                    for finetune in ('partial', 'whole'):
                        d = exp_dir / f'{loss_type}-{finetune}-{opt_type}-{mlp}'
                        if d.exists():
                            if debug:
                                if file_print:
                                    print(d, file=rfile)
                                else:
                                    print(d)
                            r = read_dir_results(d)
                            if file_print:
                                write_dir_results(r, rfile)
                            else:
                                write_dir_results(r, None)

        if gather_test:
            if file_print:
                print("Test Results", file=rfile)
            else:
                print("Test Results")
            for loss_type in ['alternating', 'joint']:
                for mlp in ('sharemlp', 'nosharemlp'):
                    for opt_type in ('single', 'multiple'):
                        for finetune in ('partial', 'whole'):
                            d = exp_dir / f'{loss_type}-{finetune}-{opt_type}-{mlp}'
                            if d.exists():
                                if debug:
                                    if file_print:
                                        print(d, file=rfile)
                                    else:
                                        print(d)
                                r = read_dir_results(d)
                                if file_print:
                                    write_dir_results(r, rfile, split='test')
                                else:
                                    write_dir_results(r, None, split='test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather experiment results")
    parser.add_argument('--lang', '-l', default='arabic', help="Language")
    parser.add_argument('--exp-name', '-e', required=True, help="Experiment Directory")
    parser.add_argument('--file-print', '-f', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--gather-test', '-t', action='store_true')
    args = parser.parse_args()
    args = vars(args)
    gather(**args)