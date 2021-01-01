from pathlib import Path
import re

exp_root = Path('exp')
exp_dir = exp_root / 'arabic' / 'ud-sud' / 'padt-tree'
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
    print(' & '.join(vanilla), file=file)
    models = [f'{i}-{j}' for i in task_names for j in task_names]
    m = 0
    for i in range(len(task_names)):
        finetune = []
        for j in range(len(task_names)):
            finetune.append(results[split][models[m]][0])
            finetune.append(results[split][models[m]][1])
            m += 1
        print(' & '.join(finetune), file=file)
    total = [results[split][f'total-{tname}'][i] for tname in task_names for i in range(2)]
    print(' & '.join(total), file=file)

# for exp in exp_dir.glob('*'):
#     if exp.is_dir():
#         print(exp.name)
#         r = read_dir_results(exp)
#         write_dir_results(r)

with open(exp_dir / 'results.txt', 'w') as rfile:
    print("Dev Results", file=rfile)
    for loss_type in ['alternating', 'joint']:
        for mlp in ('sharemlp', 'nosharemlp'):
            for opt_type in ('single', 'multiple'):
                for finetune in ('partial', 'whole'):
                    d = exp_dir / f'{loss_type}-{finetune}-{opt_type}-{mlp}'
                    if d.exists():
                        # print(d, file=rfile)
                        r = read_dir_results(d)
                        write_dir_results(r, rfile)
    print("Test Results", file=rfile)
    for loss_type in ['alternating', 'joint']:
        for mlp in ('sharemlp', 'nosharemlp'):
            for opt_type in ('single', 'multiple'):
                for finetune in ('partial', 'whole'):
                    d = exp_dir / f'{loss_type}-{finetune}-{opt_type}-{mlp}'
                    if d.exists():
                        print(d, file=rfile)
                        r = read_dir_results(d)
                        write_dir_results(r, rfile, split='test')