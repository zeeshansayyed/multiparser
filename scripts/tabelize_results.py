from pathlib import Path
import numpy as np
import re

lang = 'arabic'
exp_type = 'ud-sud'
exp_name = 'padt-tree'

exp_root = Path('exp')
exp_dir = exp_root / lang / exp_type / exp_name
result = exp_dir / 'results.txt'

with result.open('r') as result_file:
    lines = result_file.read()

lines = lines.split('\n')
results = []

for line in lines:
    if '&' in line:
        line = line.strip().split('&')
        results.append([float(word.strip()) for word in line])


results = np.array(results)
res_groups = results.reshape(-1, 4, 4)

for group in res_groups:
    # print(group)
    group_max = np.max(group, axis=0)
    for line in group:
        line = list(line)
        for i, word_max in enumerate(group_max):
            if line[i] == word_max:
                line[i] = f'\\textbf{{{line[i]}}}'
            else:
                line[i] = str(line[i])
        # print(' & '.join(line))

baseline_result = []
for tname in exp_type.split('-'):
    baseline_dir = exp_root / lang / 'single' / f'{tname}-{exp_name}'
    train_logs = sorted(baseline_dir.glob('train*.log'), reverse=True)
    dev_uas = 0
    dev_las = 0
    print(train_logs)
    for train_log in train_logs[:1]:
        print(train_log)
        with train_log.open('r') as train_log:
            lines = train_log.read()
        lines = lines.split('\n')
        for line in lines[-10:]:
            # print(line)
            if 'INFO dev:' in line:
                dev_uas = re.search(r"UAS: (.*?)%", line).group(1)
                dev_las = re.search(r"LAS:(.*?)%", line).group(1)

    baseline_result.append(float(dev_uas))
    baseline_result.append(float(dev_las))

plus = results.max(axis=0) - baseline_result
minus = baseline_result - results.min(axis=0)

to_hex = lambda rgb: '%02x%02x%02x' % rgb

str_results = results.astype(str)
for i, (column, str_column) in enumerate(zip(results.T, str_results.T)):
    column_diffs = column - baseline_result[i]
    for j, column_diff in enumerate(column_diffs):
        if column_diff >= 0:
            color_magnitude = int(50 * column_diff + 155) # Range is 155, 255
            color_hex = to_hex((0, color_magnitude, 0))
        else:
            color_magnitude = int(50 * (-column_diff) + 155)
            color_hex = to_hex((color_magnitude, 0, 0))
        str_column[j] = f"\\cellcolor[HTML]{{{color_hex}}} {str_column[j]}"


with open(exp_dir / 'results.tab', 'w') as table_file:
    baseline_result = list(map(str, baseline_result))
    print(' & '.join(baseline_result), file=table_file)
    for line in str_results:
        print(' & '.join(line), file=table_file)