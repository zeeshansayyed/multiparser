from pathlib import Path
import numpy as np
import re
import pudb
import argparse

lang = 'arabic'
exp_type = 'ud-sud'
exp_name = 'padt-tree-tag'

exp_root = Path('exp')


def tabelize(lang, exp_type, exp_name):
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

    bold_lines = []
    for group in res_groups:
        # print(group)
        group_max = np.max(group, axis=0)
        for line in group:
            line = list(line)
            for i, word_max in enumerate(group_max):
                if line[i] == word_max:
                    line[i] = f'\\textbf{{{line[i]}}}'
                else:
                    line[i] = f'{line[i]:.2f}' # str(line[i])
            bold_lines.append(line)

    baseline_result = []
    for tname in exp_type.split('-'):
        baseline_dir = exp_root / lang / 'single' / f'{tname}-{exp_name}'
        print(f"Checking for baseline results in {baseline_dir}")
        train_logs = sorted(baseline_dir.glob('train*.log'), reverse=True)
        dev_uas = 0
        dev_las = 0
        print(f"Found the following train logs: {' '.join([log.name for log in train_logs])}")
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

    plus_range = results.max(axis=0) - baseline_result
    minus_range = baseline_result - results.min(axis=0)
    color_span = 100
    color_start = 255 - color_span

    to_hex = lambda rgb: '%02x%02x%02x' % rgb
    str_results = [[f'{e:.2f}' for e in row] for row in results]
    for col_no, column in enumerate(results.T):
        column_diffs = column - baseline_result[col_no]
        for row_no, column_diff in enumerate(column_diffs):
            if column_diff >= 0:
                color_magnitude = 255 - int(column_diff/plus_range[col_no] * color_span) # Range is 155, 255
                color_hex = to_hex((0, color_magnitude, 0))
            else:
                # color_magnitude = int(50 * (-column_diff) + 155)
                color_magnitude = 255 - int(-column_diff / minus_range[col_no] * color_span)
                color_hex = to_hex((color_magnitude, 0, 0))

            # str_column[j] = f"\\cellcolor[HTML]{{{color_hex}}} {str_column[j]}"
            str_results[row_no][col_no] = f"\\cellcolor[HTML]{{{color_hex}}} {bold_lines[row_no][col_no]}"

    line_lengths = [len(' & '.join(line)) for line in str_results]
    max_line_length = max(line_lengths) + 5

    with open(exp_dir / 'results.tab', 'w') as table_file:
        baseline_result = list(map(str, baseline_result))
        print(f"{' & '.join(baseline_result):<{max_line_length}}", file=table_file)
        for line in str_results:
            print(f"{' & '.join(line):<{max_line_length}}", file=table_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather experiment results")
    parser.add_argument('--lang', '-l', default='arabic', help="Language")
    parser.add_argument('--exp-name', '-e', required=True, help="Experiment Directory")
    parser.add_argument('--exp-type', '-t', default='ud-sud', choices=['single', 'ud-sud'])
    args = parser.parse_args()
    args = vars(args)
    tabelize(**args)