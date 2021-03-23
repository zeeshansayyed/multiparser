from pathlib import Path

exp_root = Path("exp/arabic/ud-sud/padt-noproj")
exp_dir = exp_root / 'alternating-partial-multiple-nosharemlp'
train_log = exp_dir / 'train-2020-12-19--06:17:02.log'

# with train_log.open('r') as train_log:
#     log =  train_log.read()

# for paragraph in log.split('\n\n'):
#     if "INFO Task Name: " in paragraph:
#         print(paragraph, end='\n\n')



def read_file(train_log):
    result = {}
    with train_log.open('r') as train_log:
        for line in train_log:
            if "Dev Metrics for " in line:
                line = line.strip().split()
                # print(line[6][:-1], line[-3], line[-1])
                result[line[6][:-1]] = [line[-3][:-1], line[-1][:-1]]
    return result

def read_folder(root_folder):
    results = {}
    for dir in root_folder.iterdir():
        # if 'alternating' in dir.name:
            log_files = list(dir.glob("train*"))
            result = read_file(log_files[-1])
            if len(result) == 0:
                print(f"{dir} is empty")
            else:
                results[dir.name] = result
    return results

def write_results(results):
    for loss_type in ['alternating', 'joint']:
        for mlp in ('sharemlp', 'nosharemlp'):
            for opt_type in ('single', 'multiple'):
                for finetune in ('partial', 'whole'):
                    # print(f"{loss_type}-{finetune}-{opt_type}-{mlp}")
                    result = results[f"{loss_type}-{finetune}-{opt_type}-{mlp}"]
                    
                    result_line = []
                    for task in ('ud', 'sud', 'total'):
                        result_line += result[task]
                    print(' & '.join(result_line))

                    for starting in ('ud', 'sud', 'total'):
                        result_line = []
                        for ending in ('ud', 'sud', 'total'):
                            result_line += result[f'{starting}-{ending}']
                        print(' & '.join(result_line))


results = read_folder(exp_root)
print(results)
write_results(results)

