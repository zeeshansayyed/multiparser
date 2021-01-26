from pathlib import Path
import shutil
import pudb

# name = 'padt-tree-proj'
# source_root = Path("exp/arabic") / name
dest_root = Path("/N/slate/zasayyed/Projects/results")


# for source_dir in source_root.glob('*'):
#     print(source_dir.name)
#     dest_dir = dest_root / source_dir.name
#     # dest_dir.mkdir(parents=True, exist_ok=True)
#     for pred_file in source_dir.glob('*.conllu'):
#         print(pred_file)
#         # shutil.copy(pred_file, dest_dir)
#     print()


def copy_dir(source_dir, lang, exptype):
    print(f"Copying {source_dir}")
    for d in source_dir.glob('*'):
        dest_dir = dest_root / lang / exp_type / source_dir.name / d.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        d = d / '10'

        for conll in d.glob('*.conllu'):
            print(conll)
            shutil.copy(conll, dest_dir)

def copy_baseline_dir(source_dir, lang, exptype):
    dest_dir = dest_root / lang / exp_type / source_dir.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    d = source_dir / '10'
    print(f"Copying {d}")

    for conll in d.glob('*.conllu'):
        print(conll)
        shutil.copy(conll, dest_dir)


exp_root = Path('exp')
languages = ['arabic', 'chinese', 'english', 'german', 'greek', 'hungarian', 'korean', 'russian', 'turkish', 'vietnamese']
treebanks = ['padt', 'gsd', 'ewt', 'gsd', 'gdt', 'szeged', 'gsd', 'gsd', 'imst', 'vtb']
exp_types = ['baseline'] #, 'ud-sud']
exp_subtypes = ['tag', 'tag-ft', 'char', 'char-ft', 'tag-char-ft']
tasks = ['ud', 'sud']

# for lang_dir in exp_root.glob('*'):
#     if lang_dir.is_dir():
#         if lang_dir.name in languages:
#             print(lang_dir)
#             for exp_dir in lang_dir.glob('*'):
#                 if exp_dir.name in exp_types:
#                     print(exp_dir)

for lang, treebank in zip(languages, treebanks):
    for exp_type in exp_types:
        expdir = exp_root / lang / exp_type
        if exp_type == 'baseline':
            for task in tasks:
                for exp_subtype in exp_subtypes:
                    subexpdir = expdir / f'{task}-{treebank}-{exp_subtype}'
                    if subexpdir.is_dir():
                        copy_baseline_dir(subexpdir, lang, exp_type)
        else:
            for exp_subtype in exp_subtypes:
                subexpdir = expdir / f'{treebank}-{exp_subtype}'
                if subexpdir.is_dir():
                    copy_dir(subexpdir, lang, exp_type)
