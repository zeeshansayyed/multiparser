from pathlib import Path
from supar.utils.data import Dataset
from supar.utils.field import Field
from supar.utils.common import bos, pad, unk
from supar.utils.transform import CoNLL
from collections import defaultdict

data_dir = Path('data')
embed_dir = data_dir / 'embeddings'
lang = 'de'
LANG = 'German'
treebank = 'gsd'

sud_dir = data_dir / 'sud' / 'sud-treebanks-v2.6' / f'SUD_{LANG}-{treebank.upper()}'
ud_dir = data_dir / 'ud' / 'ud-treebannks-v2.6' / f'UD_{LANG}-{treebank.upper()}'

non_proj = defaultdict(list)
with open(f'tmp/{lang}_projectivity_analysis.txt', 'w') as out_file:
    for type in ('ud', 'sud'):
        for split in ('train', 'dev', 'test'):
            data_path = (data_dir / f'{type}' / f'{type}-treebanks-v2.6' /
                        f'{type.upper()}_{LANG}-{treebank.upper()}' /
                        f'{lang}_{treebank}-{type}-{split}.conllu')
            print(data_path, file=out_file)
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            FEAT = Field('tags', bos=bos)
            ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
            REL = Field('rels', bos=bos)
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)
            data = Dataset(transform, str(data_path))

            for i, sent in enumerate(data.sentences):
                if not CoNLL.isprojective(list(map(int, sent.arcs))):
                    print(f"{i}: {' '.join(sent.words)}", file=out_file)
                    non_proj[f'{type}-{split}'].append(int(i))

for split in ('train', 'dev', 'test'):
    ud_set = set(non_proj[f'ud-{split}'])
    sud_set = set(non_proj[f'sud-{split}'])
    print(f"{split}: UD={len(ud_set)} SUD={len(sud_set)} Intersection={len(ud_set.intersection(sud_set))}")