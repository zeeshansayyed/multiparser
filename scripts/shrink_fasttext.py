from pathlib import Path
from supar.utils.data import Dataset
from supar.utils.field import Field
from supar.utils.common import bos, pad, unk
from supar.utils.transform import CoNLL
from collections import defaultdict

data_dir = Path('data')
embed_dir = data_dir / 'embeddings'

sud_dir = data_dir / 'sud' / 'sud-treebanks-v2.6' / 'SUD_Arabic-PADT'
ud_dir = data_dir / 'ud' / 'ud-treebannks-v2.6' / 'UD_Arabic-PADT'

vocab = set()
with open('tmp/projectivity_analysis.txt', 'w') as out_file:
    for type in ('ud', 'sud'):
        for split in ('train', 'dev', 'test'):
            data_path = (data_dir / f'{type}' / f'{type}-treebanks-v2.6' /
                        f'{type.upper()}_Arabic-PADT' /
                        f'ar_padt-{type}-{split}.conllu')
            print(data_path, file=out_file)
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            FEAT = Field('tags', bos=bos)
            ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
            REL = Field('rels', bos=bos)
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)
            data = Dataset(transform, str(data_path))

            for i, sent in enumerate(data.sentences):
                vocab.update(sent.words)

i = 0
with open(embed_dir / 'cc.ar.300.vec', 'r') as embed_file, \
    open(embed_dir / 'cc.ar.300.vec.padt', 'w') as shrink_file:
    for line in embed_file:
        word = line.split()[0]
        if word in vocab or word == 'unk':
            shrink_file.write(line)
            i += 1