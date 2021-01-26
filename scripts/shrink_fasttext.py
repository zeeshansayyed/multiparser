from pathlib import Path

from torch._C import dtype
from supar.utils.data import Dataset
from supar.utils.field import Field
from supar.utils.common import bos, pad, unk
from supar.utils.transform import CoNLL
from collections import defaultdict
import argparse
import numpy as np
import pudb

data_dir = Path('data')
embed_dir = data_dir / 'embeddings'


def shrink(lang_name, lang_code, treebank, additional_data, new_extension=None):
    lang_name = lang_name.title()
    if lang_name == 'Hungarian':
        treebank = treebank.title()
    else:
        treebank = treebank.upper()

    if len(additional_data) > 0 and not new_extension:
        raise Exception("Specify a new extension for the shrinked file")

    if not new_extension:
        new_extension = treebank.lower()

    vocab = set()
    for type in ('ud', 'sud'):
        for split in ('train', 'dev', 'test'):
            data_path = (data_dir / f'{type}' / f'{type}-treebanks-v2.7' /
                         f'{type.upper()}_{lang_name}-{treebank}' /
                         f'{lang_code}_{treebank.lower()}-{type}-{split}.conllu')
            print(data_path)
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            FEAT = Field('tags', bos=bos)
            ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
            REL = Field('rels', bos=bos)
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)
            data = Dataset(transform, str(data_path))

            for i, sent in enumerate(data.sentences):
                vocab.update(sent.words)

    for data_path in additional_data:
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        FEAT = Field('tags', bos=bos)
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)
        data = Dataset(transform, str(data_path))

        for i, sent in enumerate(data.sentences):
            vocab.update(sent.words)

    print(f"Vocabulary Length = {len(vocab)}")
    print(f"Saving the new embeddings in cc.{lang_code}.300.vec.{new_extension}")
    found_unk = False
    with open(embed_dir / f'cc.{lang_code}.300.vec', 'r', errors='surrogateescape') as embed_file, \
        open(embed_dir / f'cc.{lang_code}.300.vec.{new_extension}', 'w') as shrink_file:
        for line in embed_file:
            word = line.split()[0]
            if word in vocab or word == 'unk':
                shrink_file.write(line)
                if word == 'unk':
                    found_unk = True

    if not found_unk:
        print("Did not find unk in the embeddings. Will create one ...")
        embeds = []
        with open(embed_dir / f'cc.{lang_code}.300.vec', 'r', errors='surrogateescape') as embed_file, \
            open(embed_dir / f'cc.{lang_code}.300.vec.{new_extension}', 'a') as shrink_file:
            next(embed_file)
            for line in embed_file:
                embeds.append(line.split()[1:])

            embeds = np.array(embeds, dtype=float)
            embed_mean = [f'{i:.4f}' for i in np.mean(embeds, axis=0).tolist()]
            unk_embed = ['unk'] + embed_mean
            shrink_file.write(' '.join(unk_embed) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shrink Fasttext embeddings")
    parser.add_argument('--lang-name', '-ln', required=True, help="Language Name")
    parser.add_argument('--lang-code', '-lc', required=True, help="Language Code")
    parser.add_argument('--treebank', '-t', required=True, help="Name of the treebank")
    parser.add_argument('--additional-data', '-d', nargs='+', default=[], help="Additional datafiles to consider")
    parser.add_argument('--new-extension', '-e', help="Extension for the shrinked file")
    args = parser.parse_args()
    args = vars(args)
    shrink(**args)