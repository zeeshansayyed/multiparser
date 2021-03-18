# -*- coding: utf-8 -*-

import argparse
from supar.parsers.segmentation import JointSegmenterStandardizer
from supar.cmds.cmd import parse

def main():
    parser = argparse.ArgumentParser(description='Create Joint Arabic Segmenter and Standardizer.')
    parser.set_defaults(Parser=JointSegmenterStandardizer)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', default='tag,char', help='additional features to use，separated by commas.')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--task-names', nargs='+', required=True, help='Name the tasks in order')
    subparser.add_argument('--train', default='data/stdseg/train.conll', help='path to train file')
    subparser.add_argument('--dev', default='data/stdseg/dev.conll', help='path to dev file')
    subparser.add_argument('--test', default='data/stdseg/test.conll', help='path to test file')
    subparser.add_argument('--embed', default=None, help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which bert model to use')
    subparser.add_argument('--share-mlp', action='store_true', help='whether to share mlp')
    subparser.add_argument('--joint-loss', action='store_true', help='Model joint loss')
    subparser.add_argument('--optimizer-type', choices=['single', 'multiple'], default='single')
    subparser.add_argument('--finetune', choices=['partial', 'whole'], default=None, help="If specified, training is followed by finetuning.")
    subparser.add_argument('--train-mode', choices=['train', 'finetune'], default='train', help="Whether to train a new model or finetune an existing one.")
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--task', action='append', required=True, help='Name the tasks in order')
    subparser.add_argument('--data', nargs='+', action='append', help='paths to datasets')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--task', action='append', required=True, help='Name the tasks in order')
    subparser.add_argument('--data', nargs='+', action='append', help='paths to datasets')
    subparser.add_argument('--pred', default=None, help='path to predicted result')
    parse(parser)


if __name__ == "__main__":
    main()
