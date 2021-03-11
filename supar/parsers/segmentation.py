# -*- coding: utf-8 -*-

import os
from supar.models.segmentation import StandardizerModel

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR

from supar.models import SegmenterModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk, eos
from supar.utils.field import Field, SubwordField
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import SegmentationMetric, StandardizationMetric
from supar.utils.transform import StdSeg

logger = get_logger(__name__)


def char_tokenize(word):
    if word in ('-LRB-', '-RRB-', '<<dquote>>'):
        return [word]
    else:
        return list(word)


def simple_tokenize(sentence):
    tokens = []
    for word in sentence:
        if word in ('<<LRB>>', '<<RRB>>', '<<dquote>>'):
            tokens.append(word)
        else:
            tokens += list(word)
        tokens.append(' ')
    tokens.pop()
    return tokens


class Segmenter(Parser):
    """Regular Segmenter

    Args:
        Parser ([type]): [description]
    """

    NAME = 'segmenter'
    MODEL = SegmenterModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR = self.transform.RAW
        self.SEGL = self.transform.SEGL
        self.puncts = torch.tensor([
            i for s, i in self.WORD.vocab.stoi.items() if ispunct(s)
        ]).to(self.args.device)

    def train(self, train, dev, test, buckets=32, batch_size=5000, punct=False,
              verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            punct (bool):
                If ``False``, ignores the punctuations during evaluation. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, punct=False,
                 verbose=True, **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            punct (bool):
                If ``False``, ignores the punctuations during evaluation. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False,
                verbose=True, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabiCHAR = SubwordField('chars', pad=pad, unk=unk, bos=bos,
        #                     tokenize=char_tokenize)lities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()
        bar, metric = progress_bar(loader), SegmentationMetric()
        for words, *feats, gold_labels in bar:
            self.optimizer.zero_grad()

            word_mask = words.ne(self.WORD.pad_index)
            word_mask[:, 0] = 0 # ignore the first token of each sentence
            chars = feats[0]
            char_mask = chars.ne(self.CHAR.pad_index)
            scores = self.model(words, feats, word_mask, char_mask)
            loss = self.model.loss(scores, gold_labels, char_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            pred_labels = self.model.decode(scores, word_mask)
            # ignore all punctuation if not specified
            if not self.args.punct:
                word_mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            char_mask[:, 0] = 0
            metric(pred_labels, gold_labels, word_mask, char_mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        metric = SegmentationMetric(space_index=self.SEGL.vocab.stoi[' '], compute_word_acc=True)
        for words, *feats, gold_labels in progress_bar(loader):
            word_mask = words.ne(self.WORD.pad_index)
            word_mask[:, 0] = 0 # ignore the first token of each sentence
            chars = feats[0]
            char_mask = chars.ne(self.CHAR.pad_index)

            scores = self.model(words, feats, word_mask, char_mask)
            loss = self.model.loss(scores, gold_labels, char_mask)
            pred_labels = self.model.decode(scores, word_mask)
            if not self.args.punct:
                word_mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            char_mask[:, 0] = 0
            metric(pred_labels, gold_labels, word_mask, char_mask)
        total_loss /= len(loader)
        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()
        preds = {}
        word_list, label_list = [], []
        for words, *feats in progress_bar(loader):
            word_mask = words.ne(self.WORD.pad_index)
            word_mask[:, 0] = 0 # ignore the first token of each sentence
            chars = feats[0]
            char_mask = chars.ne(self.CHAR.pad_index)
            scores = self.model(words, feats, word_mask, char_mask)
            pred_labels = self.model.decode(scores, word_mask)
            char_mask[:, 0] = 0
            lens = char_mask.sum(-1).tolist()
            word_list.extend(chars[char_mask].split(lens))
            label_list.extend(pred_labels[char_mask].split(lens))
        pu.db
        words = [self.CHAR.vocab[seq.tolist()] for seq in word_list]
        labels = [self.SEGL.vocab[seq.tolist()] for seq in label_list]
        words = [''.join(word).split() for word in words]
        labels = [''.join(label).split() for label in labels]
        preds = {'raw': words, 'labels': labels}
        return preds


    @classmethod
    def build(
        cls,
        path,
        optimizer_args={'lr': 2e-3, 'betas': (.9, .9), 'eps': 1e-12},
        scheduler_args={'gamma': .75**(1 / 5000)},
        min_freq=2,
        fix_len=512,
        **kwargs,
    ):

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser
        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos)
        # CHAR = SubwordField('chars', pad=pad, unk=unk, bos=bos, tokenize=char_tokenize)
        CHAR =  Field('chars', pad=pad, unk=unk, bos=bos, tokenize=simple_tokenize)
        LABELS = Field('labels', pad=pad, unk=unk, bos=bos, tokenize=simple_tokenize)
        # LABELS = SubwordField('labels', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
        transform = StdSeg(RAW=(WORD, CHAR), SEGL=LABELS)

        train = Dataset(transform, args.train)
        WORD.build(train, args.min_freq,
                   (Embedding.load(args.embed, args.unk) if args.embed else None))
        CHAR.build(train)
        LABELS.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_labels': len(LABELS.vocab),
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed).to(args.device)
        logger.info(f"{model}\n")
        optimizer = Adam(model.parameters(), **optimizer_args)
        scheduler = ExponentialLR(optimizer, **scheduler_args)

        return cls(args, model, transform, optimizer, scheduler)


class FairseqDictWrapper:

    def __init__(self, supar_field) -> None:
        self.field = supar_field
        self.vocab = supar_field.vocab

    def __len__(self):
        return len(self.vocab)

    def pad(self):
        return self.field.pad_index

    def unk(self):
        return self.field.unk_index

    def bos(self):
        return self.field.bos_index

    def eos(self):
        return self.field.eos_index


class Standardizer(Parser):

    NAME = 'standardizer'
    MODEL = StandardizerModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SWORD, self.SCHAR = self.transform.RAW
        self.TCHAR = self.transform.STD
        self.puncts = torch.tensor([
            i for s, i in self.SWORD.vocab.stoi.items() if ispunct(s)
        ]).to(self.args.device)

    def train(self, train, dev, test, buckets=32, batch_size=5000, punct=False,
              verbose=True, **kwargs):
        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, punct=False,
                 verbose=True, **kwargs):
        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False,
                verbose=True, **kwargs):
        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()
        bar, metric = progress_bar(loader), StandardizationMetric()

        for source_words, src_chars, tgt_chars in bar:
            self.optimizer.zero_grad()
            model_out = self.model(src_chars, tgt_chars)
            loss = self.model.loss(model_out, tgt_chars)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()
            # pred_labels = self.model.decode(src_chars)
            # metric(pred_labels, tgt_chars, None, tgt_chars.ne(0))
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        # metric = SegmentationMetric(space_index=self.TCHAR.vocab.stoi[' '], compute_word_acc=True)
        metric = StandardizationMetric(space_idx=self.TCHAR.vocab.stoi[' '], compute_word_acc=True)
        for source_words, src_chars, tgt_chars in progress_bar(loader):
            model_out = self.model(src_chars, tgt_chars)
            loss = self.model.loss(model_out, tgt_chars)
            pred_labels = self.model.decode(src_chars)
            metric(pred_labels, tgt_chars, None, tgt_chars.ne(0))
        total_loss /= len(loader)
        logger.info(f"Wrong: {metric.total_words - metric.correct_words} / {metric.total_words}")
        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()
        preds = {}
        input_sents, pred_sents = [], []
        for source_words, source_chars in progress_bar(loader):
            preds = self.model.decode(source_chars)
            pred_sents.extend(preds)
            # word_mask = source_words.ne(2) & source_words.ne(0)
            # word_lens = word_mask.sum(-1).tolist()
            # input_sents.extend(source_words[word_mask].split(word_lens))
            char_mask = source_chars.ne(2) & source_chars.ne(0) & source_chars.ne(3)
            word_lens = char_mask.sum(-1).tolist()
            input_sents.extend(source_chars[char_mask].split(word_lens))
        raw_words = [self.SCHAR.vocab[seq.tolist()] for seq in input_sents]
        raw_words = [''.join(seq).split() for seq in raw_words]
        std_words = [self.TCHAR.vocab[seq[:-1]] for seq in pred_sents] # Ignore <eos>
        std_words = [''.join(seq).split() for seq in std_words]
        preds = {'words': raw_words, 'stdchars': std_words}
        return preds
            

    @classmethod
    def build(
        cls,
        path,
        optimizer_args={'lr': 2e-3, 'betas': (.9, .9), 'eps': 1e-12},
        scheduler_args={'gamma': .75**(1 / 5000)},
        min_freq=1,
        fix_len=512,
        **kwargs,
    ):
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.SWORD.embed).to(args.device)
            return parser
        logger.info("Building the fields")
        SWORD = Field('words', pad=pad, unk=unk, bos=bos, eos=eos)
        SCHAR =  Field('chars', pad=pad, unk=unk, bos=bos, eos=eos, tokenize=simple_tokenize)
        TCHAR =  Field('stdchars', pad=pad, unk=unk, bos=bos, eos=eos, tokenize=simple_tokenize)
    
        transform = StdSeg(RAW=(SWORD, SCHAR), STD=TCHAR)

        train = Dataset(transform, args.train)
        SWORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
        SCHAR.build(train)
        TCHAR.build(train)
        source_dict = FairseqDictWrapper(SCHAR)
        target_dict = FairseqDictWrapper(TCHAR)
        args.update({
            'n_words': SWORD.vocab.n_init,
            'n_schars': len(SCHAR.vocab) if SCHAR is not None else None,
            'n_tchars': len(TCHAR.vocab) if TCHAR is not None else None,
            'char_pad_index': SCHAR.pad_index if SCHAR is not None else None,
            'pad_index': SCHAR.pad_index,
            'unk_index': SCHAR.unk_index,
            'bos_index': SCHAR.bos_index,
            'eos_index': SCHAR.eos_index,
            'source_dict': source_dict,
            'target_dict': target_dict
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(SWORD.embed).to(args.device)
        logger.info(f"{model}\n")
        optimizer = Adam(model.parameters(), **optimizer_args)
        scheduler = ExponentialLR(optimizer, **scheduler_args)

        return cls(args, model, transform, optimizer, scheduler)


class JointSegmenterStandardizer(Parser):
    pass