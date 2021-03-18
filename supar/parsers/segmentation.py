# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR

from supar.models import SegmenterModel
from supar.models.segmentation import StandardizerModel, JointSegmenterStandardizerModel
from supar.parsers.multiparsers import MultiTaskParser
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils import transform
from supar.utils import metric
from supar.utils.common import bos, pad, unk, eos
from supar.utils.field import Field, SubwordField
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import SegmentationMetric, StandardizationMetric
from supar.utils.transform import CoNLL, StdSeg

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
        metric = SegmentationMetric(space_idx=self.SEGL.vocab.stoi[' '], compute_word_acc=True)
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



LABELLING_TASKS = ('raw-segl', 'std-stdsegl')
SEQ2SEQ_TASKS = ('raw-std', 'seg-stdseg', 'raw-stdseg')


class JointSegmenterStandardizer(MultiTaskParser):

    NAME = "joint-segmenter-standardizer"
    MODEL = JointSegmenterStandardizerModel

    def __init__(self, args, model, transforms):
        super().__init__(args, model, transforms)
        # Since input fields are shared across tasks, we only store the one
        # coming from the first transform
        self.input_char, *self.input_feats = transforms[0].inputs
        # Each value of the target_fields dict is a tuple itself
        # with the first element being the target char field
        self.target_fields = {
            task_name: t.targets
            for task_name, t in zip(args.task_names, transforms)
        }

    def _update_data_source_names(self, train, dev, test, task_names):
        num_tasks = len(task_names)
        train = [train] * num_tasks
        dev = [dev] * num_tasks
        test = [test] * num_tasks
        return train, dev, test

    def train(self, train, dev, test, task_names, buckets=32, batch_size=5000,
              punct=False, tree=False, proj=False, partial=False, verbose=True,
              **kwargs):
        train, dev, test = self._update_data_source_names(train, dev, test, task_names)
        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, punct=False, tree=True,
                 proj=False, partial=False, verbose=True, **kwargs):
        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False,
                tree=True, proj=False, verbose=True, **kwargs):
        return super().predict(**Config().update(locals()))

    def _separate_train(self, loader, train_mode='train'):
        self.model.train()
        bar, metric = progress_bar(loader), SegmentationMetric()

        # for words, *feats, arcs, rels, task_name in bar: #new
        for src_chars, *feats, tgt_chars, task_name in bar:
            self.optimizer.set_mode(task_name, train_mode)
            self.scheduler.set_mode(task_name, train_mode)

            self.optimizer.zero_grad()
            src_mask = src_chars.ne(self.model.src_dict.pad())
            model_out = self.model(src_chars, feats, tgt_chars, task_name, src_mask)
            loss = self.model.loss(model_out, tgt_chars, task_name)
            # mask = words.ne(self.WORD.pad_index)
            # # ignore the first token of each sentence
            # mask[:, 0] = 0
            # s_arc, s_rel = self.model(words, feats, task_name)
            # loss = self.model.loss(s_arc, s_rel, arcs, rels, mask,
            #                        self.args.partial)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            # preds = self.model.decode(src_chars, feats, task_name, src_mask)
            # ignore all punctuation if not specified
            # if not self.args.punct:
            #     src_mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)

            # metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr(task_name)[0]:.4e} - loss: {loss:.4f}"
            )

    def _joint_train(self, loader, train_mode='train'):
        self.model.train()
        bar = progress_bar(loader)
        self.optimizer.set_mode(self.args.task_names, train_mode)
        self.scheduler.set_mode(self.args.task_names, train_mode)
        for inputs, targets in bar:
            src_chars, *feats = inputs.values()
            self.optimizer.zero_grad()
            shared_out = self.model.shared_forward(src_chars, feats)
            losses = []
            for t_name, t_targets in targets.items():
                tgt_chars = list(t_targets.values())[0] # TODO: Make this generic
                model_out = self.model.unshared_forward(shared_out, tgt_chars, t_name)
                loss = self.model.loss(model_out, tgt_chars, t_name)
                losses.append(loss)
            joint_loss = sum(losses)
            joint_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {joint_loss:.4f}"
            )

    def _train(self, loader, train_mode='train'):
        if self.args.joint_loss and train_mode == 'train':
            self._joint_train(loader, train_mode=train_mode)
        else:
            self._separate_train(loader, train_mode=train_mode)

    @torch.no_grad()
    def _evaluate(self, loader, task_name):
        self.model.eval()
        total_loss = 0
        if task_name in LABELLING_TASKS:
            metric = SegmentationMetric(
                space_idx=self.target_fields[task_name][0].vocab.stoi[' '],
                compute_word_acc=True)
        elif task_name in SEQ2SEQ_TASKS:
            metric = StandardizationMetric(
                space_idx=self.target_fields[task_name][0].vocab.stoi[' '],
                compute_word_acc=True)
        else:
            raise Exception(
                f"Task name ({task_name}) not supported in _evaluate()")

        for src_chars, *feats, tgt_chars in progress_bar(loader):
            src_mask = src_chars.ne(self.model.src_dict.pad())
            model_out = self.model(src_chars, feats, tgt_chars, task_name,
                                   src_mask)
            loss = self.model.loss(model_out, tgt_chars, task_name)
            preds = self.model.decode(src_chars, feats, task_name, src_mask)
            total_loss += loss.item()
            metric(preds, tgt_chars, None, src_mask)
        total_loss /= len(loader)
        return total_loss, metric

    def _predict(self, loader, task_name):
        # args.task_names was saved in parser while training
        # args.task was provided as cmd line args while evaluation
        task_id = self.args.task_names.index(task_name)
        self.model.eval()
        preds = {}
        input_sents, pred_sents = [], []
        for src_chars, *feats in loader:
            src_mask = src_chars.ne(self.model.src_dict.pad())
            preds = self.model.decode(src_chars, feats, task_name, src_mask)
            pred_sents.extend(preds)
            char_mask = src_chars.ne(2) & src_chars.ne(0) & src_chars.ne(3)
            word_lens = char_mask.sum(-1).tolist()
            input_sents.extend(src_chars[char_mask].split(word_lens))
        raw_words = [self.input_char.vocab[seq.tolist()] for seq in input_sents]
        raw_words = [''.join(seq).split() for seq in raw_words]
        pred_words = [self.target_fields[task_name][0].vocab[seq[:-1]] for seq in pred_sents] # Ignore <eos>
        pred_words = [''.join(seq).split() for seq in pred_words]
        preds = {'words': raw_words}
        preds[self.target_fields[task_name][0].name] = pred_words
        return preds


    @classmethod
    def build(cls, path, min_freq=1, fix_len=20, **kwargs):
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.SWORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        RAW_WORD = Field('raw_words', pad=pad, unk=unk, bos=bos, eos=eos)
        RAW_CHAR = Field('raw_chars', pad=pad, unk=unk, bos=bos, eos=eos, tokenize=simple_tokenize)
        SEG_CHAR = Field('seg_chars', pad=pad, unk=unk, bos=bos, eos=eos, tokenize=simple_tokenize)
        SEG_LABEL = Field('seg_labels', pad=pad, unk=unk, bos=bos, eos=eos, tokenize=simple_tokenize)
        STD_CHAR =  Field('std_chars', pad=pad, unk=unk, bos=bos, eos=eos, tokenize=simple_tokenize)
        STDSEG_CHAR =  Field('stdseg_chars', pad=pad, unk=unk, bos=bos, eos=eos, tokenize=simple_tokenize)
        STDSEG_LABEL = Field('stdseg_labels', pad=pad, unk=unk, bos=bos, eos=eos, tokenize=simple_tokenize)

        def get_transform(task_name):
            if task_name == 'raw-segl':
                transform = StdSeg(RAW=RAW_CHAR, SEGL=SEG_LABEL)
                transform.inputs = (RAW_CHAR,)
                transform.targets = (SEG_LABEL,)
            elif task_name == 'raw-std':
                transform = StdSeg(RAW=RAW_CHAR, STD=STD_CHAR)
                transform.inputs = (RAW_CHAR,)
                transform.targets = (STD_CHAR,)
            elif task_name == 'seg-stdseg':
                transform = StdSeg(SEG=SEG_CHAR, STDSEG=STDSEG_CHAR)
                transform.inputs = (SEG_CHAR,)
                transform.targets = (STDSEG_CHAR,)
            elif task_name == 'std-stdsegl':
                transform = StdSeg(STD=STD_CHAR, STDSEGL=STDSEG_LABEL)
                transform.inputs = (STD_CHAR,)
                transform.targets = (STDSEG_LABEL,)
            else:
                raise Exception(f"Task Name ({task_name}) is not supported")
            return transform

        transforms = [get_transform(task_name) for task_name in args.task_names]
        trains = [Dataset(transform, args.train, **args) for transform in transforms]
        for transform, trainD in zip(transforms, trains):
            for field in transform:
                if field is not None:
                    field.build(trainD)

        # Currently we only support character dictionaries and hance char embeddings
        # Modify this to support word dictionaries/embeddings
        src_dict = FairseqDictWrapper(transforms[0].inputs[0])
        # Every task will have its own target_dictionary
        tgt_dicts = {
            task_name: FairseqDictWrapper(transform.targets[0])
            for task_name, transform in zip(args.task_names, transforms)
        }

        args.update({
            'src_dict': src_dict,
            'tgt_dicts': tgt_dicts,
            'train': [args.train] * len(args.task_names),
            'dev': [args.dev] * len(args.task_names),
            'test': [args.test] * len(args.task_names),
        })

        logger.info("Building the model")
        # * We do not load pretrained word embeddings for now. IF you want to support
        # * then figure standardize how char and word embeddings are input to transform
        # * and the call `load_pretrained()`
        # model = cls.MODEL(**args).load_pretrained(SWORD.embed).to(args.device)
        model = cls.MODEL(**args).to(args.device)
        logger.info(f"{model}\n")
        return cls(args, model, transforms)