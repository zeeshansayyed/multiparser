# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import ExponentialLR
from supar.models import MultiBiaffineDependencyModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import Field, SubwordField
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger, progress_bar, init_logger
from supar.utils.metric import AttachmentMetric
from supar.utils.transform import CoNLL

from torch.utils.data import ConcatDataset
from torch.optim import Adam
from supar.utils.data import JointDataset, MultitaskDataLoader
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import is_master
from supar.utils.metric import Metric
from datetime import datetime, timedelta
from pathlib import Path
import pudb

logger = get_logger(__name__)


class MultiBiaffineDependencyParser(Parser):
    r"""
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """

    NAME = 'multi-biaffine-dependency'
    MODEL = MultiBiaffineDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.FORM
        else:
            self.WORD, self.FEAT = self.transform.FORM, self.transform.CPOS
        self.ARC, self.REL = self.transform.HEAD, self.transform.DEPREL
        self.puncts = torch.tensor([i for s, i in self.WORD.vocab.stoi.items() if ispunct(s)]).to(self.args.device)

    def train(self, train, dev, test, buckets=32, batch_size=5000, punct=False,
              tree=False, proj=False, partial=False, verbose=True, epochs=5000,
              **kwargs):
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
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """
        args = Config().update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Loading the data")
        train = [
            Dataset(self.transform, train_f, **args) for train_f in args.train
        ]
        dev = [Dataset(self.transform, dev_f, **args) for dev_f in args.dev]
        test = [Dataset(self.transform, test_f, **args) for test_f in args.test]
        for i in range(len(train)):
            logger.info(f"Building for {args.task_names[i]}")
            train[i].build(args.batch_size, args.buckets, True,
                           dist.is_initialized())
            dev[i].build(args.batch_size, args.buckets)
            test[i].build(args.batch_size, args.buckets)
            logger.info(
                f"\n{'train:':6} {train[i]}\n{'dev:':6} {dev[i]}\n{'test:':6} {test[i]}\n"
            )

        if args.joint_loss:
            joint_train = JointDataset(train, args.task_names)
            joint_train.build(args.batch_size)
            train_loader = joint_train.loader
        else:
            train_loader = MultitaskDataLoader(args.task_names, train)
        dev_loaders = [dataset.loader for dataset in dev]

        logger.info(f"{self.model}\n")
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[args.local_rank],
                             find_unused_parameters=True)
        self.optimizer = Adam(self.model.parameters(), args.lr,
                              (args.mu, args.nu), args.epsilon)
        self.scheduler = ExponentialLR(self.optimizer,
                                       args.decay**(1 / args.decay_steps))

        elapsed = timedelta()
        best_e = {tname: 1 for tname in args.task_names + ['total']}
        best_metrics = {tname: Metric() for tname in args.task_names + ['total']}

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")

            if args.joint_loss:
                self._joint_train(train_loader)
            else:
                train_loader = MultitaskDataLoader(args.task_names, train)
                self._train(train_loader)

            losses, metrics = self._multi_evaluate(dev_loaders)

            for tname in best_e:
                logger.info(f"{tname:6} - {'dev:':6} - loss: {losses[tname]:.4f} - {metrics[tname]}")
                if metrics[tname] > best_metrics[tname]:
                    best_e[tname] = epoch
                    best_metrics[tname] = metrics[tname]
                    if is_master():
                        model_path = args.path / f"{tname}.model"
                        self.save(model_path)
                        logger.info(f"Saved model for {tname}")

            for task_name in best_e:
                logger.info(f"{task_name:4} - Best epoch {best_e[task_name]} - {best_metrics[task_name]}")

            t = datetime.now() - start
            logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - max(best_e.values()) >= args.patience:
                break

        for task_name, test_data in zip(args.task_names, test):
            logger.info(f"Task Name: {task_name}")
            model_path = args.path / f"{tname}.model"
            model = self.load(path=model_path, **args)
            loss, metric = model._evaluate(test_data.loader, task_name)

            logger.info(f"Epoch {best_e[task_name]} saved")
            logger.info(f"{'dev:':6} - {best_metrics[task_name]}")
            logger.info(f"{'test:':6} - {metric}")
            logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, punct=False, tree=True,
                 proj=False, partial=False, verbose=True, **kwargs):
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
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False,
                tree=True, proj=False, verbose=True, **kwargs):
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
                If ``True``, outputs the probabilities. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
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

        bar, metric = progress_bar(loader), AttachmentMetric()

        for words, feats, arcs, rels, task_name in bar:
            self.optimizer.zero_grad()

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats, task_name)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask,
                                   self.args.partial)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )

    def _joint_train(self, loader):
        self.model.train()
        bar, metric = progress_bar(loader), AttachmentMetric()
        for inputs, targets in bar:
            words, feats = inputs.values()
            self.optimizer.zero_grad()
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            shared_out = self.model.shared_forward(words, feats)
            losses = []
            s_arcs, s_rels = {}, {}
            for t_name, t_targets in targets.items():
                arcs, rels = t_targets['arcs'], t_targets['rels']
                s_arc, s_rel = self.model.unshared_forward(shared_out, t_name)
                s_arcs[t_name], s_rels[t_name] = s_arc, s_rel
                loss = self.model.loss(s_arc, s_rel, arcs, rels, mask,
                                       self.args.partial)
                losses.append(loss)

            # loss = self.model.joint_loss(losses)
            joint_loss = sum(losses)
            joint_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            # for t_name, t_targets in targets.items():
            #     arcs, rels = t_targets['arcs'], t_targets['rels']
            #     s_arc, s_rel = s_arcs[t_name], s_rels[t_name]
            #     arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            #     if self.args.partial:
            #         mask &= arcs.ge(0)
            #     # ignore all punctuation if not specified
            #     if not self.args.punct:
            #         mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            #     metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {joint_loss:.4f} - {metric}"
            )

    @torch.no_grad()
    def _evaluate(self, loader, task_name):
        self.model.eval()

        total_loss, metric = 0, AttachmentMetric()

        for words, feats, arcs, rels in loader:
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats, task_name)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask,
                                   self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _multi_evaluate(self, loaders):
        """This will evalaute the dataloaders for each task and return a joint
        metric as well as task specific metrics

        Args:
            loaders (List[supar.utils.DataLoader]): List of data loaders. Currently
                only simply multitask loaders are supported.
                TODO: Also supported joint loaders. That will save computation time 
                as shared layers will be forwarded only once.
        """
        task_names = self.args.task_names + ['total']
        losses = {tname: 1 for tname in task_names}
        metrics = {tname: AttachmentMetric() for tname in task_names}

        for task_name, loader in zip(self.args.task_names, loaders):
            for words, feats, arcs, rels in loader:
                mask = words.ne(self.WORD.pad_index)
                mask[:, 0] = 0
                s_arc, s_rel = self.model(words, feats, task_name)
                loss = self.model.loss(s_arc, s_rel, arcs, rels, mask,
                                   self.args.partial)
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                        self.args.tree,
                                                        self.args.proj)
                if self.args.partial:
                    mask &= arcs.ge(0)
                # ignore all punctuation if not specified
                if not self.args.punct:
                    mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
                # Update losses and metrics
                losses['total'] += loss.item()
                losses[task_name] += loss.item()
                metrics['total'](arc_preds, rel_preds, arcs, rels, mask)
                metrics[task_name](arc_preds, rel_preds, arcs, rels, mask)

        return losses, metrics

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        arcs, rels, probs = [], [], []
        for words, feats in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(words, feats)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                arc_probs = s_arc.softmax(-1)
                probs.extend([
                    prob[1:i + 1, :i + 1].cpu()
                    for i, prob in zip(lens, arc_probs.unbind())
                ])
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and 
        model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. 
                Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece 
                will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos,
                                fix_len=args.fix_len)
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            FEAT = SubwordField('bert', pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token, bos=tokenizer.bos_token
                                or tokenizer.cls_token, fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        else:
            FEAT = Field('tags', bos=bos)
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        if args.feat in ('char', 'bert'):
            transform = CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL)
        else:
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)

        # HACK for creating a combined train object.
        # TODO: Think of something better. This is UGLY af
        train = [Dataset(transform, train_f, **args) for train_f in args.train]
        combined_train = train[0]
        for i in range(1, len(train)):
            combined_train.sentences += train[i].sentences

        WORD.build(combined_train, args.min_freq,
                   (Embedding.load(args.embed, args.unk) if args.embed else None))
        FEAT.build(combined_train)
        REL.build(combined_train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_rels': len(REL.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'feat_pad_index': FEAT.pad_index,
        })
        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform)
