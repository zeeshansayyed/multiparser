# -*- coding: utf-8 -*-

import abc
from datetime import datetime, timedelta
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from supar.parsers.parser import Parser
from supar.models.multiparsing import MultiBiaffineDependencyModel
from supar.utils import Config, Dataset, Embedding
from supar.utils.fn import ispunct
from supar.utils.logging import init_logger, logger, progress_bar
from supar.utils.data import JointDataset, MultitaskDataLoader, JointDataLoader, DataLoader
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.metric import Metric, AttachmentMetric
from supar.utils.parallel import is_master
from supar.utils.field import Field, SubwordField
from supar.utils.common import bos, pad, unk
from supar.utils.transform import CoNLL
from supar.utils.optim import MultiTaskOptimizer, MultiTaskScheduler


class MultiTaskParser(Parser):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        super().__init__(args, model, transform)
        self.transforms = self.transform

    def train_loop(self, task_names, train_loader, dev_loaders, test_loaders,
                   args, train_mode='train', save_prefix=None):
        assert len(task_names) == len(dev_loaders) == len(test_loaders)

        if train_mode == 'finetune':
            lr = args.lr / 10
        else:
            lr = args.lr
            task_names = task_names + ['total']


        self.optimizer = MultiTaskOptimizer(self.model, args.task_names,
                                            args.optimizer_type, lr, args.mu,
                                            args.nu, args.epsilon)
        self.scheduler = MultiTaskScheduler(self.optimizer, **args)

        elapsed = timedelta()
        best_e = {tname: 1 for tname in task_names}
        best_metrics = {tname: AttachmentMetric() for tname in task_names}

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()
            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train_loader, train_mode=train_mode)
            losses, metrics = self._multi_evaluate(dev_loaders, task_names)

            for tname in task_names:
                logger.info(f"{tname:6} - {'dev:':6} - loss: {losses[tname]:.4f} - {metrics[tname]}")
                if metrics[tname] > best_metrics[tname]:
                    best_e[tname] = epoch
                    best_metrics[tname] = metrics[tname]
                    if is_master():
                        if save_prefix:
                            model_name = f"{save_prefix}-{tname}.model"
                        else:
                            model_name = f"{tname}.model"
                        model_path = args.exp_dir / model_name
                        self.save(model_path)
                        logger.info(f"Saved model: {model_path}")

            for tname in task_names:
                if save_prefix:
                    model_name = f"{save_prefix}-{tname}.model"
                else:
                    model_name = f"{tname}.model"
                logger.info(f"{model_name:4} - Best epoch {best_e[tname]} - {best_metrics[tname]}")

            t = datetime.now() - start
            logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - max(best_e.values()) >= args.patience:
                break

        logger.info(f"{train_mode}ing completed")
        for tname, test_loader in zip(task_names, test_loaders):
            logger.info(f"Task Name: {tname}")
            args.path = args.exp_dir / f"{tname}.model"
            loss, metric = self.load(**args)._evaluate(test_loader, tname)

            logger.info(f"Epoch {best_e[tname]} saved")
            logger.info(f"{'dev:':6} - {best_metrics[tname]}")
            logger.info(f"{'test:':6} - {metric}")
            logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch\n")

        return best_metrics

    def train(self, train, dev, test, train_mode='train', buckets=32,
              batch_size=5000, lr=2e-3, mu=.9, nu=.9, epsilon=1e-12, clip=5.0,
              decay=.75, decay_steps=5000, epochs=5000, patience=100, verbose=True,
              **kwargs):

        args = Config().update(locals())
        init_logger(logger, verbose=args.verbose)

        for transform in self.transforms:
            transform.train()

        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()

        train, dev, test = [], [], []
        for train_f, dev_f, test_f, transform in zip(args.train, args.dev,
                                                     args.test, self.transforms):
            train.append(Dataset(transform, train_f, **args))
            dev.append(Dataset(transform, dev_f))
            test.append(Dataset(transform, test_f))

        for trainD, devD, testD, tname in zip(train, dev, test, args.task_names):
            logger.info(f"Building for {tname}")
            trainD.build(args.batch_size, args.buckets, True,
                         dist.is_initialized())
            devD.build(args.batch_size, args.buckets)
            testD.build(args.batch_size, args.buckets)
            logger.info(f"{'train:':6} {trainD}\n"
                        f"{'dev:':6} {devD}\n"
                        f"{'test:':6} {testD}\n")

        if args.joint_loss:
            joint_train = JointDataset(train, args.task_names)
            joint_train.build(args.batch_size, args.buckets, True,
                        dist.is_initialized())
            train_loader = joint_train.loader
        else:
            train_loader = MultitaskDataLoader(args.task_names, train)

        if args.loss_weights:
            assert len(args.loss_weights) == len(args.task_names)
            self.args.loss_weights = {
                task_name: float(loss_ratio)
                for task_name, loss_ratio in zip(args.task_names,
                                                 args.loss_weights)
            }
        else:
            self.args.loss_weights = { 
                task_name: 1.0
                for task_name in args.task_names
            }

        dev_loaders = [d.loader for d in dev]
        test_loaders = [d.loader for d in test]

        logger.info(f"{self.model}\n")
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[args.local_rank],
                             find_unused_parameters=True)

        if train_mode == 'train':
            train_metrics = self.train_loop(args.task_names, train_loader,
                                            dev_loaders, test_loaders, args)
        else:
            _, train_metrics = self._multi_evaluate(dev_loaders, args.task_names)
            logger.info("Dev Metrics before finetuning")
            for tname, dev_loader in zip(args.task_names, dev_loaders):
                logger.info(f"Task Name: {tname}")
                args.path = args.exp_dir / f"{tname}.model"
                loss, metric = self.load(**args)._evaluate(dev_loader, tname)
                logger.info(f"{tname:5}: {metric}")

        if args.finetune or train_mode == 'finetune':
            logger.info("Starting Finetuning ...")
            finetuned_metrics = {tname: {} for tname in train_metrics.keys()}
            for start_task in train_metrics.keys():
                for task_id, finetune_task in enumerate(args.task_names):
                    logger.info(f"Finetuning model: {start_task}.model with {finetune_task} data")
                    args.path = args.exp_dir / f"{start_task}.model"
                    args.loss_weights = self.args.loss_weights
                    parser = self.load(**args)
                    if args.finetune == 'partial':
                        parser.model.freeze_shared()
                    # Choose only one element from the list
                    finetune_task_list = args.task_names[task_id: task_id + 1]
                    finetune_train_loader = MultitaskDataLoader(finetune_task_list, train[task_id: task_id + 1])
                    finetune_dev_loaders = dev_loaders[task_id: task_id + 1]
                    finetune_test_loaders = test_loaders[task_id: task_id + 1]
                    finetuned_metrics[start_task].update(
                        parser.train_loop(
                            finetune_task_list,
                            finetune_train_loader,
                            finetune_dev_loaders,
                            finetune_test_loaders,
                            args,
                            train_mode='finetune',
                            save_prefix=f'{args.finetune}-{start_task}',
                        ))

            for start_task, tmetrics in train_metrics.items():
                logger.info(f"Dev Metrics for {start_task}: {tmetrics}")
                for finetuned_task_name, fmetrics in finetuned_metrics[start_task].items():
                    logger.info(f"Dev Metrics for {start_task}-{finetuned_task_name}: {fmetrics}")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        """
        If a model file is specified in the --path argument, then that model is 
        loaded and evaluated. Instead, if the experiment directory is specified, 
        then a random .model file is first loaded to set things up. Then all 
        .model files present in directory are evaluated in the following fashion:
            - {task}.model is tested on {task} data
            - total.model is tested on all provided data
            - Model ending with "*-{task}.model" are tested on {task}
        """
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        for transform in self.transforms:
            transform.train()

        for task, data_list in zip(args.task, args.data):
            # args.task_names was saved in parser while training
            # args.task was provided as cmd line args while evaluation
            task_id = args.task_names.index(task)
            if args.path.is_file():
                paths = [args.path]
            else:
                paths = []
                paths += list(args.path.glob(f"*-{task}.model"))
                paths += list(args.path.glob('total.model'))
                paths += list(args.path.glob(f'{task}.model'))
            for data in data_list:
                dataset = Dataset(self.transforms[task_id], data, proj=args.proj)
                dataset.build(args.batch_size, args.buckets)
                for path in paths:
                    parser = self.load(path)
                    start = datetime.now()
                    loss, metric = parser._evaluate(dataset.loader, task)
                    logger.info(f"Data={os.path.split(data)[-1]}\tModel={path.name}\tMetric={metric}")

        return loss, metric

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False,
                **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        for transform in self.transforms:
            transform.eval()
            if args.prob:
                transform.append(Field('probs'))
        
        original_path = args.path
        for task, data_list in zip(args.task, args.data):
            # args.task_names was saved in parser while training
            # args.task was provided as cmd line args while evaluation
            task_id = args.task_names.index(task)
            transform = self.transforms[task_id]
            if args.path.is_file():
                paths = [args.path]
            else:
                paths = []
                paths += list(args.path.glob(f"*-{task}.model"))
                paths += list(args.path.glob('total.model'))
                paths += list(args.path.glob(f'{task}.model'))

            for data in data_list:
                for path in paths:
                    logger.info("Loading the data")
                    dataset = Dataset(transform, data)
                    dataset.build(args.batch_size, args.buckets)
                    logger.info(f"\n{dataset}")
                    args.path = path
                    parser = self.load(path)
                    print(path)
                    logger.info(f"Making predictions on {data} using {path.name}")
                    start = datetime.now()
                    preds = parser._predict(dataset.loader, task)
                    elapsed = datetime.now() - start

                    for name, value in preds.items():
                        setattr(dataset, name, value)
                    if is_master():
                        if not hasattr(args, 'pred') or args.pred is None:
                            pred = args.exp_dir / f"{path.stem}-{os.path.split(data)[-1]}"
                        logger.info(f"Saving predicted results to {pred}")
                        transform.save(pred, dataset.sentences)
                    logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")
            args.path = original_path
        return dataset

    @classmethod
    def load(cls, path, **kwargs):
        """
        If a path to a model file is specified, then it loads that model. If the 
        experiment directory is provided in the path instead, then it randomly 
        loads the a model file (with .model ext)
        """
        if path.is_file():
            # return super(Parser, cls).load(path, **kwargs)
            return super().load(path, **kwargs)
        else:
            for p in path.glob('*.model'):
                return super().load(p, **kwargs)

    @torch.no_grad()
    @abc.abstractmethod
    def _evaluate(self, loader, task_name):
        raise NotImplementedError

    @torch.no_grad()
    def _multi_evaluate(self, loaders, task_names):
        losses = {tname: 1 for tname in task_names}
        metrics = {tname: AttachmentMetric() for tname in task_names}

        for loader, tname in zip(loaders, task_names):
            losses[tname], metrics[tname] = self._evaluate(loader, tname)

        losses['total'] = sum(losses.values())
        total_metric = AttachmentMetric()
        for m in metrics.values():
            total_metric += m
        metrics['total'] = total_metric
        return losses, metrics

class MultiBiaffineDependencyParser(MultiTaskParser):

    NAME = 'multi-biaffine-dependency'
    MODEL = MultiBiaffineDependencyModel

    def __init__(self, args, model, transforms):
        """
        A CoNLL transform object takes in the following Field objects:
	    - WORD: Word embeddings
		 - Supposed to be build on the train set.
		 - For MultitaskParser we build on the combined train set of
           all tasks. This will have to change when we use different
           languages.
	    - FEAT: Char/Bert/POS embeddings
		 - Just like WORD. Should be built on train set and for MTP,
           we build on the combined train.
	    - ARC: Arcs labels
		 - This are word indices indicating where the arc starts and
           where it ends.
	    - REL: Dependency relations
		 - These are relations different for different tasks, for
           e.g. the ones for UD and SUD are different. This is retrieved by 
           passing the ~ConLL.get_arcs~ function the Field constuctor
		 - Hence, we cannot build them on ~combined_train~ just the
           way we do for WORD and FEAT. Hence, we build it
           individually for different train sets for each
           corresponding task. This is what leads to different
           transform objects for different tasks.

        Hence while retrieving each of the fields in this constructor, we simply 
        retrieve the first WORD, FEAT and ARC from the first transform. 

	     In the future when we want to use the Parser for different
         languages or for train sets from completely different domains
         we might want to build WORD and FEAT separately for each
         train set, the way we do for RELs.

		 Also, when we move onto doing constituency and dependency, we
         will have to see how to treat ARC.
        """
        super().__init__(args, model, transforms)
        transform = transforms[0]
        self.WORD, self.CHAR, self.BERT = transform.FORM
        # if self.args.feat in ('char', 'bert'): #new
        #     self.WORD, self.FEAT = transform.FORM
        # else:
        #     self.WORD, self.FEAT = transform.FORM, transform.CPOS
        self.ARC, self.REL = transform.HEAD, [t.DEPREL for t in transforms]
        self.puncts = torch.tensor([
            i for s, i in self.WORD.vocab.stoi.items() if ispunct(s)
        ]).to(self.args.device)

    def train(self, train, dev, test, buckets=32, batch_size=5000, punct=False,
              tree=False, proj=False, partial=False, verbose=True, **kwargs):
        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, punct=False, tree=True,
                 proj=False, partial=False, verbose=True, **kwargs):
        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False,
                tree=True, proj=False, verbose=True, **kwargs):
        return super().predict(**Config().update(locals()))

    def _separate_train(self, loader, train_mode='train'):
        self.model.train()
        bar, metric = progress_bar(loader), AttachmentMetric()

        for words, *feats, arcs, rels, task_name in bar: #new
            self.optimizer.set_mode(task_name, train_mode)
            self.scheduler.set_mode(task_name, train_mode)

            self.optimizer.zero_grad()
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats, task_name)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask,
                                   self.args.partial)
            loss *= self.args.loss_weights[task_name]
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
                f"lr: {self.scheduler.get_last_lr(task_name)[0]:.4e} - loss: {loss:.4f} - {metric}"
            )

    def _joint_train(self, loader, train_mode='train'):
        self.model.train()
        bar, metric = progress_bar(loader), AttachmentMetric()
        self.optimizer.set_mode(self.args.task_names, train_mode)
        self.scheduler.set_mode(self.args.task_names, train_mode)
        for inputs, targets in bar:
            words, *feats = inputs.values()
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

            joint_loss = sum(losses)
            joint_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            # Calculating train metric. Can be skipped
            for t_name, t_targets in targets.items():
                arcs, rels = t_targets['arcs'], t_targets['rels']
                s_arc, s_rel = s_arcs[t_name], s_rels[t_name]
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
                if self.args.partial:
                    mask &= arcs.ge(0)
                # ignore all punctuation if not specified
                if not self.args.punct:
                    mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
                metric(arc_preds, rel_preds, arcs, rels, mask)

            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {joint_loss:.4f} - {metric}"
            )

    def _train(self, loader, train_mode='train'):
        if self.args.joint_loss and train_mode == 'train':
            self._joint_train(loader, train_mode=train_mode)
        else:
            self._separate_train(loader, train_mode=train_mode)

    @torch.no_grad()
    def _evaluate(self, loader, task_name):
        self.model.eval()
        total_loss, metric = 0, AttachmentMetric()

        if task_name not in self.model.args.task_names:
            return total_loss, metric

        for words, *feats, arcs, rels in loader:
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
    def _predict(self, loader, task_name):
        # args.task_names was saved in parser while training
        # args.task was provided as cmd line args while evaluation
        task_id = self.args.task_names.index(task_name)
        self.model.eval()

        preds = {}
        arcs, rels, probs = [], [], []
        for words, *feats in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(words, feats, task_name)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if hasattr(self.args, 'pred') and self.args.prob:
                arc_probs = s_arc.softmax(-1)
                probs.extend([
                    prob[1:i + 1, :i + 1].cpu()
                    for i, prob in zip(lens, arc_probs.unbind())
                ])
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL[task_id].vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if hasattr(self.args, 'pred') and self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
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
        TAG, CHAR, BERT = None, None, None
        if 'tag' in args.feat:
            TAG = Field('tags', bos=bos)
        if 'char' in args.feat:
            CHAR = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
        if 'bert' in args.feat:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            BERT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            BERT.vocab = tokenizer.get_vocab()
        # if args.feat == 'char':
        #     FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos,
        #                         fix_len=args.fix_len)
        # elif args.feat == 'bert':
        #     from transformers import AutoTokenizer
        #     tokenizer = AutoTokenizer.from_pretrained(args.bert)
        #     FEAT = SubwordField('bert', pad=tokenizer.pad_token,
        #                         unk=tokenizer.unk_token, bos=tokenizer.bos_token
        #                         or tokenizer.cls_token, fix_len=args.fix_len,
        #                         tokenize=tokenizer.tokenize)
        #     FEAT.vocab = tokenizer.get_vocab()
        # else:
        #     FEAT = Field('tags', bos=bos)
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        RELS = [Field('rels', bos=bos) for i in range(len(args.task_names))]

        transforms = [CoNLL(FORM=(WORD, CHAR, BERT), POS=TAG, HEAD=ARC, DEPREL=REL) for REL in RELS]
        # if args.feat in ('char', 'bert'):
        #     transforms = [
        #         CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL) for REL in RELS
        #     ]
        # else:
        #     transforms = [
        #         CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL) for REL in RELS
        #     ]

        # HACK for creating a combined train object.
        # TODO: Think of something better. This is UGLY af
        train = [
            Dataset(transform, train_f, **args)
            for train_f, transform in zip(args.train, transforms)
        ]
        combined_train = Dataset(transforms[0], args.train[0], **args)
        for i in range(1, len(train)):
            combined_train.sentences += train[i].sentences

        WORD.build(combined_train, args.min_freq,
                   (Embedding.load(args.embed, args.unk) if args.embed else None))
        if TAG is not None:
            TAG.build(combined_train)
        if CHAR is not None:
            CHAR.build(combined_train)
        # FEAT.build(combined_train)

        from collections import Counter
        rel_counters = []
        for REL, trainD in zip(RELS, train):
            c = Counter()
            for s in trainD.sentences:
                c.update(s.rels)
            rel_counters.append(c)
            REL.build(trainD)

        args.update({
            'n_words': WORD.vocab.n_init,
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            # 'n_feats': len(FEAT.vocab),
            'n_rels': [len(REL.vocab) for REL in RELS],
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            # 'feat_pad_index': FEAT.pad_index,
        })
        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transforms)