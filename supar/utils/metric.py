# -*- coding: utf-8 -*-

from collections import Counter
import torch

class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class AttachmentMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.eps = eps

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
        return s

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        lens = mask.sum(1)
        arc_mask = arc_preds.eq(arc_golds) & mask
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()
        return self

    def __add__(self, other):
        new_metric = AttachmentMetric()
        new_metric.n = self.n + other.n
        new_metric.n_ucm = self.n_ucm + other.n_ucm
        new_metric.n_lcm = self.n_lcm + other.n_lcm
        new_metric.total = self.total + other.total
        new_metric.correct_arcs = self.correct_arcs + other.correct_arcs
        new_metric.correct_rels = self.correct_rels + other.correct_rels
        return new_metric

    @property
    def score(self):
        return self.las

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)


class SpanMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            upred = Counter([(i, j) for i, j, label in pred])
            ugold = Counter([(i, j) for i, j, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)
        return self

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"

        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)


class ChartMetric(Metric):

    def __init__(self, eps=1e-12):
        super(ChartMetric, self).__init__()

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()
        return self

    def __repr__(self):
        return f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)

import pudb
class SegmentationMetric(Metric):

    def __init__(self, eps=1e-12, space_index=3, compute_word_acc=False) -> None:
        super().__init__()
        self.eps = eps
        self.n = 0.0
        self.correct_words = 0.0
        self.correct_chars = 0.0
        self.total_words = 0.0
        self.total_chars = 0.0
        self.space_index = space_index
        self.compute_word_acc = compute_word_acc

    def __repr__(self) -> str:
        if self.compute_word_acc:
            s = f"WordAcc: {self.word_acc:4.2%} CharAcc: {self.char_acc:4.2%} "
        else:
            s = f"CharAcc: {self.char_acc:4.2%} "
        return s

    def __call__(self, pred_labels, gold_labels, word_mask, char_mask):
        self.total_chars += char_mask.sum()
        res = pred_labels.eq(gold_labels)[char_mask]
        self.correct_chars += res.sum()
        if self.compute_word_acc:
            lens = char_mask.sum(-1).tolist()
            gold_list = [map(str, seq.tolist()) for seq in gold_labels[char_mask].split(lens)]
            pred_list = [map(str, seq.tolist()) for seq in pred_labels[char_mask].split(lens)]
            gold_labels = [''.join(gword).split(str(self.space_index)) for gword in gold_list]
            pred_labels = [''.join(pword).split(str(self.space_index)) for pword in pred_list]
            assert len(gold_labels) == len(pred_labels)
            for gsent, psent in zip(gold_labels, pred_labels):
                # assert len(gsent) == len(psent)
                self.total_words += len(gsent)
                for gword, pword in zip(gsent, psent):
                    if gword == pword:
                        self.correct_words += 1
            # lens = char_mask.sum(-1).tolist()
            # gold_sents = gold_labels[char_mask].split(lens)
            # pred_sents = pred_labels[char_mask].split(lens)
            # for gold_sent, pred_sent in zip(gold_sents, pred_sents):
            #     space_mask = torch.nonzero(gold_sent.eq(self.space_index),
            #                                as_tuple=False).view(-1)
            #     shifted_space_mask = space_mask.roll(1)
            #     shifted_space_mask[0] = 0
            #     word_lens = (space_mask - shifted_space_mask).tolist()
            #     word_lens[0] += 1
            #     word_lens.append(len(gold_sent) - sum(word_lens))
            #     gold_words = gold_sent.split(word_lens)
            #     pred_words = pred_sent.split(word_lens)
            #     self.total_words += len(gold_words)
            #     for gold_word, pred_word in zip(gold_words, pred_words):
            #         if gold_word.eq(pred_word).sum() == len(gold_word):
            #             self.correct_words += 1
        return self

    @property
    def score(self):
        return self.char_acc

    @property
    def word_acc(self):
        return self.correct_words / (self.total_words + self.eps)

    @property
    def char_acc(self):
        return self.correct_chars / (self.total_chars + self.eps)
