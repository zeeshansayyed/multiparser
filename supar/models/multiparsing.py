# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import (CharLSTM, BertEmbedding, LSTM, MLP, Biaffine)
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from supar.utils.transform import CoNLL
from supar.utils.alg import eisner, mst
import pudb


class MultiBiaffineDependencyModel(nn.Module):
    r"""
    The implementation of MultiTask and Joint dependency parsing model which uses 
    Biaffine Dependency Parser as its core parser.

    """

    def __init__(self,
                 task_names,
                 n_words,
                #  n_feats,
                 n_rels,
                 n_tags=None,
                 n_chars=None,
                 feat='tag,char,bert',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 char_pad_index=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pad_index=0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        
        self.n_input = n_embed
        if 'tag' in feat:
            self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                          embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'char' in feat:
            self.char_embed = CharLSTM(n_chars=n_chars,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=char_pad_index)
            self.n_input += n_feat_embed
        if 'bert' in feat:
            self.bert_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=bert_pad_index,
                                            dropout=mix_dropout)
            self.n_input += self.bert_embed.n_out
        # if feat == 'char':
        #     self.feat_embed = CharLSTM(n_chars=n_feats,
        #                                n_embed=n_char_embed,
        #                                n_out=n_feat_embed,
        #                                pad_index=feat_pad_index)
        # elif feat == 'bert':
        #     self.feat_embed = BertEmbedding(model=bert,
        #                                     n_layers=n_bert_layers,
        #                                     n_out=n_feat_embed,
        #                                     pad_index=feat_pad_index,
        #                                     dropout=mix_dropout)
        #     self.n_feat_embed = self.feat_embed.n_out
        # elif feat == 'tag':
        #     self.feat_embed = nn.Embedding(num_embeddings=n_feats,
        #                                    embedding_dim=n_feat_embed)
        # else:
        #     raise RuntimeError("The feat type should be in ['char', 'bert', 'tag'].")
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = LSTM(input_size=self.n_input,     #(input_size=n_embed+n_feat_embed,
                         hidden_size=n_lstm_hidden,
                         num_layers=n_lstm_layers,
                         bidirectional=True,
                         dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)

        # Create the MLP Layers
        if not self.args.share_mlp:
            mlp_arc_d, mlp_arc_h, mlp_rel_d, mlp_rel_h = {}, {}, {}, {}
            for task in task_names:
                mlp_arc_d[task] = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_arc, dropout=mlp_dropout)
                mlp_arc_h[task] = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_arc, dropout=mlp_dropout)
                mlp_rel_d[task] = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_rel, dropout=mlp_dropout)
                mlp_rel_h[task] = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_rel, dropout=mlp_dropout)
            self.mlp_arc_d = nn.ModuleDict(mlp_arc_d)
            self.mlp_arc_h = nn.ModuleDict(mlp_arc_h)
            self.mlp_rel_d = nn.ModuleDict(mlp_rel_d)
            self.mlp_rel_h = nn.ModuleDict(mlp_rel_h)
        else:
            self.mlp_arc_d = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_arc, dropout=mlp_dropout)
            self.mlp_arc_h = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_arc, dropout=mlp_dropout)
            self.mlp_rel_d = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_rel, dropout=mlp_dropout)
            self.mlp_rel_h = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_rel, dropout=mlp_dropout)

        arc_attn, rel_attn = {}, {}
        for task_n_rels, task in zip(n_rels, task_names):
            arc_attn[task] = Biaffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
            rel_attn[task] = Biaffine(n_in=n_mlp_rel, n_out=task_n_rels, bias_x=True, bias_y=True)
        self.arc_attn = nn.ModuleDict(arc_attn)
        self.rel_attn = nn.ModuleDict(rel_attn)

        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def get_shared_parameters(self):
        for name, param in self.named_parameters():
            if not (('arc_attn' in name) or ('rel_attn' in name) or
                    (self.args.share_mlp and 'mlp' in name)):
                yield param

    def get_task_specific_parameters(self, task_name):
        for name, param in self.named_parameters():
            if task_name in name.split('.'):
                yield param

    def freeze_shared(self):
        for name, param in self.named_parameters():
            if not (('arc_attn' in name) or ('rel_attn' in name) or
                    (self.args.share_mlp and 'mlp' in name)):
                param.requires_grad = False

    def shared_forward(self, words, feats):
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
            
        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)
        # feat_embed = self.feat_embed(feats)
        # word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # # concatenate the word and feat representations
        # embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)
        if self.args.share_mlp:
            arc_d = self.mlp_arc_d(x)
            arc_h = self.mlp_arc_h(x)
            rel_d = self.mlp_rel_d(x)
            rel_h = self.mlp_rel_h(x)
            return (arc_d, arc_h, rel_d, rel_h), mask
        else:
            return x, mask

    def unshared_forward(self, shared_out, task_name):
        x, mask = shared_out
        if not self.args.share_mlp:
            arc_d = self.mlp_arc_d[task_name](x)
            arc_h = self.mlp_arc_h[task_name](x)
            rel_d = self.mlp_rel_d[task_name](x)
            rel_h = self.mlp_rel_h[task_name](x)
        else:
            arc_d, arc_h, rel_d, rel_h = x

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn[task_name](arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn[task_name](rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel


    def forward(self, words, feats, task_name):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (~torch.LongTensor):
                Feat indices.
                If feat is ``'char'`` or ``'bert'``, the size of feats should be ``[batch_size, seq_len, fix_len]``.
                if ``'tag'``, the size is ``[batch_size, seq_len]``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible arcs.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each arc.
        """
        shared_out = self.shared_forward(words, feats)
        s_arc, s_rel = self.unshared_forward(shared_out, task_name)

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        if partial:
            mask = mask & arcs.ge(0)
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds