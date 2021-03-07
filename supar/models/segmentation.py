# -*- coding: utf-8 -*-

from fairseq import search
from supar.utils.seq2seq import EncoderRNN, AttnDecoderRNN, Seq2Seq, Attention
from fairseq.models.lstm import LSTMModel, LSTMEncoder, LSTMDecoder
from fairseq.sequence_generator import SequenceGenerator
from torch._C import BenchmarkConfig
from supar.modules import LSTM, MLP
from supar.utils import Config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pudb

MAX_SOURCE_POSITIONS = 4096
MAX_TARGET_POSITIONS = 4096


class SegmenterModel(nn.Module):
    """Implementation of standard segmenter

    Args:
        nn ([type]): [description]
    """
    def __init__(self, n_words, n_labels, n_tags=None, n_chars=None,
                 feat='tag,char,bert', n_embed=100, n_feat_embed=100,
                 n_char_embed=50, char_pad_index=0, bert=None, n_bert_layers=4,
                 mix_dropout=.0, bert_pad_index=0, embed_dropout=.33,
                 n_lstm_hidden=400, n_lstm_layers=3, lstm_dropout=.33,
                 top_dropout=.33, pad_index=0, unk_index=1, **kwargs):
        super().__init__()
        self.args = Config().update(locals())
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        self.char_embed = nn.Embedding(num_embeddings=n_chars,
                                       embedding_dim=n_embed)
        self.n_input = n_embed
        self.encoder = nn.LSTM(input_size=self.n_input, hidden_size=n_lstm_hidden,
                               num_layers=n_lstm_layers, bidirectional=True,
                               dropout=lstm_dropout)
        self.top_layer = MLP(n_in=n_lstm_hidden * 2, n_out=n_labels,
                             dropout=top_dropout)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.args.char_pad_index)
        # Miscellaneous
        self.pad_index = pad_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, words, feats, word_mask=None, char_mask=None):
        chars = feats[0]

        batch_size, seq_len = words.shape
        if word_mask == None:
            # get the mask and lengths of given batch
            word_mask = words.ne(self.pad_index)
            word_mask[:, 0] = 0  # ignore the first token of each sentence
        if char_mask == None:
            char_mask = chars.ne(self.args.char_pad_index)
            # char_mask[:, 0, :] = 0
        word_lens = char_mask.sum(-1)

        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        # mask = chars.ne(self.args.char_pad_index)
        # # [batch_size, seq_len]
        # lens = mask.sum(-1)
        # char_mask = lens.gt(0)

        # [n, fix_len, n_embed]
        # batch_size, seq_len, char_seq_len = chars.shape
        batch_size, seq_len = chars.shape
        # x = self.char_embed(chars[word_mask])
        x = self.char_embed(chars)
        # x = pack_padded_sequence(x, word_lens[word_mask], True, False)
        x = pack_padded_sequence(x, char_mask.sum(-1), True, False)
        x, _ = self.encoder(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.top_layer(x)
        # out = x.new_zeros(*chars.shape, self.args.n_labels)
        # out = out.masked_scatter_(char_mask.unsqueeze(-1), x)
        out = x
        return out

    def loss(self, scores, labels, char_mask):
        # char_mask = char_mask[:, 0]
        scores = scores[char_mask].view(-1, self.args.n_labels)
        labels = labels[char_mask].view(-1)
        return self.criterion(scores, labels)

    def decode(self, scores, mask):
        return scores.argmax(-1)



class StandardizerModel(nn.Module):
    def __init__(self, source_dict, target_dict, n_embed=100, n_lstm_hidden=400,
                 n_lstm_layers=3, embed_dropout=.33, lstm_dropout=.33,
                 top_dropout=.33, pad_index=0, unk_index=1, device='cpu', **kwargs):
        super().__init__()
        self.args = Config().update(locals())
        self.src_dict = source_dict
        self.tgt_dict = target_dict
        encoder = EncoderRNN(n_embed, n_lstm_hidden, n_lstm_hidden, len(source_dict), device)
        attn = Attention(n_lstm_hidden, n_lstm_hidden)
        decoder = AttnDecoderRNN(n_embed, n_lstm_hidden, n_lstm_hidden, len(target_dict), attn, device)
        self.seq2seq = Seq2Seq(encoder, decoder, device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=target_dict.pad())

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, src_chars, tgt_chars):
        return self.seq2seq(src_chars.T, tgt_chars.T)

    def loss(self, output, target_chars):
        output_size = output.shape[-1]
        output = output[1:].view(-1, output_size)
        # target_chars = target_chars[1:].view(-1)
        target_chars = target_chars.T[1:].reshape(-1)
        return self.criterion(output, target_chars)

    def decode(self):
        decoded_seqs = batch_beam_search_decoding(decoder=model.decoder,
                                                      enc_outs=enc_outs,
                                                      enc_last_h=h,
                                                      beam_width=opts.beam_width,
                                                      n_best=opts.n_best,
                                                      sos_token=TRG_SOS_IDX,
                                                      eos_token=TRG_EOS_IDX,
                                                      max_dec_steps=opts.max_dec_steps,
                                                      device=DEVICE)



class StandardizerModel2(nn.Module):
    def __init__(self, source_dict, target_dict, n_embed=100, n_lstm_hidden=400,
                 n_lstm_layers=3, embed_dropout=.33, lstm_dropout=.33,
                 top_dropout=.33, pad_index=0, unk_index=1, **kwargs):
        super().__init__()
        self.args = Config().update(locals())
        self.src_dict = source_dict
        self.tgt_dict = target_dict
        encoder = LSTMEncoder(
            dictionary=source_dict,
            embed_dim=n_embed,
            hidden_size=n_lstm_hidden,
            num_layers=n_lstm_layers,
            dropout_in=embed_dropout,
            dropout_out=lstm_dropout,
            bidirectional=True,
            left_pad=False,
            pretrained_embed=None,
            max_source_positions=MAX_SOURCE_POSITIONS,
        )
        decoder = LSTMDecoder(
            dictionary=target_dict,
            embed_dim=n_embed,
            hidden_size=n_lstm_hidden,
            out_embed_dim=n_lstm_hidden,
            num_layers=n_lstm_layers,
            dropout_in=embed_dropout,
            dropout_out=lstm_dropout,
            attention=True,
            encoder_output_units=n_lstm_hidden * 2,
            pretrained_embed=None,
            share_input_output_embed=False,
            # adaptive_softmax_cutoff=(
            #     utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
            #     if args.criterion == "adaptive_loss"
            #     else None
            # ),
            adaptive_softmax_cutoff=None,
            max_target_positions=MAX_SOURCE_POSITIONS,
            residuals=False,
        )
        self.lstm_model = LSTMModel(encoder, decoder)
        self.criterion = nn.CrossEntropyLoss(ignore_index=target_dict.pad())

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, src_chars, tgt_chars, src_mask=None):
        # if src_mask == None:
        #     src_mask = src_chars.ne(self.src_dict.pad())
        # src_lengths = src_mask.sum(-1)
        src_lengths = src_chars.ne(self.src_dict.pad()).sum(-1)
        encoder_out = self.lstm_model.encoder(src_chars, src_lengths=src_lengths,
                                              enforce_sorted=False)
        decoder_out, attn_scores = self.lstm_model.decoder(
            tgt_chars, encoder_out=encoder_out)
        return decoder_out, attn_scores

    def loss(self, decoder_out, target_chars):
        char_mask = target_chars.ne(self.tgt_dict.pad())
        lprobs = self.lstm_model.get_normalized_probs(decoder_out, log_probs=True)
        scores = lprobs[char_mask]
        labels = target_chars[char_mask]
        return self.criterion(scores, labels)

    def decode(self, src_chars, mask):
        # src_lengths = ((src_chars.ne(self.src_dict.eos())
        #                 & src_chars.ne(self.src_dict.pad())).long().sum(dim=1))
        src_lengths = src_chars.ne(self.src_dict.pad()).sum(-1)
        generator = SequenceGenerator(
            [self.lstm_model], self.tgt_dict, eos=self.tgt_dict.eos(),
            search_strategy=search.BeamSearch(self.tgt_dict), beam_size=5,
            max_len_a=1)
        fairseq_batch = {
            'net_input': {
                'src_tokens': src_chars,
                'src_lengths': src_lengths
            }
        }
        pu.db
        preds = generator.generate(self.lstm_model, fairseq_batch,
                                   bos_token=self.tgt_dict.bos())
        # # encoder_out = self.lstm_model.encoder(src_chars, src_lengths=src_lengths,
        # #                                       enforce_sorted=False)
        pu.db
        return src_chars.argmax(-1)
