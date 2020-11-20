from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F
import pdb


class BOWEncoder(nn.Module):
  def __init__(self, vocab_size, emb_size, config):
    super(BOWEncoder, self).__init__()
    self.emb_size = emb_size
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.config = config

  def forward(self, input):
    batch_size, seq_len = input.size()
    embedded = self.embedding(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
    embedded = F.dropout(embedded, self.config['bow_dropout'], self.training)  # [batch_size x seq_len x emb_size]
    output_pool = F.max_pool1d(embedded.transpose(1, 2), seq_len).squeeze(2)  # [batch_size x emb_size]
    encoding = torch.tanh(output_pool)
    return encoding


class SeqEncoder(nn.Module):
  def __init__(self, vocab_size, emb_size, hidden_size, config):
    super(SeqEncoder, self).__init__()
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.config = config

    self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
    self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
    for w in self.lstm.parameters():  # initialize the gate weights with orthogonal
      if w.dim() > 1:
        weight_init.xavier_normal_(w)

  def forward(self, input):
    batch_size, seq_len = input.size()
    embedded = self.embedding(input)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
    embedded = F.dropout(embedded, self.config['seqenc_dropout'], self.training)
    rnn_output, hidden = self.lstm(embedded)  # out:[b x seq x hid_sz*2](biRNN)
    rnn_output = F.dropout(rnn_output, self.config['seqenc_dropout'], self.training)
    encoding = torch.tanh(rnn_output)

    return encoding


class Attention(nn.Module):
  def __init__(self, input_size, emb_size):
    super(Attention, self).__init__()
    self.linear_compare = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(input_size*2, emb_size), nn.ReLU(),
                                        nn.Dropout(p=0.2), nn.Linear(emb_size, emb_size), nn.ReLU())

  def forward(self, left, right):
    """
    :param left: [batch * length_left * dim]
    :param right: [batch * length_right * dim]
    :return:
    """
    pairwise_inner_prod = torch.matmul(left, right.transpose(1, 2))  # [batch_size, length_left, length_right]
    attn_l_to_r = pairwise_inner_prod.softmax(1)
    attn_r_to_l = pairwise_inner_prod.softmax(2)
    left_to_right = torch.matmul(attn_l_to_r.transpose(1, 2), left)
    right_to_left = torch.matmul(attn_r_to_l, right)
    left_more = torch.cat([left, right_to_left], 2)
    right_more = torch.cat([right, left_to_right], 2)
    left_compared = self.linear_compare(left_more).sum(1)
    right_compared = self.linear_compare(right_more).sum(1)
    return left_compared, right_compared


class QCModel(nn.Module):
  def __init__(self, config):
    super(QCModel, self).__init__()
    self.name = "default"

    self.conf = config
    self.margin = config['margin']
    self.query_encoder = SeqEncoder(config['qt_n_words'], config['emb_size'], config['lstm_dims'], self.conf)

    if self.conf['code_encoder'] == "bilstm":
      self.cand_encoder = SeqEncoder(config['code_n_words'], config['emb_size'], config['lstm_dims'],
                                     self.conf)  # Bi-LSTM
    else:
      self.cand_encoder = BOWEncoder(config['code_n_words'], 2 * config['lstm_dims'], self.conf)  # MLP

    self.attention = Attention(config['lstm_dims'] * 2, config["lstm_dims"])
    self.output = nn.Sequential(nn.Dropout(p=0.2),
                                nn.Linear(config["lstm_dims"]*2, config['lstm_dims']),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(config['lstm_dims'], config['lstm_dims']),
                                nn.ReLU(),
                                nn.Linear(config['lstm_dims'], 1),
                                nn.Sigmoid()
                                # nn.Tanh()
                                )
    self.loss = nn.BCELoss()

  def query_encoding(self, cand):
    code_repr = self.query_encoder(cand)
    return code_repr

  def cand_encoding(self, qt):
    cand_repr = self.cand_encoder(qt)
    return cand_repr

  def scoring(self, qt_repr, cand_repr):
    qt_repr_w_attn, cand_repr_w_attn = self.attention(qt_repr, cand_repr)
    sim = self.output(torch.cat([qt_repr_w_attn, cand_repr_w_attn], 1)).squeeze(1)
    # sim = F.cosine_similarity(qt_repr_w_attn, cand_repr_w_attn)
    return sim

  def cross_scoring(self, qt_repr, cand_repr):
    # raise NotImplementedError()
    pdb.set_trace()
    
    # Get pairwise cosine similarity
    qt_repr_norm = qt_repr / qt_repr.norm(dim=1)[:, None]
    cand_repr_norm = cand_repr / cand_repr.norm(dim=1)[:, None]
    pw_sim = torch.mm(qt_repr_norm, cand_repr_norm.transpose(0, 1))

    # Sample probability (not sum to one)
    temperature = 1
    exp_pw_sim = torch.exp(pw_sim.data / temperature)
    sampled = []
    for x in range(exp_pw_sim.size(0)):
      y = torch.multinomial(exp_pw_sim[x], 1)
      sampled.append(y)

    # Choose sampled as negative instance
    sim = torch.gather(pw_sim, 1, torch.stack(sampled)).squeeze(1)
    return sim

  def get_scores(self, qt, cand):
    raise NotImplementedError()

    qt_repr = self.query_encoding(qt)
    cand_repr = self.cand_encoding(cand)

    # Get pairwise cosine similarity
    qt_repr_norm = qt_repr / qt_repr.norm(dim=1)[:, None]
    cand_repr_norm = cand_repr / cand_repr.norm(dim=1)[:, None]
    pw_sim = torch.mm(qt_repr_norm, cand_repr_norm.transpose(0, 1))
    return pw_sim

  def sample_cand(self, qt, cand, collision=None, if_norm=False):
    raise NotImplementedError()

    qt_repr = self.query_encoding(qt)
    cand_repr = self.cand_encoding(cand)

    # Get pairwise similarity
    # pdb.set_trace()
    qt_repr_norm = qt_repr / qt_repr.norm(dim=1)[:, None]
    cand_repr_norm = cand_repr / cand_repr.norm(dim=1)[:, None]
    pw_sim = torch.mm(qt_repr_norm, cand_repr_norm.transpose(0, 1))

    # Sample probability (not sum to one)
    temperature = 0.2
    exp_pw_sim = torch.exp(pw_sim / temperature)

    # sample
    if collision is not None:
      # pw_sim = pw_sim.masked_fill(collision, 0)
      exp_pw_sim = exp_pw_sim.masked_fill(collision, 0)

    sampled = torch.multinomial(exp_pw_sim.data, 1)
    if if_norm:
      norm_exp_pw_sim = exp_pw_sim / exp_pw_sim.sum(1, keepdim=True)
      sim = torch.gather(norm_exp_pw_sim, 1, sampled).squeeze(1)
    else:
      sim = torch.gather(pw_sim, 1, sampled).squeeze(1)

    return sampled, sim

  def sample_cand_cc_with_qc(self, qt, cand, collision=None, if_norm=False):
    raise NotImplementedError()

    qt_repr = self.cand_encoding(qt)
    cand_repr = self.cand_encoding(cand)

    # Get pairwise cosine similarity
    qt_repr_norm = qt_repr / qt_repr.norm(dim=1)[:, None]
    cand_repr_norm = cand_repr / cand_repr.norm(dim=1)[:, None]
    pw_sim = torch.mm(qt_repr_norm, cand_repr_norm.transpose(0, 1))

    # Sample probability (not sum to one)
    temperature = 0.2
    exp_pw_sim = torch.exp(pw_sim / temperature)

    # sample
    if collision is not None:
      # pw_sim = pw_sim.masked_fill(collision, 0)
      exp_pw_sim = exp_pw_sim.masked_fill(collision, 0)

    sampled = torch.multinomial(exp_pw_sim.data, 1)
    if if_norm:
      norm_exp_pw_sim = exp_pw_sim / exp_pw_sim.sum(1, keepdim=True)
      sim = torch.gather(norm_exp_pw_sim, 1, sampled).squeeze(1)
    else:
      sim = torch.gather(pw_sim, 1, sampled).squeeze(1)

    return sampled, sim

  def forward(self, qt, good_cand, bad_cand, adversarial_sample=False):
    good_cand_repr = self.cand_encoding(good_cand)
    bad_cand_repr = self.cand_encoding(bad_cand)

    qt_repr = self.query_encoding(qt)

    good_sim = self.scoring(qt_repr, good_cand_repr)
    if not adversarial_sample:
      bad_sim = self.scoring(qt_repr, bad_cand_repr)
    else:
      bad_sim = self.cross_scoring(qt_repr, bad_cand_repr)

    # loss = (self.margin - good_sim + bad_sim).clamp(min=1e-6).mean()
    loss = self.loss(torch.cat([good_sim, bad_sim]),
                     torch.cat([torch.ones_like(good_sim), torch.zeros_like(bad_sim)]))

    return loss, good_sim, bad_sim

  def cc_with_qc(self, qt, good_cand, bad_cand, adversarial_sample=False):
    good_cand_repr = self.cand_encoding(good_cand)
    bad_cand_repr = self.cand_encoding(bad_cand)

    qt_repr = self.cand_encoding(qt)

    good_sim = self.scoring(qt_repr, good_cand_repr)
    if not adversarial_sample:
      bad_sim = self.scoring(qt_repr, bad_cand_repr)
    else:
      bad_sim = self.cross_scoring(qt_repr, bad_cand_repr)

    loss = (self.margin - good_sim + bad_sim).clamp(min=1e-6).mean()
    return loss, good_sim, bad_sim


class CCModel(nn.Module):
  def __init__(self, config):
    super(CCModel, self).__init__()
    self.name = "default"

    self.conf = config
    self.margin = config['margin']

    if self.conf['code_encoder'] == "bilstm":
      self.query_encoder = SeqEncoder(config['code_n_words'], config['emb_size'], config['lstm_dims'], self.conf)
      self.cand_encoder = SeqEncoder(config['code_n_words'], config['emb_size'], config['lstm_dims'], self.conf)
    else:
      self.query_encoder = BOWEncoder(config['code_n_words'], 2 * config['lstm_dims'], self.conf)  # MLP
      self.cand_encoder = BOWEncoder(config['code_n_words'], 2 * config['lstm_dims'], self.conf)  # MLP

  def cand_encoding(self, cand):
    cand_repr = self.cand_encoder(cand)
    return cand_repr

  def query_encoding(self, code):
    code_repr = self.query_encoder(code)
    return code_repr

  def scoring(self, code_repr, cand_repr):
    sim = F.cosine_similarity(code_repr, cand_repr)
    return sim

  def _cross_scoring(self, code_repr, cand_repr):

    # Get pairwise cosine similarity
    code_repr_norm = code_repr / code_repr.norm(dim=1)[:, None]
    cand_repr_norm = cand_repr / cand_repr.norm(dim=1)[:, None]
    pw_sim = torch.mm(code_repr_norm, cand_repr_norm.transpose(0, 1))

    # Sample probability (not sum to one)
    temperature = 1
    exp_pw_sim = torch.exp(pw_sim.data / temperature)
    sampled = []
    for x in range(exp_pw_sim.size(0)):
      y = torch.multinomial(exp_pw_sim[x], 1)
      sampled.append(y)

    # Choose sampled as negative instance
    sim = torch.gather(pw_sim, 1, torch.stack(sampled)).squeeze(1)
    return sim

  def sample_cand(self, code, cand, collision=None, if_norm=False):
    code_repr = self.query_encoding(code)
    cand_repr = self.cand_encoding(cand)

    # Get pairwise cosine similarity
    code_repr_norm = code_repr / code_repr.norm(dim=1)[:, None]
    cand_repr_norm = cand_repr / cand_repr.norm(dim=1)[:, None]
    pw_sim = torch.mm(code_repr_norm, cand_repr_norm.transpose(0, 1))

    # Sample probability (not sum to one)
    temperature = 1
    exp_pw_sim = torch.exp(pw_sim / temperature)

    # sample
    if collision is not None:
      # pw_sim = pw_sim.masked_fill(collision, 0)
      exp_pw_sim = exp_pw_sim.masked_fill(collision, 0)

    sampled = torch.multinomial(exp_pw_sim.data, 1)
    if if_norm:
      norm_exp_pw_sim = exp_pw_sim / exp_pw_sim.sum(1, keepdim=True)
      sim = torch.gather(norm_exp_pw_sim, 1, sampled).squeeze(1)
    else:
      sim = torch.gather(pw_sim, 1, sampled).squeeze(1)

    return sampled, sim

  def forward(self, code, good_cand, bad_cand, adversarial_sample=False):
    good_cand_repr = self.cand_encoding(good_cand)
    bad_cand_repr = self.cand_encoding(bad_cand)

    code_repr = self.query_encoding(code)

    good_sim = self.scoring(code_repr, good_cand_repr)
    if not adversarial_sample:
      bad_sim = self.scoring(code_repr, bad_cand_repr)
    else:
      bad_sim = self._cross_scoring(code_repr, bad_cand_repr)

    loss = (self.margin - good_sim + bad_sim).clamp(min=1e-6).mean()
    return loss, good_sim, bad_sim


class QQModel(nn.Module):
  def __init__(self, config):
    super(QQModel, self).__init__()
    self.name = "default"

    self.conf = config
    self.margin = config['margin']
    self.query_encoder = SeqEncoder(config['qt_n_words'], config['emb_size'], config['lstm_dims'], self.conf)
    self.cand_encoder = SeqEncoder(config['qt_n_words'], config['emb_size'], config['lstm_dims'], self.conf)

  def cand_encoding(self, cand):
    cand_repr = self.cand_encoder(cand)
    return cand_repr

  def query_encoding(self, qt):
    qt_repr = self.query_encoder(qt)
    return qt_repr

  def scoring(self, qt_repr, cand_repr):
    sim = F.cosine_similarity(qt_repr, cand_repr)
    return sim

  def cross_scoring(self, qt_repr, cand_repr):

    # Get pairwise cosine similarity
    qt_repr_norm = qt_repr / qt_repr.norm(dim=1)[:, None]
    cand_repr_norm = cand_repr / cand_repr.norm(dim=1)[:, None]
    pw_sim = torch.mm(qt_repr_norm, cand_repr_norm.transpose(0, 1))

    # Sample probability (not sum to one)
    temperature = 1
    exp_pw_sim = torch.exp(pw_sim.data / temperature)
    sampled = []
    for x in range(exp_pw_sim.size(0)):
      y = torch.multinomial(exp_pw_sim[x], 1)
      sampled.append(y)

    # Choose sampled as negative instance
    sim = torch.gather(pw_sim, 1, torch.stack(sampled)).squeeze(1)
    return sim

  def forward(self, qt, good_cand, bad_cand, adversarial_sample=False):
    good_cand_repr = self.cand_encoding(good_cand)
    bad_cand_repr = self.cand_encoding(bad_cand)

    qt_repr = self.query_encoding(qt)

    good_sim = self.scoring(qt_repr, good_cand_repr)
    if not adversarial_sample:
      bad_sim = self.scoring(qt_repr, bad_cand_repr)
    else:
      bad_sim = self.cross_scoring(qt_repr, bad_cand_repr)

    loss = (self.margin - good_sim + bad_sim).clamp(min=1e-6).mean()
    return loss, good_sim, bad_sim