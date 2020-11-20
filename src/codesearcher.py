import os
import random
import numpy as np
import math
import argparse
import pickle
import pdb
# import logging

import torch
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils import *
from configs import get_config
from data import load_qc_data, load_cc_data, load_qq_data, my_collate, load_qc_data_codenn

ADD_ATTN = True
if not ADD_ATTN:
    from models import *
else:
    from models_w_attn import *

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(message)s")

# Random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)



class CodeSearcher:
    def __init__(self, conf):
        self.conf = conf

    ##########################
    # Model loading / saving #
    ##########################
    def save_model(self, model):
        if not os.path.exists(self.conf['model_directory']):
            os.makedirs(self.conf['model_directory'])
        torch.save(model.state_dict(), os.path.join(self.conf['model_directory'], 'best_model.ckpt'))

    def load_model(self, model):
        assert os.path.exists(os.path.join(self.conf['model_directory'], 'best_model.ckpt')), \
            'Weights for saved model not found'
        model.load_state_dict(torch.load(os.path.join(self.conf['model_directory'], 'best_model.ckpt')))

    def init_qq_with_qc(self, model, qc_model_path):
        assert os.path.exists(os.path.join(qc_model_path, 'best_model.ckpt')), 'Weights for saved model not found'
        statedict = torch.load(os.path.join(qc_model_path, 'best_model.ckpt'))
        statedict["cand_encoder.embedding.weight"] = statedict["query_encoder.embedding.weight"]
        statedict["cand_encoder.lstm.weight_ih_l0"] = statedict["query_encoder.lstm.weight_ih_l0"]
        statedict["cand_encoder.lstm.weight_hh_l0"] = statedict["query_encoder.lstm.weight_hh_l0"]
        statedict["cand_encoder.lstm.bias_ih_l0"] = statedict["query_encoder.lstm.bias_ih_l0"]
        statedict["cand_encoder.lstm.bias_hh_l0"] = statedict["query_encoder.lstm.bias_hh_l0"]
        statedict["cand_encoder.lstm.weight_ih_l0_reverse"] = statedict["query_encoder.lstm.weight_ih_l0_reverse"]
        statedict["cand_encoder.lstm.weight_hh_l0_reverse"] = statedict["query_encoder.lstm.weight_hh_l0_reverse"]
        statedict["cand_encoder.lstm.bias_ih_l0_reverse"] = statedict["query_encoder.lstm.bias_ih_l0_reverse"]
        statedict["cand_encoder.lstm.bias_hh_l0_reverse"] = statedict["query_encoder.lstm.bias_hh_l0_reverse"]
        model.load_state_dict(statedict)

    def load_other_model(self, model, model_path):
        assert os.path.exists(os.path.join(model_path, 'best_model.ckpt')), 'Weights for saved model not found'
        model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.ckpt')))

    ############
    # Training #
    ############
    def train(self, model, writer):
        """
        Trains an initialized model
        :param model: Initialized model
        :param writer: SummaryWriter from tensorboard
        :return: None
        """
        log_every = self.conf['log_every']
        valid_every = self.conf['valid_every']
        batch_size = self.conf['batch_size']
        nb_epoch = self.conf['nb_epoch']
        max_patience = self.conf['patience']

        # Load data
        if self.conf['negadv'] > 0:
            neg_adv_dict = {"train": "one", "dev": "all", "test": "all"}
        else:
            neg_adv_dict = {"train": "", "dev": "", "test": ""}

        if self.conf["model"] == "qc":
            data = load_qc_data(test=False, lang=self.conf["lang"], load_adv_neg=neg_adv_dict,
                                train_percentage=self.conf["train_percentage"])
        elif self.conf["model"] == "cc":
            data = load_cc_data(test=False, lang=self.conf["lang"], load_adv_neg=neg_adv_dict,
                                train_percentage=self.conf["train_percentage"])
        elif self.conf["model"] == "qq":
            data = load_qq_data(test=False, lang=self.conf["lang"])
        else:
            raise ValueError("Unknown model: %s" % self.conf["model"])
        train_loader = torch.utils.data.DataLoader(dataset=data["train"], batch_size=batch_size, shuffle=True,
                                                   drop_last=False, num_workers=1, collate_fn=my_collate)

        # MRR for the Best Saved model, if reload > 0, else -1
        if self.conf['reload'] > 0:
            _, max_mrr, _, _ = self.eval(model, 50, data["dev"], given_candidates=self.conf["negadv"] > 0)
        else:
            max_mrr = -1

        patience = 0
        for epoch in range(self.conf['reload'] + 1, nb_epoch):
            itr = 1
            losses, all_losses = [], []

            model = model.train()

            for batch in train_loader:
                if self.conf["negadv"] > 0:
                    qts, good_cands, bad_cands = batch["query"], batch["pos"], batch["adv_neg"]
                else:
                    qts, good_cands, bad_cands = batch["query"], batch["pos"], batch["neg"]
                qts, good_cands, bad_cands = gVar(qts), gVar(good_cands), gVar(bad_cands)

                loss, good_scores, bad_scores = model(qts, good_cands, bad_cands)

                losses.append(loss.item())
                all_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itr % log_every == 0:
                    print('epo:[%d/%d]  itr:%d  Loss=%.5f' % (epoch, nb_epoch, itr, np.mean(losses)))
                    losses = []
                itr = itr + 1

            # Write to tensorboard
            if writer is not None:
                writer.add_scalar("Train/%s_loss" % self.conf["model"].upper(), np.mean(all_losses), epoch)

            if epoch % valid_every == 0:
                print("validating..")
                acc1, mrr, map, ndcg = self.eval(model, 50, data["dev"], given_candidates=self.conf["negadv"] > 0)
                if mrr > max_mrr:
                    self.save_model(model)
                    patience = 0
                    print("Model improved. Saved model at %d epoch" % epoch)
                    max_mrr = mrr
                else:
                    print("Model didn't improve for ", patience + 1, " epochs")
                    patience += 1
                if writer is not None:
                    writer.add_scalar('Valid/%s_MRR' % conf['model'].upper(), mrr, epoch)
                    writer.add_scalar('Valid/%s_MAP' % conf['model'].upper(), map, epoch)
                    writer.add_scalar('Valid/%s_nDCG' % conf['model'].upper(), ndcg, epoch)

            if patience >= max_patience:
                print("Patience Limit Reached. Stopping Training")
                break

    ####################################
    # Training with adversarial sample #
    ####################################
    def train_w_adversarial_sample(self, model, writer):
        """
        Trains an initialized model, using adversarially sampled negative answers
        :param model: Initialized model
        :param writer: SummaryWriter from tensorboard
        :return: None
        """
        log_every = self.conf['log_every']
        valid_every = self.conf['valid_every']
        batch_size = self.conf['batch_size']
        nb_epoch = self.conf['nb_epoch']
        max_patience = self.conf['patience']

        # Load data
        if self.conf["model"] == "qc":
            data = load_qc_data(test=False, lang=self.conf["lang"])
        elif self.conf["model"] == "cc":
            data = load_cc_data(test=False, lang=self.conf["lang"])
        elif self.conf["model"] == "qq":
            data = load_qq_data(test=False, lang=self.conf["lang"])
        else:
            raise ValueError("Unknown model: %s" % self.conf["model"])
        train_loader = torch.utils.data.DataLoader(dataset=data["train"], batch_size=batch_size, shuffle=True,
                                                   drop_last=False, num_workers=1, collate_fn=my_collate)

        # MRR for the Best Saved model, if reload > 0, else -1
        if self.conf['reload'] > 0:
            _, max_mrr, _, _ = self.eval(model, 50, data["dev"], given_candidates=self.conf["negadv"] > 0)

        patience = 0
        for epoch in range(self.conf['reload'] + 1, nb_epoch):
            itr = 1
            losses, all_losses = [], []

            model = model.train()

            for batch in train_loader:
                qts, good_cands, bad_cands = batch["query"], batch["pos"], batch["neg"]
                qts, good_cands, bad_cands = gVar(qts), gVar(good_cands), gVar(bad_cands)

                loss, good_scores, bad_scores = model(qts, good_cands, bad_cands, adversarial_sample=True)

                losses.append(loss.item())
                all_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itr % log_every == 0:
                    print('epo:[%d/%d] itr:%d Loss=%.5f' % (epoch, nb_epoch, itr, np.mean(losses)))
                    losses = []
                itr = itr + 1

            # Write to tensorboard
            if writer is not None:
                writer.add_scalar("Train/%s_loss" % self.conf["model"].upper(), np.mean(all_losses), epoch)

            if epoch % valid_every == 0:
                print("validating..")
                acc1, mrr, map, ndcg = self.eval(model, 50, data["dev"], given_candidates=self.conf["negadv"] > 0)
                if mrr > max_mrr:
                    self.save_model(model)
                    patience = 0
                    print("Model improved. Saved model at %d epoch" % epoch)
                    max_mrr = mrr
                else:
                    print("Model didn't improve for ", patience + 1, " epochs")
                    patience += 1
                if writer is not None:
                    writer.add_scalar('Valid/%s_MRR' % conf['model'].upper(), mrr, epoch)
                    writer.add_scalar('Valid/%s_MAP' % conf['model'].upper(), map, epoch)
                    writer.add_scalar('Valid/%s_nDCG' % conf['model'].upper(), ndcg, epoch)

            if patience >= max_patience:
                print("Patience Limit Reached. Stopping Training")
                break

    #######################
    # Evaluation on StaQC #
    #######################
    def eval(self, model, poolsize, dataset, bool_collect=False, f_qual=None, given_candidates=False):
        if given_candidates:
            acc, mrr, map, ndcg = self._eval_w_given_candidates(
                model, poolsize, dataset, bool_collect=bool_collect, f_qual=f_qual)
        else:
            acc, mrr, map, ndcg = self._eval(
                model, poolsize, dataset, bool_collect=bool_collect, f_qual=f_qual)
        return acc, mrr, map, ndcg

    def _eval(self, model, poolsize, dataset, bool_collect=False, f_qual=None):
        """
        simple validation in a code pool.
        :param model: Trained Model
        :param poolsize: poolsize - size of the code pool, if -1, load the whole test set
        :param dataset: which dataset to evaluate on
        :return: Accuracy, MRR, MAP, nDCG
        """
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=poolsize, shuffle=False,
                                                  drop_last=False, num_workers=1, collate_fn=my_collate)

        model = model.eval()
        accs, mrrs, maps, ndcgs = [], [], [], []

        sims_collection = []
        for batch in data_loader:
            qts, cands, all_pos = batch["query"], batch["pos"], batch["all_pos"]
            qts, cands = gVar(qts), gVar(cands)
            qts_repr = model.query_encoding(qts)
            cands_repr = model.cand_encoding(cands)

            all_pos_number = [x.size(0) for x in all_pos]
            all_pos_stack = torch.cat(all_pos)
            all_pos_stack = gVar(all_pos_stack)
            all_pos_repr = model.cand_encoding(all_pos_stack)
            all_pos_repr = torch.split(all_pos_repr, all_pos_number)

            _poolsize = len(qts) if bool_collect else min(poolsize, len(qts))  # true poolsize
            for i in range(_poolsize):
                cands_repr_w_all_pos = torch.cat([all_pos_repr[i], cands_repr[:i], cands_repr[i+1:]])
                if not ADD_ATTN:
                    _qts_repr = qts_repr[i].expand(cands_repr_w_all_pos.size(0), -1)
                else:
                    _qts_repr = qts_repr[i].expand(cands_repr_w_all_pos.size(0), -1, -1)
                scores = model.scoring(_qts_repr, cands_repr_w_all_pos).data.cpu().numpy()
                neg_scores = np.negative(scores)
                predict = np.argsort(neg_scores)
                predict = [int(k) for k in predict]
                real = list(range(all_pos_repr[i].size(0)))  # index of positive sample
                accs.append(ACC(real, predict))
                mrrs.append(MRR(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))
                sims_collection.append(scores)

                if f_qual is not None:
                    dataset.print_qualitative(file=f_qual, batch=batch, scores=scores, index=i,
                                              mrr=mrrs[-1], map=maps[-1], ndcg=ndcgs[-1])

        if bool_collect:
            save_path = os.path.join(self.conf['model_directory'], "collect_sims_staqc_%s.pkl" % dataset.data_name)
            print("Save collection to %s" % save_path)
            pickle.dump(sims_collection, open(save_path, "wb"))

        print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            len(accs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)

    def _eval_w_given_candidates(self, model, poolsize, dataset, bool_collect=False, f_qual=None):
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=poolsize, shuffle=False,
                                                  drop_last=False, num_workers=1, collate_fn=my_collate)

        model = model.eval()
        accs, mrrs, maps, ndcgs = [], [], [], []

        sims_collection = []
        for batch in data_loader:
            qts, all_pos, adv_neg = batch["query"], batch["all_pos"], batch["adv_neg"]
            qts, adv_neg = gVar(qts), gVar(adv_neg)
            qts_repr = model.query_encoding(qts)
            adv_neg_repr = model.cand_encoding(adv_neg.view(-1, adv_neg.size(-1))).reshape(
                adv_neg.size(0), adv_neg.size(1), -1)

            all_pos_number = [x.size(0) for x in all_pos]
            all_pos_stack = torch.cat(all_pos)
            all_pos_stack = gVar(all_pos_stack)
            all_pos_repr = model.cand_encoding(all_pos_stack)
            all_pos_repr = torch.split(all_pos_repr, all_pos_number)

            _poolsize = len(qts) if bool_collect else min(poolsize, len(qts))  # true poolsize
            for i in range(_poolsize):
                cands_repr_w_adv_neg = torch.cat([all_pos_repr[i], adv_neg_repr[i][: _poolsize - 1]])
                _qts_repr = qts_repr[i].expand(cands_repr_w_adv_neg.size(0), -1)
                scores = model.scoring(_qts_repr, cands_repr_w_adv_neg).data.cpu().numpy()
                neg_scores = np.negative(scores)
                predict = np.argsort(neg_scores)
                predict = [int(k) for k in predict]
                real = list(range(all_pos_repr[i].size(0)))  # index of positive sample
                accs.append(ACC(real, predict))
                mrrs.append(MRR(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))
                sims_collection.append(scores)

                if f_qual is not None:
                    self._print_qualitative(f_qual=f_qual, qvocab=dataset.qvocab, cvocab=dataset.cvocab,
                                            query=qts[i].tolist(), pos=all_pos[i].tolist(),
                                            neg=(adv_neg[i][: _poolsize - 1].tolist()),
                                            labels=real, preds=scores,
                                            MRR=mrrs[-1], MAP=maps[-1], nDCG=ndcgs[-1])

        if bool_collect:
            save_path = os.path.join(self.conf['model_directory'], "collect_sims_staqc_%s.pkl" % dataset.data_name)
            print("Save collection to %s" % save_path)
            pickle.dump(sims_collection, open(save_path, "wb"))

        print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            len(accs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)

    #####################################################
    # Evaluation on StaQC: rank CC pairs using QC model #
    #####################################################
    def eval_cc_data_on_qc_model(self, model, poolsize, dataset, bool_collect=False):
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=poolsize, shuffle=False,
                                                  drop_last=False, num_workers=1, collate_fn=my_collate)

        model = model.eval()
        accs, mrrs, maps, ndcgs = [], [], [], []

        sims_collection = []
        for batch in data_loader:

            qts, cands, all_pos = batch["query"], batch["pos"], batch["all_pos"]
            qts, cands = gVar(qts), gVar(cands)
            qts_repr = model.cand_encoding(qts)  # NOTE: queries in this special case is code
            cands_repr = model.cand_encoding(cands)

            all_pos_number = [x.size(0) for x in all_pos]
            all_pos_stack = torch.cat(all_pos)
            all_pos_stack = gVar(all_pos_stack)
            all_pos_repr = model.cand_encoding(all_pos_stack)
            all_pos_repr = torch.split(all_pos_repr, all_pos_number)

            _poolsize = len(qts) if bool_collect else min(poolsize, len(qts))  # true poolsize
            for i in range(_poolsize):
                cands_repr_w_all_pos = torch.cat([all_pos_repr[i], cands_repr[:i], cands_repr[i+1:]])
                _qts_repr = qts_repr[i].expand(cands_repr_w_all_pos.size(0), -1)
                scores = model.scoring(_qts_repr, cands_repr_w_all_pos).data.cpu().numpy()
                neg_scores = np.negative(scores)
                predict = np.argsort(neg_scores)
                predict = [int(k) for k in predict]
                real = list(range(all_pos_repr[i].size(0)))  # index of positive sample
                accs.append(ACC(real, predict))
                mrrs.append(MRR(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))
                sims_collection.append(scores)

        if bool_collect:
            save_path = os.path.join(self.conf['model_directory'], "collect_sims_staqc_%s.pkl" % dataset.data_name)
            print("Save collection to %s" % save_path)
            pickle.dump(sims_collection, open(save_path, "wb"))

        print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            len(accs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)

    ########################
    # Evaluation on CodeNN #
    ########################
    def eval_codenn(self, model, poolsize, dataset, f_qual=None, bool_collect=False):
        """
        simple validation in a code pool.
        :param model: Trained Model
        :param poolsize: poolsize - size of the code pool, if -1, load the whole test set
        :param dataset: which dataset to evaluate on
        :return: Accuracy, MRR, MAP, nDCG
        """
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=poolsize, shuffle=False,
                                                  num_workers=1, collate_fn=my_collate)
        model = model.eval()

        sims_collection = []
        accs, mrrs, maps, ndcgs = [], [], [], []

        for batch in data_loader:
            qts, cands = batch["query"], batch["pos"]
            cands = gVar(cands)
            cands_repr = model.cand_encoding(cands)

            if isinstance(qts, list):
                assert len(qts) == 3
                qts = [gVar(qts_i) for qts_i in qts]
            else:
                qts = [gVar(qts)]

            sims_per_qts = []
            for qts_i in qts:
                qt_repr = model.query_encoding(qts_i)

                sims = model.scoring(qt_repr, cands_repr).data.cpu().numpy()
                negsims = np.negative(sims)
                predict = np.argsort(negsims)
                predict = [int(k) for k in predict]
                real = [0]  # index of the positive sample

                # save
                sims_per_qts.append(sims)

                mrrs.append(MRR(real, predict))
                accs.append(ACC(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))

            sims_collection.append(sims_per_qts)

        if bool_collect:
            save_path = os.path.join(self.conf['model_directory'], "collect_sims_codenn_%s.pkl" % dataset.data_name)
            print("Save collection to %s" % save_path)
            pickle.dump(sims_collection, open(save_path, "wb"))

        print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
                len(mrrs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search Model")
    parser.add_argument("-M", "--model", choices=["qc", "qq", "cc"], required=True,
                        help="Which model to train: QC, QQ or CC.")
    parser.add_argument("-m", "--mode", choices=["train", "eval", "collect"],
                        default='train',
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluates models on a test set;"
                             " the `collect` mode collects model scores;",
                        required=True)
    parser.add_argument("--self_adversarial", type=int, default=-1, required=True,
                        help="Should I train the model with self adversarial sampling, yes if self-adversarial>0?")
    parser.add_argument("--reload", type=int, default=-1, help="Should I reload saved model, yes if reload>0?",
                        required=True)
    parser.add_argument("--reload_path", type=str, default="", help="Enclosing folder of the to-be-reloaded model.")
    parser.add_argument("--save_qual", type=int, default=-1, help="When eval, whether save qual results.")

    # model setup
    parser.add_argument("--dropout", type=float, default=0.0, help="What is the dropout?", required=True)
    parser.add_argument("--emb_size", type=int, default=100, help="What is the embedding size?", required=True)
    parser.add_argument("--lstm_dims", type=int, default=200, help="What is the lstm dimension?", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="What is the batch size?", required=True)
    parser.add_argument("--temp", type=str, default="", help="Name of a temporary test (not affect saved model etc).")
    parser.add_argument("--init_qq_w_qc", type=int, default=0, help="Init Q-encoder in CC model with that in QC model.")

    # dataset setup
    parser.add_argument("--lang", type=str, default="SQL", help="Which language dataset to use.")
    parser.add_argument("--qn_mode", type=str, default="sl",
                        choices=["sl", "rl_bleu", "rl_mrr", "codenn_gen"], help="Data set to use.")
    parser.add_argument('--pool_size', type=int, default=50, help="candidate pool size for evaluation")
    parser.add_argument("--train_percentage", type=float, default=1.0,
                        help="The percentage of data used to train the model.")

    # optimization
    parser.add_argument("--lr", type=float, default=0.001, help="What is the learning rate?")
    parser.add_argument("--margin", type=float, default=0.05, help="Margin for pairwise loss.")
    parser.add_argument("--optimizer", type=str,
                        choices=["adam", "adagrad", "sgd", "rmsprop", "asgd", "adadelta"],
                        default="adam", help="Which optimizer to use?")

    # evaluation setup
    parser.add_argument("--negadv", type=int, default=0, help="If use TF-IDF adversarial candidates.")
    parser.add_argument("--codenn", type=int, default=0, help="If use CodeNN dataset for evaluation.")

    return parser.parse_args()


def create_model_name_string(c):
    string1 = 'qtlen_{}_codelen_{}_qtnwords_{}_codenwords_{}_batch_{}_optimizer_{}_lr_{}'. \
        format(c['qt_len'], c['code_len'], c['qt_n_words'], c['code_n_words'],
               c['batch_size'], c['optimizer'], str(c['lr'])[2:] if c['lr'] < 1.0 else str(c['lr']))
    string2 = '_embsize_{}_lstmdims_{}_bowdropout_{}_seqencdropout_{}'. \
        format(c['emb_size'], c['lstm_dims'], str(c['bow_dropout'])[2:], str(c['seqenc_dropout'])[2:])
    string3 = '_codeenc_{}'.format(c['code_encoder'])
    string = string1 + string2 + string3

    return string


if __name__ == '__main__':
    args = parse_args()
    conf = get_config(args)

    # hyper-params
    conf['model'] = args.model
    conf['bow_dropout'] = args.dropout
    conf['seqenc_dropout'] = args.dropout
    conf['emb_size'] = args.emb_size
    conf['lstm_dims'] = args.lstm_dims
    conf['batch_size'] = args.batch_size
    conf['lr'] = args.lr
    conf['reload'] = args.reload
    conf['reload_path'] = args.reload_path
    conf['optimizer'] = args.optimizer
    conf['negadv'] = args.negadv
    conf['codenn'] = args.codenn
    conf['train_percentage'] = args.train_percentage
    conf['lang'] = args.lang
    # conf['nb_epoch'] = 500
    conf['patience'] = 100

    if conf['reload'] <= 0 and args.mode in {'eval', 'collect'}:
        print("For eval/collect mode, please give reload=1. If you looking to train the model, change the mode to train. "
              "\n Note: Train overrides previously saved model, if it had exactly the same parameters")
    else:
        if args.mode == 'train':
            print("Warning: Train overrides previously saved model, if it had exactly the same parameters")
            print("If retraining the model from previous check point, set reload >0 to start training from previous "
                  "checkpoint")
        print(" Code encoder : ", conf['code_encoder'])
        print(" Dropout : ", conf['seqenc_dropout'])
        print(" Embedding size : ", conf['emb_size'])
        print(" LSTM hidden dimension : ", conf['lstm_dims'])
        print(" Margin: ", conf['margin'])
        print(" Optimizer: ", conf['optimizer'])

        # Creating unique model string based on parameters defined. Helps differentiate between different runs of model
        model_string = create_model_name_string(conf)

        model_dir_str = "%s" % args.model.upper()
        model_dir_str += "_selfadv" if args.self_adversarial > 0 else ""
        model_dir_str += "_%s" % args.temp if args.temp else ""

        conf['model_directory'] = os.path.join(conf['ckptdir'], '%s' % model_dir_str, model_string)
        conf['reload_model_directory'] = os.path.join(conf['reload_path'], model_string)

        if not os.path.exists(conf['model_directory']):
            os.makedirs(conf['model_directory'])
        print(" Model Directory : ")
        print(conf['model_directory'])

        if args.mode == 'train':
            conf['summary_directory'] = os.path.join(conf['sumdir'], model_dir_str, model_string)
            print(" Summary Directory : " + conf['summary_directory'])
            writer = SummaryWriter(conf['summary_directory'])

        searcher = CodeSearcher(conf)

        #####################
        # Define model ######
        #####################
        print('Building %s Model' % args.model.upper())
        if args.model == "qc":
            model = QCModel(conf)
        elif args.model == "cc":
            model = CCModel(conf)
        elif args.model == "qq":
            model = QQModel(conf)
        print("model: ", model)

        if conf['reload'] > 0:
            if args.mode in {'eval', 'collect'}:
                print("Reloading saved model for evaluating/collecting results")
            else:
                print("Reloading saved model for Re-training")
            if args.init_qq_w_qc:
                qc_reload_path = os.path.join(args.reload_path, model_string)
                searcher.init_qq_with_qc(model, conf['reload_model_directory'])
            else:
                if len(conf["reload_path"]) > 0:
                    searcher.load_other_model(model, conf['reload_model_directory'])
                else:
                    searcher.load_model(model)

        if torch.cuda.is_available():
            print('using GPU')
            model = model.cuda()
        else:
            print('using CPU')

        print("\nParameter requires_grad state: ")
        for name, param in model.named_parameters():
            print name, param.requires_grad
        print("")

        if conf['optimizer'] == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=conf['lr'])
            print("Recommend lr 0.01 for AdaGrad while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=conf['lr'], momentum=0.9)
            print("Recommend lr 0.1 for SGD (momentum 0.9) while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=conf['lr'])
            print("Recommend lr 0.01 for RMSprop while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'asgd':
            optimizer = optim.ASGD(model.parameters(), lr=conf['lr'])
            print("Recommend lr 0.01 for ASGD while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=conf['lr'])
            print("Recommend lr 1.00 for Adadelta while using %.5f." % conf['lr'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=conf['lr'])
            print("Recommend lr 0.001 for Adam while using %.5f." % conf['lr'])

        if args.mode == 'train':
            if not args.self_adversarial > 0:
                print('Training Model')
                searcher.train(model, writer=writer)
            else:
                print('Training Model with Adversarial Sampling')
                searcher.train_w_adversarial_sample(model, writer=writer)

        elif args.mode == 'eval':
            print('Evaluating Model')
            print('Pool size = %d' % args.pool_size)
            if conf["codenn"] > 0:
                if conf["model"] == "qc":
                    data = load_qc_data_codenn(train=False)
                    f_qual_dev, f_qual_test = None, None
                    if args.save_qual > 0:
                        f_qual_dev = open(os.path.join(conf['model_directory'], "qual_dev.txt"), "w")
                        f_qual_test = open(os.path.join(conf['model_directory'], "qual_test.txt"), "w")

                    searcher.eval_codenn(model, args.pool_size, data["dev"], f_qual=f_qual_dev)
                    searcher.eval_codenn(model, args.pool_size, data["test"], f_qual=f_qual_test)

                    if f_qual_dev is not None: f_qual_dev.close()
                    if f_qual_test is not None: f_qual_test.close()

            else:
                if conf['negadv'] > 0:
                    neg_adv_dict = {"train": "one", "dev": "all", "test": "all"}
                else:
                    neg_adv_dict = {"train": "", "dev": "", "test": ""}

                if conf["model"] == "qc":
                    data = load_qc_data(train=False, lang=args.lang, load_adv_neg=neg_adv_dict)
                elif conf["model"] == "cc":
                    data = load_cc_data(train=False, lang=args.lang, load_adv_neg=neg_adv_dict)
                elif conf["model"] == "qq":
                    data = load_qq_data(train=False, lang=args.lang)
                else:
                    raise ValueError("Unknown model: %s" % self.conf["model"])

                f_qual_dev, f_qual_test = None, None
                if args.save_qual > 0:
                    f_qual_dev = open(os.path.join(conf['model_directory'], "qual_dev.txt"), "w")
                    f_qual_test = open(os.path.join(conf['model_directory'], "qual_test.txt"), "w")

                searcher.eval(model, args.pool_size, data["dev"], f_qual=f_qual_dev,
                              given_candidates=conf["negadv"] > 0)
                searcher.eval(model, args.pool_size, data["test"], f_qual=f_qual_test,
                              given_candidates=conf["negadv"] > 0)

                if f_qual_dev is not None: f_qual_dev.close()
                if f_qual_test is not None: f_qual_test.close()

        elif args.mode == 'collect':
            raise NotImplementedError()
            print('Collecting outputs...')
            if conf["model"] == "qc":
                data = load_qc_data(train=False)
            elif conf["model"] == "cc":
                data = load_cc_data(train=False)
            elif conf["model"] == "qq":
                data = load_qq_data(train=False)
            else:
                raise ValueError("Unknown model: %s" % self.conf["model"])
            searcher.eval(model, 50, data["dev"], bool_collect=True)
            searcher.eval(model, 50, data["test"], bool_collect=True)

        else:
            print("Please provide a Valid argument for mode - train/eval")

