from __future__ import print_function
import os
import random
import numpy as np
import math
import argparse
import pickle
import pdb
import traceback
import sys
# import logging

import torch
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils import *
from configs import get_config
from data import load_qc_data, load_cc_data, load_qq_data, my_collate
from models import *

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(message)s")

# Random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class JointSearcher:
    def __init__(self, conf):
        self.conf = conf

    ##########################
    # Model loading / saving #
    ##########################
    def save_model(self, model):
        qc_model, cc_model, qq_model = model["qc"], model["cc"], model["qq"]
        if qc_model is not None:
            if not os.path.exists(self.conf['model_directory']['qc']):
                os.makedirs(self.conf['model_directory']['qc'])
            torch.save(qc_model.state_dict(), os.path.join(self.conf['model_directory']['qc'], 'best_model.ckpt'))
        if cc_model is not None:
            if not os.path.exists(self.conf['model_directory']['cc']):
                os.makedirs(self.conf['model_directory']['cc'])
            torch.save(cc_model.state_dict(), os.path.join(self.conf['model_directory']['cc'], 'best_model.ckpt'))
        if qq_model is not None:
            if not os.path.exists(self.conf['model_directory']['qq']):
                os.makedirs(self.conf['model_directory']['qq'])
            torch.save(qq_model.state_dict(), os.path.join(self.conf['model_directory']['qq'], 'best_model.ckpt'))

    def load_model(self, model):
        qc_model, cc_model, qq_model = model["qc"], model["cc"], model["qq"]
        if qc_model is not None:
            assert os.path.exists(os.path.join(self.conf['model_directory']['qc'], 'best_model.ckpt')), \
                'Weights for saved qc model not found'
            qc_model.load_state_dict(torch.load(os.path.join(self.conf['model_directory']['qc'], 'best_model.ckpt')))
        if cc_model is not None:
            assert os.path.exists(os.path.join(self.conf['model_directory']['cc'], 'best_model.ckpt')), \
                'Weights for saved cc model not found'
            cc_model.load_state_dict(torch.load(os.path.join(self.conf['model_directory']['cc'], 'best_model.ckpt')))
        # if qq_model is not None:
        #     assert os.path.exists(os.path.join(self.conf['model_directory']['qq'], 'best_model.ckpt')), \
        #         'Weights for saved qq model not found'
        #     qq_model.load_state_dict(torch.load(os.path.join(self.conf['model_directory']['qq'], 'best_model.ckpt')))

    def load_other_model(self, model, model_path):
        qc_model, cc_model, qq_model = model["qc"], model["cc"], model["qq"]
        if qc_model is not None:
            assert os.path.exists(os.path.join(model_path['qc'], 'best_model.ckpt')), \
                'Weights for saved qc model not found'
            qc_model.load_state_dict(torch.load(os.path.join(model_path['qc'], 'best_model.ckpt')))
        if cc_model is not None:
            assert os.path.exists(os.path.join(model_path['cc'], 'best_model.ckpt')), \
                'Weights for saved cc model not found'
            cc_model.load_state_dict(torch.load(os.path.join(model_path['cc'], 'best_model.ckpt')))
        # if qq_model is not None:
        #     assert os.path.exists(os.path.join(model_path['qq'], 'best_model.ckpt')), \
        #         'Weights for saved qq model not found'
        #     qq_model.load_state_dict(torch.load(os.path.join(model_path['qq'], 'best_model.ckpt')))

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
        assert (self.conf["model"] == "joint"), "For individual QC/CC/QQ model train/test, use codesearcher.py"
        data = {"qc": load_qc_data(test=False),
                "cc": load_cc_data(test=False),
                "qq": load_qq_data(test=False)}

        train_loader = {
            "qc": torch.utils.data.DataLoader(dataset=data["qc"]["train"], batch_size=batch_size, shuffle=True,
                                              drop_last=True, num_workers=1, collate_fn=my_collate),
            "cc": torch.utils.data.DataLoader(dataset=data["cc"]["train"], batch_size=batch_size, shuffle=True,
                                              drop_last=True, num_workers=1, collate_fn=my_collate),
            # "qq": torch.utils.data.DataLoader(dataset=data["qq"]["train"], batch_size=batch_size, shuffle=True,
            #                                   drop_last=True, num_workers=1, collate_fn=my_collate)
        }

        # MRR for the Best Saved model, if reload > 0, else -1
        if self.conf['reload'] > 0:
            _, max_mrr, max_map, max_ndcg = searcher.eval(model, args.pool_size, {k: v["dev"] for k, v in data.items()})
            if writer is not None:
                for k in max_mrr.keys():
                    writer.add_scalar('Valid/%s_MRR' % k.upper(), max_mrr[k], self.conf['reload'])
                    writer.add_scalar('Valid/%s_MAP' % k.upper(), max_map[k], self.conf['reload'])
                    writer.add_scalar('Valid/%s_nDCG' % k.upper(), max_ndcg[k], self.conf['reload'])
        else:
            max_mrr = {"qc": -1, "cc": -1, "qq": -1}

        patience = 0
        for epoch in range(self.conf['reload'] + 1, nb_epoch):
            itr = 1
            losses = {"qc": [], "cc": [], "qq": []}
            all_losses = {"qc": [], "cc": [], "qq": []}
            all_losses_new = {"qc_on_cc_data": [], "cc_on_cc_data": []}
            rewards = []
            all_rewards = []

            # model["qc"], model["cc"], model["qq"] = model["qc"].train(), model["cc"].train(), model["qq"].train()
            # model = {k: v.train() for k, v in model.items()}

            # Update QC model
            if self.conf["update_qc"] > 0:
                model["qc"].train()
                model["cc"].eval()
                for qc_batch in train_loader["qc"]:
                    qc_query, qc_good_cands, qc_bad_cands = qc_batch["query"], qc_batch["pos"], qc_batch["neg"]
                    qc_query, qc_good_cands, qc_bad_cands = gVar(qc_query), gVar(qc_good_cands), gVar(qc_bad_cands)

                    # "collision[i][j]=1" means that "qc_batch['query_offset'][i]=qc_batch['rand_offset']',
                    # which means that qc_bad_cands[j] is actually a positive sample for qc_query[i] or qc_good_cands[i]
                    # collision = gVar(qc_batch['query_offset'].unsqueeze(1) == qc_batch['rand_offset'].unsqueeze(0))
                    collision = None

                    # Score and sample negative code using CC model
                    sampled, sim = model["cc"].sample_cand(qc_good_cands, qc_bad_cands, collision=collision,
                                                           if_norm=True)
                    qc_bad_cands_ranked_by_cc = torch.index_select(qc_bad_cands, 0, sampled.squeeze())
                    # Use the sampled as negative samples to train QC
                    loss, good_scores, bad_scores = model["qc"](qc_query, qc_good_cands, qc_bad_cands_ranked_by_cc)
                    losses["qc"].append(loss.item())
                    all_losses["qc"].append(loss.item())
                    optimizer["qc"].zero_grad()
                    loss.backward()  # FIXME: what exactly does retain_graph do?
                    optimizer["qc"].step()

                    if itr % log_every == 0:
                        print('epo:[%d/%d]  itr:%d  QC Loss=%.2E  CC Loss=%.2E  CC Reward=%.2E' % (
                            epoch, nb_epoch, itr,
                            np.mean(losses["qc"]) if losses["qc"] else -1,
                            np.mean(losses["cc"]) if losses["cc"] else -1,
                            np.mean(rewards) if rewards else -1))
                        losses = {"qc": [], "cc": [], "qq": []}
                        rewards = []
                    itr = itr + 1

            # Update CC model, using policy gradient
            if self.conf["update_cc"] > 0:
                model["qc"].eval()
                model["cc"].train()
                for cc_batch in train_loader["cc"]:

                    cc_query, cc_good_cands, cc_bad_cands = cc_batch["query"], cc_batch["pos"], cc_batch["neg"]
                    cc_query, cc_good_cands, cc_bad_cands = gVar(cc_query), gVar(cc_good_cands), gVar(cc_bad_cands)
                    cc_qts = cc_batch["qts"]
                    cc_qts = gVar(cc_qts)

                    collision = None

                    # Score and sample negative code using CC model
                    sampled, sim = model["cc"].sample_cand(cc_query, cc_bad_cands, collision=collision,
                                                           if_norm=True)
                    cc_bad_cands_ranked_by_cc = torch.index_select(cc_bad_cands, 0, sampled.squeeze())

                    qc_loss_on_cc_data, good_scores, bad_scores = model["qc"](cc_qts, cc_good_cands, cc_bad_cands_ranked_by_cc)
                    cc_loss_on_cc_data, _, _ = model["cc"](cc_query, cc_good_cands, cc_bad_cands_ranked_by_cc)
                    all_losses_new["qc_on_cc_data"].append(qc_loss_on_cc_data.mean().item())
                    all_losses_new["cc_on_cc_data"].append(cc_loss_on_cc_data.mean().item())

                    reward = (bad_scores - good_scores).data

                    # loss = (-1 * reward * sim).mean()
                    # loss = (-1 * reward * torch.log(sim)).mean()
                    loss = (-1 * reward * cc_loss_on_cc_data).mean()
                    losses["cc"].append(loss.item())
                    all_losses["cc"].append(loss.item())
                    rewards.append(reward.mean().item())
                    all_rewards.append(reward.mean().item())
                    optimizer["cc"].zero_grad()
                    loss.backward()
                    optimizer["cc"].step()

                    if itr % log_every == 0:
                        print('epo:[%d/%d]  itr:%d  QC Loss=%.2E  CC Loss=%.2E  CC Reward=%.2E' % (
                            epoch, nb_epoch, itr,
                            np.mean(losses["qc"]) if losses["qc"] else -1,
                            np.mean(losses["cc"]) if losses["cc"] else -1,
                            np.mean(rewards) if rewards else -1))
                        losses = {"qc": [], "cc": [], "qq": []}
                        rewards = []
                    itr = itr + 1

            # for qc_batch in train_loader["qc"]:
            #
            #     qts, good_cands, bad_cands = qc_batch["query"], qc_batch["pos"], qc_batch["neg"]
            #     qts, good_cands, bad_cands = gVar(qts), gVar(good_cands), gVar(bad_cands)
            #
            #     # "collision[i][j]=1" means that "qc_batch['query_offset'][i]=qc_batch['rand_offset']',
            #     # which means that bad_cands[j] is actually a positive sample for qts[i] or good_cands[i]
            #     # collision = gVar(qc_batch['query_offset'].unsqueeze(1) == qc_batch['rand_offset'].unsqueeze(0))
            #     collision = None
            #
            #     # Rank negative code using CC model
            #     # FIXME: There is a chance that one of the bad_cands is good when cross-matching
            #     sampled, sim = model["cc"].sample_cand(good_cands, bad_cands, collision=collision, if_norm=True)
            #     bad_cands_ranked_by_cc = torch.index_select(bad_cands, 0, sampled.squeeze())
            #
            #     # Update QC model
            #     loss, good_scores, bad_scores = model["qc"](qts, good_cands, bad_cands_ranked_by_cc)
            #     if self.conf['update_qc'] > 0:
            #         losses["qc"].append(loss.item())
            #         optimizer["qc"].zero_grad()
            #         loss.backward(retain_graph=True)  # FIXME: what exactly does retain_graph do?
            #         optimizer["qc"].step()
            #
            #     # Update CC model, using policy gradient
            #     if self.conf['update_cc'] > 0:
            #         pw_scores = model["qc"].get_scores(qts, bad_cands)
            #         baseline = pw_scores.mean(1)
            #         reward = (bad_scores - baseline).data  # FIXME: IRGAN defines reward as (2 * sigmoid(logit) - 1)
            #         loss = (-1 * reward * sim).mean()
            #         losses["cc"].append(loss.item())
            #         rewards.append(reward.mean().item())
            #         optimizer["cc"].zero_grad()
            #         loss.backward()
            #         optimizer["cc"].step()
            #
            #
            #     if itr % log_every == 0:
            #         print('epo:[%d/%d]  itr:%d  QC Loss=%.2E  CC Loss=%.2E  CC Reward=%.2E' % (
            #             epoch, nb_epoch, itr,
            #             np.mean(losses["qc"]) if losses["qc"] else -1,
            #             np.mean(losses["cc"]) if losses["cc"] else -1,
            #             np.mean(rewards) if rewards else -1))
            #         losses = {"qc": [], "cc": [], "qq": []}
            #     itr = itr + 1

            # Write to tensorboard
            if writer is not None:
                for k, v in all_losses.items():
                    if v:
                        writer.add_scalar('Train/%s_loss' % k.upper(), np.mean(v), epoch)
                if rewards:
                    writer.add_scalar('Train/CC_reward', np.mean(all_rewards), epoch)

                writer.add_scalar('Train/QC_loss_on_CC_data', np.mean(all_losses_new["qc_on_cc_data"]), epoch)
                writer.add_scalar('Train/CC_loss_on_CC_data', np.mean(all_losses_new["cc_on_cc_data"]), epoch)

            model_monitored = ["qc", "cc"]
            if epoch % valid_every == 0:
                print("validating..")
                acc1, mrr, map, ndcg = self.eval(model, args.pool_size, {k: v["dev"] for k, v in data.items()})
                model_to_save = {}
                for k in mrr.keys():
                    if writer is not None:
                        writer.add_scalar('Valid/%s_MRR' % k.upper(), mrr[k], epoch)
                        writer.add_scalar('Valid/%s_MAP' % k.upper(), map[k], epoch)
                        writer.add_scalar('Valid/%s_nDCG' % k.upper(), ndcg[k], epoch)
                    model_to_save[k] = None
                    if k in model_monitored and mrr[k] > max_mrr[k]:
                        max_mrr[k] = mrr[k]
                        model_to_save[k] = model[k]
                        print("%s model improved. Saving at %d epoch." % (k.upper(), epoch))
                        patience = 0
                if all([x is None for x in model_to_save.values()]):
                    print("Model didn't improve for ", patience + 1, " epochs")
                    patience += 1
                else:
                    self.save_model(model_to_save)

            if patience >= max_patience:
                print("Patience Limit Reached. Stopping Training")
                break

    #######################
    # Evaluation on StaQC #
    #######################
    def eval(self, model, poolsize, dataset, bool_collect=False, write_qual=False):
        assert (set(model.keys()) == set(dataset.keys())), "model and dataset have mismatched keys."
        acc, mrr, map, ndcg = {}, {}, {}, {}
        for k in model.keys():
            f_qual = None
            if write_qual:
                f_qual = open(os.path.join(self.conf['model_directory'][k], "qualitative.txt"), 'w')

            print("%s:\t" % k.upper(), end="")
            acc[k], mrr[k], map[k], ndcg[k] = self._eval(model[k], poolsize, dataset[k], bool_collect=bool_collect,
                                                         f_qual=f_qual)
            if write_qual:
                f_qual.close()
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

                if f_qual is not None:
                    self._print_qualitative(f_qual=f_qual, qvocab=dataset.qvocab, cvocab=dataset.cvocab,
                                            query=qts[i].tolist(), pos=all_pos[i].tolist(),
                                            neg=(cands[:i].tolist() + cands[i+1:].tolist()),
                                            labels=real, preds=scores,
                                            MRR=mrrs[-1], MAP=maps[-1], nDCG=ndcgs[-1])

        if bool_collect:
            save_path = os.path.join(self.conf['model_directory'], "collect_sims_staqc_%s.pkl" % dataset.data_name)
            print("Save collection to %s" % save_path)
            pickle.dump(sims_collection, open(save_path, "wb"))

        print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            len(accs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)

    def _print_qualitative(self, f_qual, qvocab, cvocab, query, pos, neg, labels, preds, MRR, MAP, nDCG):
        f_qual.write("\nMRR=%.4E  MAP=%.4E  nDCG=%.4E\n" % (MRR, MAP, nDCG))
        f_qual.write("Q: %s\n" % " ".join([qvocab.vocab[_].encode('utf-8')
                                           for _ in filter(lambda x: x>0, query)]))
        zipped = zip(['+'] * len(pos) + ['-'] * len(neg), preds, pos + neg)
        zipped.sort(key=lambda t: t[1], reverse=True)
        for i, (l, p, a) in enumerate(zipped):
            f_qual.write("%-2d %s %.2E: %s\n" % (i, l, p, " ".join([cvocab.vocab[_].encode('utf-8')
                                                                    for _ in filter(lambda x: x>0, a)])))


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search Model")
    parser.add_argument("-M", "--model", choices=["qc", "qq", "cc", "joint"], required=True,
                        help="Which model to train: QC, QQ, CC or their joint.")
    parser.add_argument("-m", "--mode", choices=["train", "eval", "collect"],
                        default='train',
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluates models on a test set;"
                             " the `collect` mode collects model scores;",
                        required=True)
    parser.add_argument("--reload", type=int, default=-1, help="Should I reload saved model, yes if reload>0?",
                        required=True)
    parser.add_argument("--qc_reload_path", type=str, default="", help="Enclosing folder of the to-be-reloaded model.")
    parser.add_argument("--cc_reload_path", type=str, default="", help="Enclosing folder of the to-be-reloaded model.")
    parser.add_argument("--qq_reload_path", type=str, default="", help="Enclosing folder of the to-be-reloaded model.")

    # model setup
    parser.add_argument("--update_qc", type=int, default=1, help="If update QC model?", required=True)
    parser.add_argument("--update_cc", type=int, default=1, help="If update CC model?", required=True)
    parser.add_argument("--dropout", type=float, default=0.0, help="What is the dropout?", required=True)
    parser.add_argument("--emb_size", type=int, default=100, help="What is the embedding size?", required=True)
    parser.add_argument("--lstm_dims", type=int, default=200, help="What is the lstm dimension?", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="What is the batch size?", required=True)
    parser.add_argument("--temp", type=str, default="", help="Name of a temporary test (not affect saved model etc).")

    # dataset setup
    parser.add_argument("--qn_mode", type=str, default="sl",
                        choices=["sl", "rl_bleu", "rl_mrr", "codenn_gen"], help="Data set to use.")
    parser.add_argument('--pool_size', type=int, default=50, help="candidate pool size for evaluation")

    # optimization
    parser.add_argument("--lr", type=float, default=0.001, help="What is the learning rate?")
    parser.add_argument("--margin", type=float, default=0.05, help="Margin for pairwise loss.")
    parser.add_argument("--optimizer", type=str,
                        choices=["adam", "adagrad", "sgd", "rmsprop", "asgd", "adadelta"],
                        default="adam", help="Which optimizer to use?")
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
    conf['qc_reload_path'] = args.qc_reload_path
    conf['cc_reload_path'] = args.cc_reload_path
    conf['qq_reload_path'] = args.qq_reload_path
    conf['optimizer'] = args.optimizer
    conf['update_qc'] = args.update_qc
    conf['update_cc'] = args.update_cc

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

        model_dir_str = "%s" % args.model
        model_dir_str += "_updateQC" if args.update_qc else ""
        model_dir_str += "_updateCC" if args.update_cc else ""
        model_dir_str += "_%s" % args.temp if args.temp else ""

        conf['model_directory'] = {
            "qc": os.path.join(conf['ckptdir'], 'QC_%s' % model_dir_str, model_string),
            "cc": os.path.join(conf['ckptdir'], 'CC_%s' % model_dir_str, model_string),
            "qq": os.path.join(conf['ckptdir'], 'QQ_%s' % model_dir_str, model_string)}

        if conf['qc_reload_path'] and conf['cc_reload_path'] and conf['qq_reload_path']:
            conf['reload_model_directory'] = {
                "qc": os.path.join(conf['qc_reload_path'], model_string),
                "cc": os.path.join(conf['cc_reload_path'], model_string),
                "qq": os.path.join(conf['qq_reload_path'], model_string)}

        for dir in conf['model_directory'].values():
            if not os.path.exists(dir):
                os.makedirs(dir)
        print(" Model Directory : ")
        for k, v in conf['model_directory'].items():
            print("%s : %s" % (k, v))

        conf['summary_directory'] = os.path.join(conf['sumdir'], model_dir_str, model_string)
        # if not os.path.exists(conf['summary_directory']):
        #     os.makedirs(conf['summary_directory'])
        print(" Summary Directory : " + conf['summary_directory'])
        writer = None
        writer = SummaryWriter(conf['summary_directory'])


        searcher = JointSearcher(conf)

        #####################
        # Define model ######
        #####################
        print('Building %s Model' % args.model.upper())
        model = {"qc": QCModel(conf),
                 "cc": CCModel(conf),
                 "qq": QQModel(conf)}
        print("QC model: ", model["qc"])
        print("CC model: ", model["cc"])
        print("QQ model: ", model["qq"])

        if conf['reload'] > 0:
            if args.mode in {'eval', 'collect'}:
                print("Reloading saved model for evaluating/collecting results")
            else:
                print("Reloading saved model for Re-training")
            if "reload_model_directory" in conf:
                searcher.load_other_model(model, conf['reload_model_directory'])
            else:
                searcher.load_model(model)

        if torch.cuda.is_available():
            print('using GPU')
            model = {k: v.cuda() if v is not None else v for k, v in model.items()}
        else:
            print('using CPU')

        print("\nParameter requires_grad state: ")
        for k, m in model.items():
            if m is None: continue
            print("%s model:" % k.upper())
            for name, param in m.named_parameters():
                print(name, param.requires_grad)
        print("")

        if conf['optimizer'] == 'adagrad':
            optimizer = {"qc": optim.Adagrad(model["qc"].parameters(), lr=conf['lr']),
                         "cc": optim.Adagrad(model["cc"].parameters(), lr=conf['lr']),
                         "qq": optim.Adagrad(model["qq"].parameters(), lr=conf['lr'])}
            print("Recommend lr 0.01 for AdaGrad while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'sgd':
            optimizer = {"qc": optim.SGD(model["qc"].parameters(), lr=conf['lr'], momentum=0.9),
                         "cc": optim.SGD(model["cc"].parameters(), lr=conf['lr'], momentum=0.9),
                         "qq": optim.SGD(model["qq"].parameters(), lr=conf['lr'], momentum=0.9)}
            print("Recommend lr 0.1 for SGD (momentum 0.9) while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'rmsprop':
            optimizer = {"qc": optim.RMSprop(model["qc"].parameters(), lr=conf['lr']),
                         "cc": optim.RMSprop(model["cc"].parameters(), lr=conf['lr']),
                         "qq": optim.RMSprop(model["qq"].parameters(), lr=conf['lr'])}
            print("Recommend lr 0.01 for RMSprop while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'asgd':
            optimizer = {"qc": optim.ASGD(model["qc"].parameters(), lr=conf['lr']),
                         "cc": optim.ASGD(model["cc"].parameters(), lr=conf['lr']),
                         "qq": optim.ASGD(model["qq"].parameters(), lr=conf['lr'])}
            print("Recommend lr 0.01 for ASGD while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'adadelta':
            optimizer = {"qc": optim.Adadelta(model["qc"].parameters(), lr=conf['lr']),
                         "cc": optim.Adadelta(model["cc"].parameters(), lr=conf['lr']),
                         "qq": optim.Adadelta(model["qq"].parameters(), lr=conf['lr'])}
            print("Recommend lr 1.00 for Adadelta while using %.5f." % conf['lr'])
        else:
            optimizer = {"qc": optim.Adam(model["qc"].parameters(), lr=conf['lr'] * 0.1),
                         "cc": optim.Adam(model["cc"].parameters(), lr=conf['lr'] * 0.1),
                         "qq": optim.Adam(model["qq"].parameters(), lr=conf['lr'])}
            print("Recommend lr 0.001 for Adam while using %.5f." % conf['lr'])

        if args.mode == 'train':
            print('Training Model')
            searcher.train(model, writer=writer)

        elif args.mode == 'eval':
            print('Evaluating Model')
            print('Pool size = %d' % args.pool_size)
            assert (conf["model"] == "joint"), "For individual QC/CC/QQ model train/test, use codesearcher.py"
            data = {"qc": load_qc_data(train=False),
                    "cc": load_cc_data(train=False),
                    "qq": load_qq_data(train=False)}
            searcher.eval(model, args.pool_size, {k: v["dev"] for k, v in data.items()})
            searcher.eval(model, args.pool_size, {k: v["test"] for k, v in data.items()})

        elif args.mode == 'collect':
            print('Collecting outputs...')
            assert (conf["model"] == "joint"), "For individual QC/CC/QQ model train/test, use codesearcher.py"
            data = {"qc": load_qc_data(train=False),
                    "cc": load_cc_data(train=False),
                    "qq": load_qq_data(train=False)}
            searcher.eval(model, 50, {k: v["dev"] for k, v in data.items()}, bool_collect=True)
            searcher.eval(model, 50, {k: v["test"] for k, v in data.items()}, bool_collect=True)

        else:
            print("Please provide a Valid argument for mode - train/eval")
