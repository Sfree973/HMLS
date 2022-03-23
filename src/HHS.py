
"""

import os
import random
import shutil
from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import src.eval
from src.utils.ops import var_cuda, zeros_var_cuda
import src.utils.ops as ops


class LFramework(nn.Module):
    def __init__(self, args, kg, mdl):
        super(LFramework, self).__init__()
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.model = args.model

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None
        self.task_size = args.task_size
        self.hard_size = args.hard_size
        print("hard_size", self.hard_size)

        self.inference = not args.train
        self.run_analysis = args.run_analysis
        self.gpu_id = args.gpu

        self.kg = kg
        self.mdl = mdl
        print('{} module created'.format(self.model))

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(),
                  'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    def run_train(self, train_data, dev_data, few_shot=False, adaptation=False, adaptation_relation=None, emb=False):
        self.print_all_model_parameters()

        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []
        print(self.start_epoch)

        for epoch_id in trange(self.start_epoch, self.num_epochs):
            print('Epoch {}'.format(epoch_id))
            self.train()
            if self.rl_variation_tag.startswith('rs'):
                self.fn.eval()
                self.fn_kg.eval()
                if self.model.endswith('hypere'):
                    self.fn_secondary_kg.eval()
            self.batch_size = self.train_batch_size
            if few_shot and not emb:
                self.batch_size *= 2
                shuffle_list = []
                for x in train_data:
                    random.shuffle(train_data[x])
                    num = int((len(train_data[x]) - 1) / self.batch_size) + 1
                    for i in range(num):
                        shuffle_list.append(x)
                # print("shuffle_list",len(shuffle_list),len(train_data),shuffle_list)
                random.shuffle(shuffle_list)
                new_train_data = []
                index_list = {}
                relatin2index = {}
                for x in train_data:
                    index_list[x] = 0
                    relatin2index[x] = []
                i = 0
                for x in shuffle_list:
                    if index_list[x] * self.batch_size + self.batch_size <= len(train_data[x]):
                        new_train_data += train_data[x][index_list[x] *
                                                        self.batch_size: index_list[x] * self.batch_size + self.batch_size]
                        index_list[x] += 1
                        relatin2index[x].append(i)
                        i += 1
                    else:
                        new_train_data += train_data[x][index_list[x]
                                                        * self.batch_size: len(train_data[x])]
                        others = self.batch_size - \
                            (len(train_data[x]) -
                             index_list[x] * self.batch_size)
                        new_train_data += train_data[x][:others]
                self.batch_size = int(self.batch_size / 2)
                print("relatin2index created")               
                relation2id = self.kg.relation2id
                
                import pickle
                with open("cluster.txt", 'rb') as text:
                    clust = pickle.load(text)
                cl = []
                for i in clust:
                    tmp = []
                    for j in i:
                        tmp.append(relation2id[j])
                    cl.append(tmp)
                    
                rel2clst = {}
                for ids in cl:
                    if len(ids)==1:
                        rel2clst[ids[0]] = ids
                    else:
                        for id in ids:
                            tmp = ids.copy()
                            tmp.remove(id)
                            rel2clst[id] = tmp
                print("rel2clst_dict created")
                
                
            else:
                random.shuffle(train_data)
            batch_losses = []
            entropies = []
            if self.run_analysis:
                rewards = None
                fns = None

            if few_shot and not emb:
                start = 0
                end = self.task_size
                train_index = [i for i in range(start, end)]
                easy_size = self.task_size - self.hard_size
                full_batch = self.batch_size * 2
                len_train = len(new_train_data)

                print("trainsize", len_train)
                
                #create dict
                x = 0
                while True:
                    self.optim.zero_grad()
        
                    if end * full_batch > len_train:
            
                        #print("process")
                        for example_id in range(start * full_batch, len_train, full_batch):
                            mini_batch = new_train_data[example_id:example_id +
                                                        self.batch_size]
                            # print("FOR",mini_batch)
                            mini_batch_valid = new_train_data[example_id +
                                                              self.batch_size:example_id + 2 * self.batch_size]
                            loss, _ = self.meta_loss(
                                mini_batch, mini_batch_valid)
                            loss['model_loss'].backward()
                            if self.grad_norm > 0:
                                clip_grad_norm_(
                                    self.parameters(), self.grad_norm)
                            batch_losses.append(loss['print_loss'])
                            if 'entropy' in loss:
                                entropies.append(loss['entropy'])
                            if self.run_analysis:
                                if rewards is None:
                                    rewards = loss['reward']
                                else:
                                    rewards = torch.cat(
                                        [rewards, loss['reward']])
                                if fns is None:
                                    fns = loss['fn']
                                else:
                                    fns = torch.cat([fns, loss['fn']])

                        for example_id in index_list:
                            example_id = example_id * full_batch
                            mini_batch = new_train_data[example_id:example_id +
                                                        self.batch_size]
                            # print("FOR",mini_batch)
                            mini_batch_valid = new_train_data[example_id +
                                                              self.batch_size:example_id + 2 * self.batch_size]
                            loss, _ = self.meta_loss(
                                mini_batch, mini_batch_valid)
                            loss['model_loss'].backward()
                            if self.grad_norm > 0:
                                clip_grad_norm_(
                                    self.parameters(), self.grad_norm)
                            batch_losses.append(loss['print_loss'])
                            if 'entropy' in loss:
                                entropies.append(loss['entropy'])
                            if self.run_analysis:
                                if rewards is None:
                                    rewards = loss['reward']
                                else:
                                    rewards = torch.cat(
                                        [rewards, loss['reward']])
                                if fns is None:
                                    fns = loss['fn']
                                else:
                                    fns = torch.cat([fns, loss['fn']])
                        self.optim.step()
                        break

                    reward_list = []
                    rel_list = []
                    for example_id in train_index:
                        example_id = example_id * full_batch
                        
                        mini_batch = new_train_data[example_id:example_id +
                                                    self.batch_size]
                        rel_list.append(mini_batch[0][2])
                        # print("FOR",mini_batch)
                        mini_batch_valid = new_train_data[example_id +
                                                          self.batch_size:example_id + 2 * self.batch_size]
                        loss, prec = self.meta_loss(
                            mini_batch, mini_batch_valid)
                        reward_list.append(prec)
                        loss['model_loss'].backward()
                        if self.grad_norm > 0:
                            clip_grad_norm_(self.parameters(), self.grad_norm)
                        batch_losses.append(loss['print_loss'])
                        if 'entropy' in loss:
                            entropies.append(loss['entropy'])
                        if self.run_analysis:
                            if rewards is None:
                                rewards = loss['reward']
                            else:
                                rewards = torch.cat([rewards, loss['reward']])
                            if fns is None:
                                fns = loss['fn']
                            else:
                                fns = torch.cat([fns, loss['fn']])
                    self.optim.step()
                    # keep hardness train list
                    index = np.argpartition(
                        np.array(reward_list), self.hard_size-1)[:self.hard_size]
                    
                    if x%2 == 0:
                        # hard come from this batch
                        orgin_list = []
                        # hard come frome last batch
                        last_list = []
                        for i in index:
                            if i >= easy_size:
                                last_list.append(train_index[i])
                            else:
                                orgin_list.append(i)
                                
                        index_list = (
                            np.array(orgin_list)+start).tolist()+last_list
                    else:
                        index_list = []
                        for i in index:
                            # 1) find hard relation id(index_list -> relation id)
                            replace_rel = random.choice(rel2clst[rel_list[i]])
                            # 2) random a new relation task's index
                            index_list.append(random.choice(relatin2index[replace_rel]))
                    
                    start = end
                    end += easy_size
                    train_index = [i for i in range(
                        start, end)] + index_list
                    x+=1
                    
            else:
                print('length of train_data', len(train_data))
                for example_id in range(0, len(train_data), self.batch_size):
                    self.optim.zero_grad()
                    mini_batch = train_data[example_id:example_id +
                                            self.batch_size]
                    if len(mini_batch) < self.batch_size:
                        continue
                    loss, _ = self.loss(mini_batch)
                    loss['model_loss'].backward()
                    if self.grad_norm > 0:
                        clip_grad_norm_(self.parameters(), self.grad_norm)
                    self.optim.step()
                    batch_losses.append(loss['print_loss'])
                    if 'entropy' in loss:
                        entropies.append(loss['entropy'])
                    if self.run_analysis:
                        if rewards is None:
                            rewards = loss['reward']
                        else:
                            rewards = torch.cat([rewards, loss['reward']])
                        if fns is None:
                            fns = loss['fn']
                        else:
                            fns = torch.cat([fns, loss['fn']])
            # Check training statistics
            stdout_msg = 'Epoch {}: average training loss = {}'.format(
                epoch_id, np.mean(batch_losses))
            if entropies:
                stdout_msg += 'entropy = {}'.format(np.mean(entropies))
            print(stdout_msg)
            if epoch_id % self.num_wait_epochs == 0 or epoch_id == self.num_epochs - 1:
                if not adaptation:
                    self.save_checkpoint(
                        checkpoint_id=epoch_id, epoch_id=epoch_id)
                else:
                    self.save_checkpoint(
                        checkpoint_id=epoch_id, epoch_id=epoch_id, relation=adaptation_relation)
            if self.run_analysis:
                print('* Analysis: # path types seen = {}'.format(self.num_path_types))
                num_hits = float(rewards.sum())
                hit_ratio = num_hits / len(rewards)
                print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                num_fns = float(fns.sum())
                fn_ratio = num_fns / len(fns)
                print('* Analysis: false negative ratio = {}'.format(fn_ratio))

            # Check dev set performance
            # if (self.run_analysis or (epoch_id > 0 and epoch_id % self.num_wait_epochs == 0)) and self.model != '!TransE' and self.model != 'PTransE':
            #     self.eval()
            #     self.batch_size = self.dev_batch_size
            #     dev_scores = self.forward(dev_data, verbose=False)
            #     print('Dev set performance: (correct evaluation)')
            #     _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
            #     metrics = mrr
            #     print('Dev set performance: (include test set labels)')
            #     src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
            #     # Action dropout anneaking
            #     if self.model.startswith('point'):
            #         eta = self.action_dropout_anneal_interval
            #         if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
            #             old_action_dropout_rate = self.action_dropout_rate
            #             self.action_dropout_rate *= self.action_dropout_anneal_factor
            #             print('Decreasing action dropout rate: {} -> {}'.format(
            #                 old_action_dropout_rate, self.action_dropout_rate))
            #     # Save checkpoint
            #     if metrics > best_dev_metrics:
            #         self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
            #         best_dev_metrics = metrics
            #         with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
            #             o_f.write('{}'.format(epoch_id))
            #     else:
            #         # Early stopping
            #         pass
            #         # if epoch_id >= self.num_wait_epochs and metrics < np.mean(dev_metrics_history[-self.num_wait_epochs:]):
            #         #     break
            #     dev_metrics_history.append(metrics)
            #     if self.run_analysis:
            #         num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
            #         dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
            #         hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
            #         fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
            #         if epoch_id == 0:
            #             with open(num_path_types_file, 'w') as o_f:
            #                 o_f.write('{}\n'.format(self.num_path_types))
            #             with open(dev_metrics_file, 'w') as o_f:
            #                 o_f.write('{}\n'.format(metrics))
            #             with open(hit_ratio_file, 'w') as o_f:
            #                 o_f.write('{}\n'.format(hit_ratio))
            #             with open(fn_ratio_file, 'w') as o_f:
            #                 o_f.write('{}\n'.format(fn_ratio))
            #         else:
            #             with open(num_path_types_file, 'a') as o_f:
            #                 o_f.write('{}\n'.format(self.num_path_types))
            #             with open(dev_metrics_file, 'a') as o_f:
            #                 o_f.write('{}\n'.format(metrics))
            #             with open(hit_ratio_file, 'a') as o_f:
            #                 o_f.write('{}\n'.format(hit_ratio))
            #             with open(fn_ratio_file, 'a') as o_f:
            #                 o_f.write('{}\n'.format(fn_ratio))

    def forward(self, examples, verbose=False):
        pred_scores = []
        for example_id in range(0, len(examples), self.batch_size):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            pred_score = self.predict(mini_batch, verbose=verbose)
            pred_scores.append(pred_score[:mini_batch_size])
        scores = torch.cat(pred_scores)
        return scores

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """
        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), num_labels])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(
                batch_e2), requires_grad=False)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
        return batch_e1, batch_e2, batch_r

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, checkpoint_id, epoch_id=None, is_best=False, relation=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['epoch_id'] = epoch_id

        if not relation:
            out_tar = os.path.join(
                self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        else:
            out_tar = os.path.join(
                self.model_dir, 'checkpoint-{}-{}.tar'.format(checkpoint_id, relation))
        if is_best:
            best_path = os.path.join(self.model_dir, 'model_best.tar')
            shutil.copyfile(out_tar, best_path)
            print('=> best model updated \'{}\''.format(best_path))
        else:
            torch.save(checkpoint_dict, out_tar)
            print('=> saving checkpoint to \'{}\''.format(out_tar))

    def load_checkpoint(self, input_file, adaptation=False, emb_few=False):
        """
        Load model checkpoint.
        :param n: Neural network module.
        :param kg: Knowledge graph module.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(
                input_file, map_location=('cuda:' + str(self.gpu_id)))
            self.load_state_dict(checkpoint['state_dict'])
            if not self.inference and not adaptation and not emb_few:
                self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    def export_to_embedding_projector(self):
        """
        Export knowledge base embeddings into .tsv files accepted by the Tensorflow Embedding Projector.
        """
        vector_path = os.path.join(self.model_dir, 'vector.tsv')
        meta_data_path = os.path.join(self.model_dir, 'metadata.tsv')
        v_o_f = open(vector_path, 'w')
        m_o_f = open(meta_data_path, 'w')
        for r in self.kg.relation2id:
            if r.endswith('_inv'):
                continue
            r_id = self.kg.relation2id[r]
            R = self.kg.relation_embeddings.weight[r_id]
            r_print = ''
            for i in range(len(R)):
                r_print += '{}\t'.format(float(R[i]))
            v_o_f.write('{}\n'.format(r_print.strip()))
            m_o_f.write('{}\n'.format(r))
            print(r, '{}'.format(float(R.norm())))
        v_o_f.close()
        m_o_f.close()
        print('KG embeddings exported to {}'.format(vector_path))
        print('KG meta data exported to {}'.format(meta_data_path))

    @property
    def rl_variation_tag(self):
        parts = self.model.split('.')
        if len(parts) > 1:
            return parts[1]
        else:
            return ''
