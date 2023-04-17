import torch
import torch.nn as nn
import os
from . import utils as solver_utils
from utils.utils import to_cuda, to_onehot
from torch import optim
from . import clustering
from discrepancy.cdd import CDD
from math import ceil as ceil
from .base_solver import BaseSolver
from copy import deepcopy

class AML(BaseSolver):
    def __init__(self, net_1, net_2, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(AML, self).__init__(net_1, net_2, dataloader, \
                      bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}
        
        self.para1 = self.opt.PARAMS.CON_LOSS
        self.para2 = self.opt.PARAMS.IM_LOSS

        

    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

    def solve(self):
        stop = False
        self.max_acc = 0.0
        if self.resume:
            self.iters += 1
            self.loop += 1


        while True: 
            stop = self.complete_training()
            if stop: break
                
            self.compute_iters_per_loop()

            self.update_network()
            self.loop += 1
        save_path = self.opt.SAVE_DIR
        acc = str(round(self.max_acc,1))
        out = save_path.split('/')
        out.pop()
        newout = ''
        for m in out:
            newout = os.path.join(newout,m)
        import datetime
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        newout = os.path.join(newout,acc+'_'+str(nowTime))
        os.rename(save_path, newout)


        print('Training Done!')
        
    def update_labels(self):
        net = self.net
        net.eval()
        opt = self.opt

        source_dataloader = self.train_data[self.clustering_source_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.source_name])

        source_centers = solver_utils.get_centers(net, 
		source_dataloader, self.opt.DATASET.NUM_CLASSES, 
                self.opt.CLUSTERING.FEAT_KEY)
        init_target_centers = source_centers

        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.target_name])

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(net, target_dataloader)
        self.path2label = self.clustering.path2label

    def compute_iters_per_loop(self):
        self.iters_per_loop = int(len(self.train_data[self.source_name]['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    # def update_network(self, filtered_classes):
    def update_network(self):
        # initial configuration
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
                     iter(self.train_data[self.source_name]['loader'])
        self.train_data[self.target_name]['iterator'] = \
                     iter(self.train_data[self.target_name]['loader'])

        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net_1.train()
            self.net_2.train()

            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name)
            target_sample = self.get_samples(self.target_name) 
            source_data_1, source_data_2, source_gt = source_sample['Img_1'],\
                        source_sample['Img_2'], source_sample['Label']
            target_data_1, target_data_2 = target_sample['Img_1'], target_sample['Img_2']


            source_data_1 = to_cuda(source_data_1)
            source_data_2 = to_cuda(source_data_2)
            target_data_1 = to_cuda(target_data_1)
            target_data_2 = to_cuda(target_data_2)
            source_gt = to_cuda(source_gt)
            self.net_1.module.set_bn_domain(self.bn_domain_map[self.source_name])
            self.net_2.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_result1 = self.net_1(source_data_1)
            source_result2 = self.net_2(source_data_2)
            source_preds1 = source_result1['logits']
            source_preds2 = source_result2['logits']

            # compute the cross-entropy loss
            loss_ce = self.CELoss(source_preds1, source_gt) + self.CELoss(source_preds2, source_gt)
            

            self.net_1.module.set_bn_domain(self.bn_domain_map[self.target_name])
            self.net_2.module.set_bn_domain(self.bn_domain_map[self.target_name])
            target_result1 = self.net_1(target_data_1)
            target_result2 = self.net_2(target_data_2)

            pred1 = target_result1['probs']
            pred2 = target_result2['probs']

            im_loss1 = self.im_loss(pred1)
            im_loss2 = self.im_loss(pred2)
            with torch.no_grad():
                e1 = self.entropy(pred1)
                e2 = self.entropy(pred2)

                
                mask_1 = e1 <= e2
                mask_2 = e1 > e2
                
            pred_u1_1 = pred1[mask_1]
            pred_u2_1 = pred2[mask_1]

            pred_u1_2 = pred1[mask_2]
            pred_u2_2 = pred2[mask_2]
            loss_con_1 = -self.discrepancy(pred_u1_1, pred_u2_1.detach()) + self.discrepancy(pred_u1_2, pred_u2_2.detach())
            loss_con_2 = self.discrepancy(pred_u1_1.detach(), pred_u2_1) - self.discrepancy(pred_u1_2.detach(), pred_u2_2)

            

            loss = loss_ce + self.para1*(loss_con_1 + loss_con_2)+ self.para2*(im_loss1 + im_loss2)

            self.optimizer['G_1'].zero_grad()
            self.optimizer['G_2'].zero_grad()
            self.optimizer['FC_1'].zero_grad()
            self.optimizer['FC_2'].zero_grad()
            loss.backward()
            self.optimizer['G_1'].step()
            self.optimizer['G_2'].step()
            self.optimizer['FC_1'].step()
            self.optimizer['FC_2'].step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu = self.model_eval((source_preds1+source_preds2)/2, source_gt)
                cur_loss = {'loss_ce': loss_ce, 'loss_con_1': loss_con_1,
			'loss_con_2': loss_con_2,'loss_im_1':im_loss1,'loss_im_2':im_loss2}
                self.logging(cur_loss, accu)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net_1.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    self.net_2.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    self.temp_accu = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop, 
                              self.iters, self.opt.EVAL_METRIC, self.temp_accu))
                    a = self.max_acc if self.max_acc>self.temp_accu else self.temp_accu
                    print('max acc:' + str(a))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
        (update_iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                if self.temp_accu > self.max_acc:
                    self.max_acc = self.temp_accu
                    self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False


    def get_label(self, path):
        label = []
        l = len(path)
        for i in range(l):
            label.append(self.path2label[path[i]])
        label  = to_cuda(torch.Tensor(label).long())
        return label

    def discrepancy(self, p1, p2):
        return ((p1 - p2)**2).sum(1).mean()

    def save_ckpt(self):
        save_path = self.opt.SAVE_DIR
        ckpt_weights = os.path.join(save_path, 'ckpt_max.weights')

        torch.save({'weights1': self.net_1.module.state_dict(),
                    'weights2': self.net_2.module.state_dict(),
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_weights)

    def entropy(self,p1):
        epsilon = 1e-5
        entropy = -p1 * torch.log(p1 + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy
    
    def im_loss(self,p1):
        entropy_loss = torch.mean(self.entropy(p1))
        cls_p = torch.mean(p1,dim=0)
        cls_loss = torch.sum(-cls_p*torch.log(cls_p+1e-5))
        return entropy_loss - cls_loss



