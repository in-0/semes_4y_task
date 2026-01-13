"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, freq_path):
        super(BalancedSoftmax, self).__init__()
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    # breakpoint()
    if labels != torch.float32:
        labels = labels.type(torch.float32).to(torch.device('cuda'))
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=1280, num_classes=4):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha # contrast weight among samples
        self.beta = beta # supervise loss weight
        self.gamma = gamma # paco loss
        self.supt = supt
        self.num_classes = num_classes

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        ss = features.shape[0]
        batch_size = ( features.shape[0] - self.K ) // 2
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features[:batch_size], features.T), self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

class CBLoss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, beta, gamma):
        super(CBLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, labels):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = labels

        weights = torch.tensor(weights).float().to(labels.device)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        pred = logits.softmax(dim=1)
        labels_one_hot = labels_one_hot.type(torch.float32).to(labels.device)
        weights = weights.type(torch.float32).to(labels.device)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss

class MTMLoss(nn.Module):
    """Modality-Text Matching Loss"""
    def __init__(self, num_classes=4, lambda_weight=0.5, text_emb_path='./semi_text_embs/merged_text_embeddings.pth'):
        super(MTMLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_weight = lambda_weight
        self.class_name = ['normal', 'caution', 'warning', 'critical']
        
        # text embeddings 불러오기
        try:
            text_embeddings_dict = torch.load(text_emb_path)
            # {'class': feat} 형태를 [num_classes, num_prompts, dim_feat] 형태로 변환
            max_prompts = max([feat.shape[0] if feat.dim() > 1 else 1 for feat in text_embeddings_dict.values()])
            text_embs = torch.zeros(num_classes, max_prompts, 1024)  # [num_classes, num_prompts, dim_feat]
            count = 0
            for i in self.class_name:
                if i in text_embeddings_dict:
                    feat = text_embeddings_dict[i]
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)  # [dim_feat] -> [1, dim_feat]
                    text_embs[count, :feat.shape[0], :] = feat
                    count += 1
                else:
                    print(f"Warning: class {i} not found in text embeddings, using random embedding")
                    text_embs[i] = torch.randn(max_prompts, 1024)
            print(f"Loaded text embeddings for MTMLoss with shape: {text_embs.shape}")
        except FileNotFoundError:
            print("Warning: ./semi_text_embs/merged_text_embeddings.pth not found. Using random embeddings.")
            text_embs = torch.randn(num_classes, 10, 1024)  # 기본값으로 10개 프롬프트 사용
        except Exception as e:
            print(f"Error loading text embeddings: {e}")
            text_embs = torch.randn(num_classes, 10, 1024)
        
        self.register_buffer('text_embs', text_embs)
    
    def forward(self, img_feat, sen_feat, gt_labels):
        """
        Args:
            img_feat: [batch_size, dim_feat] - image features (can be None for sensor-only)
            sen_feat: [batch_size, dim_feat] - sensor features (can be None for vision-only)
            gt_labels: [batch_size] - ground truth class labels
        """
        batch_size = gt_labels.size(0)
        device = gt_labels.device
        
        # text embeddings를 device로 이동
        text_embs = self.text_embs.to(device)  # [num_classes, num_prompts, dim_feat]
        num_classes, num_prompts, dim_feat = text_embs.shape
        
        # vision-only 모달리티인 경우
        if sen_feat is None or torch.all(sen_feat == 0):
            # InfoNCE loss 계산 (vision feature와 text embeddings 간)
            img_feat_norm = F.normalize(img_feat, dim=1)  # [batch_size, dim_feat]
            text_embs_flat = text_embs.view(-1, dim_feat)  # [num_classes * num_prompts, dim_feat]
            text_embs_norm = F.normalize(text_embs_flat, dim=1)  # [num_classes * num_prompts, dim_feat]
            
            # similarity matrix 계산
            sim_matrix = torch.matmul(img_feat_norm, text_embs_norm.T)  # [batch_size, num_classes * num_prompts]
            
            # positive indices (GT class의 모든 prompts)
            pos_indices = []
            for i, gt_label in enumerate(gt_labels):
                start_idx = gt_label * num_prompts
                end_idx = (gt_label + 1) * num_prompts
                pos_indices.extend([(i, j) for j in range(start_idx, end_idx)])
            
            # InfoNCE loss 계산
            L1 = self._info_nce_loss(sim_matrix, pos_indices, batch_size, num_classes * num_prompts)
            
            # vision-only이므로 L2와 L_agree는 0
            L2 = torch.zeros_like(L1)
            L_agree = torch.zeros_like(L1)
            
        # sensor-only 모달리티인 경우
        elif img_feat is None or torch.all(img_feat == 0):
            # InfoNCE loss 계산 (sensor feature와 text embeddings 간)
            sen_feat_norm = F.normalize(sen_feat, dim=1)  # [batch_size, dim_feat]
            text_embs_flat = text_embs.view(-1, dim_feat)  # [num_classes * num_prompts, dim_feat]
            text_embs_norm = F.normalize(text_embs_flat, dim=1)  # [num_classes * num_prompts, dim_feat]
            
            # similarity matrix 계산
            sim_matrix = torch.matmul(sen_feat_norm, text_embs_norm.T)  # [batch_size, num_classes * num_prompts]
            
            # positive indices (GT class의 모든 prompts)
            pos_indices = []
            for i, gt_label in enumerate(gt_labels):
                start_idx = gt_label * num_prompts
                end_idx = (gt_label + 1) * num_prompts
                pos_indices.extend([(i, j) for j in range(start_idx, end_idx)])
            
            # InfoNCE loss 계산
            L2 = self._info_nce_loss(sim_matrix, pos_indices, batch_size, num_classes * num_prompts)
            
            # sensor-only이므로 L1과 L_agree는 0
            L1 = torch.zeros_like(L2)
            L_agree = torch.zeros_like(L2)
            
        # fusion 모달리티인 경우
        else:
            # L1 = InfoNCE(img_feat, text_emb) 계산
            img_feat_norm = F.normalize(img_feat, dim=1)  # [batch_size, dim_feat]
            text_embs_flat = text_embs.view(-1, dim_feat)  # [num_classes * num_prompts, dim_feat]
            text_embs_norm = F.normalize(text_embs_flat, dim=1)  # [num_classes * num_prompts, dim_feat]
            
            # similarity matrix 계산
            sim_matrix_img = torch.matmul(img_feat_norm, text_embs_norm.T)  # [batch_size, num_classes * num_prompts]
            
            # positive indices (GT class의 모든 prompts)
            pos_indices = []
            for i, gt_label in enumerate(gt_labels):
                start_idx = gt_label * num_prompts
                end_idx = (gt_label + 1) * num_prompts
                pos_indices.extend([(i, j) for j in range(start_idx, end_idx)])
            
            # InfoNCE loss 계산
            L1 = self._info_nce_loss(sim_matrix_img, pos_indices, batch_size, num_classes * num_prompts)
            
            # L2 = InfoNCE(sen_feat, text_emb) 계산
            sen_feat_norm = F.normalize(sen_feat, dim=1)  # [batch_size, dim_feat]
            sim_matrix_sen = torch.matmul(sen_feat_norm, text_embs_norm.T)  # [batch_size, num_classes * num_prompts]
            L2 = self._info_nce_loss(sim_matrix_sen, pos_indices, batch_size, num_classes * num_prompts)
            
            # L_agree = abs(cos(img_feat,text_emb) - cos(sen_feat,text_emb)) 계산 (GT class만)
            gt_text_embs = text_embs[gt_labels]  # [batch_size, num_prompts, dim_feat]
            img_feat_expanded = img_feat.unsqueeze(1).expand(-1, num_prompts, -1)  # [batch_size, num_prompts, dim_feat]
            sen_feat_expanded = sen_feat.unsqueeze(1).expand(-1, num_prompts, -1)  # [batch_size, num_prompts, dim_feat]
            
            cos_img_text = F.cosine_similarity(img_feat_expanded, gt_text_embs, dim=2)  # [batch_size, num_prompts]
            cos_sen_text = F.cosine_similarity(sen_feat_expanded, gt_text_embs, dim=2)  # [batch_size, num_prompts]
            L_agree = torch.abs(cos_img_text.mean(dim=1) - cos_sen_text.mean(dim=1))  # [batch_size]
        
        # lambda*L1 + (1-lambda)*L2 계산
        weighted_loss = self.lambda_weight * L1 + (1 - self.lambda_weight) * L2
        
        # 최종 loss
        total_loss = weighted_loss + L_agree
        
        return total_loss.mean()
    
    def _info_nce_loss(self, sim_matrix, pos_indices, batch_size, num_negatives, temperature=0.07):
        """
        InfoNCE loss 계산
        Args:
            sim_matrix: [batch_size, num_negatives] - similarity matrix
            pos_indices: list of (batch_idx, text_idx) tuples - positive pairs
            batch_size: batch size
            num_negatives: number of negative samples
            temperature: temperature parameter
        """
        if not pos_indices:
            return torch.tensor(0.0, device=sim_matrix.device)
        
        # temperature scaling
        sim_matrix = sim_matrix / temperature
        
        # 각 sample에 대해 positive와 negative 분리
        losses = []
        for i in range(batch_size):
            # 현재 sample의 positive indices
            pos_indices_i = [j for batch_idx, j in pos_indices if batch_idx == i]
            if not pos_indices_i:
                continue
                
            # positive logits
            pos_logits = sim_matrix[i, pos_indices_i]  # [num_positives]
            
            # negative logits (positive가 아닌 모든 것)
            neg_mask = torch.ones(num_negatives, dtype=torch.bool, device=sim_matrix.device)
            neg_mask[pos_indices_i] = False
            neg_logits = sim_matrix[i, neg_mask]  # [num_negatives]
            
            # InfoNCE loss 계산
            logits = torch.cat([pos_logits, neg_logits])  # [num_positives + num_negatives]
            
            # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
            # 첫 번째가 positive sample이므로 target은 0
            target = torch.zeros(1, dtype=torch.long, device=sim_matrix.device)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=sim_matrix.device)


def create_loss_functions(args, cls_num_list, samples_per_cls, use_paco=True, use_cb=True, use_mtm=False):
    """손실 함수들을 생성하는 함수"""
    # PaCoLoss 설정
    if use_paco:
        fusion_criterion = PaCoLoss(
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            temperature=args.moco_t, K=args.moco_k, num_classes=args.num_classes
        )
        
        # cls_num_list 설정
        if cls_num_list is not None:
            fusion_criterion.cal_weight_for_classes(cls_num_list)
        else:
            # cls_num_list가 없는 경우 기본값 사용
            fusion_criterion.cal_weight_for_classes([1, 1, 1, 1])  # 기본값
    else:
        fusion_criterion = None
    
    # CBLoss 설정
    if use_cb:
        aux_criterion = CBLoss(samples_per_cls=samples_per_cls, no_of_classes=4, beta=0.9999, gamma=2)
    else:
        aux_criterion = None
    
    # MTMLoss 설정
    if use_mtm:
        mtm_criterion = MTMLoss(num_classes=args.num_classes, lambda_weight=args.mtm_lambda)
    else:
        mtm_criterion = None
    
    return fusion_criterion, aux_criterion, mtm_criterion