import os
import random
import logging
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from losses import *
import Moco_fusion_mm
from utils import Meter
from logger import setup_logger
import data.dataloader as dataloader 
from models import MobileNetV2CAE, DNN, LateFusion

import time
from torchsummary import summary
import torch.cuda as cuda
from thop import profile

parser = argparse.ArgumentParser()

parser.add_argument('--save_path', type=str, default='./results', help='Directory to save the model')
parser.add_argument('--seed', type=int, default=777, help='Seed')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--image_size', type=int, default=224, help='Image size')
parser.add_argument('--scheduler', type=str, choices=['default', 'step', 'cosine'], default='step', help='Choose Scheduler')
parser.add_argument('--step_size', type=int, default=8, help='Step size for learning rate scheduler')
parser.add_argument('--loss', choices=['ce', 'reweight', 'BalancedSoftmax', 'Paco'], default='Paco', help='Loss function')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_layers', type=int, default=5, help='Number of the sensor backbone layers')
parser.add_argument('--dim', type=int, default=128, help='Model inner dimension')
parser.add_argument('--imb_ratio', type=float, default=0.02, help='imbalance ratio')
parser.add_argument('--comment', type=str, default='sensor_fusion_result', help='exp comment')
parser.add_argument('--tmax', type=float, default=20, help='cosine scheduler T_max')
parser.add_argument('--data', type=str, choices=['gdcm', 'sms'], default='gdcm', help='Choose dataset loader: gdcm for GDCMDFusion, sms for SEMIDataset')

# args for Paco
parser.add_argument('--reload', default=None, type=str, help='load supervised model')
parser.add_argument('--alpha', default=0.1, type=float, help='contrast weight among samples')
parser.add_argument('--beta', default=0.5, type=float, help='contrast weight between centers and samples')
parser.add_argument('--gamma', default=0.5, type=float, help='paco loss')
parser.add_argument('--aug', default=None, type=str, help='aug strategy')
parser.add_argument('--num_classes', default=4, type=int, help='num classes in dataset')
parser.add_argument('--feat_dim', default=1280, type=int, help='last feature dim of backbone')
parser.add_argument('--moco_k', default=1024, type=int, help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco_t', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--lamb', type=float, default=1.0, help='paco loss factor')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def create_directories(path):
    head, tail = os.path.split(path)
    if head and not os.path.exists(head):
        create_directories(head)
    if not os.path.exists(path):
        os.makedirs(path)

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
        labels_one_hot = labels_one_hot.type(torch.float32).to(device)
        weights = weights.type(torch.float32).to(device)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = parser.parse_args()
    set_seed(args.seed)

    # DDP single-process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(world_size=1, rank=0)

    # Save path
    time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    save_path = os.path.join(args.save_path, f'{args.comment}_{time}')
    plotting_save_path = os.path.join(save_path, 'plotting_results')
    create_directories(plotting_save_path)

    # Logger
    setup_logger('training_log', os.path.join(save_path, 'training.log'))
    setup_logger('testing_log', os.path.join(save_path, 'testing.log'))
    training_logger = logging.getLogger('training_log')
    test_logger = logging.getLogger('testing_log')
    training_logger.info('Arguments: %s', args)

    # ---------------------------------------------------
    # 1. Dataset / DataLoader 설정
    # ---------------------------------------------------
    training_logger.info('Loading datasets...')
    if args.data == 'gdcm':
        data_root = './data/GDCMD'
        # GDCMDFusion dataset -> (train_dataset, test_dataset)
        train_dataset, test_dataset = dataloader.build_dataset(
            data_root, mode='GDCMD', seed=args.seed, imb_ratio=args.imb_ratio
        )
        fusion_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        fusion_test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.data == 'sms':
        data_root = './data/semi'
        train_dataset, test_dataset = dataloader.build_dataset(
            data_root, mode='semi', seed=args.seed, imb_ratio=args.imb_ratio
        )
        fusion_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        fusion_test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---------------------------------------------------
    # 2. 모델 및 Loss 설정
    # ---------------------------------------------------
    num_epochs = args.num_epochs
    vision_model = MobileNetV2CAE(num_classes=args.num_classes, pretrained=True, freeze_until=15).to(device)
    vision_model2 = MobileNetV2CAE(num_classes=args.num_classes, pretrained=True, freeze_until=15).to(device)
    if args.data == 'gdcm':
        sensor_model = DNN(in_dim=7, num_layers=5, num_classes=4, dim=128, skip_connection=True)
        sensor_model2 = DNN(in_dim=7, num_layers=5, num_classes=4, dim=128, skip_connection=True)
    elif args.data == 'sms':
        sensor_model = DNN(in_dim=8, num_layers=5, num_classes=4, dim=128, skip_connection=True)
        sensor_model2 = DNN(in_dim=8, num_layers=5, num_classes=4, dim=128, skip_connection=True)

    fusion_model = LateFusion(vision_model, sensor_model, dim=args.dim).to(device)
    fusion_model2 = LateFusion(vision_model2, sensor_model2, dim=args.dim).to(device)
    model = Moco_fusion_mm.MoCo(query_encoder=fusion_model, key_encoder=fusion_model2)
    
    # fusion_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    fusion_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)


    # Scheduler
    if args.scheduler == 'default':
        scheduler = None
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(fusion_optimizer, step_size=args.step_size, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fusion_optimizer, T_max=args.tmax, eta_min=0)

    # PaCoLoss
    fusion_criterion = PaCoLoss(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        temperature=args.moco_t, K=args.moco_k, num_classes=args.num_classes
    )


    fusion_criterion.cal_weight_for_classes(train_dataset.dataset.cls_num_list)
    samples_per_cls = train_dataset.dataset.get_samples_per_cls()

    aux_criterion = CBLoss(samples_per_cls=samples_per_cls, no_of_classes=4, beta=0.9999, gamma=2)

    # ---------------------------------------------------
    # 3. 학습 루프
    # ---------------------------------------------------
    meter = Meter(num_epochs, save_path, plotting_save_path, test_logger)
    m = nn.Softmax(dim=1)
    train_loss_list = []
    train_acc_list = []
    lamb = args.lamb

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0
        train_loss_val = 0.0
        test_loss_val = 0.0
        correct = 0
        total = 0
        preds_list = []
        test_targets_list = []

        # ----------------- Training -----------------
        for i, (images, sensors, targets) in enumerate(fusion_train_loader):
            targets_oh = targets.float().to(device)
            targets_int = targets_oh.argmax(dim=1)

            images_q = images[0].to(device)  # (B, 3, 224, 224)
            images_k = images[1].to(device)
            sensors_q = sensors.to(device)
            sensors_k = sensors.to(device)

            features_moco, target_moco, logits_moco, feature_concat = model(
                im_q=images_q, im_k=images_k,
                sen_q=sensors_q, sen_k=sensors_k,
                labels=targets_int
            )

            pacoloss = fusion_criterion(features_moco, target_moco, logits_moco)
            celoss   = aux_criterion(feature_concat, targets_oh)
            loss     = lamb * pacoloss + celoss

            fusion_optimizer.zero_grad()
            loss.backward()
            fusion_optimizer.step()

            train_loss_val += loss.item()

            pred_class = logits_moco.argmax(dim=1).detach().cpu()
            gt_class   = targets_int.detach().cpu()
            correct   += pred_class.eq(gt_class).sum().item()
            total     += gt_class.size(0)

            if (i+1) % 4 == 0:
                training_logger.info(
                    f"Epoch: {epoch+1}/{num_epochs}, Step: {i+1}/{len(fusion_train_loader)}, "
                    f"Loss: {loss.item():.4f}, Pacoloss: {pacoloss.item():.4f}, CE: {celoss.item():.4f}"
                )

        if scheduler:
            scheduler.step()

        epoch_train_loss = train_loss_val / len(fusion_train_loader)
        epoch_train_acc  = correct / total * 100
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)

        # ----------------- Evaluation -----------------
        model.eval()
        with torch.no_grad():
            for images, sensors, targets in fusion_test_loader:
                target_oh = targets.float().to(device)
                images_q = images.to(device)
                sensors_q = sensors.to(device)

                logit1, logit2 = model(im_q=images_q, sen_q=sensors_q)
                fusion_outputs = (logit1 + logit2) / 2

                batch_loss = nn.CrossEntropyLoss()(fusion_outputs, target_oh)
                test_loss_val += batch_loss.item()

                preds_list.append(m(fusion_outputs.cpu()))
                test_targets_list.append(target_oh.cpu())

        epoch_test_loss = test_loss_val / len(fusion_test_loader)

        preds_cat = torch.cat(preds_list, dim=0)
        targets_cat = torch.cat(test_targets_list, dim=0)
        meter.update(model, preds_cat, targets_cat, epoch_test_loss, epoch)
        meter.plot_and_save(
            range(epoch+1), train_loss_list,
            'Epoch', 'Loss',
            os.path.join(plotting_save_path, 'train_loss.png'),
            title='Train Loss'
        )
        meter.plot_and_save(
            range(epoch+1), train_acc_list,
            'Epoch', 'Accuracy',
            os.path.join(plotting_save_path, 'train_accuracy.png'),
            title='Train Accuracy'
        )

        torch.save(model.state_dict(), os.path.join(save_path, 'last_model.pth'))

if __name__ == '__main__':
    main()
