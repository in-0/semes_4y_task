#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from in0, 0419
import os
import argparse
import numpy as np
import pandas as pd
import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# from ptflops import get_model_complexity_info
import logging

from data.dataloader import build_dataset, GDCMDFusion
from models import MobileNetV2CAE, DNN, LateFusion
from utils import Meter
from logger import setup_logger
from losses import *


parser = argparse.ArgumentParser()
parser.add_argument('data_root', type=str, default=None, help='Root directory of the data')
parser.add_argument('--save_path', type=str, default='./sensor_results', help='Directory to save the model')
parser.add_argument('--seed', type=int, default=666, help='Seed')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--image_size', type=int, default=224, help='Image size')
parser.add_argument('--scheduler', type=str, choices=['default', 'step', 'cosine'], default='default', help='Choose Scheduler')
parser.add_argument('--step_size', type=int, default=30, help='Step size for learning rate scheduler')
parser.add_argument('--loss', choices=['ce', 'reweight', 'BalancedSoftmax', 'Paco'], default='ce', help='Loss function')
parser.add_argument('--classifier', type=str, choices=['linear', 'cosine'], default='linear', help='Choose classifier type')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--num_layers', type=int, default=5, help='Number of the sensor backbone layers')
parser.add_argument('--dim', type=int, default=128, help='Sensor model inner dimension')
parser.add_argument('--imb_ratio', type=float, default=0, help='imabalance ratio')
parser.add_argument('--norm', type=str, choices=['batch', 'layer'], default = 'batch', help='normalization type')
parser.add_argument('--comment', type=str, default=None, help='exp comment')


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
    
    if head:
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
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,self.no_of_classes)

        pred = logits.softmax(dim = 1)
        
        labels_one_hot = labels_one_hot.type(torch.float32).to(device)
        weights = weights.type(torch.float32).to(device)
        
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss
    

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = parser.parse_args()
    set_seed(args.seed)


    # Save path
    time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    save_path = os.path.join(args.save_path, f'{time}_sensor_{args.comment}_classifier_{args.classifier}_{args.seed}_{args.imb_ratio}')
    plotting_save_path = os.path.join(save_path, 'plotting_results')
    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path, exist_ok=True)
    # os.makedirs(save_path, exist_ok=True)
    create_directories(plotting_save_path)
    # os.makedirs(plotting_save_path, exist_ok=True)
    
    # Set up logger
    setup_logger('training_log', os.path.join(save_path, 'training.log'))
    setup_logger('testing_log', os.path.join(save_path, 'testing.log'))
    training_logger = logging.getLogger('training_log')
    test_logger = logging.getLogger('testing_log')
    training_logger.info('Arguments: %s', args)

    # Loading datasets
    training_logger.info('Loading datasets...')

    train_sensor_dataset, test_sensor_dataset = build_dataset(args.data_root, mode='sensor', seed=args.seed, imb_ratio=args.imb_ratio, Paco = False)

    num_epochs = args.num_epochs
    learning_rate = args.lr

    # Instantiate models
    sensor_model = DNN(num_layers=args.num_layers, num_classes=4, dim=args.dim).to(device)


    # Optimizer
    sensor_optimizer = torch.optim.SGD(sensor_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    if args.scheduler == 'default':
        scheduler = None
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(sensor_optimizer, step_size=args.step_size, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sensor_optimizer, T_max = 400, eta_min = 0)

    # Load the sensor dataset / dataloader
    sensor_train_loader = DataLoader(train_sensor_dataset, batch_size=args.batch_size, shuffle=True)
    sensor_test_loader = DataLoader(test_sensor_dataset, batch_size=args.batch_size, shuffle=False)

    if args.loss == 'ce':
        sensor_criterion = nn.CrossEntropyLoss()
    elif args.loss == 'reweight':
        num_classes = 4
        beta = 0.9999
        gamma = 2
        samples_per_cls = train_sensor_dataset.get_samples_per_cls()
        sensor_criterion = CBLoss(samples_per_cls=samples_per_cls, no_of_classes=num_classes, beta=beta, gamma=gamma)
    elif args.loss == 'BalancedSoftmax':
        sensor_criterion = BalancedSoftmax('./data/GDCMD/freq.json') # imbalance 비율 바뀔 때 freq.json 파일 수정


    # Training the sensor model
    meter = Meter(num_epochs, save_path, plotting_save_path, test_logger)
    m = nn.Softmax(dim=1)
    train_loss_list = []
    train_acc_list = []
    
    model = sensor_model
    for epoch in range(num_epochs):
        train_acc = 0
        train_loss = 0
        test_loss = 0
        correct = 0
        total = 0
        preds = []
        test_targets = []
      
        model.train()
        for i, (images, targets) in enumerate(sensor_train_loader):
            images = images.to(device)
            targets = targets.float().to(device)
                                    
            sensor_outputs, _ = model(images)
            loss = sensor_criterion(sensor_outputs, targets)

            sensor_optimizer.zero_grad()
            loss.backward()
            sensor_optimizer.step()
            train_loss += loss.item()
            train_preds = sensor_outputs.detach().clone().to('cpu')
            tmp_targets = targets.detach().clone().to('cpu').argmax(dim=1)

            correct = train_preds.argmax(dim=1).eq(tmp_targets).sum().item()
            tmp_acc = correct / tmp_targets.shape[0]
            total += tmp_targets.shape[0]
            train_acc += correct

            training_logger.info(f'loss: {loss.item():.3f}')
            if (i+1) % 4 == 0:
                training_logger.info(f'Mode: Training sensor\tEpoch: [{epoch+1}/{num_epochs}]\tStep: [{i+1}/{len(sensor_train_loader)}]\tAccuracy: {(tmp_acc*100):.4f}\tLoss: {loss.item():.4f}')

        if scheduler:
            scheduler.step()
        train_loss_list.append((train_loss / len(sensor_train_loader)))
        train_acc_list.append((train_acc/total)*100)

        # Evaluation
        model.eval()
        with torch.no_grad():
            for images, targets in sensor_test_loader:
                images = images.to(device)
                target = targets.float().to(device)
                               
                sensor_outputs, _ = model(images)
                test_loss += nn.CrossEntropyLoss()(sensor_outputs, target)

                preds.append(m(sensor_outputs.to('cpu')))
                test_targets.append(target.to('cpu'))

        preds = torch.cat(preds, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        test_loss = test_loss.item() / len(sensor_test_loader)
        meter.update(model, preds, test_targets, test_loss, epoch)
        meter.plot_and_save(range(epoch+1), train_loss_list, 'Epoch', 'Loss', os.path.join(plotting_save_path, 'train_loss.png'), title='Train Loss')
        meter.plot_and_save(range(epoch+1), train_acc_list, 'Epoch', 'Accuracy', os.path.join(plotting_save_path, 'train_accuracy.png'), title='Train Accuracy')
        torch.save(model.state_dict(), os.path.join(save_path, 'last_sensor_model.pth'))
        

if __name__=='__main__':
    main()
