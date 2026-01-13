import os
import random
import logging
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses import *
import Moco_fusion_mm
from utils import Meter
from data.GDCMD import build_dataset
from models import MobileNetV2CAE, DNN, LateFusion

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./data/GDCMD', help='Root directory of the data')
parser.add_argument('--save_path', type=str, default='./results', help='Directory to save the model')
parser.add_argument('--seed', type=int, default=666, help='Seed')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=16, help='Number of epochs')
parser.add_argument('--image_size', type=int, default=224, help='Image size')
parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
parser.add_argument('--num_layers', type=int, default=5, help='Number of the sensor backbone layers')
parser.add_argument('--dim', type=int, default=128, help='Model inner dimension')
parser.add_argument('--imb_ratio', type=float, default=0.1, help='imbalance ratio')
parser.add_argument('--comment', type=str, default='paco_multi_modal', help='exp comment')


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

def main():
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    set_seed(args.seed)

    # Save path
    time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    save_path = os.path.join(args.save_path, f'fusion_{args.comment}_lr{args.lr}_{time}_{args.seed}')
    plotting_save_path = os.path.join(save_path, 'plotting_results')

    # Set up logger
    training_logger = logging.getLogger('training_log')
    test_logger = logging.getLogger('testing_log')
    training_logger.info('Arguments: %s', args)

    # Loading datasets
    training_logger.info('Loading datasets...')
    train_fusion_dataset, test_fusion_dataset = build_dataset(args.data_root, mode='fusion', seed=args.seed, imb_ratio=args.imb_ratio)
    num_epochs = args.num_epochs
    
    # Instantiate models
    vision_model = MobileNetV2CAE().to(device)
    vision_model2 = MobileNetV2CAE().to(device)
    sensor_model = DNN(num_layers=args.num_layers, num_classes=4, dim=args.dim).to(device)
    sensor_model2 = DNN(num_layers=args.num_layers, num_classes=4, dim=args.dim).to(device)
    fusion_model = LateFusion(vision_model, sensor_model, dim=args.dim).to(device)
    fusion_model2 = LateFusion(vision_model2, sensor_model2, dim=args.dim).to(device)            
    model = Moco_fusion_mm.MoCo(query_encoder = fusion_model, key_encoder = fusion_model2)  
    
    # Load the fusion dataset / dataloader
    fusion_test_loader = DataLoader(test_fusion_dataset, batch_size=args.batch_size, shuffle=False)

    # Training the fusion model
    meter = Meter(num_epochs, save_path, plotting_save_path, test_logger)
    m = nn.Softmax(dim=1)

    test_loss = 0
    preds = []
    test_targets = []
        
    # load model
    model_path = './results/trained_model/best_model.pth'
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(model_path, map_location=device))

    # # Evaluation
    model.eval()
    with torch.no_grad():
        for images, sensors, targets in fusion_test_loader:
            target_oh = targets.float().to(device)
            
            images_q = images.to(device)
            sensors_q = sensors.to(device)
            logit1, logit2 = model(im_q=images_q, sen_q = sensors_q)                  
            fusion_outputs = (logit1+logit2)/2
            
            test_loss += nn.CrossEntropyLoss()(fusion_outputs, target_oh)             
            preds.append(m(fusion_outputs.to('cpu')))
            test_targets.append(target_oh.to('cpu'))

    preds = torch.cat(preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    test_loss = test_loss.item() / len(fusion_test_loader)
    test_acc = (preds.argmax(dim=1).eq(test_targets.argmax(dim=1)).sum().item()/test_targets.argmax(dim=1).shape[0])
    breakpoint()
    test_acc_per_cls = meter.compute_class_accuracy(test_targets.argmax(dim=1), preds.argmax(dim=1))
    print(f'\nTest accuracy: {test_acc}, Test accuracy per class: {test_acc_per_cls}\n')


if __name__=='__main__':
    main()
