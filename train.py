import os
import random
import logging
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import *
import Moco_fusion_mm
from utils import Meter
from logger import setup_logger
import data.dataloader as dataloader 
from models import create_model, create_optimizer_and_scheduler
from losses import create_loss_functions

parser = argparse.ArgumentParser()

parser.add_argument('--save_path', type=str, default='./results', help='Directory to save the model')
parser.add_argument('--seed', type=int, default=666, help='Seed')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--scheduler', type=str, choices=['default', 'step', 'cosine'], default='cosine', help='Choose Scheduler')
parser.add_argument('--step_size', type=int, default=8, help='Step size for learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_layers', type=int, default=5, help='Number of the sensor backbone layers')
parser.add_argument('--dim', type=int, default=128, help='Model inner dimension')
parser.add_argument('--imb_ratio', type=float, default=0.02, help='imbalance ratio')
parser.add_argument('--comment', type=str, default='sensor_fusion_result', help='exp comment')
parser.add_argument('--data', type=str, choices=['gdcm', 'sms'], default='gdcm', help='Choose dataset loader: gdcm for GDCMDFusion, sms for SEMIDataset')
parser.add_argument('--modality', type=str, choices=['fusion', 'sensor', 'vision'], default='fusion', help='Choose modality: fusion, sensor only, or vision only')
parser.add_argument('--use_textemb', action='store_true', help='Use text embeddings for similarity-based weighting in LateFusion')
parser.add_argument('--use_dim_matching_layer', action='store_true', help='Use dimension matching layer in LateFusion (target_dim=1024)')

# args for Paco
parser.add_argument('--alpha', default=0.1, type=float, help='contrast weight among samples')
parser.add_argument('--beta', default=0.5, type=float, help='contrast weight between centers and samples')
parser.add_argument('--gamma', default=0.5, type=float, help='paco loss')
parser.add_argument('--aug', default=None, type=str, help='aug strategy')
parser.add_argument('--num_classes', default=4, type=int, help='num classes in dataset')
parser.add_argument('--feat_dim', default=1280, type=int, help='last feature dim of backbone')
parser.add_argument('--moco_k', default=1024, type=int, help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco_t', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--lamb_paco_fusion', type=float, default=1.0, help='paco loss factor for fusion')
parser.add_argument('--lamb_ce_fusion', type=float, default=1.0, help='ce loss factor for fusion')
parser.add_argument('--lamb_mtm_fusion', type=float, default=1.0, help='mtm loss factor for fusion')
parser.add_argument('--mtm_lambda', type=float, default=0.5, help='lambda weight for mtm loss')

# args for loss functions
parser.add_argument('--use_paco', action='store_true', help='Use PaCoLoss')
parser.add_argument('--use_cb', action='store_true', help='Use CBLoss')
parser.add_argument('--use_mtm', action='store_true', help='Use MTMLoss (Modality-Text Matching Loss)')

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = parser.parse_args()
    set_seed(args.seed)

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
    
    # use_textemb 관련 로깅
    if args.use_textemb:
        training_logger.info('Using text embeddings for similarity-based weighting')
        if args.modality == 'fusion':
            training_logger.info('Text embeddings will be used in LateFusion')
        elif args.modality == 'sensor':
            training_logger.info('Text embeddings will be used in SensorTextFusion')
        elif args.modality == 'vision':
            training_logger.info('Text embeddings will be used in VisionTextFusion')
        else:
            training_logger.warning('use_textemb is set but may not be fully utilized with current settings')
        training_logger.info('Text embeddings will be loaded from ./semi_text_embs/merged_text_embeddings.pth')
    else:
        training_logger.info('Not using text embeddings (use_textemb=False)')

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
    elif args.data == 'sms':
        data_root = './data/semi'
        train_dataset, test_dataset = dataloader.build_dataset(
            data_root, mode='semi', seed=args.seed, imb_ratio=args.imb_ratio
        )

    # 데이터로더 생성
    fusion_train_loader, fusion_test_loader = dataloader.create_data_loaders(args, train_dataset, test_dataset)

    # ---------------------------------------------------
    # 2. 모델 및 Loss 설정
    # ---------------------------------------------------
    num_epochs = args.num_epochs
    
    # 모델 생성
    model = create_model(args, DEVICE)
    
    # 옵티마이저와 스케줄러 생성
    fusion_optimizer, scheduler = create_optimizer_and_scheduler(model, args)

    # 데이터셋 정보 가져오기
    cls_num_list, samples_per_cls = dataloader.get_dataset_info(train_dataset)
    
    # 손실 함수 생성
    fusion_criterion, aux_criterion, mtm_criterion = create_loss_functions(args, cls_num_list, samples_per_cls, use_paco=args.use_paco, use_cb=args.use_cb, use_mtm=args.use_mtm)

    # ---------------------------------------------------
    # 3. 학습 루프
    # ---------------------------------------------------
    meter = Meter(num_epochs, save_path, plotting_save_path, test_logger)
    m = nn.Softmax(dim=1)
    train_loss_list = []
    train_acc_list = []

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
        for i, data in enumerate(fusion_train_loader):
            if args.modality == 'fusion':
                images, sensors, targets = data
                targets_oh = targets.float().to(DEVICE)
                targets_int = targets_oh.argmax(dim=1)

                images_q = images[0].to(DEVICE)  # (B, 3, 224, 224)
                images_k = images[1].to(DEVICE)
                sensors_q = sensors.to(DEVICE)
                sensors_k = sensors.to(DEVICE)

                features_moco, target_moco, logits_moco, feature_concat, vision_feature, sensor_feature = model(
                    im_q=images_q, im_k=images_k,
                    sen_q=sensors_q, sen_k=sensors_k,
                    labels=targets_int
                )
                
            elif args.modality == 'sensor':
                sensors, targets = data
                targets_oh = targets.float().to(DEVICE)
                targets_int = targets_oh.argmax(dim=1)

                sensors_q = sensors.to(DEVICE)
                sensors_k = sensors.to(DEVICE)

                features_moco, target_moco, logits_moco, feature_concat, vision_feature, sensor_feature = model(
                    sen_q=sensors_q, sen_k=sensors_k,
                    labels=targets_int
                )
                
            elif args.modality == 'vision':
                images, targets = data
                targets_oh = targets.float().to(DEVICE)
                targets_int = targets_oh.argmax(dim=1)

                images_q = images[0].to(DEVICE)  # (B, 3, 224, 224)
                images_k = images[1].to(DEVICE)

                features_moco, target_moco, logits_moco, feature_concat, vision_feature, sensor_feature = model(
                    im_q=images_q, im_k=images_k,
                    labels=targets_int
                )

            pacoloss = fusion_criterion(features_moco, target_moco, logits_moco) if fusion_criterion is not None else torch.tensor(0.0).to(DEVICE)
            celoss = aux_criterion(feature_concat, targets_oh) if aux_criterion is not None else torch.tensor(0.0).to(DEVICE)
            
            # MTMLoss 계산
            mtmloss = torch.tensor(0.0).to(DEVICE)
            if mtm_criterion is not None:
                if args.modality == 'fusion' and vision_feature is not None and sensor_feature is not None:
                    # fusion 모달리티: vision과 sensor 피쳐 모두 사용
                    mtmloss = mtm_criterion(vision_feature, sensor_feature, targets_int)
                elif args.modality == 'sensor' and sensor_feature is not None:
                    # sensor 모달리티: sensor 피쳐만 사용 (vision 피쳐는 None으로)
                    mtmloss = mtm_criterion(None, sensor_feature, targets_int)
                elif args.modality == 'vision' and vision_feature is not None:
                    # vision 모달리티: vision 피쳐만 사용 (sensor 피쳐는 None으로)
                    mtmloss = mtm_criterion(vision_feature, None, targets_int)
            
            # 최종 loss 계산
            loss = args.lamb_paco_fusion * pacoloss + args.lamb_ce_fusion * celoss + args.lamb_mtm_fusion * mtmloss

            fusion_optimizer.zero_grad()
            loss.backward()
            fusion_optimizer.step()

            train_loss_val += loss.item()

            pred_class = logits_moco.argmax(dim=1).detach().cpu()
            gt_class   = targets_int.detach().cpu()
            correct   += pred_class.eq(gt_class).sum().item()
            total     += gt_class.size(0)

            if (i+1) % 4 == 0:
                celoss_value = celoss.item() if isinstance(celoss, torch.Tensor) else celoss
                mtmloss_value = mtmloss.item() if isinstance(mtmloss, torch.Tensor) else mtmloss
                training_logger.info(
                    f"Epoch: {epoch+1}/{num_epochs}, Step: {i+1}/{len(fusion_train_loader)}, "
                    f"Loss: {loss.item():.4f}, Pacoloss: {pacoloss.item():.4f}, CE: {celoss_value:.4f}, MTM: {mtmloss_value:.4f}"
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
            for data in fusion_test_loader:
                if args.modality == 'fusion':
                    images, sensors, targets = data
                    target_oh = targets.float().to(DEVICE)
                    images_q = images.to(DEVICE)
                    sensors_q = sensors.to(DEVICE)

                    logit1, logit2 = model(im_q=images_q, sen_q=sensors_q)
                    
                elif args.modality == 'sensor':
                    sensors, targets = data
                    target_oh = targets.float().to(DEVICE)
                    sensors_q = sensors.to(DEVICE)

                    logit1, logit2 = model(sen_q=sensors_q)
                    
                elif args.modality == 'vision':
                    images, targets = data
                    target_oh = targets.float().to(DEVICE)
                    images_q = images.to(DEVICE)

                    logit1, logit2 = model(im_q=images_q)
                
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