#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import numpy as np
import Moco_fusion_mm

# SMS 클래스명 순서 정의
SMS_CLASS_NAME = ['normal', 'caution', 'warning', 'critical']

# MobileNet for image data
class MobileNetV2CAE(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, freeze_until=None):

        super(MobileNetV2CAE, self).__init__()

        self.pre_train = pretrained
        self.mobilenet = mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
        # mobilenet의 features 모듈을 encoder로 사용
        self.encoder = self.mobilenet.features
        
        if freeze_until is not None:
            # freeze_until 이전까지의 레이어만 freeze
            for idx, layer in enumerate(self.encoder):
                if idx < freeze_until:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            # 기본적으로 모든 파라미터를 고정 (freeze)
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # mobilenet_v2의 마지막 출력 채널은 1280입니다.
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x), x

class DNN(nn.Module):
    def __init__(self, in_dim, num_layers, num_classes, dim, skip_connection=True) -> None:
        super(DNN, self).__init__()
        self.skip_connection = skip_connection
        self.dim = dim
        self.act = nn.ReLU()
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(in_dim, self.dim))  # input layer
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(self.dim, self.dim))  # hidden layers
        
        self.classifier = nn.Linear(self.dim, num_classes)

    def forward(self, x):
        x = x.float()
        residual = 0
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if self.skip_connection and i % 3 == 2:
                x = x + residual
                residual = x
        return self.classifier(x), x

# Sensor-only model with text embeddings
class SensorTextFusion(nn.Module):
    def __init__(self, sensor_model, dim, num_classes=4, use_textemb=False):
        super(SensorTextFusion, self).__init__()
        self.sensor_model = sensor_model
        self.use_textemb = use_textemb
        self.dim = dim
        self.num_classes = num_classes
        
        # use_textemb가 True일 때 text embeddings 불러오기
        if self.use_textemb:
            try:
                text_embs = torch.from_numpy(np.load('./text_embeddings.npy')).float()
                print(f"Loaded text embeddings with shape: {text_embs.shape}")
            except FileNotFoundError:
                print("Warning: text_embeddings.npy not found. Using random embeddings.")
                text_embs = torch.randn(num_classes, dim)  # sensor 차원으로 설정
            except Exception as e:
                print(f"Error loading text embeddings: {e}")
                text_embs = torch.randn(num_classes, dim)
            self.register_buffer('text_embs', text_embs)
        
        # 분류기
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        _, x = self.sensor_model(x)
        
        # use_textemb가 True이고 training 모드일 때만 similarity 계산
        if self.use_textemb and self.training:
            # text embeddings를 x와 같은 device로 이동
            text_embs = self.text_embs.to(x.device)
            
            # x와 text_embs 간의 cosine similarity 계산
            x_similarities = F.cosine_similarity(
                x.unsqueeze(1),  # [batch_size, 1, dim]
                text_embs.unsqueeze(0),  # [1, num_classes, dim]
                dim=2
            )  # [batch_size, num_classes]
            
            # Softmax로 가중치 계산
            x_weights = F.softmax(x_similarities, dim=1)  # [batch_size, num_classes]
            
            # 가중 평균 계산 (각 클래스별로)
            x_weighted = torch.zeros_like(x)
            for i in range(self.num_classes):
                x_weighted += x_weights[:, i:i+1] * x
            
            # 가중치가 적용된 feature로 분류
            return self.classifier(x_weighted), x_weighted
        else:
            # test 모드이거나 use_textemb가 False인 경우 원래 stream
            return self.classifier(x), x

# Vision-only model with text embeddings
class VisionTextFusion(nn.Module):
    def __init__(self, vision_model, num_classes=4, use_textemb=False):
        super(VisionTextFusion, self).__init__()
        self.vision_model = vision_model
        self.use_textemb = use_textemb
        self.num_classes = num_classes
        
        # use_textemb가 True일 때 text embeddings 불러오기
        if self.use_textemb:
            try:
                # merged_text_embeddings.pth 파일 불러오기
                text_embeddings_dict = torch.load('./semi_text_embs/merged_text_embeddings.pth')
                # {'class': feat} 형태를 [num_classes, num_prompts_feats, 1280] 형태로 변환
                max_prompts = max([feat.shape[0] if feat.dim() > 1 else 1 for feat in text_embeddings_dict.values()])
                text_embs = torch.zeros(num_classes, max_prompts, 1280)  # [num_classes, num_prompts_feats, 1280]
                for i in range(num_classes):
                    if i in text_embeddings_dict:
                        feat = text_embeddings_dict[i]
                        if feat.dim() == 1:
                            feat = feat.unsqueeze(0)  # [1280] -> [1, 1280]
                        text_embs[i, :feat.shape[0], :] = feat
                    else:
                        print(f"Warning: class {i} not found in text embeddings, using random embedding")
                        text_embs[i] = torch.randn(max_prompts, 1280)
                print(f"Loaded text embeddings with shape: {text_embs.shape}")
            except FileNotFoundError:
                print("Warning: ./semi_text_embs/merged_text_embeddings.pth not found. Using random embeddings.")
                text_embs = torch.randn(num_classes, 10, 1280)  # 기본값으로 10개 프롬프트 사용
            except Exception as e:
                print(f"Error loading text embeddings: {e}")
                text_embs = torch.randn(num_classes, 10, 1280)
            self.register_buffer('text_embs', text_embs)
        
        # 분류기
        self.classifier = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        _, x = self.vision_model(x)
        
        # use_textemb가 True이고 training 모드일 때만 similarity 계산
        if self.use_textemb and self.training:
            # text embeddings를 x와 같은 device로 이동
            text_embs = self.text_embs.to(x.device)
            
            # x와 text_embs 간의 cosine similarity 계산
            x_similarities = F.cosine_similarity(
                x.unsqueeze(1),  # [batch_size, 1, 1280]
                text_embs.unsqueeze(0),  # [1, num_classes, 1280]
                dim=2
            )  # [batch_size, num_classes]
            
            # Softmax로 가중치 계산
            x_weights = F.softmax(x_similarities, dim=1)  # [batch_size, num_classes]
            
            # 가중 평균 계산 (각 클래스별로)
            x_weighted = torch.zeros_like(x)
            for i in range(self.num_classes):
                x_weighted += x_weights[:, i:i+1] * x
            
            # 가중치가 적용된 feature로 분류
            return self.classifier(x_weighted), x_weighted
        else:
            # test 모드이거나 use_textemb가 False인 경우 원래 stream
            return self.classifier(x), x

# Fusion model that combine the vision and sensor model
class LateFusion(nn.Module):
    def __init__(self, vision_model, sensor_model, dim, num_classes=4, use_textemb=False, use_dim_matching_layer=False):
        super(LateFusion, self).__init__()
        self.vision_model = vision_model
        self.sensor_model = sensor_model
        self.num_classes = num_classes
        self.dim = dim
        self.use_dim_matching_layer = use_dim_matching_layer
        
        # use_textemb는 use_dim_matching_layer가 있을 경우에만 사용 가능
        if use_textemb and not use_dim_matching_layer:
            raise ValueError("use_textemb can only be used when use_dim_matching_layer is set")
        self.use_textemb = use_textemb if use_dim_matching_layer else False
        
        # use_dim_matching_layer가 있을 때만 dimension matching layer 생성
        if self.use_dim_matching_layer:
            # dimension matching layer의 출력 차원을 1024로 고정
            target_dim = 1024
            self.vision_dimension_adapter = nn.Sequential(
                nn.Linear(1280, target_dim),
                nn.BatchNorm1d(target_dim),
                nn.ReLU()
            )
            self.sensor_dimension_adapter = nn.Sequential(
                nn.Linear(dim, target_dim),
                nn.BatchNorm1d(target_dim),
                nn.ReLU()
            )
            fusion_dim = target_dim + target_dim  # vision(1024) + sensor(1024)
        else:
            # 일반 fusion 방식: dimension matching layer 없음
            self.vision_dimension_adapter = None
            self.sensor_dimension_adapter = None
            fusion_dim = 1280 + dim  # vision(1280) + sensor(dim=128)

        if self.use_textemb:
            try:
                # merged_text_embeddings.pth 파일 불러오기
                text_embeddings_dict = torch.load('./semi_text_embs/merged_text_embeddings.pth')
                # {'class_name': feat} 형태를 [num_classes, 1024] 형태로 변환
                # 각 클래스의 prompts를 평균내어 하나의 embedding으로 만듦
                # SMS_CLASS_NAME 순서대로 텐서 구성
                text_embs = torch.zeros(num_classes, 1024)  # [num_classes, 1024]
                
                for idx, class_name in enumerate(SMS_CLASS_NAME):
                    if class_name in text_embeddings_dict:
                        feat = text_embeddings_dict[class_name]
                        if feat.dim() == 1:
                            # 이미 [1024] 형태면 그대로 사용
                            text_embs[idx] = feat
                        else:
                            # [num_prompts, 1024] 형태면 prompts 차원에 대해 평균
                            text_embs[idx] = feat.mean(dim=0)  # [num_prompts, 1024] -> [1024]
                    else:
                        print(f"Warning: class '{class_name}' not found in text embeddings, using random embedding")
                        text_embs[idx] = torch.randn(1024)
                
                print(f"Loaded text embeddings with shape: {text_embs.shape}")
            except FileNotFoundError:
                print("Warning: ./semi_text_embs/merged_text_embeddings.pth not found. Using random embeddings.")
                text_embs = torch.randn(num_classes, 1024)  # [num_classes, 1024]
            except Exception as e:
                print(f"Error loading text embeddings: {e}")
                text_embs = torch.randn(num_classes, 1024)
            self.register_buffer('text_embs', text_embs)

        self.fusion_dim = fusion_dim
        linear_add = nn.Linear(fusion_dim, fusion_dim)
        norm_add = nn.BatchNorm1d(fusion_dim)
        self.add_layer = nn.Sequential(linear_add, norm_add, nn.ReLU())
        self.norm1 = nn.BatchNorm1d(1280)  # multimodal
        self.norm2 = nn.BatchNorm1d(dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)  # multimodal

    def forward(self, x1, x2):
        _, x1 = self.vision_model(x1)
        _, x2 = self.sensor_model(x2)       
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        
        # use_dim_matching_layer가 있을 때만 dimension matching layer를 통과
        if self.use_dim_matching_layer:
            if self.vision_dimension_adapter is not None:
                x1 = self.vision_dimension_adapter(x1)
            if self.sensor_dimension_adapter is not None:
                x2 = self.sensor_dimension_adapter(x2)
        
        if self.use_textemb and self.training:
            # text embeddings를 x1과 x2와 같은 device로 이동
            text_embs = self.text_embs.to(x1.device)
            
            # x1과 text_embs 간의 cosine similarity 계산
            x1_similarities = F.cosine_similarity(
                x1.unsqueeze(1),  # [batch_size, 1, 1024]
                text_embs.unsqueeze(0),  # [1, num_classes, 1024]
                dim=2
            )
            x2_similarities = F.cosine_similarity(
                x2.unsqueeze(1),  # [batch_size, 1, 1024]
                text_embs.unsqueeze(0),  # [1, num_classes, 1024]
                dim=2
            )  # [batch_size, num_classes]
            
            # 각 샘플에 대해 가장 큰 similarity score만 사용
            x1_sim_max = x1_similarities.max(dim=1)[0]  # [batch_size] - 가장 큰 similarity
            x2_sim_max = x2_similarities.max(dim=1)[0]  # [batch_size] - 가장 큰 similarity
            
            # alpha 계산 (가장 큰 similarity를 기준으로)
            alpha = x1_sim_max / (x1_sim_max + x2_sim_max + 1e-8)  # [batch_size]
            alpha = alpha.unsqueeze(1)  # [batch_size, 1] - 브로드캐스팅을 위해 차원 추가
            
            # 가중치를 적용하여 fusion features 조정
            x1_weighted = x1 + alpha * x1  # [batch_size, 1] * [batch_size, 1024] -> [batch_size, 1024]
            x2_weighted = x2 + (1 - alpha) * x2  # [batch_size, 1] * [batch_size, 1024] -> [batch_size, 1024]
            x = torch.cat((x1_weighted, x2_weighted), dim=1)
        else:
            x = torch.cat((x1, x2), dim=1)
        
        x = self.add_layer(x)
        return self.classifier(x), x, x1, x2

def create_model(args, device):
    """모델을 생성하는 함수"""
    if args.modality == 'fusion':
        # 기존 fusion 모델
        vision_model = MobileNetV2CAE(num_classes=args.num_classes, pretrained=True, freeze_until=15).to(device)
        vision_model2 = MobileNetV2CAE(num_classes=args.num_classes, pretrained=True, freeze_until=15).to(device)
        if args.data == 'gdcm':
            sensor_model = DNN(in_dim=7, num_layers=5, num_classes=4, dim=128, skip_connection=True)
            sensor_model2 = DNN(in_dim=7, num_layers=5, num_classes=4, dim=128, skip_connection=True)
        elif args.data == 'sms':
            sensor_model = DNN(in_dim=8, num_layers=5, num_classes=4, dim=128, skip_connection=True)
            sensor_model2 = DNN(in_dim=8, num_layers=5, num_classes=4, dim=128, skip_connection=True)

        # fusion_type에 따라 다른 fusion 모델 사용
        fusion_model = LateFusion(vision_model, sensor_model, dim=args.dim, use_textemb=args.use_textemb, use_dim_matching_layer=args.use_dim_matching_layer).to(device)
        fusion_model2 = LateFusion(vision_model2, sensor_model2, dim=args.dim, use_textemb=args.use_textemb, use_dim_matching_layer=args.use_dim_matching_layer).to(device)
        
        # use_dim_matching_layer가 있을 때 feat_dim 계산
        if args.use_dim_matching_layer:
            target_dim = 1024
            feat_dim = target_dim * 2  # vision(1024) + sensor(1024)
        else:
            feat_dim = args.feat_dim
        
        if args.use_paco:
            model = Moco_fusion_mm.MoCo(query_encoder=fusion_model, key_encoder=fusion_model2, modality=args.modality, feat_dim=feat_dim, use_dim_matching_layer=args.use_dim_matching_layer, m=args.moco_m, T=args.moco_t, K=args.moco_k)
        else:
            model = fusion_model
        
    elif args.modality == 'sensor':
        # 센서만 사용하는 모델
        if args.data == 'gdcm':
            sensor_model = DNN(in_dim=7, num_layers=args.num_layers, num_classes=args.num_classes, dim=args.dim, skip_connection=True).to(device)
            sensor_model2 = DNN(in_dim=7, num_layers=args.num_layers, num_classes=args.num_classes, dim=args.dim, skip_connection=True).to(device)
        elif args.data == 'sms':
            sensor_model = DNN(in_dim=8, num_layers=args.num_layers, num_classes=args.num_classes, dim=args.dim, skip_connection=True).to(device)
            sensor_model2 = DNN(in_dim=8, num_layers=args.num_layers, num_classes=args.num_classes, dim=args.dim, skip_connection=True).to(device)
        
        # use_textemb가 True이면 SensorTextFusion 사용
        if args.use_textemb:
            sensor_model = SensorTextFusion(sensor_model, dim=args.dim, num_classes=args.num_classes, use_textemb=True).to(device)
            sensor_model2 = SensorTextFusion(sensor_model2, dim=args.dim, num_classes=args.num_classes, use_textemb=True).to(device)
        
        model = Moco_fusion_mm.MoCo(query_encoder=sensor_model, key_encoder=sensor_model2, modality=args.modality, feat_dim=args.dim, use_dim_matching_layer=args.use_dim_matching_layer, m=args.moco_m, T=args.moco_t, K=args.moco_k)
        
    elif args.modality == 'vision':
        # 이미지만 사용하는 모델
        vision_model = MobileNetV2CAE(num_classes=args.num_classes, pretrained=True, freeze_until=15).to(device)
        vision_model2 = MobileNetV2CAE(num_classes=args.num_classes, pretrained=True, freeze_until=15).to(device)
        
        # use_textemb가 True이면 VisionTextFusion 사용
        if args.use_textemb:
            vision_model = VisionTextFusion(vision_model, num_classes=args.num_classes, use_textemb=True).to(device)
            vision_model2 = VisionTextFusion(vision_model2, num_classes=args.num_classes, use_textemb=True).to(device)
        
        model = Moco_fusion_mm.MoCo(query_encoder=vision_model, key_encoder=vision_model2, modality=args.modality, feat_dim=args.feat_dim, use_dim_matching_layer=args.use_dim_matching_layer, m=args.moco_m, T=args.moco_t, K=args.moco_k)
    
    return model

def create_optimizer_and_scheduler(model, args):
    """옵티마이저와 스케줄러를 생성하는 함수"""
    import torch.optim as optim
    
    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # 스케줄러
    if args.scheduler == 'default':
        scheduler = None
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    
    return optimizer, scheduler
