# 모달리티별 학습 가이드

이 코드는 다양한 모달리티(fusion, sensor만, vision만)로 학습할 수 있도록 확장되었습니다.

## 주요 기능

1. **Fusion 모델**: 이미지와 센서 데이터를 모두 사용 (기본)
2. **Sensor만**: 센서 데이터만 사용하여 학습
3. **Vision만**: 이미지 데이터만 사용하여 학습

## 사용법

### 1. 기본 사용법

```bash
# Fusion 모델 (기본)
python train.py --modality fusion

# Sensor만 학습
python train.py --modality sensor

# Vision만 학습
python train.py --modality vision
```

### 2. 상세 옵션

```bash
python train.py \
    --modality sensor \           # fusion, sensor, vision 중 선택
    --data gdcm \                 # gdcm 또는 sms
    --num_epochs 30 \             # 학습 에포크 수
    --batch_size 32 \             # 배치 크기
    --lr 0.001 \                  # 학습률
    --num_layers 5 \              # 센서 모델 레이어 수 (sensor 모달리티에서만 사용)
    --dim 128 \                   # 모델 차원
    --imb_ratio 0.02 \            # 불균형 비율
    --comment "sensor_only_test"  # 실험 이름
```

### 3. 예제 스크립트 실행

```bash
# 모든 모달리티 예제 실행
python run_examples.py
```

## 모달리티별 특징

### Fusion 모델 (`--modality fusion`)
- **입력**: 이미지 + 센서 데이터
- **모델**: LateFusion (MobileNetV2CAE + DNN)
- **특징**: 두 모달리티의 정보를 결합하여 최적의 성능 달성

### Sensor만 (`--modality sensor`)
- **입력**: 센서 데이터만
- **모델**: DNN (Deep Neural Network)
- **특징**: 
  - GDCM 데이터셋: 7차원 센서 데이터
  - SMS 데이터셋: 8차원 센서 데이터
  - `--num_layers`로 레이어 수 조정 가능

### Vision만 (`--modality vision`)
- **입력**: 이미지 데이터만
- **모델**: MobileNetV2CAE (사전 훈련된 모델)
- **특징**: 
  - ImageNet 사전 훈련 가중치 사용
  - 15번째 레이어까지 고정 (freeze_until=15)

## 데이터셋 메서드 활용

코드는 데이터셋의 다음 메서드들을 활용합니다:

- `get_image_only(index)`: 이미지 데이터만 반환
- `get_sensor_only(index)`: 센서 데이터만 반환  
- `get_label_only(index)`: 라벨만 반환

## 모델 구조

### Sensor 모달리티
```python
sensor_model = DNN(
    in_dim=7,  # GDCM: 7, SMS: 8
    num_layers=args.num_layers,
    num_classes=args.num_classes,
    dim=args.dim,
    skip_connection=True
)
```

### Vision 모달리티
```python
vision_model = MobileNetV2CAE(
    num_classes=args.num_classes,
    pretrained=True,
    freeze_until=15
)
```

### Fusion 모달리티
```python
fusion_model = LateFusion(
    vision_model=vision_model,
    sensor_model=sensor_model,
    dim=args.dim
)
```

## 학습 과정

1. **데이터 로딩**: 모달리티에 따라 적절한 데이터만 로드
2. **모델 초기화**: 선택된 모달리티에 맞는 모델 생성
3. **MoCo 학습**: PaCo Loss를 사용한 contrastive learning
4. **평가**: 테스트 데이터로 성능 평가

## 결과 저장

학습 결과는 다음 경로에 저장됩니다:
```
./results/{comment}_{timestamp}/
├── training.log
├── testing.log
├── last_model.pth
└── plotting_results/
    ├── train_loss.png
    ├── train_accuracy.png
    └── ...
```

## 주의사항

1. **메모리 사용량**: Vision 모달리티는 이미지 처리로 인해 더 많은 메모리가 필요할 수 있습니다.
2. **배치 크기**: 모달리티에 따라 적절한 배치 크기를 설정하세요.
3. **학습률**: Sensor 모달리티는 일반적으로 더 높은 학습률이 필요할 수 있습니다.

## 예제 명령어

```bash
# Sensor만 학습 (빠른 테스트)
python train.py --modality sensor --num_epochs 5 --batch_size 64 --lr 0.01

# Vision만 학습 (이미지 처리)
python train.py --modality vision --num_epochs 10 --batch_size 16 --lr 0.001

# Fusion 모델 학습 (전체 성능)
python train.py --modality fusion --num_epochs 30 --batch_size 32 --lr 0.001
``` 