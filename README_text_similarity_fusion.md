# 텍스트 임베딩 기반 Similarity Weighting Fusion

## 개요

이 프로젝트는 ["Normal", "Caution", "Warning", "Critical"] 4개 클래스의 텍스트 임베딩과 이미지/센서 피쳐 간의 유사도를 계산하여 가중치를 적용하는 fusion 모델을 구현합니다.

## 핵심 아이디어

### 1. 텍스트 임베딩 생성
- 각 클래스("Normal", "Caution", "Warning", "Critical")에 대해 학습 가능한 텍스트 임베딩을 생성
- `TextEmbeddingGenerator` 클래스에서 클래스별 임베딩을 학습 가능한 파라미터로 관리

### 2. 유사도 계산
- 이미지 피쳐와 센서 피쳐를 텍스트 임베딩 공간으로 투영
- 코사인 유사도를 사용하여 각 모달리티와 클래스별 텍스트 임베딩 간의 유사도 계산
- Temperature scaling을 적용하여 유사도 분포 조정

### 3. 가중치 적용
- Softmax를 사용하여 유사도를 가중치로 변환
- 각 클래스별로 이미지와 센서 피쳐에 가중치 적용
- 최종적으로 학습 가능한 파라미터를 통해 이미지와 센서의 중요도 조정

## 모델 구조

### TextEmbeddingGenerator
```python
class TextEmbeddingGenerator(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=4):
        # 클래스별 텍스트 임베딩을 학습 가능한 파라미터로 정의
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, embedding_dim))
        # 텍스트 임베딩을 피쳐 공간으로 매핑하는 레이어
        self.text_projection = nn.Sequential(...)
```

### TextSimilarityFusion
```python
class TextSimilarityFusion(nn.Module):
    def __init__(self, vision_model, sensor_model, dim, num_classes=4, 
                 embedding_dim=512, temperature=0.1):
        # 텍스트 임베딩 생성기
        self.text_embedding_generator = TextEmbeddingGenerator(embedding_dim, num_classes)
        # 이미지와 센서 피쳐를 텍스트 임베딩 공간으로 매핑
        self.vision_projection = nn.Linear(1280, embedding_dim)
        self.sensor_projection = nn.Linear(dim, embedding_dim)
        # 가중치 조정을 위한 학습 가능한 파라미터
        self.vision_weight = nn.Parameter(torch.tensor(0.5))
        self.sensor_weight = nn.Parameter(torch.tensor(0.5))
```

## 사용법

### 1. 기본 실행
```bash
python train.py \
    --modality fusion \
    --fusion_type text_similarity \
    --embedding_dim 512 \
    --temperature 0.1 \
    --data gdcm \
    --num_epochs 30 \
    --batch_size 32 \
    --lr 0.001
```

### 2. 예제 스크립트 사용
```bash
# 기본 설정으로 실행
python run_text_similarity_fusion.py

# 다양한 설정으로 실험 실행
python run_text_similarity_fusion.py multi
```

### 3. 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--fusion_type` | Fusion 방식 선택 (`late` 또는 `text_similarity`) | `late` |
| `--embedding_dim` | 텍스트 임베딩 차원 | `512` |
| `--temperature` | 유사도 계산 시 temperature scaling | `0.1` |
| `--modality` | 모달리티 선택 (`fusion`, `sensor`, `vision`) | `fusion` |

## 학습 과정

### 1. Forward Pass
1. 이미지와 센서 피쳐 추출
2. 텍스트 임베딩 생성
3. 피쳐를 텍스트 임베딩 공간으로 투영
4. 코사인 유사도 계산
5. Temperature scaling 및 Softmax로 가중치 계산
6. 가중치를 적용한 피쳐 fusion
7. 최종 분류

### 2. Loss Function
- PaCoLoss: Contrastive learning을 위한 loss
- CBLoss: Class-balanced loss (선택적)

## 장점

1. **해석 가능성**: 텍스트 임베딩을 통해 각 클래스와의 유사도를 명시적으로 계산
2. **적응적 가중치**: 데이터에 따라 이미지와 센서의 중요도가 자동으로 조정
3. **클래스별 특화**: 각 클래스에 대해 다른 가중치 적용 가능
4. **학습 가능한 텍스트 표현**: 텍스트 임베딩이 학습 과정에서 최적화됨

## 실험 설정

### 기본 실험
- Embedding dimension: 512
- Temperature: 0.1
- Learning rate: 0.001
- Batch size: 32
- Epochs: 30

### 하이퍼파라미터 튜닝
- Temperature: 0.05, 0.1, 0.5
- Embedding dimension: 256, 512, 1024
- Learning rate: 0.0001, 0.001, 0.01

## 결과 분석

### 1. 유사도 시각화
각 클래스별 텍스트 임베딩과 이미지/센서 피쳐 간의 유사도를 시각화하여 모델의 동작을 분석할 수 있습니다.

### 2. 가중치 분석
학습된 이미지/센서 가중치를 통해 각 모달리티의 중요도를 확인할 수 있습니다.

### 3. 클래스별 성능
각 클래스("Normal", "Caution", "Warning", "Critical")별 성능을 분석하여 모델의 효과를 평가합니다.

## 파일 구조

```
semes_3y/semes_code/
├── models.py                    # 모델 정의 (TextSimilarityFusion 포함)
├── train.py                     # 학습 스크립트
├── run_text_similarity_fusion.py # 실행 예제 스크립트
└── README_text_similarity_fusion.md # 이 문서
```

## 참고사항

- 텍스트 임베딩은 학습 가능한 파라미터로 초기화되며, 학습 과정에서 최적화됩니다.
- Temperature 파라미터는 유사도 분포의 sharpness를 조절합니다.
- Embedding dimension은 텍스트 임베딩의 표현력을 결정합니다.
- 모델은 기존의 LateFusion과 호환되며, `--fusion_type` 인자로 선택할 수 있습니다. 