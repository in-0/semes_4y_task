#!/usr/bin/env python3
"""
다양한 모달리티로 학습을 실행하는 예제 스크립트

사용법:
1. Fusion 모델 (기본): python run_examples.py --modality fusion
2. Sensor만: python run_examples.py --modality sensor
3. Vision만: python run_examples.py --modality vision
"""

import subprocess
import sys
import os

def run_training(modality, data='gdcm', num_epochs=10, batch_size=32, lr=0.001):
    """
    지정된 모달리티로 학습을 실행합니다.
    
    Args:
        modality (str): 'fusion', 'sensor', 또는 'vision'
        data (str): 'gdcm' 또는 'sms'
        num_epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        lr (float): 학습률
    """
    
    # 기본 명령어 구성
    cmd = [
        'python', 'train.py',
        '--modality', modality,
        '--data', data,
        '--num_epochs', str(num_epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--comment', f'{modality}_only_training'
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print(f"모달리티: {modality}")
    print(f"데이터셋: {data}")
    print(f"에포크: {num_epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"학습률: {lr}")
    print("-" * 50)
    
    try:
        # 학습 실행
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("학습이 성공적으로 완료되었습니다!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"학습 중 오류가 발생했습니다: {e}")
        print(f"에러 출력: {e.stderr}")
        return False

def main():
    """메인 함수"""
    
    print("=" * 60)
    print("다양한 모달리티 학습 예제")
    print("=" * 60)
    
    # 예제 1: Fusion 모델 (기본)
    print("\n1. Fusion 모델 학습 (이미지 + 센서)")
    print("이 명령어는 이미지와 센서 데이터를 모두 사용하여 fusion 모델을 학습합니다.")
    run_training('fusion', num_epochs=5, batch_size=16)
    
    # 예제 2: Sensor만
    print("\n2. Sensor만 학습")
    print("이 명령어는 센서 데이터만 사용하여 모델을 학습합니다.")
    run_training('sensor', num_epochs=5, batch_size=32)
    
    # 예제 3: Vision만
    print("\n3. Vision만 학습")
    print("이 명령어는 이미지 데이터만 사용하여 모델을 학습합니다.")
    run_training('vision', num_epochs=5, batch_size=16)
    
    print("\n" + "=" * 60)
    print("모든 예제 실행 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main() 