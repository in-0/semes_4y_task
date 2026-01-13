#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
텍스트 임베딩 기반 similarity weighting fusion 실행 스크립트

이 스크립트는 ["Normal", "Caution", "Warning", "Critical"] 4개 클래스의 
텍스트 임베딩과 이미지/센서 피쳐 간의 유사도를 계산하여 가중치를 적용하는 
fusion 모델을 학습합니다.
"""

import subprocess
import sys
import os

def run_text_similarity_fusion():
    """텍스트 임베딩 기반 fusion 모델 학습 실행"""
    
    # 기본 설정
    base_cmd = [
        "python", "train.py",
        "--modality", "fusion",
        "--fusion_type", "text_similarity",
        "--data", "gdcm",  # 또는 "sms"
        "--num_epochs", "30",
        "--batch_size", "32",
        "--lr", "0.001",
        "--dim", "128",
        "--embedding_dim", "512",
        "--temperature", "0.1",
        "--loss", "Paco",
        "--scheduler", "step",
        "--step_size", "8",
        "--seed", "666",
        "--comment", "text_similarity_fusion"
    ]
    
    print("텍스트 임베딩 기반 fusion 모델 학습을 시작합니다...")
    print(f"실행 명령어: {' '.join(base_cmd)}")
    print()
    
    try:
        # 명령어 실행
        result = subprocess.run(base_cmd, check=True, capture_output=True, text=True)
        print("학습이 성공적으로 완료되었습니다!")
        print("출력:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"오류가 발생했습니다: {e}")
        print("오류 출력:")
        print(e.stderr)
        return False
    
    return True

def run_with_different_settings():
    """다양한 설정으로 텍스트 similarity fusion 실행"""
    
    settings = [
        {
            "name": "기본 설정",
            "args": {
                "--embedding_dim": "512",
                "--temperature": "0.1",
                "--lr": "0.001"
            }
        },
        {
            "name": "높은 temperature",
            "args": {
                "--embedding_dim": "512", 
                "--temperature": "0.5",
                "--lr": "0.001"
            }
        },
        {
            "name": "낮은 temperature",
            "args": {
                "--embedding_dim": "512",
                "--temperature": "0.05", 
                "--lr": "0.001"
            }
        },
        {
            "name": "큰 embedding dimension",
            "args": {
                "--embedding_dim": "1024",
                "--temperature": "0.1",
                "--lr": "0.001"
            }
        },
        {
            "name": "작은 embedding dimension", 
            "args": {
                "--embedding_dim": "256",
                "--temperature": "0.1",
                "--lr": "0.001"
            }
        }
    ]
    
    for i, setting in enumerate(settings):
        print(f"\n=== {i+1}. {setting['name']} ===")
        
        # 기본 명령어
        cmd = [
            "python", "train.py",
            "--modality", "fusion",
            "--fusion_type", "text_similarity", 
            "--data", "gdcm",
            "--num_epochs", "30",
            "--batch_size", "32",
            "--dim", "128",
            "--loss", "Paco",
            "--scheduler", "step",
            "--step_size", "8",
            "--seed", "666",
            "--comment", f"text_similarity_{setting['name'].replace(' ', '_')}"
        ]
        
        # 추가 설정 적용
        for key, value in setting["args"].items():
            cmd.extend([key, str(value)])
        
        print(f"실행 명령어: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {setting['name']} 완료!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {setting['name']} 실패: {e}")
            print(e.stderr)

if __name__ == "__main__":
    print("텍스트 임베딩 기반 Similarity Weighting Fusion")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        print("다양한 설정으로 실험을 실행합니다...")
        run_with_different_settings()
    else:
        print("기본 설정으로 실험을 실행합니다...")
        run_text_similarity_fusion() 