import torch
import os
from pathlib import Path

def merge_text_embeddings():
    """
    semi_text_embs 폴더의 모든 pth 파일을 불러와서 하나의 pth 파일로 합칩니다.
    파일명에서 '_'를 기준으로 나눈 마지막 단어를 클래스명으로 사용합니다.
    """
    
    # semi_text_embs 폴더 경로
    folder_path = Path('./semi_text_embs')
    
    # 결과를 저장할 딕셔너리
    merged_embeddings = {}
    
    # 폴더 내의 모든 pth 파일 찾기
    pth_files = list(folder_path.glob('*.pth'))
    
    print(f"Found {len(pth_files)} pth files:")
    
    for pth_file in pth_files:
        # 파일명에서 클래스명 추출 (마지막 '_' 이후)
        filename = pth_file.stem  # 확장자 제외
        class_name = filename.split('_')[-1]
        
        print(f"Loading {pth_file.name} -> class: {class_name}")
        
        try:
            # pth 파일 불러오기
            embeddings = torch.load(pth_file, map_location='cpu')
            
            # 딕셔너리에 저장
            merged_embeddings[class_name] = embeddings
            
            print(f"  - Shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")
            print(f"  - Type: {type(embeddings)}")
            
        except Exception as e:
            print(f"  - Error loading {pth_file.name}: {e}")
    
    # 결과 출력
    print(f"\nMerged embeddings keys: {list(merged_embeddings.keys())}")
    
    # 하나의 pth 파일로 저장
    output_path = './merged_text_embeddings.pth'
    torch.save(merged_embeddings, output_path)
    
    print(f"\nSaved merged embeddings to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    # 저장된 파일 확인
    print("\nVerifying saved file...")
    loaded_embeddings = torch.load(output_path, map_location='cpu')
    print(f"Loaded keys: {list(loaded_embeddings.keys())}")
    
    return merged_embeddings

if __name__ == "__main__":
    merged_embeddings = merge_text_embeddings() 