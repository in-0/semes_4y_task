import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm  # 진행률 표시용
import argparse

def save_pkl_data_as_png(pkl_file, output_dir):
    """PKL 파일 내 .npy 데이터를 로드하여 hot 컬러맵 적용 후 .png 형식으로 저장하는 함수"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for i, (img_data, _, _, base_name) in enumerate(tqdm(data, desc="Saving Heatmap PNGs", unit="image")):
        # Load numpy array from .npy file
        img = np.load(img_data)  # (120,160)

        # Normalize to 0-255 and convert to uint8
        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply HOT colormap
        img_colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_HOT)

        # Save as PNG
        output_path = os.path.join(output_dir, f"{base_name}_{i}.png")
        cv2.imwrite(output_path, img_colored)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PKL 데이터 PNG 변환 (HOT 컬러맵 적용)")
    parser.add_argument('--mode', type=str, choices=['train', 'val'], required=True, help="데이터셋 종류 (train 또는 val)")
    args = parser.parse_args()

    if args.mode == 'train':
        pkl_path = "data/semi/semes_train.pkl"
        output_directory = "data/semi/vis/train"
    else:
        pkl_path = "data/semi/semes_val.pkl"
        output_directory = "data/semi/vis/val"

    save_pkl_data_as_png(pkl_path, output_directory)
