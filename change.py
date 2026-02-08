import os
import pickle
import argparse
from tqdm import tqdm

def fix_pkl_paths(pkl_file, output_pkl, mode):
    """기존 pkl 파일의 경로를 train/raw 또는 val/raw로 자동 수정하여 다시 저장"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    fixed_data = []
    for item in tqdm(data, desc="Fixing PKL Paths", unit="entry"):
        # pkl 포맷 호환:
        # - 기존: (img_data, sensor_values, target, base_name)
        # - 신규: (img_data, sensor_values, target, base_name, mtf_path)
        img_data, sensor_values, target, base_name, *rest = item
        mtf_path = rest[0] if rest else None
        # 🔹 경로 수정
        if "data/semi/val/raw" in img_data and mode == "train":
            corrected_img_data = img_data.replace("data/semi/val/raw", "data/semi/train/raw")
        elif "data/semi/train/raw" in img_data and mode == "val":
            corrected_img_data = img_data.replace("data/semi/train/raw", "data/semi/val/raw")
        else:
            corrected_img_data = img_data  # 이미 올바르면 그대로 유지

        corrected_mtf = None
        if isinstance(mtf_path, str):
            corrected_mtf = mtf_path
            if "data/semi/val/raw" in corrected_mtf and mode == "train":
                corrected_mtf = corrected_mtf.replace("data/semi/val/raw", "data/semi/train/raw")
            elif "data/semi/train/raw" in corrected_mtf and mode == "val":
                corrected_mtf = corrected_mtf.replace("data/semi/train/raw", "data/semi/val/raw")
        fixed_data.append((corrected_img_data, sensor_values, target, base_name, corrected_mtf))

    # 수정된 데이터를 새로운 pkl 파일로 저장
    with open(output_pkl, 'wb') as f:
        pickle.dump(fixed_data, f)

    print(f"🔄 {pkl_file} → {output_pkl} 로 저장 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PKL 파일 경로 수정")
    parser.add_argument('--mode', type=str, choices=['train', 'val'], required=True, help="수정할 데이터셋 종류 (train 또는 val)")
    args = parser.parse_args()

    if args.mode == "train":
        input_pkl = "data/semi/semes_train_fixed.pkl"
        output_pkl = "data/semi/semes_train_fixed.pkl"  # 수정된 pkl 파일 저장 경로
    else:
        input_pkl = "data/semi/semes_val.pkl"
        output_pkl = "data/semi/semes_val_fixed.pkl"  # 수정된 pkl 파일 저장 경로

    fix_pkl_paths(input_pkl, output_pkl, args.mode)
