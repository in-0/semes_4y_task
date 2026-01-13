import os
import json
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class SemiDataset(Dataset):
    def __init__(self, raw_root, label_root, transform=None, cache_file=None):
        self.raw_root = raw_root  
        self.label_root = label_root
        self.transform = transform
        self.cache_file = cache_file
        self.mode = "train" if "train" in cache_file else "val" 

        print(f"[{self.mode.upper()}] 데이터 전처리 진행합니다.")
        self.pkl_data = []
        json_paths = []
        for dirpath, _, filenames in os.walk(label_root):
            for filename in filenames:
                if filename.endswith('.json'):
                    json_paths.append(os.path.join(dirpath, filename))

        total_json_count = len(json_paths)
        bin_count = 0

        for label_path in tqdm(json_paths, desc=f"Preprocessing {self.mode.upper()}", unit="json"):
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            base_name = os.path.splitext(os.path.basename(label_path))[0]

            modality = "agv" if "agv" in label_path.lower() else "oht"
            bin_path = self.find_raw_file(base_name, ".bin", modality)

            if bin_path and os.path.exists(bin_path):
                bin_path = os.path.join(self.raw_root, os.path.relpath(bin_path, self.raw_root))
                bin_count += 1
            
            sensor_keys = ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4"]
            sensor_values = []
            sensor_data = label_data.get("sensor_data", [{}])[0]
            for key in sensor_keys:
                if key in sensor_data and len(sensor_data[key]) > 0:
                    sensor_values.append(sensor_data[key][0].get("value", 0))
                else:
                    sensor_values.append(0)
            sensor_values = np.array(sensor_values, dtype=np.float32)

            annotations = label_data.get("annotations", [{}])[0]
            tagging = annotations.get("tagging", [])
            target = int(tagging[0].get("state", 0)) if tagging else 0

            self.pkl_data.append((bin_path, sensor_values, target, base_name))

        print(f"총 JSON 파일 갯수: {total_json_count}, BIN 파일 갯수: {bin_count}")
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.pkl_data, f)
        print(f"캐시 파일 저장 완료: {self.cache_file}")

    def find_raw_file(self, base_name, extension, modality):
        pattern = os.path.join(self.raw_root, modality, '**', base_name + extension)
        files = glob.glob(pattern, recursive=True)
        return files[0] if files else None

    def __len__(self):
        return len(self.pkl_data)

    def __getitem__(self, index):
        bin_path, sensor_values, target, base_name = self.pkl_data[index]

        bin_array = None
        if bin_path:
            abs_bin_path = os.path.join(self.raw_root, os.path.relpath(bin_path, os.path.join("data/semi", self.mode, "raw")))

            if os.path.exists(abs_bin_path):
                with open(abs_bin_path, 'rb') as fbin:
                    bin_array = np.load(fbin)

        if bin_array is not None and self.transform is not None:
            bin_array = self.transform(bin_array)

        return bin_array, sensor_values, target, base_name

if __name__ == '__main__':
    import argparse
    from torchvision import transforms

    parser = argparse.ArgumentParser(description="SemiDataset Pickle 캐시 생성")
    parser.add_argument('--mode', type=str, choices=['train', 'val'], required=True, help="데이터셋 종류 (train 또는 val)")
    args = parser.parse_args()

    base_dir = "/home/js/workspace/semes/data/semi"
    if args.mode == "train":
        raw_root = os.path.join(base_dir, "train", "raw")
        label_root = os.path.join(base_dir, "train", "label")
        cache_file = os.path.join(base_dir, "semes_train.pkl")
    else:
        raw_root = os.path.join(base_dir, "val", "raw")
        label_root = os.path.join(base_dir, "val", "label")
        cache_file = os.path.join(base_dir, "semes_val.pkl")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = SemiDataset(
        raw_root=raw_root,
        label_root=label_root,
        transform=transform,
        cache_file=cache_file
    )
    print(f"데이터셋 길이: {len(dataset)}")
    print(f"캐시 파일 경로: {dataset.cache_file}")
