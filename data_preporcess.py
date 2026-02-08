import os
import json
import glob
import pickle
import numpy as np
from tqdm import tqdm

try:
    from torch.utils.data import Dataset  # type: ignore
except ModuleNotFoundError:
    # MTF 생성만 할 때는 torch가 없어도 되도록 처리
    Dataset = object  # type: ignore

def _nan_to_num(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    # NaN이 섞인 경우 평균으로 대체 (전부 NaN이면 0)
    mean = np.nanmean(x)
    if np.isnan(mean):
        mean = 0.0
    return np.nan_to_num(x, nan=float(mean), posinf=float(mean), neginf=float(mean))

def _quantize_to_bins(ts: np.ndarray, n_bins: int = 8, strategy: str = "quantile") -> np.ndarray:
    """
    1D 시계열을 [0, n_bins-1] 정수 bin index로 양자화.
    strategy:
      - "quantile": 분위수 기반 (분포가 치우친 경우 안정적)
      - "uniform": 최소~최대 구간을 균등 분할
    """
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    ts = _nan_to_num(ts)
    if ts.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if n_bins <= 1:
        return np.zeros_like(ts, dtype=np.int64)
    if np.all(ts == ts[0]):
        return np.zeros_like(ts, dtype=np.int64)

    if strategy == "uniform":
        lo = float(np.min(ts))
        hi = float(np.max(ts))
        if hi == lo:
            return np.zeros_like(ts, dtype=np.int64)
        edges = np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
    else:
        # quantile (기본)
        qs = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
        edges = np.quantile(ts, qs)

    # 중복 edge가 많으면 digitize가 비정상 동작할 수 있으니 유니크 처리
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros_like(ts, dtype=np.int64)

    # digitize는 bins 길이가 (k)일 때 결과는 [0..k]가 될 수 있으므로 clip
    # edges[1:-1]만 사용하면 결과는 [0..len(edges)-2]
    inner = edges[1:-1]
    q = np.digitize(ts, inner, right=False).astype(np.int64)
    max_bin = min(n_bins - 1, int(np.max(q)) if q.size else 0)
    return np.clip(q, 0, max_bin).astype(np.int64)

def markov_transition_field(
    ts: np.ndarray,
    n_bins: int = 8,
    strategy: str = "quantile",
    out_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Markov Transition Field(MTF) 생성.
    - 입력: 1D 시계열 (길이 T)
    - 출력: (T, T) 이미지, 값은 전이확률
    """
    ts = np.asarray(ts)
    q = _quantize_to_bins(ts, n_bins=n_bins, strategy=strategy)
    T = int(q.size)
    if T == 0:
        return np.zeros((0, 0), dtype=out_dtype)
    if T == 1:
        return np.zeros((1, 1), dtype=out_dtype)

    n_states = int(np.max(q)) + 1 if q.size else 1
    n_states = max(n_states, 1)
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for a, b in zip(q[:-1], q[1:]):
        counts[int(a), int(b)] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

    # MTF: M[i, j] = P[q[i], q[j]]
    M = P[q[:, None], q[None, :]].astype(out_dtype, copy=False)
    return M

def _derive_1d_timeseries(x: np.ndarray, time_axis: int = 0) -> np.ndarray:
    """
    입력이 2D 이상인 경우 MTF를 위한 1D 시계열로 축약.
    - 기본: 2D면 time_axis를 시간축으로 보고 나머지 축 평균
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        if time_axis not in (0, 1):
            time_axis = 0
        return x.mean(axis=1 - time_axis)
    # 그 이상은 일단 flatten
    return x.reshape(-1)

def generate_mtf_bins_inplace(
    data_root: str,
    splits: tuple[str, ...] = ("train", "val"),
    overwrite: bool = False,
    n_bins: int = 8,
    strategy: str = "quantile",
    time_axis: int = 0,
    out_dtype: str = "float32",
):
    """
    ./data/semi 구조에서 raw 아래의 *.bin(np.load로 읽는 파일)을 대상으로
    동일 폴더에 *.mtf.bin 파일을 생성한다.
    저장 포맷: np.save로 저장된 numpy 바이너리 (확장자만 .bin)
    """
    data_root = os.path.abspath(data_root)
    out_dtype_np = np.dtype(out_dtype)
    raw_roots = [os.path.join(data_root, split, "raw") for split in splits]

    targets: list[str] = []
    for raw_root in raw_roots:
        if not os.path.isdir(raw_root):
            continue
        for dirpath, _, filenames in os.walk(raw_root):
            for fn in filenames:
                if not fn.lower().endswith(".bin"):
                    continue
                # 이미 생성된 mtf 파일은 스킵
                if fn.lower().endswith(".mtf.bin"):
                    continue
                targets.append(os.path.join(dirpath, fn))

    if not targets:
        print(f"[MTF] 대상 .bin 파일을 찾지 못했습니다. (data_root={data_root})")
        return

    print(f"[MTF] 변환 대상: {len(targets)} files")
    for in_path in tqdm(targets, desc="Generating MTF bins", unit="file"):
        out_path = in_path[:-4] + ".mtf.bin"
        if (not overwrite) and os.path.exists(out_path):
            continue

        try:
            with open(in_path, "rb") as f:
                arr = np.load(f, allow_pickle=False)
        except Exception as e:
            print(f"[MTF][SKIP] 로드 실패: {in_path} ({e})")
            continue

        ts = _derive_1d_timeseries(arr, time_axis=time_axis)
        mtf = markov_transition_field(ts, n_bins=n_bins, strategy=strategy, out_dtype=out_dtype_np)

        try:
            # np.save는 확장자와 무관하게 numpy 바이너리를 저장함
            with open(out_path, "wb") as f:
                np.save(f, mtf, allow_pickle=False)
        except Exception as e:
            print(f"[MTF][SKIP] 저장 실패: {out_path} ({e})")
            continue

def augment_pkl_with_mtf_paths(pkl_in: str, pkl_out: str | None = None, overwrite: bool = False):
    """
    기존 semes_*.pkl에 mtf 경로를 '추가'한다.
    - 입력 포맷(기존): (img_bin_path, sensor, target, base_name)
    - 출력 포맷(신규): (img_bin_path, sensor, target, base_name, mtf_bin_path_or_None)
    """
    if overwrite:
        pkl_out = pkl_in
    if pkl_out is None:
        root, ext = os.path.splitext(pkl_in)
        pkl_out = root + "_mtf" + ext

    with open(pkl_in, "rb") as f:
        data = pickle.load(f)

    fixed = []
    updated = 0
    kept = 0
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            img_path, sensor, target, base_name = item
            mtf_path = None
            if isinstance(img_path, str) and img_path.lower().endswith(".bin"):
                cand = img_path[:-4] + ".mtf.bin"
                if os.path.exists(cand):
                    mtf_path = cand
            fixed.append((img_path, sensor, target, base_name, mtf_path))
            updated += 1
        else:
            fixed.append(item)
            kept += 1

    with open(pkl_out, "wb") as f:
        pickle.dump(fixed, f)
    print(f"[PKL] 저장 완료: {pkl_out} (updated={updated}, kept={kept})")

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

            mtf_path = None
            if bin_path and isinstance(bin_path, str) and bin_path.lower().endswith(".bin"):
                cand = bin_path[:-4] + ".mtf.bin"
                if os.path.exists(cand):
                    mtf_path = cand
            
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

            # 기존 포맷을 유지하면서(앞 4개), MTF 경로를 5번째로 "추가"
            self.pkl_data.append((bin_path, sensor_values, target, base_name, mtf_path))

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

    parser = argparse.ArgumentParser(description="SEME(Semi) 전처리/캐시/MTF 생성 유틸")
    parser.add_argument('--mode', type=str, choices=['train', 'val'], help="캐시 생성 시 데이터셋 종류 (train 또는 val)")
    parser.add_argument('--base_dir', type=str, default="data/semi", help="데이터 루트 경로 (기본: ./data/semi)")
    parser.add_argument('--make_mtf', action='store_true', help="raw/*.bin -> raw/*.mtf.bin 생성")
    parser.add_argument('--mtf_splits', type=str, default="train,val", help="MTF 생성 split 콤마구분 (기본: train,val)")
    parser.add_argument('--mtf_n_bins', type=int, default=8, help="MTF bin 개수 (기본: 8)")
    parser.add_argument('--mtf_strategy', type=str, choices=['quantile', 'uniform'], default='quantile', help="MTF 양자화 방식")
    parser.add_argument('--mtf_time_axis', type=int, default=0, help="2D 입력에서 시간축(0 또는 1), 기본 0")
    parser.add_argument('--mtf_overwrite', action='store_true', help="이미 존재하는 *.mtf.bin 덮어쓰기")
    parser.add_argument('--augment_pkl_with_mtf', action='store_true', help="기존 semes_*.pkl에 mtf 경로 필드 추가")
    parser.add_argument('--augment_overwrite', action='store_true', help="augment 결과를 같은 pkl에 덮어쓰기")
    args = parser.parse_args()

    if args.make_mtf:
        splits = tuple([s.strip() for s in args.mtf_splits.split(",") if s.strip()])
        generate_mtf_bins_inplace(
            data_root=args.base_dir,
            splits=splits,
            overwrite=args.mtf_overwrite,
            n_bins=args.mtf_n_bins,
            strategy=args.mtf_strategy,
            time_axis=args.mtf_time_axis,
            out_dtype="float32",
        )
        raise SystemExit(0)

    if args.augment_pkl_with_mtf:
        # torch 없이도 동작
        train_pkl = os.path.join(args.base_dir, "semes_train.pkl")
        val_pkl = os.path.join(args.base_dir, "semes_val.pkl")
        if os.path.exists(train_pkl):
            augment_pkl_with_mtf_paths(train_pkl, overwrite=args.augment_overwrite)
        if os.path.exists(val_pkl):
            augment_pkl_with_mtf_paths(val_pkl, overwrite=args.augment_overwrite)
        raise SystemExit(0)

    if not args.mode:
        raise SystemExit("--mode(train/val) 또는 --make_mtf 중 하나는 필요합니다.")

    # 캐시 생성 모드는 torchvision/torch에 의존
    from torchvision import transforms  # type: ignore

    base_dir = args.base_dir
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
