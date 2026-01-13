import pickle
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import random
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

LABEL_NAMES = ['NoGas', 'Perfume', 'Smoke', 'Mixture']

def one_hot_encode(labels):
    """각 int 라벨들을 원-핫으로 바꿔주는 함수"""
    classes, label_indices = np.unique(labels, return_inverse=True)
    one_hot_vectors = np.eye(len(classes))[label_indices]
    return one_hot_vectors

class SensorOnlyDataset(torch.utils.data.Dataset):
    """센서 데이터만 사용하는 래퍼 데이터셋"""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
    def __getitem__(self, index):
        if hasattr(self.original_dataset, 'get_sensor_only') and hasattr(self.original_dataset, 'get_label_only'):
            sensors = self.original_dataset.get_sensor_only(index)
            labels = self.original_dataset.get_label_only(index)
            return sensors, labels
        else:
            # 기존 데이터셋에서 센서와 라벨만 추출
            if isinstance(self.original_dataset[index], tuple):
                if len(self.original_dataset[index]) == 3:
                    _, sensors, labels = self.original_dataset[index]
                else:
                    sensors, labels = self.original_dataset[index]
            else:
                raise ValueError("데이터셋 형식을 인식할 수 없습니다.")
            return sensors, labels
    
    def __len__(self):
        return len(self.original_dataset)
    
    @property
    def cls_num_list(self):
        """원본 데이터셋의 cls_num_list에 접근"""
        if hasattr(self.original_dataset, 'cls_num_list'):
            return self.original_dataset.cls_num_list
        else:
            # cls_num_list가 없는 경우 samples_per_cls로부터 계산
            samples_per_cls = self.get_samples_per_cls()
            return samples_per_cls
    
    def get_samples_per_cls(self):
        if hasattr(self.original_dataset, 'get_samples_per_cls'):
            return self.original_dataset.get_samples_per_cls()
        else:
            # 기본 구현
            labels = []
            for i in range(len(self)):
                _, label = self[i]
                labels.append(label.argmax().item() if hasattr(label, 'argmax') else label)
            unique_labels, counts = np.unique(labels, return_counts=True)
            return counts

class VisionOnlyDataset(torch.utils.data.Dataset):
    """이미지 데이터만 사용하는 래퍼 데이터셋"""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
    def __getitem__(self, index):
        if hasattr(self.original_dataset, 'get_image_only') and hasattr(self.original_dataset, 'get_label_only'):
            images = self.original_dataset.get_image_only(index)
            labels = self.original_dataset.get_label_only(index)
            return images, labels
        else:
            # 기존 데이터셋에서 이미지와 라벨만 추출
            if isinstance(self.original_dataset[index], tuple):
                if len(self.original_dataset[index]) == 3:
                    images, _, labels = self.original_dataset[index]
                else:
                    images, labels = self.original_dataset[index]
            else:
                raise ValueError("데이터셋 형식을 인식할 수 없습니다.")
            return images, labels
    
    def __len__(self):
        return len(self.original_dataset)
    
    @property
    def cls_num_list(self):
        """원본 데이터셋의 cls_num_list에 접근"""
        if hasattr(self.original_dataset, 'cls_num_list'):
            return self.original_dataset.cls_num_list
        else:
            # cls_num_list가 없는 경우 samples_per_cls로부터 계산
            samples_per_cls = self.get_samples_per_cls()
            return samples_per_cls
    
    def get_samples_per_cls(self):
        if hasattr(self.original_dataset, 'get_samples_per_cls'):
            return self.original_dataset.get_samples_per_cls()
        else:
            # 기본 구현
            labels = []
            for i in range(len(self)):
                _, label = self[i]
                labels.append(label.argmax().item() if hasattr(label, 'argmax') else label)
            unique_labels, counts = np.unique(labels, return_counts=True)
            return counts

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

class GDCMDFusion(Dataset):
    num_classes = 4
    def __init__(self, images, sensors, labels, path_, image_size=224, train=None, imb_ratio=None, seed=666):
        self.img_dir = path_
        self.images = images
        self.sensors = sensors
        
        # 라벨을 원-핫 인코딩
        self.labels = one_hot_encode(labels)
        # 센서 데이터 정규화
        self.sensors = (self.sensors - np.mean(self.sensors, axis=0)) / (np.std(self.sensors, axis=0) + 1e-8)
        self.train = train
        
        # Imbalanced dataset 생성
        if imb_ratio is not None and imb_ratio > 0.0 and train:
            sample_indices = sample_by_ratio(labels, imb_ratio, seed)
            self.images = [self.images[i] for i in sample_indices]
            self.sensors = self.sensors[sample_indices]
            self.labels = self.labels[sample_indices]
        
        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize((image_size, image_size)), 
                transforms.RandomCrop((image_size, image_size), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            self.transform2 = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.RandomResizedCrop(image_size),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
                ], p=1.0),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
        
        # 클래스별 데이터 인덱스 저장
        self.class_data = [[] for _ in range(self.num_classes)]
        for i in range(len(self.labels)):
            y = self.labels[i].argmax()
            self.class_data[y].append(i)
        self.cls_num_list = [len(c) for c in self.class_data]

    def get_samples_per_cls(self):
        unique_labels, counts = np.unique(self.labels.argmax(axis=1), return_counts=True)
        return counts

    def __getitem__(self, index):
        # 이미지 파일명에서 가스 타입 추출 (예시: 'xxx_NoGas.png'라면 'NoGas')
        gas_type = self.images[index].split('_')[1][:-4]
        img_path = os.path.join(self.img_dir, gas_type, self.images[index])
        images_np = np.array(Image.open(img_path))
        
        images_q = self.transform(images_np)
        sensors = self.sensors[index]
        labels = self.labels[index]

        if self.train:
            images_k = self.transform2(images_np)
            return [images_q, images_k], sensors, labels
        else:
            return images_q, sensors, labels

    def __len__(self):
        return len(self.labels)

    def get_image_only(self, index):
        """이미지만 반환하는 메서드"""
        gas_type = self.images[index].split('_')[1][:-4]
        img_path = os.path.join(self.img_dir, gas_type, self.images[index])
        images_np = np.array(Image.open(img_path))
        
        if self.train:
            images_q = self.transform(images_np)
            images_k = self.transform2(images_np)
            return [images_q, images_k]
        else:
            images_q = self.transform(images_np)
            return images_q

    def get_sensor_only(self, index):
        """센서 데이터만 반환하는 메서드"""
        return self.sensors[index]
    
    def get_label_only(self, index):
        """라벨만 반환하는 메서드"""
        return self.labels[index]

def sample_by_ratio(labels, ratio, seed):
    np.random.seed(seed)
    unique_labels, counts = np.unique(labels, return_counts=True)
    first_class_indices = np.where(labels == unique_labels[0])[0]
    other_labels = [label for label in unique_labels[1:]]
    other_classes_indices = [np.where(labels == label)[0] for label in other_labels]
    num_samples_other_classes = int(counts[unique_labels[0]] * ratio)
    sampled_indices = list(first_class_indices)
    for indices in other_classes_indices:
        sampled_indices_other_class = np.random.choice(indices, num_samples_other_classes, replace=False)
        sampled_indices.extend(sampled_indices_other_class)
    return np.array(sampled_indices)

class SEMIDataset(Dataset):
    """
    - (img, sensor, label, 기타) 형태의 pickle 파일을 읽어 데이터셋 구성
    - 라벨은 원-핫 인코딩
    - train 모드에서는 [img_q, img_k] 두 장 반환, test 모드에서는 단일 이미지만 반환
    - 이미지 파일이 경로일 경우 PIL.Image를 이용해 읽어오며, 3채널(RGB)로 변환해 사전학습 모델과 호환
    - 센서 데이터는 전체 평균/표준편차로 정규화함
    """
    def __init__(self, pkl_file, train=True, image_size=224, imb_ratio=None, seed=666):
        super().__init__()
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        self.train = train
        raw_labels = np.array([item[2] for item in self.data])
        
        # Imbalanced dataset 생성
        if imb_ratio is not None and imb_ratio > 0.0 and train:
            sample_indices = sample_by_ratio(raw_labels, imb_ratio, seed)
            self.data = [self.data[i] for i in sample_indices]
            raw_labels = np.array([item[2] for item in self.data])
        
        self.labels = one_hot_encode(raw_labels)  # shape (N, 4)

        sensors = np.array([item[1] for item in self.data])
        self.sensor_mean = np.mean(sensors, axis=0)
        self.sensor_std = np.std(sensors, axis=0)

        _, counts = np.unique(self.labels.argmax(axis=1), return_counts=True)
        self.cls_num_list = counts.tolist()

        if self.train:
            self.transform_q = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop((image_size, image_size), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            self.transform_k = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
                ], p=1.0),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform_q = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
            self.transform_k = None

    def __len__(self):
        return len(self.data)

    def get_samples_per_cls(self):
        unique_labels, counts = np.unique(self.labels.argmax(axis=1), return_counts=True)
        return counts

    def __getitem__(self, index):
        img_data, sensor, _int_label, _ = self.data[index]
        img = np.load(img_data)  # (120,160)
        # float64일 경우 uint8로 변환
        if img.dtype == np.float64:
            img = img.astype(np.uint8)
        sensor = (np.array(sensor) - self.sensor_mean) / (self.sensor_std + 1e-8)
        sensor = torch.tensor(sensor, dtype=torch.float32)

        one_hot_label = torch.tensor(self.labels[index], dtype=torch.float32)

        if self.train:
            img_q = self.transform_q(img)
            img_k = self.transform_k(img) if self.transform_k is not None else img_q
            return [img_q, img_k], sensor, one_hot_label
        else:
            img_q = self.transform_q(img)
            return img_q, sensor, one_hot_label

    def get_image_only(self, index):
        """이미지만 반환하는 메서드"""
        img_data, _, _, _ = self.data[index]
        img = np.load(img_data)
        # float64일 경우 uint8로 변환
        if img.dtype == np.float64:
            img = img.astype(np.uint8)
        
        if self.train:
            img_q = self.transform_q(img)
            img_k = self.transform_k(img) if self.transform_k is not None else img_q
            return [img_q, img_k]
        else:
            img_q = self.transform_q(img)
            return img_q
    
    def get_sensor_only(self, index):
        """센서 데이터만 반환하는 메서드"""
        _, sensor, _, _ = self.data[index]
        sensor = (np.array(sensor) - self.sensor_mean) / (self.sensor_std + 1e-8)
        return torch.tensor(sensor, dtype=torch.float32)
    
    def get_label_only(self, index):
        """라벨만 반환하는 메서드"""
        return torch.tensor(self.labels[index], dtype=torch.float32)
        
def build_dataset(data_root, mode=None, imb_ratio=None, seed=666):
    if mode == "GDCMD":
        image_path = os.path.join(data_root, 'images/')
        sensor_path = os.path.join(data_root, 'sensors/Gas_Sensors_Measurements.csv')
        df_sensor = pd.read_csv(sensor_path)
        sensor = df_sensor.iloc[:, 1:-2].values
        file_names = df_sensor.iloc[:, -1].values
        str_labels = df_sensor.iloc[:, -2].values
        image_names = [f"{file_name}.png" for file_name in file_names]

        labels = pd.DataFrame(str_labels, columns=['label'])
        for i, label in enumerate(LABEL_NAMES):
            labels = labels.replace(label, i)
        labels = labels.values

        indices = np.arange(len(labels))
        X_train_indices, X_test_indices, y_train, y_test = train_test_split(
            indices, labels, test_size=0.2, stratify=labels, random_state=seed
        )
        
        train_dataset = GDCMDFusion(
            [image_names[i] for i in X_train_indices],
            sensor[X_train_indices],
            y_train,
            path_=image_path,
            train=True,
            imb_ratio=imb_ratio,
            seed=seed
        )
        test_dataset = GDCMDFusion(
            [image_names[i] for i in X_test_indices],
            sensor[X_test_indices],
            y_test,
            path_=image_path,
            train=False
        )

        print(f"[GDCMD] Train size: {len(train_dataset)}")
        print(f"[GDCMD] Test size:  {len(test_dataset)}")
        print(f"[GDCMD] Class Distribution (Train): {train_dataset.get_samples_per_cls()}")

    elif mode == "semi":
        train_pkl = os.path.join(data_root, 'semes_train.pkl')
        val_pkl = os.path.join(data_root, 'semes_val.pkl')

        train_dataset = SEMIDataset(pkl_file=train_pkl, train=True, image_size=224, imb_ratio=imb_ratio, seed=seed)
        test_dataset = SEMIDataset(pkl_file=val_pkl, train=False, image_size=224)

        print(f"[semi] Train size: {len(train_dataset)}")
        print(f"[semi] Test size:  {len(test_dataset)}")
        print(f"[semi] Class Distribution (Train): {train_dataset.get_samples_per_cls()}")

    else:
        raise ValueError("Invalid mode! Choose between 'GDCMD' and 'semi'.")

    return train_dataset, test_dataset

def create_data_loaders(args, train_dataset, test_dataset):
    """데이터셋을 DataLoader로 변환하는 함수"""
    from torch.utils.data import DataLoader
    
    # 모달리티에 따라 데이터셋 래핑
    if args.modality == 'sensor':
        train_dataset = SensorOnlyDataset(train_dataset)
        test_dataset = SensorOnlyDataset(test_dataset)
    elif args.modality == 'vision':
        train_dataset = VisionOnlyDataset(train_dataset)
        test_dataset = VisionOnlyDataset(test_dataset)
    
    fusion_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    fusion_test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return fusion_train_loader, fusion_test_loader

def get_dataset_info(train_dataset):
    """데이터셋 정보를 추출하는 함수"""
    # 원본 데이터셋에서 cls_num_list와 samples_per_cls 가져오기 (모든 모달리티에 동일하게 적용)
    if hasattr(train_dataset, 'dataset'):
        # Subset 데이터셋인 경우 (build_dataset에서 생성된 경우)
        original_dataset = train_dataset.dataset
    elif hasattr(train_dataset, 'original_dataset'):
        # 래퍼 데이터셋인 경우 (sensor 또는 vision 모달리티)
        original_dataset = train_dataset.original_dataset
    else:
        # 원본 데이터셋인 경우
        original_dataset = train_dataset
    
    # cls_num_list 설정
    if hasattr(original_dataset, 'cls_num_list'):
        cls_num_list = original_dataset.cls_num_list
    else:
        # cls_num_list가 없는 경우 samples_per_cls로부터 계산
        if hasattr(original_dataset, 'get_samples_per_cls'):
            cls_num_list = original_dataset.get_samples_per_cls()
        else:
            # get_samples_per_cls도 없는 경우 기본값 사용
            cls_num_list = [1, 1, 1, 1]  # 기본값
    
    # samples_per_cls 가져오기 (CBLoss용)
    if hasattr(original_dataset, 'get_samples_per_cls'):
        samples_per_cls = original_dataset.get_samples_per_cls()
    else:
        # get_samples_per_cls가 없는 경우 기본값 사용
        samples_per_cls = [1, 1, 1, 1]  # 기본값
    
    return cls_num_list, samples_per_cls
