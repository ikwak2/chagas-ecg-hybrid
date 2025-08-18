import os
import torch
from torch.utils.data import Dataset
import numpy as np
from helper_code import *
from util_ho import *
from scipy.signal import resample_poly

class ECGDataset(Dataset):
    def __init__(
        self,
        data_folder,
        sr=400,
        samp_sec=10,
        transform=None,
        mod="median",
        method="oc_co_wo_let",
        is_record=False,
        sample=True,
        file_list=None
    ):
        self.data_folder = data_folder
        self.is_record = is_record
        self.sr = sr
        self.samp_sec = samp_sec
        self.seq_length = int(sr * samp_sec)
        self.transform = transform
        self.mod = mod
        self.method = method
        self.sample = sample

        if file_list is None:
            self.file_list = self._load_file_list()
        else:
            self.file_list = [
                os.path.join(self.data_folder, f) for f in file_list
            ]

    def _load_file_list(self):
        # 확장자 제거한 파일 경로 리스트
        if not self.is_record:
            files = []
            for root, _, filenames in os.walk(self.data_folder):
                for filename in filenames:
                    if filename.endswith('.hea'):
                        files.append(os.path.join(root, filename[:-4]))
        else:
            files = [self.data_folder]
        return files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # 1. 신호 및 헤더 로드
        header = load_header(file_path)

        # 신호 로딩 + 리샘플링
        signal, fields = load_signals_and_resample(
            file_path, target_fs=self.sr
        )


        # 채널 reorder 
        channels = fields['sig_name']
        reference_channels = [
            'I', 'II', 'III', 'AVR', 'AVL', 'AVF',
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
        ]
        signal = reorder_signal(signal, channels, reference_channels)

        # 신호 Augumentation signal 0.9~1.1 사이 수를 랜덤하게 곱하기
        signal = signal * np.random.uniform(0.9, 1.1, 1)

        # 랜덤하게 시퀀스 길이 조정 0~0.5초정도 사이에서 시작위치 정하기. s1은 0~200 사이의 랜덤 정수
        if len(signal) > 1500:
            s1 = np.random.randint(0, 201)
            signal = signal[s1:,:]

#        handcrafted_feat1 = extract_handcrafted_features(signal, self.sr)
#        handcrafted_feat1 = torch.tensor(handcrafted_feat1, dtype=torch.float32)
        handcrafted_feat2 = extract_handcrafted_features2(signal, self.sr)
        handcrafted_feat = torch.tensor(handcrafted_feat2, dtype=torch.float32)

#        handcrafted_feat = torch.cat((handcrafted_feat1, handcrafted_feat2), dim=0)

        # process_pipeline 적용
        processed_signal = process_pipeline(
            signal, mod=self.mod, method=self.method
        )

        processed_signal = processed_signal.T  # (leads, samples)

        # 길이 조정
        if processed_signal.shape[1] < self.seq_length:
            pad_len = self.seq_length - processed_signal.shape[1]
            processed_signal = np.pad(
                processed_signal, ((0, 0), (0, pad_len)), mode='constant'
            )
        else:
            if self.sample:
                max_start = processed_signal.shape[1] - self.seq_length
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                processed_signal = processed_signal[:, start:start + self.seq_length]
            else:
                processed_signal = processed_signal[:, :self.seq_length]

        # 2. 메타데이터 추출
        age = get_age(header)
        sex = get_sex(header)

        source = get_source2(header)
        '''
        if 'code15' in file_path.lower():
            source = 'code15'
        elif 'samitrop' in file_path.lower():
            source = 'samitrop'
        elif 'ptbxl' in file_path.lower():
            source = 'ptbxl'
        else:
            source = 'unknown'
        '''

        # 3. 메타데이터 인코딩
        sex_encoding = self._encode_sex(sex)
        # source_encoding = self._encode_source(source)

        # 4. Tensor 변환
        sample = {
            'signal': torch.tensor(processed_signal.copy(), dtype=torch.float32),
            'meta': {
                'age': torch.tensor([age], dtype=torch.float32),
                'sex': torch.tensor(sex_encoding, dtype=torch.float32)                
            },
            'handcrafted': handcrafted_feat,
            'source': source
        }

        if not self.is_record:
            label = get_label(header)
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(-1, dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def _process_signal(self, signal):
        signal = signal.T
        target_len = self.seq_length

        if signal.shape[1] < target_len:
            pad_len = target_len - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_len)), mode='constant')
        else:
            signal = signal[:, :target_len]

        return signal

    def _encode_sex(self, sex):
        encoding = np.zeros(3)
        if sex == 'Female':
            encoding[0] = 1
        elif sex == 'Male':
            encoding[1] = 1
        else:
            encoding[2] = 1
        return encoding

    def _encode_source(self, source):
        encoding = np.zeros(3)
        if source == 'code15':
            encoding[0] = 1
        elif source == 'samitrop':
            encoding[1] = 1
        else:
            encoding[2] = 1
        return encoding
