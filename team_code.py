#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import joblib
from helper_code import *
from Generators import ECGDataset
from params import params1
from models import ResNet_BiGRU_Net
from tqdm import tqdm


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    os.makedirs(model_folder, exist_ok=True)

    print(f"[LOG] Scanning .hea records in: {data_folder}")
    all_records = find_records(data_folder)
    print(f"[LOG] Total records found: {len(all_records)}")

#    code15_records, code15_labels = [], []
#    other_records = []

#    for rec in all_records:
#        header = load_header(os.path.join(data_folder, rec))
#        label = get_label(header)
#        source = get_source(header)
#        if source == 'CODE-15%':
#            code15_records.append(rec)
#            code15_labels.append(int(label))
#        else:
#            other_records.append(rec)

#    from sklearn.model_selection import train_test_split
#    _, code15_sampled_idx = train_test_split(
#        list(range(len(code15_records))),
#        test_size=0.3,
#        stratify=code15_labels,
#        random_state=42
#    )
 #   code15_sampled = [code15_records[i] for i in code15_sampled_idx]

#    filtered_records = other_records + code15_sampled
#    print(f"[LOG] Records used after filtering: {len(filtered_records)}")

    print(f"[LOG] Initializing ECGDataset...")
    dataset = ECGDataset(
        data_folder=data_folder,
        sr=params1['sr'],
        samp_sec=params1['samp_sec'],
        mod=params1['mod'],
        method=params1['method'],
        is_record=False,
        sample=True,
        file_list=all_records
    )

    all_labels = []
    for rec in all_records:
        header = load_header(os.path.join(data_folder, rec))
        label = get_label(header)
        all_labels.append(int(label))

    train_set_indices, val_set_indices = train_test_split(
        list(range(len(all_records))),
        test_size=0.1,
        stratify=all_labels,
        shuffle=True,
        random_state=42
    )
    print(f"[LOG] Train size: {len(train_set_indices)}, Val size: {len(val_set_indices)}")


    train_loader = DataLoader(Subset(dataset, train_set_indices), batch_size=params1['batch_size'], shuffle=True, num_workers=2)
    val_loader_set = DataLoader(Subset(dataset, val_set_indices), batch_size=params1['batch_size'], shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet_BiGRU_Net(meta_features=4, handcrafted_dim=4).to(device)



    pos_count, neg_count = 0, 0
#    for _, labels in train_loader:
#        pos_count += (labels == 1).sum().item()
#        neg_count += (labels == 0).sum().item()
#    pos_weight_value = neg_count / (pos_count + 1e-5)
    pos_weight_value = 5.0
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    print(f"[LOG] pos_count: {pos_count}, neg_count: {neg_count}, pos_weight: {pos_weight_value:.4f}")

    from losses import SourceAwareBCELoss, EmbeddingAlignLoss

    # 손실 함수 초기화
    criterion = SourceAwareBCELoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Early stopping variables
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    wait = 0

    print(f"[LOG] Starting training for {params1['epochs']} epochs...")
    for epoch in range(params1['epochs']):
        print(f"\n[LOG] -------- Epoch {epoch+1}/{params1['epochs']} --------")
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{params1['epochs']}]", disable=not verbose)

        for inputs, labels in loop:
            signals = inputs['signal'].to(device)
            ages = inputs['meta']['age'].to(device)
            sexes = inputs['meta']['sex'].to(device)
            handcrafted = inputs['handcrafted'].to(device)
            labels = labels.unsqueeze(1).to(device)
            sources = inputs['source']  # 소스 정보

            outputs = model(signals, ages, sexes, handcrafted)
            loss = criterion(outputs, labels, sources)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"[LOG] Epoch {epoch+1} training complete. Avg Train Loss: {avg_train_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader_set:
                signals = inputs['signal'].to(device)
                ages = inputs['meta']['age'].to(device)
                sexes = inputs['meta']['sex'].to(device)
                handcrafted = inputs['handcrafted'].to(device)
                labels = labels.unsqueeze(1).to(device)
                sources = inputs['source']

                outputs = model(signals, ages, sexes, handcrafted)
                loss = criterion(outputs, labels, sources)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader_set)
        print(f"[LOG] Validation Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            wait = 0
            print("[LOG] New best model found. Saving...")
        else:
            wait += 1
            print(f"[LOG] No improvement. Wait count: {wait}/{patience}")
            if wait >= patience:
                print("[LOG] Early stopping triggered.")
                break

    # Save
    final_model_path = os.path.join(model_folder, 'model.sav')
    joblib.dump({'model_state': best_state}, final_model_path)
    print(f"[LOG] Final best model saved at {final_model_path}")


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(model_folder, 'model.sav')
    checkpoint = joblib.load(path)
    model = ResNet_BiGRU_Net(meta_features=4, handcrafted_dim=4).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    if verbose:
        print(f"Loaded model from {path}")
    return model




# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_data = ECGDataset(
        data_folder=record,
        sr=params1['sr'],
        samp_sec=params1['samp_sec'],
        mod=params1['mod'],
        method=params1['method'],
        is_record=True,
        sample=False
    )
    inputs, _ = eval_data[0]
    signals = inputs['signal'].unsqueeze(0).to(device)
    ages = inputs['meta']['age'].unsqueeze(0).to(device)
    sexes = inputs['meta']['sex'].unsqueeze(0).to(device)
    handcrafted = inputs['handcrafted'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(signals, ages, sexes, handcrafted)
        prob = torch.sigmoid(output).item()
        binary = int(prob >= 0.5)

    if verbose:
        print(f"Prediction: {binary}, Probability: {prob:.4f}")
    return binary, prob

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)

    age = get_age(header)
    age = np.array([age])

    sex = get_sex(header)
    sex_one_hot_encoding = np.zeros(3, dtype=bool)
    if sex.casefold().startswith('f'):
        sex_one_hot_encoding[0] = 1
    elif sex.casefold().startswith('m'):
        sex_one_hot_encoding[1] = 1
    else:
        sex_one_hot_encoding[2] = 1

    source = get_source(header)

    signal, fields = load_signals(record)
    channels = fields['sig_name']

    # Reorder the channels
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    signal = reorder_signal(signal, channels, reference_channels)

    # Compute two per-channel features as examples.
    signal_mean = np.zeros(num_channels)
    signal_std = np.zeros(num_channels)

    for i in range(num_channels):
        num_finite_samples = np.sum(np.isfinite(signal[:, i]))
        if num_finite_samples > 0:
            signal_mean[i] = np.nanmean(signal)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal)
        else:
            signal_std = 0.0

    # Return the features.

    return age, sex_one_hot_encoding, source, signal_mean, signal_std

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
    

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.stop_epoch = None

    def __call__(self, val_loss, current_epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.stop_epoch = current_epoch
        else:
            self.best_loss = val_loss
            self.counter = 0