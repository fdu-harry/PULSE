import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm
from MODEL_STRUCTURE import Transformer
import time
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
from ERL_LOSS import compute_erl_loss

def device_nvidia(num):
    """Set device to GPU or CPU"""
    device = torch.device(f'cuda:{num}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device


def trans_tensor(dataset):
    """Convert numpy array to torch tensor"""
    return torch.FloatTensor(dataset)


def cal_statistic(cm):
    """Calculate accuracy metrics from confusion matrix"""
    tp = np.diagonal(cm)
    gt_num = np.sum(cm, axis=1)
    pre_num = np.sum(cm, axis=0)
    fp = pre_num - tp
    num0 = np.sum(gt_num)
    gt_num0 = num0 - gt_num
    tn = gt_num0 - fp

    with np.errstate(divide='ignore', invalid='ignore'):
        sen = np.true_divide(tp, gt_num)  # sensitivity
        spe = np.true_divide(tn, gt_num0)  # specificity
        ppv = np.true_divide(tp, pre_num)  # precision
        F1 = 2 * (sen * ppv) / (sen + ppv)
        acc = np.true_divide(np.sum(tp), num0)

    # Handle division by zero
    sen[np.isnan(sen)] = 0
    spe[np.isnan(spe)] = 0
    ppv[np.isnan(ppv)] = 0
    F1[np.isnan(F1)] = 0

    print('Sensitivity:', sen)
    print('Precision:', ppv)
    print('F1 Score:', F1)

    return acc, sen, spe, ppv, F1


def data_maker_test(path):
    """Load test dataset"""
    test = 'test'
    all_data = []
    all_labels = []

    # Load different classes
    with open(f'{path}{test}_N.pkl', 'rb') as file1:
        DCG_N = list(pickle.load(file1))
    with open(f'{path}{test}_S.pkl', 'rb') as file2:
        DCG_S = list(pickle.load(file2))
    with open(f'{path}{test}_V.pkl', 'rb') as file3:
        DCG_V = list(pickle.load(file3))

    # Combine data and labels
    all_data.extend(DCG_N + DCG_S + DCG_V)
    all_labels.extend([0] * len(DCG_N) + [1] * len(DCG_S) + [2] * len(DCG_V))

    # Print class distribution
    print("Test set class distribution:", Counter(all_labels))

    # Convert to tensors
    all_data = trans_tensor(np.array(all_data).astype(np.float32))
    all_labels = trans_tensor(np.array(all_labels))

    return all_data, all_labels


def data_maker(years, path):
    """Load training dataset from multiple years"""
    all_data = []
    all_labels = []

    for year in years:
        # Load data for each year
        with open(f'{path}{year}_N.pkl', 'rb') as file1:
            DCG_N = list(pickle.load(file1))
        with open(f'{path}{year}_S.pkl', 'rb') as file2:
            DCG_S = list(pickle.load(file2))
        with open(f'{path}{year}_V.pkl', 'rb') as file3:
            DCG_V = list(pickle.load(file3))

        # Add data from additional sources if year not 2019
        if year != '2019':
            with open(f'{path}{year}_8159_S8159-2.pkl', 'rb') as file10:
                DCG_S = list(pickle.load(file10))[0:4574] + DCG_S
            with open(f'{path}{year}_8159_V8159-2.pkl', 'rb') as file11:
                DCG_V = list(pickle.load(file11))[0:22787] + DCG_V

        # Combine data and labels
        all_data.extend(DCG_N + DCG_S + DCG_V)
        all_labels.extend([0] * len(DCG_N) + [1] * len(DCG_S) + [2] * len(DCG_V))

    print("Training set class distribution:", Counter(all_labels))

    # Convert to tensors
    all_data = trans_tensor(np.array(all_data).astype(np.float32))
    all_labels = trans_tensor(np.array(all_labels))

    return all_data, all_labels


def plot_roc(all_labels, all_pred):
    """Calculate ROC curve points"""
    enc = OneHotEncoder()
    all_labels = np.array(all_labels)[:, np.newaxis]
    label_h = enc.fit_transform(all_labels).toarray()
    fpr, tpr, _ = roc_curve(label_h.ravel(), all_pred.ravel())
    return fpr, tpr


def train_epoch(model, train_loader, optimizer, criterion_entropy, w1, device):
    """Training for one epoch"""
    model.train()
    epoch_all_labels = []
    epoch_all_res = []
    epoch_total_loss = 0

    for batch in tqdm(train_loader, desc='- (Training)', leave=False):
        sig, label = map(lambda x: x.to(device), batch)

        # Forward pass
        features, pred = model(sig)

        # Calculate losses
        ce_loss = criterion_entropy(pred, label)
        erl_loss = compute_erl_loss(features, label, num_classes=3)
        loss = ce_loss + w1 * erl_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record results
        epoch_total_loss += loss.item()
        epoch_all_labels.extend(label.cpu().numpy())
        epoch_all_res.extend(pred.max(1)[1].cpu().numpy())

    return epoch_total_loss, epoch_all_labels, epoch_all_res


def validate_epoch(model, val_loader, criterion_entropy, w1, device):
    """Validation for one epoch"""
    model.eval()
    epoch_all_labels = []
    epoch_all_res = []
    epoch_total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='- (Validation)', leave=False):
            sig, label = map(lambda x: x.to(device), batch)

            features, pred = model(sig)

            ce_loss = criterion_entropy(pred, label)
            erl_loss = compute_erl_loss(features, label, num_classes=3)
            loss = ce_loss + w1 * erl_loss

            epoch_total_loss += loss.item()
            epoch_all_labels.extend(label.cpu().numpy())
            epoch_all_res.extend(pred.max(1)[1].cpu().numpy())

    return epoch_total_loss, epoch_all_labels, epoch_all_res


def test_epoch(model, test_loader, criterion_entropy, w1, device):
    """Testing for one epoch"""
    model.eval()
    epoch_all_labels = []
    epoch_all_res = []
    epoch_all_pre = []
    epoch_total_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='- (Testing)', leave=False):
            sig, label = map(lambda x: x.to(device), batch)

            features, pred = model(sig)

            ce_loss = criterion_entropy(pred, label)
            erl_loss = compute_erl_loss(features, label, num_classes=3)
            loss = ce_loss + w1 * erl_loss

            epoch_total_loss += loss.item()
            epoch_all_labels.extend(label.cpu().numpy())
            epoch_all_res.extend(pred.max(1)[1].cpu().numpy())
            epoch_all_pre.extend(pred.cpu().numpy())

    return epoch_total_loss, epoch_all_labels, epoch_all_res, np.array(epoch_all_pre)


if __name__ == '__main__':
    # Setup
    device = device_nvidia(0)
    epoches = 80
    path = 'I:/balanced_dataset_8channel/'
    years_train = ['2019', '2020', '2021', '2022', '2023_2024']
    save_base_path = 'I:/model_outputs/'

    # Load data
    all_data, all_labels = data_maker(years_train, path)
    test_data, test_labels = data_maker_test(path)

    # Prepare test loader
    test_dataset = TensorDataset(test_data, test_labels.to(dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=512)

    # 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, all_labels)):
        print(f"Fold {fold + 1}")

        # Initialize model and training components
        model = Transformer(device=device, num_tokens=3, dim=512, heads=8, dim_head=64).to(device)
        w1 = nn.Parameter(torch.tensor(0.005, requires_grad=True))
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': [w1]}
        ], lr=0.0001)
        criterion_entropy = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]).to(device))

        # Prepare data loaders
        X_train, X_val = all_data[train_idx], all_data[val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]
        y_train = y_train.to(dtype=torch.long)
        y_val = y_val.to(dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512)

        # Training loop
        val_beat_acc = 0
        Loss_train = []
        Loss_val = []
        Acc_train = []
        Acc_val = []

        for epoch in range(epoches):
            print(f'[ Epoch {epoch} ]')

            # Training phase
            start_time0 = time.time()
            epoch_total_loss, epoch_all_labels, epoch_all_res = train_epoch(
                model, train_loader, optimizer, criterion_entropy, w1, device
            )

            epoch_train_loss = epoch_total_loss / len(train_loader)
            epoch_cm = confusion_matrix(epoch_all_labels, epoch_all_res)
            epoch_train_acc, _, _, _, _ = cal_statistic(epoch_cm)

            Loss_train.append(epoch_train_loss)
            Acc_train.append(epoch_train_acc)

            print(f'{epoch} - (Training) loss: {epoch_train_loss:8.5f}, '
                  f'accuracy: {100 * epoch_train_acc:3.3f} %, time: {(time.time() - start_time0) / 60:3.3f} min')

            # Validation phase
            start_time1 = time.time()
            epoch_total_loss, epoch_all_labels, epoch_all_res = validate_epoch(
                model, val_loader, criterion_entropy, w1, device
            )

            epoch_val_loss = epoch_total_loss / len(val_loader)
            epoch_cm = confusion_matrix(epoch_all_labels, epoch_all_res)
            epoch_val_acc, _, _, _, _ = cal_statistic(epoch_cm)

            Loss_val.append(epoch_val_loss)
            Acc_val.append(epoch_val_acc)

            print(f'{epoch} - (Validation) loss: {epoch_val_loss:8.5f}, '
                  f'accuracy: {100 * epoch_val_acc:3.3f} %, time: {(time.time() - start_time1) / 60:3.3f} min')

            # Testing phase
            start_time2 = time.time()
            epoch_total_loss, epoch_all_labels, epoch_all_res, epoch_all_pre = test_epoch(
                model, test_loader, criterion_entropy, w1, device
            )

            # Calculate ROC curve
            FPR, TPR = plot_roc(epoch_all_labels, epoch_all_pre)

            epoch_test_loss = epoch_total_loss / len(test_loader)
            epoch_cm = confusion_matrix(epoch_all_labels, epoch_all_res)
            epoch_test_acc, _, _, _, _ = cal_statistic(epoch_cm)

            print(f'{epoch} - (Testing) loss: {epoch_test_loss:8.5f}, '
                  f'accuracy: {100 * epoch_test_acc:3.3f} %, time: {(time.time() - start_time2) / 60:3.3f} min')

            # Save best model
            if epoch_val_acc > val_beat_acc:
                save_path = f'{save_base_path}/fold_{fold}'
                os.makedirs(save_path, exist_ok=True)

                # Save model
                torch.save(model, f'{save_path}/best_model.pkl')

                # Save ROC curve data
                np.save(f'{save_path}/FPR.npy', FPR)
                np.save(f'{save_path}/TPR.npy', TPR)

                val_beat_acc = epoch_val_acc

                # Save evaluation metrics
                class_report = classification_report(epoch_all_labels, epoch_all_res, output_dict=True)
                conf_matrix_df = pd.DataFrame(
                    epoch_cm,
                    index=[f"Actual_{i}" for i in range(epoch_cm.shape[0])],
                    columns=[f"Predicted_{i}" for i in range(epoch_cm.shape[1])]
                )
                class_report_df = pd.DataFrame(class_report).transpose()

                with pd.ExcelWriter(f'{save_path}/model_evaluation.xlsx') as writer:
                    conf_matrix_df.to_excel(writer, sheet_name="Confusion_Matrix")
                    class_report_df.to_excel(writer, sheet_name="Classification_Report")

                print(f"Saved best model and evaluation metrics for fold {fold}")

        # Save training history
        np.save(f'{save_path}/loss_train.npy', Loss_train)
        np.save(f'{save_path}/loss_val.npy', Loss_val)
        np.save(f'{save_path}/acc_train.npy', Acc_train)
        np.save(f'{save_path}/acc_val.npy', Acc_val)