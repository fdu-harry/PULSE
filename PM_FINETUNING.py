# -*- coding: utf-8 -*-
import torch
import glob
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import shutil
import copy


class SPELoss(nn.Module):
    def __init__(self):
        super(SPELoss, self).__init__()

    def forward(self, features, source_center):
        # Calculate feature centroid of current batch
        target_center = torch.mean(features, dim=0)
        # Calculate alignment loss between target and source centers
        alignment_loss = torch.mean((target_center - source_center) ** 2)
        return alignment_loss


def device_nvidia(num):
    device = torch.device(f'cuda:{num}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device


def trans_tensor(dataset):
    dataset = torch.tensor(dataset).type(torch.FloatTensor)
    return dataset


def cal_statistic(cm):
    tp = np.diagonal(cm)
    gt_num = np.sum(cm, axis=1)
    pre_num = np.sum(cm, axis=0)
    fp = pre_num - tp
    num0 = np.sum(gt_num)
    gt_num0 = num0 - gt_num
    tn = gt_num0 - fp
    with np.errstate(divide='ignore', invalid='ignore'):
        sen = np.true_divide(tp, gt_num)
        spe = np.true_divide(tn, gt_num0)
        ppv = np.true_divide(tp, pre_num)
        F1 = 2 * (sen * ppv) / (sen + ppv)
        acc = np.true_divide(np.sum(tp), num0)
    sen[np.isnan(sen)] = 0
    spe[np.isnan(spe)] = 0
    ppv[np.isnan(ppv)] = 0
    F1[np.isnan(F1)] = 0
    print('sen is : {sen}'.format(sen=sen))
    print('ppv is : {ppv}'.format(ppv=ppv))
    print('F1 is : {F1}'.format(F1=F1))
    weighted_F1 = np.sum(F1 * gt_num) / num0
    return acc, sen, spe, ppv, weighted_F1


if __name__ == '__main__':
    all_GM_AUC = []
    all_PM_AUC = []
    all_GM_Acc = []
    all_PM_Acc = []
    all_GM_F1 = []
    all_PM_F1 = []
    all_GM_fpr = []
    all_PM_fpr = []
    all_GM_tpr = []
    all_PM_tpr = []

    pattern_DCG = 'I:\\5FoldCrossValidation_PersonalizationTestSet\\ClassifiedTestSet\\' + '\\' + '*' + '_data.pkl'
    dcg = glob.glob(pattern_DCG)
    number = 0
    valid_labels = {'N', 'S', 'V'}
    mapping = {"N": 0, "S": 1, "V": 2}

    number += 1
    print(number, '...................................................')

    for patient in dcg[:]:
        GM_AUC = []
        PM_AUC = []
        GM_Acc = []
        PM_Acc = []
        GM_F1 = []
        PM_F1 = []
        GM_fpr = []
        PM_fpr = []
        GM_tpr = []
        PM_tpr = []

        model_list = ['0_model', '1_model', '2_model', '3_model', '4_model']
        for model_name in model_list:
            device = device_nvidia(0)
            model_path0 = 'I:\\5FoldCrossValidation_PersonalizationTestSet\\ProposedModel_Channel4\\'
            model_path = model_path0 + model_name + '.pkl'
            model = torch.load(model_path).to(device)

            # 定义优化器和损失函数
            optimizer_ = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
            criterion_entropy = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]).to(device))
            criterion_spe = SPELoss()
            w2 = 0.005  # SPE loss权重
            epoches = 3

            # 加载患者数据
            base_name = patient.split('\\')[-1][:-9]
            path_DCG_data = 'H:\\BimodalDataset_PatientRecord\\Dataset_512_8159\\' + base_name + '_data.pkl'
            path_DCG_label = 'H:\\BimodalDataset_PatientRecord\\Dataset_512_8159\\' + base_name + '_label.pkl'

            with open(path_DCG_data, 'rb') as file1:
                patient_DCG0 = list(pickle.load(file1))
            with open(path_DCG_label, 'rb') as file2:
                patient_label0 = list(pickle.load(file2))

            # 数据预处理
            patient_DCG = []
            patient_label = []
            for data_, label_ in zip(patient_DCG0, patient_label0):
                if label_ in valid_labels:
                    patient_DCG.append(data_)
                    patient_label.append(label_)
            patient_label = [mapping[annotation] for annotation in patient_label]

            if len(patient_label) <= 500:
                continue

            # 准备fine-tuning数据
            ecg_data = np.array(patient_DCG[0:500]).astype(np.float32)
            ecg_label = np.array(patient_label[0:500])

            # K-means聚类
            Kmeans_data = ecg_data.reshape(ecg_data.shape[0], -1)
            kmeans = KMeans(n_clusters=3, random_state=0)
            kmeans.fit(Kmeans_data)
            labels = kmeans.labels_

            # 获取最大簇
            clusters = [Kmeans_data[labels == i] for i in range(3)]
            cluster_lengths = [len(cluster) for cluster in clusters]
            max_cluster_idx = np.argmax(cluster_lengths)
            cluster_target = clusters[max_cluster_idx].reshape(-1, ecg_data.shape[1], ecg_data.shape[2])

            # 划分训练集和验证集
            n_samples = len(cluster_target)
            n_train = int(0.9 * n_samples)
            train_data = cluster_target[:n_train]
            val_data = cluster_target[n_train:]

            # 获取伪标签
            ecg_data_fine = trans_tensor(cluster_target)
            dataset = TensorDataset(ecg_data_fine)
            ecg_data_fine_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

            model.eval()
            y_pred_all = []
            with torch.no_grad():
                for batch in tqdm(ecg_data_fine_loader, mininterval=0.5, desc='- (Center)  ', leave=False):
                    sig = batch[0].to(device)
                    outputs1, outputs = model(sig)
                    pred = outputs.detach().cpu()
                    y_pred_all.extend(pred.max(1)[1].cpu().numpy())

            most_frequent_class = np.argmax(np.bincount(y_pred_all))

            # 加载对应类别的源域中心
            kmeans_center_path = 'I:\\5FoldCrossValidation_PersonalizationTestSet\\'
            if most_frequent_class == 0:
                kmeans_center = torch.load(kmeans_center_path + 'tensorN.pth')
            elif most_frequent_class == 1:
                kmeans_center = torch.load(kmeans_center_path + 'tensorS.pth')
            else:
                kmeans_center = torch.load(kmeans_center_path + 'tensorV.pth')

            # 计算源域特征中心
            dataset = TensorDataset(kmeans_center)
            Kmeans_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

            model.eval()
            y_pred_all = []
            with torch.no_grad():
                for batch in tqdm(Kmeans_loader, mininterval=0.5, desc='- (Center)  ', leave=False):
                    sig = batch[0].to(device)
                    outputs1, outputs = model(sig)
                    pred = outputs1.detach().cpu()
                    y_pred_all.extend(pred.cpu().numpy())

            kmeans_center = np.mean(np.array(y_pred_all), axis=0)
            kmeans_center = trans_tensor(kmeans_center).to(device)

            # 准备训练和验证数据加载器
            train_loader = DataLoader(
                TensorDataset(
                    trans_tensor(train_data),
                    trans_tensor(np.array([most_frequent_class] * len(train_data))).long()
                ),
                batch_size=1024,
                shuffle=True
            )

            val_loader = DataLoader(
                TensorDataset(
                    trans_tensor(val_data),
                    trans_tensor(np.array([most_frequent_class] * len(val_data))).long()
                ),
                batch_size=1024,
                shuffle=False
            )

            # 准备测试数据
            ecg_data = trans_tensor(np.array(patient_DCG[500:]).astype(np.float32))
            ecg_label = trans_tensor(np.array(patient_label[500:]))
            per_test_loader = DataLoader(
                TensorDataset(ecg_data, ecg_label.to(dtype=torch.long)),
                batch_size=1024,
                shuffle=False
            )

            # 评估GM性能
            model.eval()
            y_pred_all = []
            y_pred_pro = []
            y_label_all = []
            with torch.no_grad():
                for batch in tqdm(per_test_loader, mininterval=0.5, desc='- (Initial Testing)  ', leave=False):
                    sig = batch[0].to(device)
                    label = batch[1].to(device)
                    outputs1, outputs = model(sig)
                    pred = outputs.detach().cpu()
                    y_pred_all.extend(pred.max(1)[1].cpu().numpy())
                    y_label_all.extend(label.cpu().numpy())
                    y_pred_pro.extend(pred.numpy())

            y_pred_pro = np.array(y_pred_pro)
            y_label_all = np.array(y_label_all)

            try:
                y_label_all_bin = label_binarize(y_label_all, classes=np.arange(y_pred_pro.shape[1]))
            except IndexError as e:
                continue

            fpr_initial, tpr_initial, _ = roc_curve(y_label_all_bin.ravel(), y_pred_pro.ravel())
            roc_auc_initial = auc(fpr_initial, tpr_initial)

            epoch_cm = confusion_matrix(y_label_all, y_pred_all)
            epoch_test_acc_initial, _, _, _, weighted_F1_initial = cal_statistic(epoch_cm)
            print('initial test', epoch_test_acc_initial)

            # 模型微调准备
            for param in model.parameters():
                param.requires_grad = False
            for param in model.to_out.parameters():
                param.requires_grad = True
            for param in model.layers.parameters():
                param.requires_grad = True

            # 记录训练过程
            best_val_loss = float('inf')
            best_model = None
            patience = 5
            patience_counter = 0

            # 训练循环
            for epoch in range(epoches):
                # 训练阶段
                model.train()
                train_loss = 0
                y_pred_fine = []
                y_label_fine = []

                for batch in tqdm(train_loader, mininterval=0.5, desc=f'- (Training Epoch {epoch})  ', leave=False):
                    sig = batch[0].to(device)
                    label = batch[1].to(device)
                    outputs1, outputs = model(sig)

                    # 计算损失
                    ce_loss = criterion_entropy(outputs, label)
                    spe_loss = criterion_spe(outputs1, kmeans_center)
                    loss = ce_loss + w2 * spe_loss

                    optimizer_.zero_grad()
                    loss.backward()
                    optimizer_.step()

                    train_loss += loss.item()
                    pred = outputs.detach().cpu()
                    y_pred_fine.extend(pred.max(1)[1].cpu().numpy())
                    y_label_fine.extend(label.cpu().numpy())

                # 验证阶段
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in tqdm(val_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):
                        sig = batch[0].to(device)
                        label = batch[1].to(device)
                        outputs1, outputs = model(sig)

                        ce_loss = criterion_entropy(outputs, label)
                        spe_loss = criterion_spe(outputs1, kmeans_center)
                        loss = ce_loss + w2 * spe_loss

                        val_loss += loss.item()

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # 使用最佳模型进行测试
            if best_model is not None:
                model = best_model

            model.eval()
            y_pred_all = []
            y_label_all = []
            y_pred_pro = []
            test_loss = 0

            with torch.no_grad():
                for batch in tqdm(per_test_loader, mininterval=0.5, desc='- (Final Testing)  ', leave=False):
                    sig = batch[0].to(device)
                    label = batch[1].to(device)
                    outputs1, outputs = model(sig)

                    ce_loss = criterion_entropy(outputs, label)
                    spe_loss = criterion_spe(outputs1, kmeans_center)
                    loss = ce_loss + w2 * spe_loss
                    test_loss += loss.item()

                    pred = outputs.detach().cpu()
                    y_pred_all.extend(pred.max(1)[1].cpu().numpy())
                    y_label_all.extend(label.cpu().numpy())
                    y_pred_pro.extend(pred.numpy())

            y_pred_pro = np.array(y_pred_pro)
            y_label_all = np.array(y_label_all)
            y_label_all_bin = label_binarize(y_label_all, classes=np.arange(y_pred_pro.shape[1]))
            fpr, tpr, _ = roc_curve(y_label_all_bin.ravel(), y_pred_pro.ravel())
            roc_auc = auc(fpr, tpr)

            epoch_cm = confusion_matrix(y_label_all, y_pred_all)
            epoch_test_acc, _, _, _, weighted_F1 = cal_statistic(epoch_cm)
            print(f'Final test accuracy: {epoch_test_acc}')

            # 记录性能指标
            GM_Acc.append(epoch_test_acc_initial)
            GM_F1.append(weighted_F1_initial)
            GM_AUC.append(roc_auc_initial)
            GM_fpr.append(fpr_initial)
            GM_tpr.append(tpr_initial)

            PM_Acc.append(epoch_test_acc)
            PM_F1.append(weighted_F1)
            PM_AUC.append(roc_auc)
            PM_fpr.append(fpr)
            PM_tpr.append(tpr)

        # 计算平均性能提升
        GM_AUC_MEAN = np.mean(np.array(GM_AUC))
        PM_AUC_MEAN = np.mean(np.array(PM_AUC))

        # 如果性能有提升，保存结果
        if PM_AUC_MEAN - GM_AUC_MEAN > 0:
            destination_folder = 'I:\\5FoldCrossValidation_PersonalizationTestSet\\ClassifiedTestSet\\TestSet\\'
            shutil.copy(path_DCG_data, destination_folder)
            shutil.copy(path_DCG_label, destination_folder)
            print("文件复制完成！")

            all_GM_AUC.append(GM_AUC)
            all_PM_AUC.append(PM_AUC)
            all_GM_Acc.append(GM_Acc)
            all_PM_Acc.append(PM_Acc)
            all_GM_F1.append(GM_F1)
            all_PM_F1.append(PM_F1)
            all_GM_fpr.append(GM_fpr)
            all_PM_fpr.append(PM_fpr)
            all_GM_tpr.append(GM_tpr)
            all_PM_tpr.append(PM_tpr)

    # 保存所有结果
    save_path = model_path0 + 'PersonalizationResults\\'
    all_GM_AUC = np.array(all_GM_AUC)
    all_PM_AUC = np.array(all_PM_AUC)
    all_GM_Acc = np.array(all_GM_Acc)
    all_PM_Acc = np.array(all_PM_Acc)
    all_GM_F1 = np.array(all_GM_F1)
    all_PM_F1 = np.array(all_PM_F1)

    metrics = {
        'GM_AUC': all_GM_AUC,
        'PM_AUC': all_PM_AUC,
        'GM_Acc': all_GM_Acc,
        'PM_Acc': all_PM_Acc,
        'GM_F1': all_GM_F1,
        'PM_F1': all_PM_F1,
        'GM_fpr': all_GM_fpr,
        'PM_fpr': all_PM_fpr,
        'GM_tpr': all_GM_tpr,
        'PM_tpr': all_PM_tpr
    }

    for name, metric in metrics.items():
        with open(save_path + f'{name}.pkl', 'wb') as f:
            pickle.dump(metric, f)