import numpy as np
import pandas as pd
import torchmetrics
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score,roc_curve,classification_report
import copy
import csv
import os
import torch

class metric_lg():
    def __init__(self,metric_dir):
        self.metric_dir=metric_dir
    def init_metric_dir(self):
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)

    def multi_label_roc(self,labels, predictions, num_classes):
        fprs = []
        tprs = []
        thresholds = []
        thresholds_optimal = []
        aucs = []
        if len(predictions.shape) == 1:
            predictions = predictions[:, None]
        if num_classes>1:
            one_hot = np.zeros((len(labels), int(labels.max()) + 1), dtype=int)
            one_hot[np.arange(len(labels)), labels.astype(int)] = 1
        for c in range(0, num_classes):
            label = labels if num_classes ==1 else one_hot[:, c]
            if sum(label) == 0:
                continue
            prediction = predictions[:, c]
            fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
            fpr_optimal, tpr_optimal, threshold_optimal = self.optimal_thresh(fpr, tpr, threshold)
            c_auc = roc_auc_score(label, prediction)
            aucs.append(c_auc)
            thresholds.append(threshold)
            thresholds_optimal.append(threshold_optimal)
        return aucs, thresholds, thresholds_optimal

    def optimal_thresh(self,fpr, tpr, thresholds, p=0):
        loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
        idx = np.argmin(loss, axis=0)
        return fpr[idx], tpr[idx], thresholds[idx]

    def get_reslut(self,test_predictions,test_labels,epoch=-1,nclasses=1,csv_path=None,mean=False):

        auc_value, _, thresholds_optimal = self.multi_label_roc(test_labels, test_predictions,
                                                                     nclasses)

        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag.squeeze()
        test_labels = np.squeeze(test_labels)
        bag_score = 0
        # average acc of all labels
        for i in range(0, len(test_labels)):
            bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
        avg_score = bag_score / len(test_labels)  # ACC
        cls_report = classification_report(test_labels, test_predictions, digits=4)
        print('\n Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100,
                                                      sum(auc_value) / len(auc_value) * 100))
        print('\n', cls_report)
        precision = precision_score(test_labels, test_predictions, average='macro')
        recall = recall_score(test_labels, test_predictions, average='macro')
        f1 = f1_score(test_labels, test_predictions, average='macro')

        # y_pred = torch.softmax(test_predictions, dim=-1)
        #
        # AUROC_metric = torchmetrics.AUROC(num_classes=nclasses, average='macro', task="multiclass").to("cuda")
        # AUROC_metric(y_pred, test_labels)
        # auroc = AUROC_metric.compute().item()
        #
        # F1_metric = torchmetrics.F1Score(num_classes=nclasses, average='macro', task="multiclass").to("cuda")
        # F1_metric(y_pred, test_labels)
        # f1_score = F1_metric.compute().item()
        #
        # acc_metric = torchmetrics.Accuracy(num_classes=nclasses, average="macro", task="multiclass").to("cuda")
        # acc_metric(y_pred, test_labels)
        # acc = acc_metric.compute().item()
        #
        # recall_metric = torchmetrics.Recall(num_classes=nclasses, average="macro", task="multiclass").to("cuda")
        # recall_metric(y_pred, test_labels)
        # recall = recall_metric.compute().item()
        #
        # pre_metric = torchmetrics.Precision(num_classes=nclasses, average="macro", task="multiclass").to("cuda")
        #
        # pre_metric(y_pred, test_labels)
        # pre = pre_metric.compute().item()
        # test_score = {'acc': acc * 100, 'auc': auroc * 100,
        #               'precision': pre * 100,
        #               'recall': recall * 100,
        #               'f1': f1_score * 100}
        # print(test_score)
        test_score = {'acc': avg_score * 100, 'auc': sum(auc_value) / len(auc_value) * 100,
                       'precision': precision * 100,
                       'recall': recall * 100,
                       'f1': f1 * 100}

        if csv_path is not None:
            w_a="w" if epoch==0 else "a"
            self.write_dict_to_csv(csv_path,test_score,w_a)
        if mean:
            self.compute_mean(csv_path)
        return test_score

    def write_dict_to_csv(self,filename, data_dict,w_a):
        self.init_metric_dir()
        # 写入字典到 CSV 文件
        with open(filename, w_a, newline='') as csvfile:
            fieldnames = list(data_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 如果文件为空，写入表头
            if os.stat(filename).st_size == 0:
                writer.writeheader()

            # 写入数据
            writer.writerow(data_dict)
    def compute_mean(self,csv_path):
        df = pd.read_csv(csv_path)

        df_numeric = df.iloc[1:]  # 排除第一行
        mean_values = df_numeric.mean(numeric_only=True).round(2)

        mean_row = pd.DataFrame([mean_values])
        df = pd.concat([df, mean_row], ignore_index=True)

        df.to_csv(csv_path, index=False)
    def roc_threshold(self, label,prediction,th=False):
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = self.optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        if th:
            return c_auc, threshold_optimal
        else:
            return c_auc
