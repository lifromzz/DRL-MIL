import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score,roc_curve
import csv
import os
import torch
from torchmetrics import AUROC, F1Score, Recall, Precision, Accuracy,MatthewsCorrCoef
class metric_lg():
    def __init__(self,metric_dir):
        self.metric_logger = {
            "Accuracy": accuracy_score,
            "auc":self.roc_threshold,
            "Precision": precision_score,
            "Recall": recall_score,
            "F1": f1_score,

        }
        self.metric_dir=metric_dir
    def init_metric_dir(self):
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)
    def Accuracy(self,predictions,targets):
        # 获取预测结果中概率最大的类别作为预测类别
        _, predicted_labels = torch.max(predictions, 1)
        if len(targets.size())>=2:
            _,targets=torch.max(targets,1)

        # 比较预测类别与真实标签，统计预测正确的数量
        correct = (predicted_labels == targets).sum().item()

        # 计算正确率
        accuracy = correct / targets.size(0)

        return accuracy

    def get_reslut(self,epoch,preds, target, task_type="binary", num_classes=None,csv_path=None,mean=False):

        """

         参数:
             preds (torch.Tensor): 模型预测值（概率或 logits）
             target (torch.Tensor): 真实标签
             task_type (str): 任务类型，可选 'binary', 'multiclass', 'multilabel'
             num_classes (int): 类别数（仅多分类或多标签任务需要）

         返回:
             dict: 包含 AUC、F1、recall、precision、accuracy 的字典


         注意: binary_preds = torch.tensor([0.8, 0.2, 0.9, 0.1])
            binary_target = torch.tensor([1, 0, 1, 0])
            binary_metrics = calculate_metrics(binary_preds, binary_target, task_type="binary")
            print("二分类指标:", binary_metrics)

            # 多分类示例
            multi_preds = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
            multi_target = torch.tensor([0, 1, 2])
            multi_metrics = calculate_metrics(multi_preds, multi_target, task_type="multiclass", num_classes=3)
            print("多分类指标:", multi_metrics)
         """
        # 确保输入是 PyTorch 张量
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)

        # 初始化指标
        metrics = {}

        # 设置任务参数
        task_kwargs = {}
        if task_type == "multiclass":
            task_kwargs["num_classes"] = num_classes
        elif task_type == "multilabel":
            task_kwargs["num_labels"] = num_classes

        if task_type =="binary":
            preds=preds[:,1]

        accuracy = Accuracy(task=task_type, **task_kwargs)
        metrics["accuracy"] = accuracy(preds, target).item() * 100

        # 计算 AUC
        auroc = AUROC(task=task_type, **task_kwargs)
        metrics["auc"] = auroc(preds, target).item()*100


        # 计算 F1 分数
        f1 = F1Score(task=task_type, **task_kwargs,average="macro")
        metrics["f1"] = f1(preds, target).item()*100

        mcc=  MatthewsCorrCoef(task=task_type, **task_kwargs)
        metrics["mcc"] = mcc(preds,target).item()*100

        # 计算召回率
        recall = Recall(task=task_type, **task_kwargs, average="macro")
        metrics["recall"] = recall(preds, target).item() * 100

        # 计算精确率
        precision = Precision(task=task_type, **task_kwargs, average="macro")
        metrics["precision"] = precision(preds, target).item() * 100
        print(metrics)
        if csv_path is not None:
            w_a="w" if epoch==0 else "a"
            self.write_dict_to_csv(csv_path,metrics,w_a)
        if mean:
            self.compute_mean(csv_path)
        return metrics

    def compute_mean(self, csv_path):
        df = pd.read_csv(csv_path)

        df_numeric = df.iloc[:]  # 排除第一行
        mean_values = df_numeric.mean(numeric_only=True).round(2)

        mean_row = pd.DataFrame([mean_values])
        df = pd.concat([df, mean_row], ignore_index=True)

        df.to_csv(csv_path, index=False)
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

    def roc_threshold(self, label,prediction,th=False):
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = self.optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        if th:
            return c_auc, threshold_optimal
        else:
            return c_auc

    def optimal_thresh(self,fpr, tpr, thresholds, p=0):
        loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
        idx = np.argmin(loss, axis=0)
        return fpr[idx], tpr[idx], thresholds[idx]

    def eval_metric(self,oprob, label):

        auc, threshold = self.roc_threshold(label,oprob)
        prob = oprob > threshold
        label = label > threshold

        TP = (prob & label).sum(0)
        TN = ((~prob) & (~label)).sum(0)
        FP = (prob & (~label)).sum(0)
        FN = ((~prob) & label).sum(0)

        accuracy = np.mean((TP + TN) / (TP + TN + FP + FN + 1e-12))
        precision = np.mean(TP / (TP + FP + 1e-12))
        recall = np.mean(TP / (TP + FN + 1e-12))
        specificity = np.mean(TN / (TN + FP + 1e-12))
        F1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        Tsocre = {"Accuracy":accuracy,
                  "AUC":auc,
                  "F1":F1,
                  "Recall":recall,
                  "Precision":precision,
                  "specificity":specificity}
        return Tsocre