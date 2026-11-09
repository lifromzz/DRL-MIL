import os
import json
import shutil
import time

import pandas as pd
import torch

import random
import numpy as np

def fix_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic=True

def get_subpath(dirpath,sort=False):
    path_list=os.listdir(dirpath)
    for i,path in enumerate(path_list):
        path_list[i]=os.path.normpath("%s/%s"%(dirpath,path))
    if sort:
        path_list.sort()
    return path_list
def get_subfolder_names(folder_path):
    #输出文件夹下不加后缀的文件夹名称
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and '.' not in item:
            subfolders.append(item)
    return subfolders
def join_path(first_path,second_path):
    path=os.path.normpath("%s/%s"%(first_path,second_path))
    return path

def dir_check(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_json(json_path,mode="r"):
    with open(json_path,mode) as f:
        cnnjson=json.load(f)
        f.close()
    return cnnjson

def read_txt(path):
    txt=open(path,encoding="utf-8")
    txt_list=[]
    for line in txt.readlines():
        line=line.strip("\n")
        line=line.split("\t")
        txt_list.append(line)
    return txt_list

def onehot_2_number(output):
    return torch.max(output,1)[1]
def number_2_onehot(number,nclass):
    batchsize=number.size[0]
    return torch.zeros(batchsize,nclass).scatter_(1,number,1)
def load_to_device(data,device,train=False,eval=False):
    for k,v in data.items():
        if train:
            data[k]=v.train()
        if eval:
            data[k]=v.eval()
        data[k]=v.to(device)
    return data

def Max_MIN_Tensor(input):
    max_value=torch.max(input)
    min_value=torch.min(input)
    return (input-min_value)/(max_value-min_value+1e-8)


def zscore_standardization(data,miu=0):
    """
    对数据进行Z-score标准化处理

    参数:
    data (numpy.ndarray): 原始数据，二维数组，其中每一行是一个样本，每一列是一个特征

    返回:
    numpy.ndarray: 标准化后的数据
    """
    # 计算每个特征的均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # 避免除以零的情况（虽然在实际应用中标准差为零的情况很少见）
    std[std == 0] = 1

    # 计算Z-score
    z_scores = (data - mean) / std+miu

    return z_scores

def copy_files_from_dict( source_folder, destination_folder,data_dict):
    # 遍历每一列
    for column, file_name_list in data_dict.items():
        for file_name in file_name_list:
            source_file_path = os.path.join(source_folder, file_name)
            destination_subfolder = os.path.join(destination_folder, column)

            if not os.path.exists(destination_subfolder):
                os.makedirs(destination_subfolder)
            if os.path.exists(source_file_path):
                shutil.copy(source_file_path, destination_subfolder)

def save_dict_to_csv(dict_name,csv_file_path):
    max_length = max(len(lst) for lst in dict_name.values())
    for keys, values in dict_name.items():
        if len(values) >= max_length:
            continue
        else:
            for i in range(max_length - len(values)):
                values.append(0)
    df = pd.DataFrame(dict_name)
    df.to_csv(csv_file_path, index=False)

def sort_instance_by_attention(instance_feat,attn):
    attn_clone = attn.clone().detach().squeeze()
    instance_feat = instance_feat.clone().detach()
    sorted_attn, index = attn_clone.sort()  # 升序
    sorted_instance_feat = instance_feat[index]
    return sorted_instance_feat

def random_sample(x,ramdom_per):
    random_num=int(ramdom_per*len(x))
    random_path=random.sample(x,random_num)
    return random_path