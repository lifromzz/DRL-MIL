import numpy as np
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
import h5py
import torch
from tqdm import tqdm



class BagDataset_gpu(Dataset):
    def __init__(self, data,split_path,device="cuda") -> None:
        super(BagDataset_gpu).__init__()
        self.device=device
        file_labels_df = pd.read_csv(split_path)
        train_slide_names = file_labels_df['path'].tolist()
        pt_labels = file_labels_df['STAS'].tolist()

        self.x ,self.y= [],[]
        for i,(slide_name,pt_label) in tqdm(enumerate(zip(train_slide_names,pt_labels))):
            h5_path = os.path.join(data,slide_name + ".h5")
            pt_path = os.path.join(data,slide_name + ".pt")

            if os.path.exists(pt_path):
                bag=torch.load(pt_path,map_location=self.device)
                label = torch.tensor([int(pt_label)],dtype=torch.long).to(self.device)
                self.x.append(bag)
                self.y.append(label)
            elif os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as file:
                    bag = torch.from_numpy(file['features'][:]).to(self.device)
                    label = torch.tensor([int(pt_label)],dtype=torch.long).to(self.device)
                    self.x.append(bag)
                    self.y.append(label)
            else:
                print(h5_path)



    def __getitem__(self, idx):
        label, feats = self.y[idx].to("cuda"), self.x[idx].to("cuda")
        return label, feats

    def __len__(self):
        return len(self.x)

def load_dataset(train,args):
    if train == "train":
        dataset=BagDataset_gpu(
            data=args.train_data,
            split_path=args.train_excel_path,
            device="cuda")

        return DataLoader(dataset,shuffle=True,batch_size=1)
    elif train == "val":
        dataset = BagDataset_gpu(
            data=args.val_data,
            split_path=args.val_excel_path,
            device="cpu"
        )
        return DataLoader(dataset,shuffle=False,batch_size=1)
    else:
        dataset = BagDataset_gpu(
            data=args.test_data,
            split_path=args.test_excel_path,
            device="cpu"
        )
        return DataLoader(dataset,shuffle=False,batch_size=1)