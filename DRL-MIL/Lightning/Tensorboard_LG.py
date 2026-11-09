from torch.utils.tensorboard import SummaryWriter
from club.util import get_subpath,get_subfolder_names
import os

class tensorboard_lg():
    def __init__(self,tensorboard_folder):
        self.tensorboard_folder=tensorboard_folder
        self.recoder_dict={}
        self.mini_batch_count = 0

        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)
    def init_tensorbard(self,seed):
        next_tensorboard_log = os.path.join(self.tensorboard_folder, str(seed))
        if not os.path.exists(next_tensorboard_log):
            os.makedirs(next_tensorboard_log)
        # 创建写入器
        self.writer = SummaryWriter(next_tensorboard_log)
    def next_tensorbard(self):
        subfolder_names_list=get_subfolder_names(self.tensorboard_folder)
        if len(subfolder_names_list)==0:
            max_index=0
        else:
            max_index=max([int(i) for i in subfolder_names_list])
        next_tensorboard_log=os.path.join(self.tensorboard_folder,str(max_index+1))
        if not os.path.exists(next_tensorboard_log):
            os.makedirs(next_tensorboard_log)
        #创建写入器
        self.writer = SummaryWriter(next_tensorboard_log)


    def refresh_log(self, epoch,recoder_dict,step):

        for k in recoder_dict.keys():
            if ("acc" in k) or ("loss" in k):
                v=recoder_dict[k]/step
                self.writer.add_scalar(k, v, epoch)
                print(f"Epoch {epoch+1},{k}:{v:.4f}")
                recoder_dict[k] = v
        return recoder_dict