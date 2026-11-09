import numpy as np
import sys
sys.path.append("/root/autodl-tmp/sata_aaai/")
import torch
from tqdm import tqdm
from Lightning.BagDataset_GPU import load_dataset
from Lightning.Tensorboard_LG import tensorboard_lg
from Lightning.metric_LG import metric_lg
from Lightning.stop_early_LG import stop_early_lg
from torch import nn
import torch.nn.functional as F
from Models.attn import ResidualGatedFusion


class Lightning():
    def __init__(self,
                 intervention_model,observation_model,
                 args

                ):
        self.melg=metric_lg(metric_dir=args.metic_dir)
        self.stlg=stop_early_lg(metric=args.metric, patient=args.patient)
        self.tlg=tensorboard_lg(tensorboard_folder=args.tensorboard_dir)
        self.criterion = nn.CrossEntropyLoss()


        self.args=args

        self.intervention_model = intervention_model.to(args.device)
        self.observation_model = observation_model.to(args.device)
        self.intervention_model.load_state_dict(observation_model.state_dict())

        self.attn = ResidualGatedFusion(args.hidden_dim).to(args.device)
        self.observation_classifier = nn.Linear(args.hidden_dim, args.nclass).to(args.device)
        self.intervention_classifier = nn.Linear(args.hidden_dim, args.nclass).to(args.device)


        causal_params = list(self.intervention_model.parameters()) + \
                        list(self.intervention_classifier.parameters()) +\
                        list(self.attn.parameters()) + \
                        list(self.observation_model.parameters()) +\
                        list(self.observation_classifier.parameters())

        self.optimizer_causal = torch.optim.Adam(
            filter(lambda p: p.requires_grad, causal_params),
            lr=args.lr, weight_decay=args.wd)

        self.scheduler_causal = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_causal, args.num_epochs, 0.000005)


    def max_log_prob_of_wrong_classes(self,prob,y_true):

        truth_mask = F.one_hot(y_true, num_classes=prob.shape[1]).float()  # [B, C]
        wrong_probs = prob * (1.0 - truth_mask)
        max_wrong_probs, max_idx = torch.max(wrong_probs, dim=1)
        log_probs = F.log_softmax(prob, dim=1)  # [B, C]

        loss_trivial = -torch.mean(log_probs[0,max_idx])
        return loss_trivial



    def train_step(self,x,y):

        feat_obs = self.observation_model(x)
        feat_int = self.intervention_model(x)

        fused_feat, c = self.attn(feat_int, feat_obs.detach())

        int_score = self.intervention_classifier(fused_feat).view(1, -1)
        obs_score = self.observation_classifier(feat_obs).view(1, -1)

        obs_int_score = self.intervention_classifier(feat_obs).view(1, -1)
        int_obs_score = self.observation_classifier(feat_int).view(1, -1)

        loss = 1 * self.criterion(obs_score, y) \
               + 1 * self.criterion(int_score, y) \
               + self.args.l * self.max_log_prob_of_wrong_classes(obs_int_score, y) \
                + self.args.l * self.max_log_prob_of_wrong_classes(int_obs_score, y)

        self.optimizer_causal.zero_grad()
        loss.backward()
        self.optimizer_causal.step()


    def train(self,best_ckp,last_ckp):

        train_dataloder = load_dataset(train="train", args=self.args)
        val_dataloader = load_dataset(train="val", args=self.args)

        self.tlg.init_tensorbard(self.args.seed)
        best_score=0

        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            self.set_train_mode()

            for bag_label, bag_feats in tqdm(train_dataloder, desc="{} {} Seed {} Training {}".format(self.args.model_name,
                                                                                                     self.args.dataname,
                                                                                                     self.args.seed,epoch)):
                self.train_step(bag_feats.squeeze(0), bag_label.squeeze(0))
            self.scheduler_causal.step()

            acc = self.val(val_dataloader)

            if acc > best_score:
                best_score = acc
                torch.save(self.get_checkpoint_state(), best_ckp)
        torch.save(self.get_checkpoint_state(), last_ckp)

    def val_inference(self,bag_feats,y):
        feat_trivial = self.observation_model(bag_feats)
        feat_causal = self.intervention_model(bag_feats)

        fused_feat,c = self.attn(feat_causal,feat_trivial.detach())

        true_score = self.intervention_classifier(fused_feat)
        true_score1 = self.observation_classifier(feat_trivial)

        true_score=(torch.softmax(true_score,dim=1)+torch.softmax(true_score1,dim=1))/2

        return true_score

    def val(self,val_dataloader,epoch=0,csv_path=None,mean=False,Test=True):
        self.set_eval_mode()

        val_labels = []
        val_predictions = []

        with torch.no_grad():
            for bag_label, bag_feats in val_dataloader:

                causal_score = self.val_inference(bag_feats.squeeze(0),bag_label)
                val_labels.extend(bag_label.cpu())
                val_predictions.extend(causal_score.cpu())

        test_labels = torch.cat(val_labels)
        test_predictions = torch.stack(val_predictions)


        current_result = self.melg.get_reslut(epoch,test_predictions, test_labels,num_classes=self.args.nclass,
                         task_type="binary" if self.args.nclass == 2 else "multiclass",csv_path=csv_path,mean=mean)

        return current_result["auc"]

    def test(self,
             epoch,
             checkpoint_path=None,
             csv_path=None,
             mean=False
             ):
        self.set_eval_mode()

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.load_checkpoint_state(checkpoint)

        test_dataloader = load_dataset(train="test", args=self.args)
        test_score=self.val(test_dataloader,epoch=epoch,csv_path=csv_path,mean=mean,Test=False)

        return test_score

    def get_checkpoint_state(self):
        return {
            "intervention_model": self.intervention_model.state_dict(),
            "observation_model": self.observation_model.state_dict(),
            "observation_classifier": self.observation_classifier.state_dict(),
            "intervention_classifier": self.intervention_classifier.state_dict(),
            "optimizer_causal": self.optimizer_causal.state_dict(),
            "attn":self.attn.state_dict()
        }

    def load_checkpoint_state(self, checkpoint):

        self.intervention_model.load_state_dict(checkpoint["intervention_model"])
        self.observation_model.load_state_dict(checkpoint["observation_model"])

        self.observation_classifier.load_state_dict(checkpoint["observation_classifier"])
        self.intervention_classifier.load_state_dict(checkpoint["intervention_classifier"])
        self.attn.load_state_dict(checkpoint["attn"])

    def set_train_mode(self):
        self.intervention_model.train()
        self.observation_model.train()
        self.observation_classifier.train()
        self.intervention_classifier.train()
        self.attn.train()

    def set_eval_mode(self):
        self.intervention_model.eval()
        self.observation_model.eval()
        self.observation_classifier.eval()
        self.intervention_classifier.eval()
        self.attn.eval()

