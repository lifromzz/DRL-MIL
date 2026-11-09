import argparse

class_dict={
    "cptac_lung":3,
    "BRACS":7,
    "BRACS_": 7

}
def get_argp(dataname,model_name,vit=None):

    argp = argparse.ArgumentParser()
    argp.add_argument("--model_name", default=model_name,type=str)
    argp.add_argument("--dataname", default=dataname,type=str)

    if "_" in dataname and "BRACS" in dataname:
        dataname = dataname.replace("_", "")

    if vit=="BT":
        argp.add_argument("--train_data", default=f"/root/autodl-tmp/Dataset/{dataname}/ssl_BT_r50_level1_224/stage4", type=str)
        argp.add_argument("--test_data", default=f"/root/autodl-tmp/Dataset/{dataname}/ssl_BT_r50_level1_224/stage4", type=str)
        argp.add_argument("--val_data", default=f"/root/autodl-tmp/Dataset/{dataname}/ssl_BT_r50_level1_224/stage4", type=str)
        argp.add_argument("--input_dim", default=2048, type=int)  # 1536
    else:
        argp.add_argument("--train_data", default=f"/root/autodl-tmp/Dataset/{dataname}/features_gigapath/", type=str)
        argp.add_argument("--test_data", default=f"/root/autodl-tmp/Dataset/{dataname}/features_gigapath/", type=str)
        argp.add_argument("--val_data", default=f"/root/autodl-tmp/Dataset/{dataname}/features_gigapath/", type=str)
        argp.add_argument("--input_dim", default=1536, type=int)  # 1536

    argp.add_argument("--val",default=True,type=bool)
    argp.add_argument("--nclass", default=class_dict[dataname], type=int)

    if model_name =="ilra" or  model_name =="wikg" or model_name =="transmil":
        argp.add_argument("--hidden_dim", default=512, type=int)
    else:
        argp.add_argument("--hidden_dim", default=2048 if vit=="BT" else 1536, type=int)

    argp.add_argument("--l",default=0.1,type=float)

    if model_name =="ilra":
        argp.add_argument("--lr", default=1e-4)
        argp.add_argument("--wd", default=0)
    else:
        argp.add_argument("--lr", default=1e-4)
        argp.add_argument("--wd", default=1e-5)

    argp.add_argument("--start_epoch", default=0, type=int)
    argp.add_argument("--num_epochs", default=50, type=int)
    argp.add_argument("--start_seed",default=0,type=int)
    argp.add_argument("--end_seed",default=5,type=int)
    argp.add_argument("--epoch_frq",default=-1,type=str)
    argp.add_argument("--patient", default=100)


    argp.add_argument("--tensorboard_dir", default="./{}/log".format(dataname))
    argp.add_argument("--checkpoint_dir", default="./{}/checkpoint".format(dataname))
    argp.add_argument("--metic_dir", default="./{}/results".format(dataname))
    argp.add_argument("--metric",default="auc")
    argp.add_argument("--seed", default=0)
    argp.add_argument("--device", default="cuda")

    return argp