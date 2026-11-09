import os.path
from Config.base import *
from Models.model_loader import model_load

def main(args,train,test,best):
    from Lightning.Lightning_single_opti import Lightning

    for i in range(args.start_seed,args.end_seed):
        args.seed = i
        set_data(args)
        intervention_model,observation_model = model_load(args)

        lg = Lightning(intervention_model,observation_model,args)

        best_ckp = os.path.join(args.checkpoint_dir,
                                "{}_best_seed{}.pth".format(args.model_name, args.seed))
        last_ckp = os.path.join(args.checkpoint_dir,
                                "{}_Last_seed{}.pth".format(args.model_name, args.seed))

        if train:
            lg.train(best_ckp,last_ckp)

        if test:
            lg.test(epoch=0 if i == 0 else 200,
                    checkpoint_path=best_ckp if best else last_ckp,
                    csv_path=args.test_best_csv_path if best else args.test_last_csv_path,
                    mean= True if i ==args.end_seed-1 else False)