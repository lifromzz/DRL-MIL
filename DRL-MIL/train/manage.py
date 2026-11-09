from train import train_test
from Config import train_args as args_base
from Config.base import *


def train(dataname,model_name,vit):
    args = args_base.get_args(dataname=dataname, model_name=model_name,vit=vit)

    run_root = f"result"

    reset_run_root(args, run_root,vit=vit)
    train_test.main(args, train=True, test=True, best=True)

def main(model_name,vit=None):

    train("BRACS", model_name,vit=vit)
    train("BRACS_", model_name,vit=vit)
    train("cptac_lung", model_name,vit=vit)
#
if __name__ == '__main__':

    main("abmil","PG")
    main("transmil", "PG")
    main("ilra", "PG")
    main("wikg", "PG")
    main("abmil","BT")
    main("transmil","BT")
    main("ilra","BT")
    main("wikg","BT")

