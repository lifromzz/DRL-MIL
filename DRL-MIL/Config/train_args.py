from Config.args_base import get_argp


def get_args(dataname,model_name,vit=False):
    argp=get_argp(dataname,model_name=model_name,vit=vit)
    args=argp.parse_args()

    return args