
def model_load(args):
    if args.model_name == 'abmil':
        import Models.abmil as mil
        intervention_model = mil.Attention(in_size=args.input_dim).cuda()
        observation_model = mil.Attention(in_size=args.input_dim).cuda()
    elif args.model_name == "ilra":
        from Models.ILRA import ILRA
        intervention_model = ILRA(feat_dim=args.input_dim, hidden_feat=args.hidden_dim).cuda()
        observation_model = ILRA(feat_dim=args.input_dim, hidden_feat=args.hidden_dim).cuda()
    elif args.model_name == "wikg":
        from  Models.WiKG import WiKG
        intervention_model = WiKG(dim_in=args.input_dim).cuda()
        observation_model = WiKG(dim_in=args.input_dim).cuda()
    elif args.model_name == "transmil":
        from Models.TransMIL import TransMIL
        intervention_model = TransMIL(input_dim=args.input_dim,n_classes=args.nclass).cuda()
        observation_model = TransMIL(input_dim=args.input_dim,n_classes=args.nclass).cuda()
    return intervention_model,observation_model