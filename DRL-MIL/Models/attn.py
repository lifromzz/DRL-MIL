from torch import nn

class ResidualGatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_a = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        self.attn_b = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, feat_causal, feat_trivial):
        a=self.attn_a(feat_causal)
        b=self.attn_b(feat_trivial)
        fused_feat = b*feat_trivial+(1-b)*a
        return fused_feat,b