import torch.nn as nn
class BaseEncoder(nn.Module):
	def __init__(self, feat_input, latent_dim):
        super().__init__()
		self.feat_input = feat_input
		self.latent_dim = latent_dim

	def forward(self, *input):
        raise NotImplementedError