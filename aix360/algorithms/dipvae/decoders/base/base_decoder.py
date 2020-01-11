import torch.nn as nn
class BaseDecoder(nn.Module):
	def __init__(self, latent_input, protected_input, latent_dim):
        super().__init__()
        self.latent_input = latent_input
		self.protected_input = protected_input
		self.feat_dim = feat_dim

	def forward(self, *input):
        raise NotImplementedError