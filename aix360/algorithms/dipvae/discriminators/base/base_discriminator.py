import torch.nn as nn
class BaseDiscriminator(nn.Module):
	def __init__(self, latent_input, protected_dim):
                super().__init__()
                self.latent_input = latent_input
	        self.protected_dim = protected_dim

	def forward(self, *input):
                raise NotImplementedError