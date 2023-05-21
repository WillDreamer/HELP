import torch.nn as nn
import lib.utils as utils
import numpy as np
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim,decoder_network = None):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space
        if decoder_network == None:
            decoder = nn.Sequential(
                nn.Linear(latent_dim, latent_dim//2),
                nn.ReLU(),
                nn.Linear(latent_dim//2,output_dim)
            )
            # utils.init_network_weights(decoder)
        else:
            decoder = decoder_network
        self.norm = torch.nn.LayerNorm(output_dim)
        self.decoder = decoder

    def forward(self, data):
        data = self.decoder(data)
        data = self.norm(data)
        data = F.softmax(data, dim=1)
        return data





