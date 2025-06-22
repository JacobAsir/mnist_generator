# File: model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Hyperparameters (must match training) ---
LATENT_DIM = 20
IMG_SIZE = 28 * 28
NUM_CLASSES = 10

# --- CVAE Model Definition ---
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        
        # Encoder (not needed for generation, but part of the saved state_dict)
        self.fc1 = nn.Linear(IMG_SIZE + NUM_CLASSES, 400)
        self.fc21 = nn.Linear(400, LATENT_DIM)  # mu
        self.fc22 = nn.Linear(400, LATENT_DIM)  # log_var

        # Decoder
        self.fc3 = nn.Linear(LATENT_DIM + NUM_CLASSES, 400)
        self.fc4 = nn.Linear(400, IMG_SIZE)

    def encode(self, x, y):
        inputs = torch.cat([x, y], 1)
        h1 = F.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        inputs = torch.cat([z, y], 1)
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, log_var = self.encode(x.view(-1, IMG_SIZE), y)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, y), mu, log_var
