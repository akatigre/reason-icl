import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from data_template import DATA_MAP
from data_embed import load_data_and_embeddings
from datasets import load_dataset
import os
        
        
class VAEEncoder(nn.Module):
    def __init__(self, orig_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        hidden = 2 * latent_dim
        self.fc = nn.Linear(orig_dim, hidden)
        self.mu_c = nn.Linear(hidden, latent_dim)
        self.logvar_c = nn.Linear(hidden, latent_dim)
        self.mu_l = nn.Linear(hidden, latent_dim)
        self.logvar_l = nn.Linear(hidden, latent_dim)
        
    def forward(self, z_orig):
        h = F.relu(self.fc(z_orig))
        
        mu_c = self.mu_c(h)
        lv_c = self.logvar_c(h)
        
        mu_l = self.mu_l(h)
        lv_l = self.logvar_l(h)
        
        return mu_c, lv_c, mu_l, lv_l

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, orig_dim):
        super(VAEDecoder, self).__init__()
        hidden = 2*latent_dim
        self.fc = nn.Linear(latent_dim*2, hidden)
        self.out = nn.Linear(hidden, orig_dim)
        
    def forward(self, z_c, z_l):
        cat_z = torch.cat([z_c, z_l], dim=-1)
        h = F.relu(self.fc(cat_z))
        recon = self.out(h)  # => tries to reconstruct z_orig
        return recon

# ---------------------------------------------------------------------
# 4) A Full VAE Module
# ---------------------------------------------------------------------
class ContentLogicVAE(nn.Module):
    def __init__(self, orig_dim, latent_dim):
        super(ContentLogicVAE, self).__init__()
        self.encoder = VAEEncoder(orig_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, orig_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, z_orig):
        """
        z_orig: [batch_size, orig_dim]
        Returns:
         z_c, z_l, recon, (mu_c, logvar_c, mu_l, logvar_l)
        """
        mu_c, lv_c, mu_l, lv_l = self.encoder(z_orig)
        
        z_c = self.reparameterize(mu_c, lv_c)
        z_l = self.reparameterize(mu_l, lv_l)
        
        recon = self.decoder(z_c, z_l)  # reconstruct z_orig
        return z_c, z_l, recon, (mu_c, lv_c, mu_l, lv_l)

# ---------------------------------------------------------------------
# 5) Loss function: standard VAE + logic-consistency for augmented data
# ---------------------------------------------------------------------
def vae_logic_loss(
    model, 
    z_orig, 
    z_orig_aug, 
    beta=1.0,              # weight for KL
    lambda_logic=0.1       # weight for logic consistency
):
    """
    model        : ContentLogicVAE
    z_orig       : [batch_size, orig_dim]    (original hidden rep)
    z_orig_aug   : [batch_size, orig_dim]    (augmented hidden rep)
    beta         : scale factor for KL term
    lambda_logic : scale factor for logic-consistency term
    
    Returns total loss (reconstruction + KL + logic consistency).
    """
    # --- Forward pass: original
    z_c, z_l, recon, (mu_c, lv_c, mu_l, lv_l) = model(z_orig)
    
    # Reconstruction loss (MSE or L1, etc.)
    recon_loss = F.mse_loss(recon, z_orig, reduction='mean')
    
    # KL for content part
    kl_c = -0.5 * torch.sum(1 + lv_c - mu_c.pow(2) - lv_c.exp(), dim=1)
    kl_c = kl_c.mean()
    # KL for logic part
    kl_l = -0.5 * torch.sum(1 + lv_l - mu_l.pow(2) - lv_l.exp(), dim=1)
    kl_l = kl_l.mean()
    kl_total = kl_c + kl_l
    
    # --- Forward pass: augmented
    z_c_aug, z_l_aug, recon_aug, (mu_cA, lv_cA, mu_lA, lv_lA) = model(z_orig_aug)
    
    # We won't necessarily do recon loss on augmented if we want it too,
    # but let's do it for completeness:
    recon_loss_aug = F.mse_loss(recon_aug, z_orig_aug, reduction='mean')
    
    # KL for augmented
    kl_cA = -0.5 * torch.sum(1 + lv_cA - mu_cA.pow(2) - lv_cA.exp(), dim=1).mean()
    kl_lA = -0.5 * torch.sum(1 + lv_lA - mu_lA.pow(2) - lv_lA.exp(), dim=1).mean()
    kl_total_aug = kl_cA + kl_lA
    
    # --- Logic consistency: z_l vs. z_l_aug
    # If the augmented sample has the *same logic*, we want them close
    logic_consistency = F.mse_loss(z_l, z_l_aug, reduction='mean')
    
    # Combine everything
    total_recon = recon_loss + recon_loss_aug
    total_kl = kl_total + kl_total_aug
    total_loss = total_recon + beta * total_kl + lambda_logic * logic_consistency
    
    return total_loss

if __name__ == "__main__":
    data_type = "gsm8k"

    data_config = DATA_MAP[data_type]
    dataset = load_dataset(data_config["data_path"], 'main')

    emb_model = "Qwen/Qwen3-Embedding-4B"
    gen_model = "Qwen/Qwen2.5-7B-Instruct"

    keys_to_embed = data_config["input_columns"] + [data_config["output_column"]]


    raw_path = f"data/{data_type}/train/train.jsonl"
    enhanced_path = f"data/{data_type}/train/train_enhanced.jsonl"

    ds_references = load_dataset("json", data_files=enhanced_path)["train"]
    ds_references = load_data_and_embeddings(ds_references, os.path.dirname(raw_path), emb_model, keys_to_embed)

    

