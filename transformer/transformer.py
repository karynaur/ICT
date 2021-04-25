import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class Config:
   """
      Assign dropout values as given in the paper
   """
   embed_drop = 0
   residual_drop = 0
   attention_drop = 0

   def __init__(self, vocab_size, block_size, **kwargs):
      self.vocab_size = vocab_size
      self.block_size = block_size

      for k,v in kwargs.items():
         setattr(self, k, v)


class Attention(nn.module):
   def __init__(self, config):
      super().__init__()
      assert config.n_embed % config.n_head == 0, "embeddings and heads dont match"

      self.key = nn.Linear(config.n_embed, config.n_embed)
      self.query = nn.Linear(config.n_embed, config.n_embed)
      self.value = nn.Linear(config.n_embed, config.n_embed)

      self.attention_drop = nn.Dropout(config.attention_drop)
      self.residual_drop = nn.Dropout(config.residual_drop)
      
      self.proj = nn.Linear(config.n_embed, config.n_embed)




