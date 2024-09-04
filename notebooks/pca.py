# %%
import faiss
import torch

feature = torch.randn(640, 100, 100)

faiss.PCAMatrix(d_in=640, d_out=3)