from einops import rearrange
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
import faiss


def reduce_feature_map(feature_map: Tensor):

    D, H, W = feature_map.shape

    # reshape to flat
    feature_flat = rearrange(feature_map, 'd h w -> (h w) d')

    # fit PCA matrix
    pca = faiss.PCAMatrix(D, 3)
    pca.train(feature_flat)

    # apply PCA matrix
    reduced_flat = pca.apply(feature_flat)

    # scale for visualization
    scaler = MinMaxScaler()
    reduced_flat_scaled = scaler.fit_transform(reduced_flat)

    # reshape to square
    reduced = rearrange(reduced_flat_scaled, '(h w) d -> d h w', h=H, w=W)
    reduced = Tensor(reduced)

    return reduced
