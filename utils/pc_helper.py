import torch 
from pytorch3d.ops import knn_points
from pytorch3d.ops import estimate_pointcloud_normals
from pytorch3d.ops import sample_farthest_points

@torch.no_grad()
def knn(mean: torch.Tensor, K: int, mean2: torch.Tensor = None):
    if(mean2 is None):
        K += 1
        dists, idx, nn = knn_points(mean[None, ...], mean[None, ...], K=K, return_nn=True)
        return dists[0, :, 1:], nn[0, :, 1:, :], idx[0, :, 1:]
    else:
        dists, idx, nn = knn_points(mean[None, ...], mean2[None, ...], K=K, return_nn=True)
        return dists[0, :, :], nn[0, :, :, :], idx[0, :, :]
    
@torch.no_grad()
def farthest_point_sampling(mean: torch.Tensor, K, random_start_point=False):

    if mean.ndim == 2:
        L = torch.tensor(mean.shape[0], dtype=torch.long).to(mean.device)
        pts, indices = sample_farthest_points(
            mean[None, ...],
            L[None, ...],
            K,
            random_start_point=random_start_point,
        )
        return pts[0], indices[0]
    elif mean.ndim == 3:
        # mean: [B, L, 3]
        B = mean.shape[0]
        L = torch.tensor(mean.shape[1], dtype=torch.long).to(mean.device)
        pts, indices = sample_farthest_points(
            mean,
            L[None, ...].repeat(B),
            K,
            random_start_point=random_start_point,
        )

        return pts, indices