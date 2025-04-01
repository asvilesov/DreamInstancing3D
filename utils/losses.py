import torch
from utils.pc_helper import knn, farthest_point_sampling

#----------------- NN Loss -----------------#

def lower_bound_knn(mean: torch.Tensor, K: int, radius: torch.Tensor):
    _, _, idx = knn(mean, K) # computed with no_grad, so can't use dists
    radius_knn = radius[idx] + radius[:, None]
    dists_grad = torch.norm(mean[idx] - mean[:, None], dim=-1, keepdim=False)

    penalty = dists_grad - radius_knn
    pos_penalty = penalty[penalty > 0]
    neg_penalty = torch.abs(penalty[penalty < 0])
    if(len(pos_penalty) == 0):
        pos_penalty = torch.tensor(0, dtype=torch.float32, device=mean.device)
    if(len(neg_penalty) == 0):
        neg_penalty = torch.tensor(0, dtype=torch.float32, device=mean.device)
    return pos_penalty, neg_penalty

def knn_loss(mean, radius, mean2, K):
    _, _, idx = knn(mean, K, mean2) # computed with no_grad, so can't use dists
    radius_knn = radius[:, None]
    dists_grad = torch.norm(mean2[idx] - mean[:, None], dim=-1, keepdim=False)

    penalty = dists_grad - radius_knn
    pos_penalty = penalty[penalty > 0]
    if(len(pos_penalty) == 0):
        pos_penalty = torch.tensor(0, dtype=torch.float32, device=mean.device)
    return torch.sqrt(torch.sum(torch.square(pos_penalty))) # not batch dependent, but is # of gaussians dependent

def knn_rigidity_loss(mean, mean2, K):
    _, _, idx = knn(mean, K, mean) # computed with no_grad, so can't use dists
    dists_grad = mean[idx] - mean[:, None]
    dists_grad2 = mean2[idx] - mean2[:, None]

    penalty = torch.norm(dists_grad - dists_grad2, dim=-1, keepdim=False)
    
    return torch.sum(penalty)

def generate_mask_batched(batched_indices_set1, batched_indices_set2):
    # Convert the batched indices sets to PyTorch tensors
    set1_tensor = torch.tensor(batched_indices_set1)
    set2_tensor = torch.tensor(batched_indices_set2)

    # Create a boolean mask for indices not in set2 along the last dimension
    mask = torch.all(set1_tensor.unsqueeze(2) != set2_tensor.unsqueeze(1), dim=-1)
    numerical_mask = mask.type(torch.float32)

    return numerical_mask

def knn_proximity_loss(mean, mean2, K):
    _, _, idx = knn(mean, K, mean) # computed with no_grad, so can't use dists
    _, _, idx2 = knn(mean2, K, mean2) # computed with no_grad, so can't use dists
    mask = generate_mask_batched(idx, idx2)

    dists_grad2 = mean2[idx] - mean2[:, None]
    penalty = torch.norm(dists_grad2, dim=-1, keepdim=False)
    penalty = penalty * mask
    
    # print(idx[500:505])
    # print(idx2[500:505])
    # print(mask[500:505])
    # print(mean[500:505])
    # print(mean2[500:505])
    
    return torch.sum(penalty)

#----------------- Depth Loss -----------------#

def gradient_x(img):
    gx = img[...,:,:-1] - img[...,:,1:]
    return gx

def gradient_y(img):
    gy = img[...,:-1,:] - img[...,1:,:]
    return gy

def smooth_depth_loss(depth, img, masking=True):

    depth_gradients_x = gradient_x(depth)
    depth_gradients_y = gradient_y(depth)

    if(masking):
        mask = (depth > 0).float()
        maskx = torch.logical_and(mask[...,:,:-1], mask[...,:,1:]).float()
        masky = torch.logical_and(mask[...,:-1,:], mask[...,1:,:]).float()
        depth_gradients_x = depth_gradients_x * maskx
        depth_gradients_y = depth_gradients_y * masky

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = depth_gradients_x * weights_x
    smoothness_y = depth_gradients_y * weights_y

    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))