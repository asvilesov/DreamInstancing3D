import alphashape
import numpy as np
from utils.pc_helper import farthest_point_sampling
import torch
import trimesh

def alpha_hull_ids(xyz, alpha=5):
    pts, idx = farthest_point_sampling(torch.tensor(xyz), 4000)
    pts = pts.cpu().numpy()
    idx = idx.cpu().numpy()
    shape = alphashape.alphashape(pts, alpha)
    ########## Remove inner vertices
    graphs = trimesh.graph.connected_component_labels(shape.face_adjacency)
    group, counts = np.unique(graphs, return_counts=True)
    largest_group_idx = group[np.argmax(counts)]
    largest_group = shape.faces[graphs == largest_group_idx].flatten()
    new_nodes = np.unique(largest_group)
    alpha_arr = np.array(shape.vertices[new_nodes], dtype=np.float32)
    ##########
    alpha_idx_fps = inNd(pts, alpha_arr)
    alpha_idx = idx[alpha_idx_fps]
    all_idxs = torch.zeros(xyz.shape[0], dtype=bool)
    all_idxs[alpha_idx] = True

    return all_idxs


def asvoid(arr):
    """
    Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
    View the array as dtype np.void (bytes). The items along the last axis are
    viewed as one value. This allows comparisons to be performed on the entire row.
    """
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        """ Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

def inNd(a, b, assume_unique=False):
    a = asvoid(a)
    b = asvoid(b)
    return np.in1d(a, b, assume_unique)