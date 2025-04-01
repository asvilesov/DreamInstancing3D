import numpy as np

def load4npy(nFiles):
    scenes = []
    for i in range(nFiles):
        scene = np.load(str(i) + '.npy', allow_pickle=True)
        scenes.append(scene)
    return scenes