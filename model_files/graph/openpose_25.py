import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np
from . import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]

# From OpenPose BODY_25 format
inward = [
    (1, 0),    # Neck -> Nose
    (2, 1), (3, 2), (4, 3),           # Right arm
    (5, 1), (6, 5), (7, 6),           # Left arm
    (8, 1),                           # MidHip -> Neck
    (9, 8), (10, 9),                  # Right leg
    (11, 8), (12, 11), (13, 12),      # Left leg
    (14, 0), (15, 0), (16, 14), (17, 15),  # Eyes + ears from nose
    (22, 10), (23, 22), (24, 10),     # Right foot
    (19, 13), (20, 19), (21, 13)      # Left foot
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = self_link
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)

if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.title("OpenPose BODY_25 Graph Structure")
    plt.show()