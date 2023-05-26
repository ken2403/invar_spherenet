import inspect


class GraphKeys:
    """Class that holds the name of the data key.

    B: batch, N: node, E: edge, NB: n_neighbor_basis
    """

    Lattice = "lattice"  # lattice matrix of (B, 3, 3) shape
    PBC = "pbc"  # periodic boudary condition of (B, 3) shape
    Neighbors = "neighbors"  # number of neighbor index per each image of (B) shape

    Batch_idx = "batch"  # batch index of (N) shape
    Z = "z"  # atomic number of (N) shape
    Pos = "pos"  # atomic position of (N, 3) shape
    Rot_mat = "rotation_matrix"  # atomic rotation matrix of (N, NB, 3, 3) shape
    Basis_edge_idx1 = "basis_edge_idx1"  # edge index of the first proximity out of the neighbor basis of (N, NB) shape
    Basis_edge_idx2 = "basis_edge_idx2"  # edge index of the second proximity out of the neighbor basis of (N, NB) shape

    # Attributes marked with "index" are automatically incremented in batch processing
    Edge_idx = "edge_index"  # edge index with [idx_s, idx_t] of (2, E) shape
    Edge_idx_swap = "edge_swap"  # indices to map s->t to t->s of (E) shape.
    Edge_shift = "edge_shift"  # edge shift of cell of (E, 3) shape
    Edge_dist_st = "edge_dist"  # edge distances ||r_st|| of (E) shape
    Edge_vec_st = "edge_vec"  # edge vectors r_st of (E, 3) shape

    Theta = "theta"  # azimuthal angles of (E, NB) shape
    Phi = "phi"  # polar angles of (E, NB) shape


KEYS = [
    a[1]
    for a in inspect.getmembers(GraphKeys, lambda a: not (inspect.isroutine(a)))
    if not (a[0].startswith("__") and a[0].endswith("__"))
]
