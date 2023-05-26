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

    # Attributes marked with "index" are automatically incremented in batch processing
    Edge_idx = "edge_index"  # edge index with [idx_j, idx_i] of (2, E) shape
    Edge_idx_swap = "edge_swap"  # indices to map i->j to j->i of (E) shape.
    Edge_shift = "edge_shift"  # edge shift of cell of (E, 3) shape
    Edge_dist_ji = "edge_dist"  # edge distances ||r_ji|| of (E) shape
    Edge_vec_ji = "edge_vec"  # edge vectors r_ji of (E, 3) shape

    Theta = "theta"  # azimuthal angles of (NB, E) shape
    Phi = "phi"  # polar angles of (NB, E) shape
