class GraphKeys:
    """Class that holds the name of the data key."""

    Lattice = "lattice"  # (B, 3, 3) shape
    PBC = "pbc"  # (B, 3) shape

    Batch_idx = "batch"  # (N) shape
    Z = "z"  # (N) shape
    Pos = "pos"  # (N, 3) shape
    Rot_mat = "rotation_matrix"  # (N, nb, 3, 3) shape

    # Attributes marked with "index" are automatically incremented in batch processing
    Edge_idx = "edge_index"  # edge index with [idx_j, idx_i] (2, E) shape
    Edge_shift = "edge_shift"  # edge shift of cell (E, 3) shape
    Edge_dist = "edge_dist"  # edge distances (E) shape
    Edge_vec = "edge_vec"  # edge vectors (E, 3) shape
    Edge_dir = "edge_dir"  # edge direction (Unit vector of edge vectors) (E, 3) shape
    Transformed_vec = "transformed_vec"  # transformed vector (E, 3) shape
    Theta = "theta"  # azimuthal angles with (E) shape
    Phi = "phi"  # polar angles with (E) shape
