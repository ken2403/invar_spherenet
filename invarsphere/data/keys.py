import inspect


class GraphKeys:
    """Class that holds the name of the data key.

    B: batch, N: node, E: edge, NB: n_neighbor_basis, E_NB: n_edge_neighbor_basis
    """

    # graph-wise information
    Lattice = "lattice"  # lattice matrix of (B, 3, 3) shape
    PBC = "pbc"  # periodic boudary condition of (B, 3) shape
    Neighbors = "neighbors"  # number of neighbor index per each image of (B) shape

    # node-wise information
    Z = "z"  # atomic number of (N) shape
    Pos = "pos"  # atomic position of (N, 3) shape
    Batch_idx = "batch"  # batch index of (N) shape, use to aggregate N->B

    # Attributes marked with "index" are automatically incremented in batch processing
    Edge_idx = "edge_index"  # edge index with [idx_s, idx_t] of (2, E) shape

    # edge-wise information
    Edge_idx_swap = "edge_swap"  # index to map s->t to t->s of (E) shape.
    Edge_shift = "edge_shift"  # edge shift of lattice of (E, 3) shape
    Edge_dist_st = "edge_dist"  # edge distances ||r_st|| of (E) shape
    Edge_vec_st = "edge_vec"  # edge vectors r_st of (E, 3) shape

    # neighbor basis information
    Rot_mat = "rotation_matrix"  # atomic rotation matrix of (NB, 3, 3) shape
    Basis_node_idx = "basis_node_index"  # node index of the neighbor basis of (NB) shape, used to aggregate NB->N
    Basis_edge_idx1 = "basis_edge_idx_1"  # edge index of the first proximity of (NB) shape, used to extend E->NB
    Basis_edge_idx2 = "basis_edge_idx_2"  # edge index of the second proximity (NB) shape, used to extend E->NB
    Basis_edge_idx3 = "basis_edge_idx_3"  # edge index between the first and second proximity of (NB) shape, used to extend E->NB # noqa: E501

    # edge neighbor basis information
    Theta = "theta"  # azimuthal angles of (E_NB) shape
    Phi = "phi"  # polar angles of (E_NB) shape
    Edge_nb_idx = "edge_nb_idx"  # edge neighbor index of (E_NB) shape, used to aggregate E_NB->E or extend E->E_NB
    Nb_edge_idx = "nb_edge_idx"  # neighbor edge index of (E_NB) shape, used to extend NB->E_NB


KEYS = [
    a[1]
    for a in inspect.getmembers(GraphKeys, lambda a: not (inspect.isroutine(a)))
    if not (a[0].startswith("__") and a[0].endswith("__"))
]
