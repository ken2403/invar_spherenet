from __future__ import annotations

import pytest
import torch
from torch import Tensor

from invarsphere.nn.basis import combine_sbb_shb


@pytest.fixture(scope="module")
def sbb_shb():
    def sign(x: int):
        return (x >= 0) - (x < 0)

    def _sbb_shb(edge: int, max_n: int, max_l: int, use_phi: bool) -> tuple[Tensor, Tensor]:
        device = "cpu"
        sbb = [nq + lq * 10 for lq in range(max_l) for nq in range(max_n)]
        if use_phi:
            shb = [mq * 10 + sign(mq) * lq for lq in range(max_l) for mq in range(-lq, lq + 1)]
        else:
            shb = [lq for lq in range(max_l)]

        sbb_tensor = torch.tensor([sbb for _ in range(edge)], device=device)
        shb_tensor = torch.tensor([shb for _ in range(edge)], device=device)
        return sbb_tensor, shb_tensor

    return _sbb_shb


param_combine_sbb_shb = [
    (1, 1, True),
    (1, 1, False),
    (1, 2, True),
    (1, 3, True),
    (1, 4, True),
    (1, 5, True),
    (1, 6, True),
    (1, 7, True),
    (1, 7, False),
    (2, 1, True),
    (2, 1, False),
    (2, 2, True),
    (2, 3, True),
    (2, 4, True),
    (2, 5, True),
    (2, 6, True),
    (2, 7, True),
    (2, 7, False),
    (3, 1, True),
    (3, 1, False),
    (3, 2, True),
    (3, 3, True),
    (3, 4, True),
    (3, 5, True),
    (3, 6, True),
    (3, 7, True),
    (3, 7, False),
    (4, 1, True),
    (4, 1, False),
    (4, 2, True),
    (4, 3, True),
    (4, 4, True),
    (4, 5, True),
    (4, 6, True),
    (4, 7, True),
    (4, 7, False),
    (5, 1, True),
    (5, 1, False),
    (5, 2, True),
    (5, 3, True),
    (5, 4, True),
    (5, 5, True),
    (5, 6, True),
    (5, 7, True),
    (5, 7, False),
    (6, 1, True),
    (6, 1, False),
    (6, 2, True),
    (6, 3, True),
    (6, 4, True),
    (6, 5, True),
    (6, 6, True),
    (6, 7, True),
    (6, 7, False),
    (7, 1, True),
    (7, 1, False),
    (7, 2, True),
    (7, 3, True),
    (7, 4, True),
    (7, 5, True),
    (7, 6, True),
    (7, 7, True),
    (7, 7, False),
]


@pytest.mark.parametrize("max_n, max_l, use_phi", param_combine_sbb_shb)
def test_combine_sbb_shb(
    sbb_shb,
    max_n: int,
    max_l: int,
    use_phi: bool,
):
    E = 1  # number of edge is not important
    sbb, shb = sbb_shb(E, max_n, max_l, use_phi)
    out = combine_sbb_shb(sbb, shb, max_n, max_l, use_phi)
    # shape test
    if use_phi:
        assert out.size() == (E, max_n * max_l * max_l)
    else:
        assert out.size() == (E, max_n * max_l)

    # value test
    for nq in range(max_n):
        for lq in range(max_l):
            if use_phi:
                for num_m, _ in enumerate(range(-lq, lq + 1)):
                    expected = sbb[..., nq + lq * max_n] * shb[..., lq**2 + num_m]
                    assert torch.allclose(out[..., max_n * lq**2 + (2 * lq + 1) * nq + num_m], expected)
            else:
                expected = sbb[..., max_n * lq + nq] * shb[..., lq]
                assert torch.allclose(out[..., max_n * lq + nq], expected)
