import torch

from llm_fas.sinkhorn import gumbel_sinkhorn, hard_topk_ports, ports_to_selection_matrix


def test_gumbel_sinkhorn_shape_and_row_sums():
    logits = torch.zeros(2, 4, 16)
    relaxed = gumbel_sinkhorn(logits, tau=1.0, iters=10, add_noise=False)
    assert relaxed.shape == (2, 4, 16)
    assert torch.allclose(relaxed.sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert torch.isfinite(relaxed).all()


def test_hard_topk_ports_are_unique():
    scores = torch.zeros(2, 4, 16)
    scores[:, :, 3] = 10.0
    scores[:, :, 7] = 9.0
    scores[:, :, 11] = 8.0
    scores[:, :, 15] = 7.0
    ports = hard_topk_ports(scores, n_active=4)
    assert ports.shape == (2, 4)
    assert ports.dtype == torch.long
    for row in ports.tolist():
        assert len(set(row)) == 4


def test_ports_to_selection_matrix_returns_one_hot_rows():
    ports = torch.tensor([[3, 7, 11, 15], [0, 1, 2, 3]])
    selection = ports_to_selection_matrix(ports, N=16)
    assert selection.shape == (2, 4, 16)
    assert torch.allclose(selection.sum(dim=-1), torch.ones(2, 4))
    assert selection[0, 0, 3] == 1.0
    assert selection[0, 1, 7] == 1.0
    assert selection[1, 3, 3] == 1.0
