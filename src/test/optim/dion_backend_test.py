import copy

import pytest
import torch

from olmo_core.optim.dion_backend import dion_update, logical_matrix_views
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.optim.muon_math import MuonLayout
from olmo_core.optim.shared_muon import SharedMuon, SharedMuonConfig


@pytest.mark.parametrize(
    "shape,layout",
    [
        ((24,), MuonLayout((3, 8))),
        ((3, 12, 8), MuonLayout((3, 12, 8), split_dim=-2)),
        ((8, 36), MuonLayout((8, 3, 12), split_dim=-1, shared_experts=True)),
        ((3, 6, 8), MuonLayout((3, 6, 8))),
    ],
)
def test_logical_views_cover_storage_without_overlap(shape, layout):
    storage = torch.zeros(shape)
    views = logical_matrix_views(storage, "expert.w_up_gate", shape, layout)
    for v in views:
        v.add_(1)
    torch.testing.assert_close(storage, torch.ones_like(storage), atol=0, rtol=0)
    assert all(v.ndim == 2 for v in views)
    assert sum(v.numel() for v in views) == storage.numel()


def test_backend_config_validation():
    assert SharedMuonConfig().muon_backend == "shared"
    assert SharedMuonConfig(muon_backend="dion").merge([]).muon_backend == "dion"
    with pytest.raises(ValueError, match="Unknown Muon backend"):
        SharedMuon([torch.nn.Parameter(torch.ones(4, 4))], muon_backend="typo")
    with pytest.raises(ValueError, match="fixed RMS"):
        SharedMuon([torch.nn.Parameter(torch.ones(4, 4))], muon_backend="dion", rms=0.3)


def test_dion_matches_native_updates_and_preserves_skip_and_checkpoint():
    dion = pytest.importorskip("dion")
    torch.manual_seed(12)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = [torch.nn.Parameter(torch.randn(s, device=device)) for s in [(8, 4), (8, 4), (4, 8)]]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    actual = SharedMuon(params, muon_backend="dion")
    native = dion.Muon(
        refs, lr=0.003, mu=0.95, nesterov=True, adjust_lr="rms_norm", weight_decay=0.1
    )
    for step in range(4):
        for p, q in zip(params, refs):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        before = copy.deepcopy(actual.state_dict())
        actual.get_step_factor = lambda: torch.tensor(float(step != 2), device=device)
        actual.step()
        if step != 2:
            native.step()
        else:
            for k, v in before["state"].items():
                for field, tensor in v.items():
                    torch.testing.assert_close(
                        actual.state_dict()["state"][k][field], tensor, atol=0, rtol=0
                    )
        for p, q in zip(params, refs):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-6)
    clone = [torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed = SharedMuon(clone, muon_backend="dion")
    resumed.load_state_dict(copy.deepcopy(actual.state_dict()))
    actual.get_step_factor = lambda: torch.tensor(1.0, device=device)
    for p, q in zip(params, clone):
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
    actual.step()
    resumed.step()
    for p, q in zip(params, clone):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    with pytest.raises(ValueError, match="backend differs"):
        SharedMuon(clone).load_state_dict(actual.state_dict())


def test_dion_independent_shared_expert_projections():
    pytest.importorskip("dion")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    shape = (8, 24)
    layout = MuonLayout((8, 2, 12), split_dim=-1, shared_experts=True)
    p = torch.randn(shape, device=device)
    g = torch.randn_like(p)
    m = torch.zeros_like(p)
    pv = logical_matrix_views(p, "shared_experts.w_up_gate", shape, layout)
    gv = logical_matrix_views(g, "shared_experts.w_up_gate", shape, layout)
    mv = logical_matrix_views(m, "shared_experts.w_up_gate", shape, layout)
    refs = [torch.nn.Parameter(v.clone()) for v in pv]
    opt = SharedMuon(refs, muon_backend="dion")
    for q, grad in zip(refs, gv):
        q.grad = grad.clone()
    opt.step()
    dion_update(pv, gv, mv, lr=0.003, momentum=0.95, weight_decay=0.1)
    for view, q in zip(pv, refs):
        torch.testing.assert_close(view, q, atol=1e-6, rtol=1e-6)


def test_moe_backend_checkpoint_metadata():
    opt = object.__new__(OLMoDDPOptimizer)
    opt.algorithm = "muon_adamw"
    opt.muon_backend = "dion"
    opt.param_groups = []
    opt.states = {}
    opt._losses = []
    opt._grad_norms = []
    opt.reset_optimizer_moments_on_load = False
    opt.rolling_interval_length = 128
    checkpoint = opt.state_dict()
    assert checkpoint["_muon_backend"] == "dion"
    opt.load_state_dict(copy.deepcopy(checkpoint))
    opt.muon_backend = "shared"
    with pytest.raises(ValueError, match="backend differs"):
        opt.load_state_dict(copy.deepcopy(checkpoint))
    # Legacy checkpoints remain loadable with the default backend.
    del checkpoint["_muon_backend"]
    opt.load_state_dict(copy.deepcopy(checkpoint))
    opt.algorithm = "adamw"
    assert "_muon_backend" not in opt.state_dict()


def test_dion_skip_preserves_auxiliary_adamw():
    pytest.importorskip("dion")
    p = torch.nn.Parameter(torch.ones(4))
    opt = SharedMuon([dict(params=[p], algorithm="adamw")], muon_backend="dion")
    p.grad = torch.ones_like(p)
    opt.step()
    before = p.detach().clone()
    state = copy.deepcopy(opt.state[p])
    opt.get_step_factor = lambda: torch.tensor(0.0)
    opt.step()
    torch.testing.assert_close(p, before, atol=0, rtol=0)
    for key in state:
        torch.testing.assert_close(opt.state[p][key], state[key], atol=0, rtol=0)
