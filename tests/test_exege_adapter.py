"""The exege adapters on a tiny random ViT and the smoke data (CPU).

Mirrors what exege asks of every Patchable (exege tests/test_adapter_contracts.py,
"Patchable") and of an Intervenable, plus circuits running end to end through the registry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from exege.circuits import (
    ComponentPatch,
    EdgeGradients,
    EdgePatch,
    LinearMetric,
    Patchable,
    attribute,
    field_mean,
    open_patchable,
    prune,
    residual_graph,
)
from exege.core.errors import AdapterError, RequestError
from exege.latents import Hook, Intervenable, open_intervenable
from vit_interp.circuits import graph_of
from vit_interp.data import N_ALL, N_PROG, Stats
from vit_interp.emulator import Emulator
from vit_interp.model import ViTConfig

SMOKE = Path("/lcrc/globalscratch/ac.ngmahfouz/vit-interp/smoke")
pytestmark = pytest.mark.skipif(not (SMOKE / "1991.npy").exists(), reason="smoke data not here")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    import json

    torch.manual_seed(0)
    meta = json.loads((SMOKE / "meta.json").read_text())
    cfg = ViTConfig(in_chans=N_ALL, out_chans=N_PROG, patch=4, dim=32, depth=2, heads=2)
    em = Emulator(cfg, Stats(str(SMOKE)), meta["lat"])
    torch.nn.init.normal_(em.net.head.weight, std=0.05)  # a zero head would hide bugs
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    em.save(path, epoch=0, step=0)
    return str(path)


@pytest.fixture(scope="module")
def system(ckpt):
    s = open_patchable(ckpt, "vit-interp", data=str(SMOKE), year=1991, device="cpu", dtype="float64")
    assert isinstance(s, Patchable) and isinstance(s, EdgeGradients)
    return s


def _pair(s):
    return s.states([0, 1]), s.states([5, "1991-01-03T00:00:00"])


def _same(a, b, rtol=1e-9, atol=1e-9):
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_allclose(a[name], b[name], rtol=rtol, atol=atol, err_msg=name)


def test_the_graph_is_exeges_and_the_models(system):
    g = system.graph()
    assert g == residual_graph(2, 2)
    model = graph_of(system.em.net)
    assert list(g.senders) == model.senders and list(g.receivers) == model.receivers


def test_a_step_writes_every_field_per_batch_and_node(system):
    base, _ = _pair(system)
    out = system.step(base)
    assert tuple(out) == system.field_names()
    for x in out.values():
        assert x.shape == (2, system.grid().n_nodes) and np.issubdtype(x.dtype, np.floating)
        assert np.isfinite(x).all()


def test_edge_patching_is_exact_at_both_ends(system):
    base, source = _pair(system)
    g = system.graph()
    _same(system.step(base, EdgePatch(source, g.mask(1.0))), system.step(base))
    _same(system.step(base, EdgePatch(source, g.mask(0.0))), system.step(source))


def test_component_patching_is_exact_at_both_ends(system):
    base, source = _pair(system)
    _same(system.step(base, ComponentPatch(source, ())), system.step(base))
    _same(system.step(base, ComponentPatch(source, system.graph().senders)), system.step(source))


def test_the_input_alone_carries_everything_and_the_rest_leaves_the_skip_alone(system):
    base, source = _pair(system)
    # embed feeds every receiver and carries the persistence skip: patching it is the source
    _same(system.step(base, ComponentPatch(source, ("embed",))), system.step(source))
    # without it, the base's x and the base's direct embed->out path stay: not the source
    rest = system.step(base, ComponentPatch(source, system.graph().senders[1:]))
    assert not np.allclose(rest["T_7"], system.step(source)["T_7"])


def test_a_perturbation_adds_to_the_fields_and_leaves_the_state_alone(system):
    base, _ = _pair(system)
    before = {k: v.copy() for k, v in system.fields(base).items()}
    out_before = system.step(base)
    delta = np.linspace(0, 1, system.grid().n_nodes)
    moved = system.perturb(base, {"U_3": delta})
    np.testing.assert_allclose(system.fields(moved)["U_3"], before["U_3"] + delta)
    _same(system.fields(base), before)
    _same(system.step(system.perturb(base, {"U_3": 0 * delta})), out_before)
    system.step(base, EdgePatch(moved, system.graph().mask(0.5)))
    _same(system.fields(base), before)


def test_what_cannot_be_had_is_refused(system):
    base, source = _pair(system)
    g = system.graph()
    with pytest.raises(RequestError):
        system.step(base, EdgePatch(source, np.ones((2, 2))))
    with pytest.raises(RequestError):
        system.step(base, ComponentPatch(source, ("no-such-component",)))
    with pytest.raises(RequestError):
        system.step(base, EdgePatch(system.states([0]), g.mask()))
    with pytest.raises(RequestError):
        system.perturb(base, {"no-such-field": np.zeros(system.grid().n_nodes)})
    with pytest.raises(RequestError):
        system.perturb(base, {"U_3": np.zeros(7)})
    with pytest.raises(RequestError):
        system.states([10**6])
    with pytest.raises(RequestError):
        system.states(["0001-01-01T00:00:00"])


def test_a_missing_checkpoint_is_an_adapter_error():
    with pytest.raises(AdapterError):
        open_patchable("/no/such/ckpt.pt", "vit-interp", data=str(SMOKE), year=1991)


def test_edge_gradients_are_the_derivative_of_the_patched_metric(system):
    base, source = _pair(system)
    g = system.graph()
    n = system.grid().n_nodes
    weights = {"T_7": np.linspace(-1, 1, n) / n, "U_3": np.cos(np.arange(n)) / n}
    metric = LinearMetric(weights)
    grad = system.edge_gradients(base, source, weights)
    assert grad.shape == g.shape and not grad[~g.edges].any()
    for e in g.edge_index()[:: max(1, g.n_edges // 6)] + [(g.sender("embed"), g.receiver("out"))]:
        up, down = g.mask(), g.mask()
        up[e] += 1e-3
        down[e] -= 1e-3
        fd = (
            metric(system.step(base, EdgePatch(source, up)))
            - metric(system.step(base, EdgePatch(source, down)))
        ) / 2e-3
        # central differences: O(eps^2) curvature error, plus roundoff of a 64800-node sum
        assert grad[e] == pytest.approx(fd, rel=1e-3, abs=1e-4 * np.abs(grad).max()), g.edge_name(e)


def test_circuits_run_end_to_end(system):
    base, source = _pair(system)
    grid = system.grid()
    metric = field_mean(grid, "V_3", where=grid.within(45, 180, 3000))
    a = attribute(system, base, source, metric)
    assert a.method == "eap" and a.scores.shape == system.graph().shape
    exact = attribute(system, base, source, metric, method="patch")
    assert np.corrcoef(a.scores[system.graph().edges], exact.scores[system.graph().edges])[0, 1] > 0.5
    c = prune(system, base, source, metric, tau=0.05)
    assert 0 < c.n_edges <= system.graph().n_edges and abs(1 - c.recovery) < 0.5


def test_float32_matches_float64_closely(ckpt):
    s32 = open_patchable(ckpt, "vit-interp", data=str(SMOKE), year=1991, device="cpu")
    s64 = open_patchable(ckpt, "vit-interp", data=str(SMOKE), year=1991, device="cpu", dtype="float64")
    a, b = s32.step(s32.states([3])), s64.step(s64.states([3]))
    for n in ("PS", "T_7", "U_3"):
        np.testing.assert_allclose(a[n], b[n], rtol=1e-4, atol=1e-3 * np.abs(b[n]).std())
    g = s32.graph()
    base, source = s32.states([0]), s32.states([5])
    _same(s32.step(base, EdgePatch(source, g.mask(0.0))), s32.step(source), rtol=1e-4, atol=1e-2)


# -- Intervenable on the token grid -----------------------------------------------------------


@pytest.fixture(scope="module")
def tokens(ckpt):
    t = open_intervenable(ckpt, "vit-interp-tokens", data=str(SMOKE), year=1991, device="cpu")
    assert isinstance(t, Intervenable)
    return t


def test_a_run_writes_token_fields_and_records_float32_latents(tokens):
    info = tokens.info()
    assert info.n_nodes == tokens.grid().n_nodes == 45 * 90
    assert [entry.n_channels for entry in info.layers] == [32, 32]
    state = tokens.initial_state(0)
    r = tokens.run(state, 2, record=[(1, 0), (0, 1)])
    assert r.fields["T_7"].shape == (2, 4050) and np.isfinite(r.fields["T_7"]).all()
    assert set(r.latents) == {(1, 0), (0, 1)}
    assert r.latents[(1, 0)].dtype == np.float32 and r.latents[(1, 0)].shape == (4050, 32)


def test_a_hook_changes_what_follows_and_the_run_is_deterministic(tokens):
    state = tokens.initial_state("1991-01-02T00:00:00")
    plain = tokens.run(state, 2, noise_seed=1)
    again = tokens.run(state, 2, noise_seed=2)
    np.testing.assert_array_equal(plain.fields["U_3"], again.fields["U_3"])
    seen = {}

    def edit(h):
        seen["shape"] = h.shape
        return h + np.linspace(-1, 1, h.shape[1], dtype=np.float32)  # LayerNorm removes a constant

    hooked = tokens.run(state, 2, [Hook(0, 0, edit)], record=[(0, 0)])
    assert seen["shape"] == (4050, 32)
    assert not np.allclose(hooked.fields["U_3"][0], plain.fields["U_3"][0])
    clean = tokens.run(state, 1, record=[(0, 0)])
    np.testing.assert_allclose(hooked.latents[(0, 0)], clean.latents[(0, 0)] + np.linspace(-1, 1, 32, dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_places_that_do_not_exist_are_refused(tokens):
    state = tokens.initial_state(0)
    with pytest.raises(RequestError):
        tokens.run(state, 2, [Hook(2, 0, lambda h: h)])
    with pytest.raises(RequestError):
        tokens.run(state, 2, record=[(0, 2)])
    with pytest.raises(RequestError):
        tokens.run(state, 0)
    with pytest.raises(RequestError):
        tokens.run(tokens.initial_state(199), 5)


def test_a_unique_prefix_of_a_time_label_is_a_start(system):
    times = system._times
    full = system.states([times[3]])
    short = system.states([times[3][:13]])  # "YYYY-MM-DDTHH"
    import numpy as np

    np.testing.assert_array_equal(system.fields(full)["PS"], system.fields(short)["PS"])
    from exege.core.errors import RequestError

    with pytest.raises(RequestError, match="matches"):
        system.states([times[3][:10]])  # a whole day is four times
