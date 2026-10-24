import numpy as np

from clarity_tse.synth import scene


def test_render_deterministic_and_snr():
    spec = scene.sample_spec("kitchen", "test", 3, snr_db=5.0, duration_s=2.0)
    spec["room"]["max_order_cap"] = 8
    a = scene.render(spec, enroll_seconds=2.0)
    b = scene.render(spec, enroll_seconds=2.0)
    np.testing.assert_array_equal(a.mix, b.mix)
    measured = 10 * np.log10(np.mean(a.target_reverb[0] ** 2) / np.mean(a.interference[0] ** 2))
    assert abs(measured - 5.0) < 0.1
    assert a.mix.shape == (2, 32000) and a.enrollment.shape == (32000,)
    np.testing.assert_allclose(a.mix, a.target_reverb + a.interference, atol=1e-5)


def test_head_turn_changes_relative_azimuth():
    spec = scene.sample_spec("restaurant", "test", 4, duration_s=4.0, head_motion="turn")
    spec["room"]["max_order_cap"] = 4
    r = scene.render(spec, enroll_seconds=1.0)
    assert abs(float(r.yaw[-1] - r.yaw[0])) >= 20
    rel = scene.relative_azimuth(r.meta["target_world_az"], r.yaw)
    assert rel.shape == r.yaw.shape


def test_specs_differ_by_split_and_seed():
    a = scene.sample_spec("party", "test", 0)
    b = scene.sample_spec("party", "test", 1)
    c = scene.sample_spec("party", "val", 0)
    assert len({scene.spec_hash(a), scene.spec_hash(b), scene.spec_hash(c)}) == 3
