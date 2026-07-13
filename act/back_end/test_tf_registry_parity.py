"""Pin the intended keyset relationship between the IntervalTF and HybridzTF
layer registries so future divergence fails loudly.

Both transfer functions keep a ``_LAYER_REGISTRY`` over the same ``LayerKind``
keyset but map to different handlers. The keysets are intentionally asymmetric by
exactly one kind: ``MEAN`` is interval-only (HybridZ raises NotImplementedError
for it). This is a deliberate design choice, not a bug, so merging the two into a
single canonical list would silently change dispatch behavior. This guard asserts
the exact intended difference; any drift (a kind added/removed from one registry
only) makes it fail instead of silently changing which layers each TF supports.
"""

from act.back_end.interval_tf.interval_tf import IntervalTF
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from act.back_end.layer_schema import LayerKind


def _registry_keys(tf_cls) -> set[str]:
    return set(tf_cls._LAYER_REGISTRY.keys())


def test_interval_hybridz_registry_parity() -> None:
    interval = _registry_keys(IntervalTF)
    hybridz = _registry_keys(HybridzTF)

    interval_only = interval - hybridz
    hybridz_only = hybridz - interval

    assert interval_only == {LayerKind.MEAN.value}, (
        "IntervalTF/HybridzTF _LAYER_REGISTRY drift: interval-only kinds changed "
        f"from {{'MEAN'}} to {sorted(interval_only)}"
    )
    assert hybridz_only == set(), (
        "HybridzTF gained registry kinds absent from IntervalTF: "
        f"{sorted(hybridz_only)}"
    )


if __name__ == "__main__":
    test_interval_hybridz_registry_parity()
    print("OK: IntervalTF/HybridzTF registry parity guard passed (intended diff = {MEAN})")
