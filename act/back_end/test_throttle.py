from __future__ import annotations

import logging

from act.back_end import bounds_dispatch
from act.back_end.bab.branching import babsr


def test_warn_once_second_call_uses_debug(caplog) -> None:
    bounds_dispatch.reset_conv_materialization_count()
    with caplog.at_level(logging.DEBUG):
        bounds_dispatch._warn_once("dispatch_conv_forward", 2, "dispatch fallback")
        bounds_dispatch._warn_once("dispatch_conv_forward", 2, "dispatch fallback")

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    debugs = [record for record in caplog.records if record.levelno == logging.DEBUG]
    assert len([record for record in warnings if record.message == "dispatch fallback"]) == 1
    assert len([record for record in debugs if record.message == "dispatch fallback"]) == 1


def test_warn_once_different_layer_id_warns_again(caplog) -> None:
    bounds_dispatch.reset_conv_materialization_count()
    with caplog.at_level(logging.DEBUG):
        bounds_dispatch._warn_once("dispatch_conv_forward", 2, "layer-two")
        bounds_dispatch._warn_once("dispatch_conv_forward", 4, "layer-four")

    warnings = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == ["layer-two", "layer-four"]


def test_reset_conv_materialization_count_emits_summary_and_flushes_tracking(caplog) -> None:
    bounds_dispatch.reset_conv_materialization_count()
    with caplog.at_level(logging.WARNING):
        bounds_dispatch._record_conv_materialization(
            "dispatch_conv_forward",
            2,
            "dispatch fallback",
        )
        babsr._record_lA_materialization(5)
        bounds_dispatch.reset_conv_materialization_count()
        bounds_dispatch._warn_once("dispatch_conv_forward", 2, "dispatch fallback")

    messages = [record.message for record in caplog.records]
    assert "[bounds_dispatch] conv materializations during run:" in messages
    assert "  layer_id=2  site=dispatch_conv_forward  count=1" in messages
    assert "[bab.branching.babsr] lA materializations during run:" in messages
    assert "  layer_id=5 count=1" in messages
    assert len(
        [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and record.message == "dispatch fallback"
        ]
    ) == 2
    assert 5 not in babsr._MATERIALIZATION_COUNTS
