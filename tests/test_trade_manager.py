from datetime import UTC, datetime, timedelta

from tradingbot.config import TargetsConfig
from tradingbot.core.models import (
    CloseNow,
    EntrySignal,
    ManagedPosition,
    ModifyStop,
    Side,
    SignalKind,
)
from tradingbot.management.trade_manager import manage_position

NOW = datetime(2025, 6, 5, 8, 0, tzinfo=UTC)
FLAT = datetime(2025, 6, 5, 15, 20, tzinfo=UTC)


def make_pos(entry: float = 20100.0, stop: float = 20050.0, side: Side = Side.LONG) -> ManagedPosition:
    sig = EntrySignal(
        kind=SignalKind.FIRST, side=side, time=NOW, entry_ref=entry,
        stop_loss=stop, take_profit=None, tp_rr=None,
        risk_points=abs(entry - stop), reason="test",
    )
    return ManagedPosition(
        signal=sig, entry_price=entry, volume=1.0, opened_at=NOW,
        initial_stop=stop, stop_loss=stop, take_profit=None,
    )


def trailing_cfg(**kw) -> TargetsConfig:
    return TargetsConfig(tp_mode="trailing", trail_start_r=0.5, trail_step_r=0.5, **kw)


def test_no_action_below_trail_start():
    pos = make_pos()  # R = 50
    actions = manage_position(pos, 20120.0, NOW, FLAT, trailing_cfg())  # +0.4R
    assert actions == []


def test_breakeven_at_half_r():
    pos = make_pos()
    actions = manage_position(pos, 20125.0, NOW, FLAT, trailing_cfg())  # +0.5R
    assert len(actions) == 1 and isinstance(actions[0], ModifyStop)
    assert actions[0].new_stop == 20100.0  # breakeven


def test_step_locks_profit():
    pos = make_pos()
    actions = manage_position(pos, 20150.0, NOW, FLAT, trailing_cfg())  # +1.0R
    assert isinstance(actions[0], ModifyStop)
    assert actions[0].new_stop == 20125.0  # +0.5R locked
    pos.stop_loss = actions[0].new_stop
    actions = manage_position(pos, 20175.0, NOW, FLAT, trailing_cfg())  # +1.5R
    assert actions[0].new_stop == 20150.0  # +1.0R locked


def test_stop_never_loosens():
    pos = make_pos()
    pos.stop_loss = 20130.0  # already trailed above the computed level
    actions = manage_position(pos, 20150.0, NOW, FLAT, trailing_cfg())  # target would be 20125
    assert actions == []


def test_short_side_mirrors():
    pos = make_pos(entry=20000.0, stop=20050.0, side=Side.SHORT)  # R = 50
    actions = manage_position(pos, 19950.0, NOW, FLAT, trailing_cfg())  # +1.0R
    assert isinstance(actions[0], ModifyStop)
    assert actions[0].new_stop == 19975.0  # entry - 0.5R


def test_flat_time_closes_in_any_mode():
    pos = make_pos()
    late = FLAT + timedelta(minutes=1)
    for mode in ("fixed_rr", "trailing"):
        actions = manage_position(pos, 20100.0, late, FLAT, TargetsConfig(tp_mode=mode))
        assert len(actions) == 1 and isinstance(actions[0], CloseNow)


def test_fixed_rr_mode_never_trails():
    pos = make_pos()
    actions = manage_position(pos, 20200.0, NOW, FLAT, TargetsConfig(tp_mode="fixed_rr"))
    assert actions == []


def test_breakeven_buffer():
    pos = make_pos()
    cfg = trailing_cfg(breakeven_buffer_points=3.0)
    actions = manage_position(pos, 20125.0, NOW, FLAT, cfg)  # +0.5R -> BE + buffer
    assert isinstance(actions[0], ModifyStop)
    assert actions[0].new_stop == 20103.0


def test_trail_step_zero_rejected_by_config():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TargetsConfig(tp_mode="trailing", trail_step_r=0.0)
