import pytest

from tradingbot.data.mt5_client import SymbolSpec
from tradingbot.risk.position_sizing import SizingError, calc_volume

DAX_SPEC = SymbolSpec(
    name="GER40", point=1.0, tick_size=1.0, tick_value=1.0,
    volume_min=0.01, volume_max=100.0, volume_step=0.01, digits=1, contract_size=1.0,
)


def test_basic_sizing():
    # 100k balance, 1% risk = 1000; stop 50 points; 1 lot loses 1/point -> 20 lots.
    r = calc_volume(balance=100_000, risk_pct=1.0, stop_distance=50.0, spec=DAX_SPEC)
    assert r.volume == pytest.approx(20.0)
    assert r.risk_money_actual == pytest.approx(1000.0)


def test_rounds_down_to_step():
    # 1000 / 70 = 14.2857 lots -> 14.28 (never round up).
    r = calc_volume(balance=100_000, risk_pct=1.0, stop_distance=70.0, spec=DAX_SPEC)
    assert r.volume == pytest.approx(14.28)
    assert r.risk_money_actual <= 1000.0


def test_below_minimum_raises():
    tiny = SymbolSpec(
        name="GER40", point=1.0, tick_size=1.0, tick_value=1.0,
        volume_min=1.0, volume_max=100.0, volume_step=1.0, digits=1, contract_size=1.0,
    )
    with pytest.raises(SizingError):
        calc_volume(balance=1_000, risk_pct=1.0, stop_distance=50.0, spec=tiny)


def test_below_minimum_fallback():
    tiny = SymbolSpec(
        name="GER40", point=1.0, tick_size=1.0, tick_value=1.0,
        volume_min=1.0, volume_max=100.0, volume_step=1.0, digits=1, contract_size=1.0,
    )
    r = calc_volume(balance=1_000, risk_pct=1.0, stop_distance=50.0, spec=tiny,
                    min_volume_fallback=True)
    assert r.volume == 1.0


def test_invalid_inputs():
    with pytest.raises(SizingError):
        calc_volume(balance=0, risk_pct=1.0, stop_distance=50.0, spec=DAX_SPEC)
    with pytest.raises(SizingError):
        calc_volume(balance=1000, risk_pct=1.0, stop_distance=0, spec=DAX_SPEC)
