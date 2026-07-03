"""Price-alert model + edge-triggered evaluation.

Qt-free: ``src.ui.alerts`` imports no PySide6 at module load (QSettings is lazy),
so these run on CI's headless runner. Only the pure logic is exercised here.
"""

from types import SimpleNamespace

from src.ui.alerts import Alert, evaluate


def q(last):
    return SimpleNamespace(last=last)


def test_alert_normalizes_fields():
    a = Alert(" aapl ", "A", "199.5")
    assert a.ticker == "AAPL" and a.direction == "above" and a.price == 199.5
    assert Alert("x", "B", 10).direction == "below"


def test_alert_roundtrip():
    a = Alert("MSFT", "below", 300.0, enabled=False)
    assert Alert.from_dict(a.to_dict()) == a


def test_edge_trigger_above():
    a = Alert("AAPL", "above", 200)
    # First observation below the level only seeds state — never fires.
    assert evaluate([a], {"AAPL": q(190)}) == []
    assert a.last_met is False
    # Crossing up fires exactly once.
    assert evaluate([a], {"AAPL": q(205)}) == [a]
    # Staying above does not re-fire.
    assert evaluate([a], {"AAPL": q(210)}) == []
    # Dropping back below re-arms; crossing up again fires.
    assert evaluate([a], {"AAPL": q(195)}) == []
    assert evaluate([a], {"AAPL": q(201)}) == [a]


def test_no_fire_when_already_met_on_first_sight():
    a = Alert("X", "above", 100)
    assert evaluate([a], {"X": q(150)}) == []  # seed only
    assert a.last_met is True
    assert evaluate([a], {"X": q(160)}) == []  # still above, no fire


def test_edge_trigger_below():
    a = Alert("X", "below", 50)
    assert evaluate([a], {"X": q(60)}) == []   # seed
    assert evaluate([a], {"X": q(45)}) == [a]  # crossed down


def test_disabled_alert_never_fires():
    a = Alert("X", "above", 10, enabled=False)
    assert evaluate([a], {"X": q(100)}) == []


def test_missing_quote_leaves_state_untouched():
    a = Alert("X", "above", 10)
    evaluate([a], {"X": q(5)})            # seed False
    assert evaluate([a], {}) == []        # no quote this poll — skip
    assert a.last_met is False
    assert evaluate([a], {"X": q(20)}) == [a]  # still fires when it returns


def test_nan_price_is_ignored():
    a = Alert("X", "above", 10)
    assert evaluate([a], {"X": q(float("nan"))}) == []
    assert a.last_met is None  # never observed a real price
