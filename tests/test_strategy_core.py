from strategy_sim.models import TyreCompound, DriverParams, Stint, Strategy
from strategy_sim.simulator import simulate_single_driver


# Expectation: Lap time should improve as fuel burns off
def test_total_time_monotonic_with_fuel():
    # define environment for a short 10-lap test race:
    comp = {"C3": TyreCompound("C3", deg_rate=0.02, base_offset=0.0)}  # one tyre compound "C3" that degrades at 0.02s/lap
    drv = DriverParams(base_pace=90.0, fuel_penalty=0.02)  # driver with base lap time and fuel penalty
    strat = Strategy([Stint("C3", 10)])  # simple strategy -> 1 stint of 10 laps on "C3"
    res = simulate_single_driver(10, strat, drv, comp)
    assert res.total_time > 0  # sanity check (race total must be positive)
    assert res.laps[0].lap_time > res.laps[-1].lap_time  # heavier at start, checks the first lap is slower than the last lap