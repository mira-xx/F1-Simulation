from typing import Dict, List
from .models import TyreCompound, DriverParams, Strategy


# Represent a single lap in the race
class Lap:
    def __init__(self, i: int, tyre_age: int, fuel_load: int, lap_time: float):
        self.i, self.tyre_age, self.fuel_load, self.lap_time = i, tyre_age, fuel_load, lap_time


# Stores the final result of the race simulation for a driver
class RaceResult:
    def __init__(self, laps: List[Lap], total_time: float, pit_laps: List[int]):
        self.laps, self.total_time, self.pit_laps = laps, total_time, pit_laps


# Ensure the chosen strategy is valid
def validate_strategy(strategy: Strategy, race_laps: int):
    assert strategy.total_laps() == race_laps, "stints must sum to race length"
    assert all(s.laps > 0 for s in strategy.stints)


# Simulate a driver's race lap by lap, factoring in tyres, fuel and strategy, then returns the toal race time and pit stop info
def simulate_single_driver(race_laps: int, strategy: Strategy, driver: DriverParams, compounds: Dict[str, TyreCompound]) -> RaceResult:
    validate_strategy(strategy, race_laps)
    fuel = race_laps
    laps: List[Lap] = []
    pit_laps: List[int] = []
    li = 0


    def base_time(comp, tyre_age, fuel_load):
        return driver.base_pace + comp.base_offset + comp.deg_rate*tyre_age + driver.fuel_penalty*fuel_load


    for si, stint in enumerate(strategy.stints):
        comp = compounds[stint.compound]
        for a in range(stint.laps):
            t = base_time(comp, a, fuel)
            laps.append(Lap(li, a, fuel, t))
            li += 1
            fuel -= 1
        if si < len(strategy.stints) - 1:
            pit_laps.append(li-1)  # pit after last lap of stint
    total = sum(l.lap_time for l in laps)
    return RaceResult(laps, total, pit_laps)