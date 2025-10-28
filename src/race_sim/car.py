from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from strategy_sim.models import TyreCompound, DriverParams, Stint

@dataclass
class CarState:
    code: str
    driver: DriverParams
    stints: List[Stint]
    compounds: Dict[str, TyreCompound]
    lap_index: int = 0
    stint_i: int = 0
    tyre_age: int = 0
    fuel_left: int = 0
    time_total: float = 0.0
    in_pit: bool = False
    finished: bool = False

    # simple per-lap base model
    def base_lap_time(self) -> float:
        st = self.stints[self.stint_i]
        comp = self.compounds[st.compound]
        return (self.driver.base_pace
                + comp.base_offset
                + comp.deg_rate * self.tyre_age
                + self.driver.fuel_penalty * self.fuel_left)

    def advance_stint_if_needed(self):
        st = self.stints[self.stint_i]
        if self.tyre_age+1 >= st.laps:  # pit after completing this lap
            return True
        return False
