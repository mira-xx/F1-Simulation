from dataclasses import dataclass
from typing import List


# Represent a tyre type e.g: C3
@dataclass(frozen=True)  # Makes it immutable (values cannot be changed after creation)
class TyreCompound:
    name: str  # Identifier e.g. 'C3'
    deg_rate: float  # How much slower the lap gets per lap due to tyre degradation (seconds per lap per tyre_age)
    base_offset: float  # Perfomance difference compared to a ref tyre (positive means slower, negative means faster) s/lap vs reference


# Capture the base perfomance profile of a driver
@dataclass(frozen=True)
class DriverParams:
    base_pace: float # Best lap time possible on reference tyre with no fuel (s/lap on reference tyre, 0 fuel)
    fuel_penalty: float # Extra time added per lap for each unit of fuel carried (heavier car -> slower, s/lap per unit fuel)


# Represent one section of a driver's race between pit stops
@dataclass(frozen=True)
class Stint:
    compound: str  # Which tyre compound is used
    laps: int  # How many laps that stint lasts


# Represent an entire race strategy as a list of stints.
@dataclass(frozen=True)
class Strategy:
    stints: List[Stint]
    def total_laps(self) -> int:  # Calculates the total race distance covered by summing the laps of each stint.
        return sum(s.laps for s in self.stints)

####

# Capture all random noise in the model
from dataclasses import dataclass
@dataclass(frozen=True)
class Stochastic:
    pit_mean: float   # avrage pit stop time loss
    pit_std: float   # standard deviation of pit stop time loss
    p_sc: float   # per-Lap probability of a full safety car
    p_vsc: float   # per-Lap probability of a virtual safety car
    sc_lap_delta: float   # time added to each lap under sc
    vsc_lap_delta: float   # time added to each lap uner vsc
    sc_pit_mult: float   # pit stop time multiplier under sc
    vsc_pit_mult: float   #it stop time multiplier under vsc