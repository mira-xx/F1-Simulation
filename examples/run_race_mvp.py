from strategy_sim.models import TyreCompound, DriverParams, Stint, Strategy
from race_sim.race import RaceSim, SCParams

compounds = {
    "C2": TyreCompound("C2", 0.018, +0.20),
    "C3": TyreCompound("C3", 0.022, 0.00),
    "C4": TyreCompound("C4", 0.028, -0.20),
}

race = RaceSim(race_laps=58, compounds=compounds, sc=SCParams())
race.add_car("CAR1", DriverParams(90.0, 0.015), Strategy([Stint("C3", 30), Stint("C3", 28)]))
race.add_car("CAR2", DriverParams(90.3, 0.015), Strategy([Stint("C4", 24), Stint("C3", 34)]))
race.start()

for c in race.leaderboard():
    print(c.code, c.time_total, c.finished, c.stint_i)
