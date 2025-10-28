from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .event_loop import EventLoop
from .car import CarState
from strategy_sim.models import TyreCompound, DriverParams, Stint, Strategy

@dataclass
class SCParams:
    sc_lap_delta: float = 8.0
    vsc_lap_delta: float = 4.0
    p_sc_start: float = 0.004     # per-lap start chance under green
    p_vsc_start: float = 0.008
    sc_min_laps: int = 3
    sc_max_laps: int = 6
    vsc_min_laps: int = 2
    vsc_max_laps: int = 4

class RaceSim:
    def __init__(self, race_laps: int, compounds: Dict[str, TyreCompound], sc: SCParams):
        self.loop = EventLoop()
        self.race_laps = race_laps
        self.compounds = compounds
        self.sc = sc

        self.cars: List[CarState] = []
        self.neutral = "GREEN"   # GREEN | VSC | SC
        self.neutral_laps_left = 0  # decremented on leader-lap completions

        # pits & randomness
        self.pit_mean = 22.0
        self.pit_std  = 0.8

        # overtake / blue flags (coarse)
        self.drs_gap_s = 1.0         # DRS eligibility gap at line
        self.overtake_tau = 0.35     # need this much pace delta (s/lap) vs car ahead
        self.p_pass_drs = 0.55       # probability if DRS + pace delta
        self.p_pass_nodrs = 0.12     # probability without DRS (small)
        self.overtake_leader_loss = 0.8  # time added to passed car

        self.blue_flag_loss = 0.30   # added to backmarker when lapped by faster car

        # logs
        self.pit_log: List[dict] = []
        self.event_log: List[dict] = []

        # RNG for session events
        self._rng = np.random.default_rng(42)

    # ------------- Public API -------------
    def add_car(self, code: str, driver: DriverParams, strategy: Strategy):
        cs = CarState(code=code, driver=driver, stints=strategy.stints, compounds=self.compounds)
        cs.fuel_left = self.race_laps
        self.cars.append(cs)

    def start(self):
        # schedule first lap completion for all cars
        for i, car in enumerate(self.cars):
            first = car.base_lap_time() + self._neutral_delta()
            self.loop.schedule(first, lambda i=i: self._lap_event(i), name=f"lap_{car.code}")
        self.loop.run()

    def leaderboard(self) -> List[CarState]:
        # finished cars first by total time; then running by current time
        return sorted(self.cars, key=lambda c: (not c.finished, c.time_total))

    # ------------- Internals -------------
    def _neutral_delta(self) -> float:
        if self.neutral == "SC":
            return self.sc.sc_lap_delta
        elif self.neutral == "VSC":
            return self.sc.vsc_lap_delta
        return 0.0

    def _maybe_start_neutral(self):
        if self.neutral != "GREEN":
            return
        r = self._rng.random()
        if r < self.sc.p_sc_start:
            self.neutral = "SC"
            self.neutral_laps_left = self._rng.integers(self.sc.sc_min_laps, self.sc.sc_max_laps + 1)
            self.event_log.append({"t": self.loop.now, "event": "SC_START", "laps": int(self.neutral_laps_left)})
        elif r < self.sc.p_sc_start + self.sc.p_vsc_start:
            self.neutral = "VSC"
            self.neutral_laps_left = self._rng.integers(self.sc.vsc_min_laps, self.sc.vsc_max_laps + 1)
            self.event_log.append({"t": self.loop.now, "event": "VSC_START", "laps": int(self.neutral_laps_left)})

    def _maybe_end_neutral_on_leader(self):
        # Called when the current leader completes a lap
        if self.neutral == "GREEN":
            return
        self.neutral_laps_left -= 1
        if self.neutral_laps_left <= 0:
            ev = f"{self.neutral}_END"
            self.neutral = "GREEN"
            self.event_log.append({"t": self.loop.now, "event": ev})

    def _estimate_base_pace(self, car: CarState) -> float:
        # Expected next-lap time absent neutralization
        st = car.stints[car.stint_i]
        comp = car.compounds[st.compound]
        return (car.driver.base_pace
                + comp.base_offset
                + comp.deg_rate * car.tyre_age
                + car.driver.fuel_penalty * car.fuel_left)

    def _apply_overtaking_and_flags(self, finisher_idx: int):
        """
        Called when car i completes a lap.
        - If within DRS gap of the car ahead at the line, and has pace advantage > tau,
          attempt pass and add time loss to the passed car (leader).
        - If the finisher just lapped a car (gap > 1 lap), add blue-flag loss to that backmarker.
        """
        # Build order at this instant
        order = sorted(range(len(self.cars)), key=lambda k: self.cars[k].time_total)
        pos = order.index(finisher_idx)
        car = self.cars[finisher_idx]

        # --- Overtake attempt on car ahead (same lap) ---
        if pos > 0:
            ahead_idx = order[pos - 1]
            ahead = self.cars[ahead_idx]
            same_lap = (car.lap_index == ahead.lap_index)
            gap = car.time_total - ahead.time_total  # positive if finisher behind
            if same_lap and 0.0 <= gap <= self.drs_gap_s:
                # pace delta estimated from base lap predictions
                pace_finisher = self._estimate_base_pace(car)
                pace_ahead = self._estimate_base_pace(ahead)
                pace_delta = (pace_ahead - pace_finisher)  # > 0 means finisher faster
                if pace_delta > self.overtake_tau:
                    p = self.p_pass_drs
                else:
                    p = self.p_pass_nodrs
                if self._rng.random() < p:
                    # pass succeeds: add loss to the car that got passed
                    ahead.time_total += self.overtake_leader_loss
                    # NOTE: we don't reschedule an already-scheduled lap; this
                    # small time shove is enough to reflect position swap downstream
                    self.event_log.append({
                        "t": self.loop.now, "event": "PASS",
                        "by": car.code, "on": ahead.code, "gap_s": float(gap),
                        "pace_delta": float(pace_delta)
                    })

        # --- Blue flags: if car just lapped someone, add a small loss to the lapped car ---
        # find any car more than 1 lap behind and within ~3s of being caught at the line
        for k in order[pos+1:]:
            back = self.cars[k]
            if car.lap_index >= back.lap_index + 1:
                # Finisher is at least a lap ahead
                approach_gap = back.time_total - car.time_total
                if 0.0 <= approach_gap <= 3.0:
                    back.time_total += self.blue_flag_loss
                    self.event_log.append({
                        "t": self.loop.now, "event": "BLUE_FLAG_DELAY",
                        "who": back.code, "by": car.code
                    })

    def _lap_event(self, idx: int):
        car = self.cars[idx]
        if car.finished:
            return

        # (1) Draw lap time (base + neutral)
        base = car.base_lap_time()
        lap_time = base + self._neutral_delta()

        # (2) Advance state
        car.time_total += lap_time
        car.lap_index += 1
        car.fuel_left -= 1
        car.tyre_age  += 1

        # (3) Pit if stint complete (except after final stint)
        do_pit = (car.tyre_age >= car.stints[car.stint_i].laps) and (car.stint_i < len(car.stints)-1)
        if do_pit:
            pit_loss = float(max(0.0, np.random.normal(self.pit_mean, self.pit_std)))
            car.time_total += pit_loss
            self.pit_log.append({
                "t": self.loop.now, "code": car.code,
                "stint_to": car.stint_i + 1, "loss": pit_loss
            })
            car.stint_i += 1
            car.tyre_age = 0

        # (4) Finish or schedule next lap
        if car.lap_index >= self.race_laps:
            car.finished = True
        else:
            self.loop.schedule(car.time_total, lambda i=idx: self._lap_event(i), name=f"lap_{car.code}")

        # (5) Race-control: only the **leader** ticking a lap changes neutral phase
        leader_idx = min(range(len(self.cars)), key=lambda k: self.cars[k].time_total)
        if idx == leader_idx:
            # decrement neutral laps if active; else maybe start SC/VSC
            if self.neutral != "GREEN":
                self._maybe_end_neutral_on_leader()
            else:
                self._maybe_start_neutral()

        # (6) Overtaking + blue flags at the line
        self._apply_overtaking_and_flags(idx)
