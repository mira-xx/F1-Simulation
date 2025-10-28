from itertools import combinations, product
from typing import Dict, List
from .models import Strategy, Stint, TyreCompound

def generate_strategies(race_laps: int, compounds: Dict[str, TyreCompound], n_stops: List[int], step: int = 3) -> List[Strategy]:
    out: List[Strategy] = []
    names = list(compounds)
    for stops in n_stops:
        n_stints = stops + 1
        for split in combinations(range(step, race_laps, step), stops):
            lens, last = [], 0
            for s in split + (race_laps,):
                lens.append(s - last); last = s
            for combo in product(names, repeat=n_stints):
                out.append(Strategy([Stint(c, l) for c, l in zip(combo, lens)]))
    # optional: dedupe
    return out