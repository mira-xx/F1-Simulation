from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import fastf1
import statsmodels.api as sm

# ---------- Utilities ----------

def enable_cache(cache_dir: str = "./f1cache"):
    fastf1.Cache.enable_cache(cache_dir)

def load_session(year: int, gp: str, session_code: str = "R"):
    """session_code: 'R' (race), 'Q', 'SQ', etc."""
    session = fastf1.get_session(year, gp, session_code)
    session.load()
    return session

def build_lap_frame(session, drivers: List[str]) -> pd.DataFrame:
    """Return cleaned laps with derived features per selected drivers."""
    laps_all = session.laps.pick_drivers(drivers).copy()

    # Drop in/out laps and laps with safety car/vsc flags
    def is_clean(lap):
        # FastF1 exposes LapFlags; also use conveniences:
        return (not lap["PitInTime"]) and (not lap["PitOutTime"]) \
               and (not lap["SC"]) and (not lap["VSC"])

    # FastF1 already provides helpers; the below is robust selection:
    # Keep laps with a proper LapTime and not in/out
    laps = laps_all[(~laps_all["PitInTime"].notna()) & (~laps_all["PitOutTime"].notna())]
    # Remove laps with SC or VSC states (if available)
    if "TrackStatus" in laps.columns:
        # TrackStatus '1' = green in many events; remove anything else
        laps = laps[laps["TrackStatus"].fillna("1") == "1"]

    # Basic fields
    keep_cols = ["Driver", "LapNumber", "LapTime", "Compound", "TyreLife", "Stint", "IsAccurate"]
    keep_cols = [c for c in keep_cols if c in laps.columns]
    df = laps[keep_cols].copy()
    df = df[df["LapTime"].notna()]
    # Convert LapTime (Timedelta) to seconds
    df["lap_time_s"] = df["LapTime"].dt.total_seconds()

    # Stint and tyre age
    if "Stint" not in df.columns:
        # fallback: create stint numbers when TyreLife decreases
        df["Stint"] = df.groupby("Driver")["TyreLife"].apply(
            lambda s: (s.diff().fillna(0) < 0).cumsum() + 1
        )
    # tyre_age within stint
    df["tyre_age"] = df.groupby(["Driver", "Stint"]).cumcount()

    # fuel proxy: laps remaining in race (approx)
    race_max = int(df["LapNumber"].max())
    df["fuel_load"] = race_max - df["LapNumber"] + 1

    # Keep only plausible rows
    df = df[df["IsAccurate"] == True] if "IsAccurate" in df.columns else df
    df = df.dropna(subset=["Compound", "lap_time_s", "fuel_load", "tyre_age"])
    return df[["Driver","LapNumber","Compound","Stint","tyre_age","fuel_load","lap_time_s"]].reset_index(drop=True)

# ---------- Modeling ----------

@dataclass
class DriverCalib:
    base_pace: float
    fuel_penalty: float
    compound_deg: Dict[str, float]     # deg_rate per compound
    compound_offset: Dict[str, float]  # base_offset per compound (vs reference)

@dataclass
class EventCalib:
    drivers: Dict[str, DriverCalib]

def fit_driver(df: pd.DataFrame, driver: str, ref_compound: str = "C3") -> DriverCalib:
    """
    Fit: lap_time = base + fuel_penalty*fuel_load + sum_c [ offset_c*I_c ] + sum_c [ deg_c*(tyre_age*I_c) ]
    Reference compound (default C3) has offset_c=0 and deg_c is estimated directly by its interaction term.
    """
    d = df[df["Driver"] == driver].copy()
    
    # Check if driver has any data
    if len(d) == 0:
        raise ValueError(f"No data available for driver {driver}")
    
    # Check if we have enough data points for regression (minimum 5 laps)
    if len(d) < 5:
        raise ValueError(f"Insufficient data for driver {driver} (only {len(d)} laps)")
    
    # One-hot compounds (with reference)
    d["compound"] = d["Compound"].astype("category")
    
    # Check if there are any compound categories
    if len(d["compound"].cat.categories) == 0:
        raise ValueError(f"No compound data available for driver {driver}")
    
    if ref_compound not in d["compound"].cat.categories:
        # fallback to first seen as reference
        ref_compound = d["compound"].cat.categories[0]
    d["compound"] = d["compound"].cat.reorder_categories(
        [ref_compound] + [c for c in d["compound"].cat.categories if c != ref_compound], ordered=True
    )

    # Build design matrix with interactions: tyre_age * compound
    X = pd.get_dummies(d["compound"], drop_first=True)
    # Offsets for non-reference compounds (columns already represent I_c for c != ref)
    X_offsets = X.copy()
    # Add fuel
    X["fuel_load"] = d["fuel_load"]
    # Add deg interactions: tyre_age * I_c for non-ref, plus separate column for ref compound tyre_age
    X_deg = pd.DataFrame(index=d.index)
    X_deg[f"deg_{ref_compound}"] = d["tyre_age"]
    for c in X_offsets.columns:
        X_deg[f"deg_{c}"] = d["tyre_age"] * X_offsets[c]
    # Combine
    X_full = pd.concat([X[["fuel_load"]], X_offsets, X_deg], axis=1)
    X_full = sm.add_constant(X_full)
    
    # Ensure all columns are numeric and handle data type issues
    for col in X_full.columns:
        X_full[col] = pd.to_numeric(X_full[col], errors='coerce')
    
    # Drop rows with NaN values after conversion
    X_full = X_full.dropna()
    y = d.loc[X_full.index, "lap_time_s"].values
    
    # Check if we still have enough data after cleaning
    if len(X_full) < 3:
        raise ValueError(f"Insufficient clean data for driver {driver} after processing (only {len(X_full)} valid laps)")
    
    # Convert to float64 to ensure consistent data types
    X_full = X_full.astype('float64')
    y = y.astype('float64')

    model = sm.OLS(y, X_full).fit()
    params = model.params

    base_pace = float(params["const"])
    fuel_penalty = float(params["fuel_load"])

    # compound offsets: ref=0 by definition
    compound_offset = {ref_compound: 0.0}
    for c in X_offsets.columns:
        compound_offset[c] = float(params.get(c, 0.0))

    # compound deg: from deg columns
    compound_deg = {}
    for col in [k for k in params.index if k.startswith("deg_")]:
        comp = col.replace("deg_", "")
        compound_deg[comp] = float(params[col])

    return DriverCalib(base_pace, fuel_penalty, compound_deg, compound_offset)

def calibrate_event(year: int, gp: str, session_code: str, drivers: List[str], ref_compound: str = "C3") -> EventCalib:
    enable_cache("./f1cache")
    ses = load_session(year, gp, session_code)
    df = build_lap_frame(ses, drivers)
    out: Dict[str, DriverCalib] = {}
    for drv in drivers:
        try:
            out[drv] = fit_driver(df, drv, ref_compound=ref_compound)
        except ValueError as e:
            print(f"[warning] Skipping driver {drv}: {e}")
            continue
    return EventCalib(drivers=out)

def save_calibration(calib: EventCalib, path: str):
    obj = {"drivers": {}}
    for drv, dc in calib.drivers.items():
        obj["drivers"][drv] = {
            "base_pace": dc.base_pace,
            "fuel_penalty": dc.fuel_penalty,
            "compound_deg": dc.compound_deg,
            "compound_offset": dc.compound_offset,
        }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_calibration(path: str) -> EventCalib:
    with open(path, "r") as f:
        raw = json.load(f)
    drivers = {}
    for drv, d in raw["drivers"].items():
        drivers[drv] = DriverCalib(
            base_pace=d["base_pace"],
            fuel_penalty=d["fuel_penalty"],
            compound_deg=d["compound_deg"],
            compound_offset=d["compound_offset"],
        )
    return EventCalib(drivers=drivers)
