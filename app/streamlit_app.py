import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Dict, List

# If you're running with a src/ layout, make sure PYTHONPATH includes ./src when launching:
# $env:PYTHONPATH = "$PWD/src"; streamlit run app/streamlit_app.py

from race_sim.race import RaceSim, SCParams
from strategy_sim.models import TyreCompound, DriverParams, Stint, Strategy
from strategy_sim.simulator import simulate_single_driver
from strategy_sim.search import generate_strategies
from strategy_sim.mc import Stochastic, monte_carlo_pair

# ---------------------- Helper ----------------------
# Helper: one stochastic run that returns per-lap flags for coloring
def single_stochastic_run(race_laps, strategy, driver, compounds, stoch, seed):
    rng = np.random.default_rng(int(seed))
    det = simulate_single_driver(race_laps, strategy, driver, compounds)
    rows = []
    for lap in det.laps:
        # SC precedence with independent Bernoulli per lap
        r = rng.random()
        if r < stoch.p_sc:
            neutral = "SC"; delta = stoch.sc_lap_delta
        elif r < stoch.p_sc + stoch.p_vsc:
            neutral = "VSC"; delta = stoch.vsc_lap_delta
        else:
            neutral = "Green"; delta = 0.0
        rows.append({
            "lap": lap.i + 1,
            "lap_time": lap.lap_time + delta,
            "tyre_age": lap.tyre_age,
            "fuel_load": lap.fuel_load,
            "neutralized": neutral,
        })
    # Note: we don’t add pit-time to lap chart; that’s a between-laps loss
    return pd.DataFrame(rows)


st.set_page_config(page_title="F1 Strategy Simulator", layout="wide")
st.title("F1 Strategy Simulator")

# ---------------------- Sidebar: Inputs ----------------------
with st.sidebar:
    st.header("Race & Drivers")
    race_laps = st.number_input("Race length (laps)", 10, 100, 58, 1)

    st.subheader("Driver A")
    a_base = st.number_input("A: base_pace (s/lap)", value=90.0, step=0.1)
    a_fuel = st.number_input("A: fuel_penalty (s/lap per lap of fuel)", value=0.015, step=0.001, format="%.3f")
    driver_a = DriverParams(a_base, a_fuel)

    st.subheader("Driver B")
    b_base = st.number_input("B: base_pace (s/lap)", value=90.4, step=0.1)
    b_fuel = st.number_input("B: fuel_penalty (s/lap per lap of fuel)", value=0.015, step=0.001, format="%.3f")
    driver_b = DriverParams(b_base, b_fuel)

    st.subheader("Compounds enabled + params")
    def comp_block(name, deg, off):
        c1, c2, c3 = st.columns([1,1,0.6])
        with c1:
            use = st.checkbox(f"Use {name}", True, key=f"use_{name}")
        with c2:
            deg_rate = st.number_input(f"{name} deg_rate", value=deg, step=0.001, format="%.3f", key=f"deg_{name}")
        with c3:
            base_off = st.number_input(f"{name} offset", value=off, step=0.01, format="%.2f", key=f"off_{name}")
        return use, TyreCompound(name, deg_rate, base_off)

    defaults = [("C1", 0.015, +0.40), ("C2", 0.018, +0.20), ("C3", 0.022, 0.00), ("C4", 0.028, -0.20), ("C5", 0.035, -0.40)]
    compounds: Dict[str, TyreCompound] = {}
    for nm, d, o in defaults:
        use, c = comp_block(nm, d, o)
        if use: compounds[nm] = c

        # --- Calibration loader ---
    st.subheader("Calibration JSON")
    calib_file = st.file_uploader("Load calibration (optional)", type=["json"])
    loaded = None
    if calib_file is not None:
        import json
        loaded = json.load(calib_file)
        st.success("Calibration loaded")

    # ----- Map FastF1 compounds (HARD/MEDIUM/SOFT) -> your C1..C5 names -----
    st.caption("If your calibration uses HARD/MEDIUM/SOFT, map them to the event's C-compounds.")
    available_c = list(compounds.keys())  # e.g., ["C1","C2","C3","C4","C5"] (or a subset)
    default_h, default_m, default_s = ("C2", "C3", "C4")  # common nomination; change if needed

    map_col1, map_col2, map_col3 = st.columns(3)
    with map_col1:
        hard_map = st.selectbox("HARD →", options=available_c, index=available_c.index(default_h) if default_h in available_c else 0, key="map_HARD")
    with map_col2:
        med_map  = st.selectbox("MEDIUM →", options=available_c, index=available_c.index(default_m) if default_m in available_c else 0, key="map_MEDIUM")
    with map_col3:
        soft_map = st.selectbox("SOFT →", options=available_c, index=available_c.index(default_s) if default_s in available_c else 0, key="map_SOFT")

    alias_map = {"HARD": hard_map, "MEDIUM": med_map, "SOFT": soft_map}
    st.caption(f"Mapping used: HARD→{alias_map['HARD']}, MEDIUM→{alias_map['MEDIUM']}, SOFT→{alias_map['SOFT']}")

    # replace your current apply_driver_calib with this version
    def apply_driver_calib(label, driver_params):
        if not (loaded and "drivers" in loaded):
            return driver_params  # unchanged

        codes = list(loaded["drivers"].keys())
        code = st.selectbox(f"{label}: pick calibrated driver",
                            options=codes, key=f"sel_{label}")
        d = loaded["drivers"][code]
        st.caption(f"Using calibration of {code}")

        # Build a NEW DriverParams (don't mutate frozen instance)
        new_dp = DriverParams(
            base_pace=float(d.get("base_pace", driver_params.base_pace)),
            fuel_penalty=float(d.get("fuel_penalty", driver_params.fuel_penalty)),
        )

        # Apply compound mappings/overrides (you already had this part)
        for raw_name, deg in d.get("compound_deg", {}).items():
            name = alias_map.get(raw_name, raw_name)
            if name in compounds:
                c = compounds[name]
                compounds[name] = TyreCompound(c.name, float(deg), c.base_offset)
        for raw_name, off in d.get("compound_offset", {}).items():
            name = alias_map.get(raw_name, raw_name)
            if name in compounds:
                c = compounds[name]
                compounds[name] = TyreCompound(c.name, c.deg_rate, float(off))

        return new_dp

    if loaded:
        driver_a = apply_driver_calib("Driver A", driver_a)
        driver_b = apply_driver_calib("Driver B", driver_b)


    st.subheader("Stochastic (A)")
    a_pit_mean = st.number_input("A pit_mean (s)", value=22.0, step=0.1)
    a_pit_std  = st.number_input("A pit_std (s)",  value=0.8, step=0.1)
    a_p_sc     = st.number_input("A p_SC per lap", value=0.02, step=0.005, format="%.3f")
    a_p_vsc    = st.number_input("A p_VSC per lap", value=0.03, step=0.005, format="%.3f")
    a_sc_d     = st.number_input("A SC lap delta (s)", value=8.0, step=0.5)
    a_vsc_d    = st.number_input("A VSC lap delta (s)", value=4.0, step=0.5)
    a_sc_mult  = st.number_input("A pit mult under SC", value=0.6, step=0.05)
    a_vsc_mult = st.number_input("A pit mult under VSC", value=0.8, step=0.05)
    stoch_a = Stochastic(a_pit_mean, a_pit_std, a_p_sc, a_p_vsc, a_sc_d, a_vsc_d, a_sc_mult, a_vsc_mult)

    st.subheader("Stochastic (B)")
    b_pit_mean = st.number_input("B pit_mean (s)", value=22.5, step=0.1)
    b_pit_std  = st.number_input("B pit_std (s)",  value=0.8, step=0.1)
    b_p_sc     = st.number_input("B p_SC per lap", value=0.02, step=0.005, format="%.3f")
    b_p_vsc    = st.number_input("B p_VSC per lap", value=0.03, step=0.005, format="%.3f")
    b_sc_d     = st.number_input("B SC lap delta (s)", value=8.0, step=0.5)
    b_vsc_d    = st.number_input("B VSC lap delta (s)", value=4.0, step=0.5)
    b_sc_mult  = st.number_input("B pit mult under SC", value=0.6, step=0.05)
    b_vsc_mult = st.number_input("B pit mult under VSC", value=0.8, step=0.05)
    stoch_b = Stochastic(b_pit_mean, b_pit_std, b_p_sc, b_p_vsc, b_sc_d, b_vsc_d, b_sc_mult, b_vsc_mult)

    st.subheader("Search space")
    step = st.number_input("Pit window step (laps)", 1, 10, 8, 1)
    stops = st.multiselect("Stops to consider", [0,1,2,3], default=[1])

    st.caption("Tip: Start with few compounds & coarse step to keep things snappy.")

tab_auto, tab_manual = st.tabs(["Auto-optimize (MC)", "Manual strategy (deterministic)"])

# ---------------------- Auto-optimize tab ----------------------
with tab_auto:
    st.subheader("Two-stage search")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        runs1 = st.number_input("Stage-1 runs", 10, 500, 80, 10)
    with c2:
        runs2 = st.number_input("Stage-2 runs", 50, 2000, 300, 50)
    with c3:
        sampleA1 = st.number_input("Sample A (stage-1)", 10, 2000, 300, 10)
    with c4:
        sampleB1 = st.number_input("Sample B (stage-1)", 10, 2000, 180, 10)
    topK = st.number_input("Shortlist size (stage-2)", 10, 2000, 120, 10)
    seed = st.number_input("RNG seed", value=42, step=1)

    go = st.button("Search best strategies", type="primary")
    if go:
        if len(compounds) == 0 or len(stops) == 0:
            st.error("Enable at least one compound and select at least one stop count.")
        else:
            # Build grids
            grid_a: List[Strategy] = generate_strategies(race_laps, compounds, n_stops=stops, step=int(step))
            grid_b: List[Strategy] = generate_strategies(race_laps, compounds, n_stops=stops, step=int(step))
            st.info(f"Generated {len(grid_a)} A strategies and {len(grid_b)} B strategies.")

            rng = np.random.default_rng(int(seed))
            idx_a = rng.choice(len(grid_a), size=min(int(sampleA1), len(grid_a)), replace=False)
            idx_b = rng.choice(len(grid_b), size=min(int(sampleB1), len(grid_b)), replace=False)

            # Stage 1
            coarse = []
            prog = st.progress(0.0, text="Stage 1 running...")
            total_loops = len(idx_a) * len(idx_b)
            done = 0
            for i in idx_a:
                sa = grid_a[i]
                for j in idx_b:
                    sb = grid_b[j]
                    mc = monte_carlo_pair(
                        race_laps, sa, sb, driver_a, driver_b, compounds, stoch_a, stoch_b,
                        n_runs=int(runs1), seed=int(rng.integers(0, 1_000_000))
                    )
                    coarse.append((mc.p_win_a, mc.mean_a, mc.p5_a, mc.p95_a, mc.mean_b, mc.p5_b, mc.p95_b, i, j))
                    done += 1
                    if done % 20 == 0:
                        prog.progress(min(1.0, done / total_loops))

            coarse.sort(key=lambda r: (-r[0], r[1]))
            shortlist = coarse[:int(topK)]
            st.success(f"Stage 1 complete. Shortlisted {len(shortlist)} pairs.")

            # Stage 2
            refined = []
            prog2 = st.progress(0.0, text="Stage 2 refining...")
            for k, row in enumerate(shortlist, 1):
                _, _, _, _, _, _, _, i, j = row
                sa, sb = grid_a[i], grid_b[j]
                mc = monte_carlo_pair(
                    race_laps, sa, sb, driver_a, driver_b, compounds, stoch_a, stoch_b,
                    n_runs=int(runs2), seed=int(rng.integers(0, 1_000_000))
                )
                refined.append((mc.p_win_a, mc.mean_a, mc.p5_a, mc.p95_a, mc.mean_b, mc.p5_b, mc.p95_b, sa, sb))
                prog2.progress(min(1.0, k / len(shortlist)))

            refined.sort(key=lambda r: (-r[0], r[1]))

            # Display top table
            def s2text(s: Strategy) -> str:
                return " | ".join(f"{stint.compound}×{stint.laps}" for stint in s.stints)

            rows = []
            for row in refined[:50]:
                pwin, mean_a, p5a, p95a, mean_b, p5b, p95b, sa, sb = row
                rows.append({
                    "A strategy": s2text(sa),
                    "B strategy": s2text(sb),
                    "A win %": f"{pwin*100:.1f}%",
                    "A mean (s)": round(mean_a, 2),
                    "A p5 (s)": round(p5a, 2),
                    "A p95 (s)": round(p95a, 2),
                    "B mean (s)": round(mean_b, 2),
                    "B p5 (s)": round(p5b, 2),
                    "B p95 (s)": round(p95b, 2),
                })
            st.markdown("### Top strategies (A-favored)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # After st.dataframe(...) in Auto-optimize tab:
            import plotly.graph_objects as go

            # Build a plot for the top N A strategies showing p5–p95 bands
            N = min(10, len(refined))
            if N > 0:
                fig_rb = go.Figure()
                xs = list(range(1, N+1))
                means = [refined[i][1] for i in range(N)]    # mean_a
                p5s   = [refined[i][2] for i in range(N)]    # p5_a
                p95s  = [refined[i][3] for i in range(N)]    # p95_a

                # Shade p5–p95
                fig_rb.add_trace(go.Scatter(x=xs, y=p95s, line=dict(width=0), showlegend=False,
                                hoverinfo="skip"))
                fig_rb.add_trace(go.Scatter(x=xs, y=p5s,  fill='tonexty', name='A p5–p95',
                                hoverinfo="skip"))
                # Mean markers
                fig_rb.add_trace(go.Scatter(x=xs, y=means, mode='markers+lines', name='A mean'))
                fig_rb.update_layout(title="Top A strategies — risk bands",
                                    xaxis_title="Strategy rank (A-favored)",
                                    yaxis_title="A time (s)")
                st.plotly_chart(fig_rb, use_container_width=True)

            # Simple cost curve around first pit of the best A
            if refined:
                best = refined[0]
                best_a, best_b = best[-2], best[-1]
                st.markdown("### Cost of pitting earlier/later (first stint of best A)")
                xs, ys = [], []
                # vary first stint length ±6 laps, keeping race length constant
                for delta in range(-6, 7):
                    if delta == 0:
                        continue
                    stints = best_a.stints
                    if len(stints) < 2:
                        continue
                    l0 = stints[0].laps + delta
                    if l0 <= 0:
                        continue
                    rest = sum(s.laps for s in stints[1:])
                    if l0 + rest != race_laps:
                        continue
                    trial = Strategy([Stint(stints[0].compound, l0)] + stints[1:])
                    mc_delta = monte_carlo_pair(
                        race_laps, trial, best_b, driver_a, driver_b, compounds,
                        stoch_a, stoch_b, n_runs=max(60, runs1), seed=int(rng.integers(0, 1_000_000))
                    )
                    xs.append(delta)
                    ys.append(mc_delta.mean_a - best[1])
                if xs:
                    fig = px.line(x=xs, y=ys, labels={"x":"Δ laps (earlier ←  pit → later)", "y":"Δ A mean time (s) vs best"})
                    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Manual tab ----------------------
with tab_manual:
    st.subheader("Manual strategy sandbox (deterministic)")

    cols = st.columns(2)
    # A strategy
    with cols[0]:
        st.markdown("**Driver A strategy**")
        nA = st.number_input("# of stints (A)", 1, 4, 2, 1)
        stintsA, remA = [], race_laps
        for i in range(nA):
            comp = st.selectbox(f"A stint {i+1} compound", list(compounds.keys()), index=min(i, max(0, len(compounds)-1)), key=f"A_comp_{i}")
            laps = st.number_input(f"A stint {i+1} laps", 1, race_laps, max(1, remA//(nA-i)), 1, key=f"A_laps_{i}")
            stintsA.append(Stint(comp, int(laps))); remA -= int(laps)

    # B strategy
    with cols[1]:
        st.markdown("**Driver B strategy**")
        nB = st.number_input("# of stints (B)", 1, 4, 2, 1)
        stintsB, remB = [], race_laps
        for i in range(nB):
            comp = st.selectbox(f"B stint {i+1} compound", list(compounds.keys()), index=min(i, max(0, len(compounds)-1)), key=f"B_comp_{i}")
            laps = st.number_input(f"B stint {i+1} laps", 1, race_laps, max(1, remB//(nB-i)), 1, key=f"B_laps_{i}")
            stintsB.append(Stint(comp, int(laps))); remB -= int(laps)

    run = st.button("Run deterministic sim")
    if run:
        try:
            SA, SB = Strategy(stintsA), Strategy(stintsB)
            resA = simulate_single_driver(race_laps, SA, driver_a, compounds)
            resB = simulate_single_driver(race_laps, SB, driver_b, compounds)
            winner = "A" if resA.total_time < resB.total_time else "B"
            st.success(f"A total: {resA.total_time:.2f} s | B total: {resB.total_time:.2f} s → **{winner}**")

            # Plot lap times
            def to_df(res, label):
                return pd.DataFrame({
                    "lap": [l.i + 1 for l in res.laps],
                    "lap_time": [l.lap_time for l in res.laps],
                    "driver": label,
                    "tyre_age": [l.tyre_age for l in res.laps],
                })
            df = pd.concat([to_df(resA, "A"), to_df(resB, "B")])
            fig = px.line(df, x="lap", y="lap_time", color="driver", title="Deterministic Lap Times")
            st.plotly_chart(fig, use_container_width=True)

            # Optional: stochastic single-run timeline colored by SC/VSC
            stochastic_view = st.checkbox("Show stochastic single run with SC/VSC coloring", value=False)
            if stochastic_view:
                dfA = single_stochastic_run(race_laps, SA, driver_a, compounds, stoch_a, seed)
                dfB = single_stochastic_run(race_laps, SB, driver_b, compounds, stoch_b, seed)
                dfA["driver"] = "A"; dfB["driver"] = "B"
                dfS = pd.concat([dfA, dfB], ignore_index=True)
                fig_s = px.line(
                    dfS, x="lap", y="lap_time",
                    color="neutralized", facet_row="driver",
                    title="Stint timeline (stochastic run; SC/VSC coloring)"
                )
                st.plotly_chart(fig_s, use_container_width=True)

        except AssertionError as e:
            st.error(str(e))

# ---------------------- Phase 5 Lap‑time physics sim (single car) tab ---------------------
from physics_laptime.track_tools import load_track_csv
from physics_laptime.laptime_solver import CarParams, corner_speed_cap, forward_backward_profile

tab_phys, = st.tabs(["Lap-time physics (single car)"])

with tab_phys:
    st.subheader("Physics lap-time (deterministic)")

    # Track input
    trk_col1, trk_col2 = st.columns([2,1])
    with trk_col1:
        track_file = st.text_input("Track CSV path", value="data/simple_oval.csv", help="CSV with x,y columns in meters")
        ds_m = st.number_input("Resample step ds (m)", 0.5, 10.0, 2.0, 0.5)
    with trk_col2:
        drs_a = st.number_input("DRS start s (m)", 0.0, 5000.0, 0.0, 10.0)
        drs_b = st.number_input("DRS end s (m)",   0.0, 5000.0, 0.0, 10.0)

    # Car params
    st.markdown("**Car & tyre params**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mass = st.number_input(
            "Mass (kg)", 500.0, 900.0, 800.0, 10.0,
            help="Car+driver mass. Heavier = slower accel & braking."
        )
        mu   = st.number_input(
            "Tyre μ", 0.8, 3.5, 2.5, 0.1,
            help="Tyre-road grip coefficient. Higher = more cornering & braking grip."
        )

    with c2:
        CdA  = st.number_input(
            "CdA (m²)", 0.5, 2.5, 1.1, 0.05,
            help="Drag area. Higher = more drag, slower on straights."
        )
        ClA  = st.number_input(
            "ClA (m²)", 0.5, 8.0, 5.5, 0.1,
            help="Downforce area. Higher = more cornering grip, but also more drag."
        )

    with c3:
        Pmax = st.number_input(
            "P_max (kW)", 100.0, 900.0, 600.0, 10.0,
            help="Engine+ERS max power. Higher = faster straight-line acceleration."
        ) * 1000.0
        Fbrk = st.number_input(
            "F_brake_max (kN)", 5.0, 80.0, 45.0, 1.0,
            help="Max braking force hardware can deliver. Tyre grip may still be the limit."
        ) * 1000.0

    with c4:
        rho  = st.number_input(
            "Air density ρ (kg/m³)", 0.8, 1.4, 1.2, 0.01,
            help="Air density. Lower at high altitude/hot → less drag and less downforce."
        )

    run_phys = st.button("Solve lap")
    if run_phys:
        try:
            drs_ranges = None
            if (drs_b > drs_a) and (drs_b > 0.0):
                drs_ranges = [(float(drs_a), float(drs_b))]

            track = load_track_csv(track_file, ds=float(ds_m), drs_ranges_m=drs_ranges)
            car = CarParams(
                mass=float(mass), CdA=float(CdA),
                ClA=float(ClA), mu=float(mu),
                P_max=float(Pmax), F_brake_max=float(Fbrk),
                rho=float(rho)
            )

            vcap = corner_speed_cap(track, car)
            sol = forward_backward_profile(track, car, vcap)

            st.success(f"Predicted lap time: **{sol.lap_time:.3f} s**")

            # Plots
            dfp = pd.DataFrame({
                "s": track.s,
                "v_mps": sol.v,
                "v_kph": sol.v * 3.6,
                "a_long": sol.a_long,
                "kappa": track.kappa
            })
            fig_v = px.line(dfp, x="s", y="v_kph", title="Speed vs distance", labels={"v_kph":"Speed (km/h)", "s":"Distance (m)"})
            st.plotly_chart(fig_v, use_container_width=True)

            fig_a = px.line(dfp, x="s", y="a_long", title="Longitudinal accel vs distance", labels={"a_long":"a_long (m/s²)"})
            st.plotly_chart(fig_a, use_container_width=True)

            # Track preview
            dft = pd.DataFrame({"x": track.x, "y": track.y})
            fig_xy = px.line(dft, x="x", y="y", title="Track centerline (preview)", labels={"x":"x (m)", "y":"y (m)"})
            fig_xy.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig_xy, use_container_width=True)

        except Exception as e:
            st.error(str(e))


tab_race, = st.tabs(["Full race sim (MVP)"])
with tab_race:
    st.subheader("RaceSim MVP — timing + logs")

    laps = st.number_input("Race laps", 10, 100, 58, 1)
    pit_mean = st.number_input("Pit mean (s)", 10.0, 30.0, 22.0, 0.1)
    pit_std  = st.number_input("Pit std (s)",  0.1, 5.0, 0.8, 0.1)

    st.markdown("**SC/VSC probabilities** (per leader lap under green)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p_sc = st.number_input("p(SC start)", 0.0, 0.05, 0.004, 0.001, format="%.3f")
    with c2:
        p_vsc = st.number_input("p(VSC start)", 0.0, 0.05, 0.008, 0.001, format="%.3f")
    with c3:
        sc_min = st.number_input("SC min laps", 1, 10, 3)
        sc_max = st.number_input("SC max laps", 1, 10, 6)
    with c4:
        vsc_min = st.number_input("VSC min laps", 1, 10, 2)
        vsc_max = st.number_input("VSC max laps", 1, 10, 4)

    # Use compounds you already defined in the app, or define a small set here:
    cmp = {
        "C2": TyreCompound("C2", 0.018, +0.20),
        "C3": TyreCompound("C3", 0.022, 0.00),
        "C4": TyreCompound("C4", 0.028, -0.20),
    }

    # demo two cars; you can expand to more later or connect to your Phase-1 strategy generator
    st.markdown("**Demo grid (2 cars)**")
    colA, colB = st.columns(2)
    with colA:
        baseA = st.number_input("A base_pace (s/lap)", 60.0, 105.0, 90.0, 0.1)
        fuelA = st.number_input("A fuel_penalty", 0.0, 0.05, 0.015, 0.001, format="%.3f")
        sA1 = st.number_input("A stint1 laps", 1, 98, 30)
        sA2 = st.number_input("A stint2 laps", 1, 98, max(1, int(laps - sA1)))
        cA1 = st.selectbox("A stint1 compound", list(cmp.keys()), index=1)
        cA2 = st.selectbox("A stint2 compound", list(cmp.keys()), index=1)
    with colB:
        baseB = st.number_input("B base_pace (s/lap)", 60.0, 105.0, 90.3, 0.1)
        fuelB = st.number_input("B fuel_penalty", 0.0, 0.05, 0.015, 0.001, format="%.3f")
        sB1 = st.number_input("B stint1 laps", 1, 98, 24)
        sB2 = st.number_input("B stint2 laps", 1, 98, max(1, int(laps - sB1)))
        cB1 = st.selectbox("B stint1 compound", list(cmp.keys()), index=2)
        cB2 = st.selectbox("B stint2 compound", list(cmp.keys()), index=1)

    if st.button("Run race"):
        scp = SCParams(
            p_sc_start=float(p_sc), p_vsc_start=float(p_vsc),
            sc_min_laps=int(sc_min), sc_max_laps=int(sc_max),
            vsc_min_laps=int(vsc_min), vsc_max_laps=int(vsc_max),
        )
        sim = RaceSim(race_laps=int(laps), compounds=cmp, sc=scp)
        sim.pit_mean = float(pit_mean); sim.pit_std = float(pit_std)

        sim.add_car("CAR1", DriverParams(float(baseA), float(fuelA)),
                    Strategy([Stint(cA1, int(sA1)), Stint(cA2, int(sA2))]))
        sim.add_car("CAR2", DriverParams(float(baseB), float(fuelB)),
                    Strategy([Stint(cB1, int(sB1)), Stint(cB2, int(sB2))]))

        sim.start()

        # Timing screen
        lb = sim.leaderboard()
        rows = [{"Pos": i+1, "Code": c.code, "Total (s)": round(c.time_total, 2),
                 "Lap": c.lap_index, "Stint": c.stint_i+1, "Finished": c.finished}
                for i, c in enumerate(lb)]
        st.markdown("### Timing (finish order)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Pit log
        if sim.pit_log:
            st.markdown("### Pit log")
            st.dataframe(pd.DataFrame(sim.pit_log), use_container_width=True)

        # Event log
        if sim.event_log:
            st.markdown("### Race control events")
            st.dataframe(pd.DataFrame(sim.event_log), use_container_width=True)