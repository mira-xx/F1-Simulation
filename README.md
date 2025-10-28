# F1-Simulation
This project models F1 race strategy from first principles and data:

- **Phase 1:** Deterministic single-driver simulator (tyre degradation + fuel mass).
- **Phase 2:** Randomness (SC/VSC, pit loss distributions) + **Monte Carlo** strategy search (CLI).
- **Phase 3:** **Streamlit** MVP: compare two strategies, visualize lap times.
- **Phase 4:** **FastF1** calibration to estimate deg/fuel and tyre offsets from real sessions.
- **Phase 5:** Physics lap-time solver (v(s) forward/backward pass).
- **Phase 6:** Full race sim (multi-car, event-driven).
- **Phase 7:** Analytics dashboards (delta laps, tyre deg slopes, pit timeline, simple car visual).

## Quick start

```bash
# 1) Create & activate a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Run the Streamlit app
export PYTHONPATH=$(pwd)/src  # Windows PowerShell: $env:PYTHONPATH="$PWD/src"
streamlit run app/streamlit_app.py
