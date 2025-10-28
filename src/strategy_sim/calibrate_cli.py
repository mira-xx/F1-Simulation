from __future__ import annotations
import argparse
from typing import List, Set
from datetime import datetime

from .calibration import calibrate_event, save_calibration, load_session, enable_cache

def _normalize_codes(items: str | None) -> List[str]:
    if not items:
        return []
    return [x.strip().upper() for x in items.split(",") if x.strip()]

def _session_driver_codes(year: int, gp: str, session_code: str) -> List[str]:
    enable_cache("./f1cache")
    ses = load_session(year, gp, session_code)  # already loads the session
    # Prefer official abbreviations from results (handles replacements/DNQs properly)
    if ses.results is not None and "Abbreviation" in ses.results.columns:
        codes = ses.results["Abbreviation"].dropna().astype(str).str.upper().tolist()
        # Some events include reserves who didn’t set a lap; keep them anyway.
        return sorted(list(dict.fromkeys(codes)))
    # Fallback: try driver list from FastF1
    try:
        # ses.drivers can be driver numbers; translate to codes if needed
        codes = []
        for drv in ses.drivers:
            try:
                info = ses.get_driver(drv)
                codes.append(info["Abbreviation"].upper())
            except Exception:
                pass
        return sorted(list(dict.fromkeys(codes)))
    except Exception:
        return []

def main():
    p = argparse.ArgumentParser(description="FastF1 calibration → JSON (single or all drivers)")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--gp", type=str, required=True, help="e.g., Monaco, Bahrain, Abu Dhabi")
    p.add_argument("--session", type=str, default="R", help="R,Q,SQ,FP1,FP2,FP3 (default R)")
    # selection
    p.add_argument("--drivers", type=str, help="Comma-separated codes, e.g., VER,LEC,PER")
    p.add_argument("--all", action="store_true", help="Calibrate all drivers in the session")
    p.add_argument("--only", type=str, help="Limit to these drivers (comma-separated)")
    p.add_argument("--exclude", type=str, help="Exclude these drivers (comma-separated)")
    # modeling
    p.add_argument("--ref", type=str, default="C3", help="Reference compound name to anchor offsets (default C3)")
    # output
    p.add_argument("--out", type=str, help="Output JSON filename")
    args = p.parse_args()

    # Determine target driver codes
    candidates: List[str] = []
    if args.drivers:
        candidates = _normalize_codes(args.drivers)
    elif args.all:
        candidates = _session_driver_codes(args.year, args.gp, args.session)
        if not candidates:
            raise SystemExit("[error] Could not auto-detect drivers from session results.")
    else:
        raise SystemExit("[error] Provide --drivers VER,LEC or use --all.")

    only = set(_normalize_codes(args.only))
    exclude = set(_normalize_codes(args.exclude))

    # Apply filters (only takes precedence if provided)
    codes: List[str]
    if only:
        codes = [c for c in candidates if c in only]
    else:
        codes = [c for c in candidates if c not in exclude]

    if not codes:
        raise SystemExit("[error] No drivers selected after applying filters.")

    # Build default output name if not provided
    if args.out:
        out_path = args.out
    else:
        tag = "ALL" if (not args.drivers or args.all) else "_".join(codes)
        out_path = f"{args.gp}_{args.year}_{args.session}_{tag}.json"

    print(f"[info] Calibrating {len(codes)} drivers: {', '.join(codes)}")
    calib = calibrate_event(args.year, args.gp, args.session, codes, ref_compound=args.ref)
    save_calibration(calib, out_path)
    print(f"[ok] Wrote calibration to {out_path}")

if __name__ == "__main__":
    main()

