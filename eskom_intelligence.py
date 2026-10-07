from __future__ import annotations

import argparse
import html
import json
import math
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk

# =============================================================================
# 0. CONFIG
# =============================================================================
SEED = 7
rng = np.random.default_rng(SEED)
TODAY = pd.Timestamp.today().normalize()
DATES = pd.date_range(TODAY - pd.Timedelta(days=729), TODAY, freq="D")
N = len(DATES)
HORIZON = 30  # days of forward outlook
OUT = Path(__file__).resolve().parent / "output"
OP_RESERVE_MW = 2000
OFFLINE = False  # set by --offline: inline plotly.js + deck.gl so charts work without internet

THEME = dict(bg="#0b1220", panel="#121b2f", line="#22304d", text="#e6edf7",
             muted="#8da2c0", accent="#36c5f0", ok="#2ecc71", warn="#f5a623",
             bad="#ff5c5c", crit="#d0021b")
TYPE_COLORS = {"Coal": "#9aa5b1", "Nuclear": "#b388ff", "Gas (OCGT)": "#ff8a65",
               "Pumped storage": "#4fc3f7", "Hydro": "#26a69a", "Wind": "#aed581",
               "IPP renewables": "#ffd54f"}

# name, type, province, lat, lon, capacity MW, commissioned, units, dry-cooled
STATIONS = [
    ("Kusile", "Coal", "Mpumalanga", -25.92, 28.92, 4800, 2017, 6, True),
    ("Medupi", "Coal", "Limpopo", -23.70, 27.56, 4764, 2015, 6, True),
    ("Kendal", "Coal", "Mpumalanga", -26.09, 28.97, 4116, 1988, 6, True),
    ("Majuba", "Coal", "Mpumalanga", -27.10, 29.77, 4110, 1996, 6, False),
    ("Matimba", "Coal", "Limpopo", -23.67, 27.61, 3990, 1987, 6, True),
    ("Lethabo", "Coal", "Free State", -26.74, 27.97, 3708, 1985, 6, False),
    ("Tutuka", "Coal", "Mpumalanga", -26.78, 29.35, 3654, 1985, 6, False),
    ("Duvha", "Coal", "Mpumalanga", -25.96, 29.34, 3600, 1980, 6, False),
    ("Matla", "Coal", "Mpumalanga", -26.28, 29.14, 3600, 1979, 6, False),
    ("Kriel", "Coal", "Mpumalanga", -26.25, 29.18, 3000, 1976, 6, False),
    ("Arnot", "Coal", "Mpumalanga", -25.94, 29.79, 2352, 1971, 6, False),
    ("Hendrina", "Coal", "Mpumalanga", -26.03, 29.60, 1893, 1970, 9, False),
    ("Camden", "Coal", "Mpumalanga", -26.62, 30.09, 1561, 1967, 8, False),
    ("Grootvlei", "Coal", "Mpumalanga", -26.77, 28.50, 1180, 1969, 6, False),
    ("Koeberg", "Nuclear", "Western Cape", -33.68, 18.43, 1940, 1984, 2, False),
    ("Ankerlig", "Gas (OCGT)", "Western Cape", -33.59, 18.46, 1338, 2007, 9, False),
    ("Gourikwa", "Gas (OCGT)", "Western Cape", -34.17, 22.10, 746, 2007, 5, False),
    ("Drakensberg", "Pumped storage", "KwaZulu-Natal", -28.60, 29.07, 1000, 1981, 4, False),
    ("Ingula", "Pumped storage", "KwaZulu-Natal", -28.27, 29.58, 1332, 2017, 4, False),
    ("Palmiet", "Pumped storage", "Western Cape", -34.33, 18.97, 400, 1988, 2, False),
    ("Gariep", "Hydro", "Free State", -30.62, 25.51, 360, 1971, 4, False),
    ("Vanderkloof", "Hydro", "Northern Cape", -29.99, 24.73, 240, 1977, 2, False),
    ("Sere", "Wind", "Western Cape", -31.53, 18.27, 100, 2015, 46, False),
]
STN = pd.DataFrame(STATIONS, columns=["station", "type", "province", "lat", "lon",
                                      "capacity_mw", "commissioned", "units", "dry_cooled"])
STN["age"] = TODAY.year - STN["commissioned"]

PROVINCES = {  # centroid lat, lon, mean tmax, seasonal amplitude, rain regime, mean rain mm/d
    "Gauteng": (-26.2, 28.05, 23.5, 4.5, "summer", 2.0),
    "Mpumalanga": (-25.8, 30.0, 24.0, 4.5, "summer", 2.3),
    "Limpopo": (-23.9, 29.5, 27.0, 4.0, "summer", 1.6),
    "Free State": (-28.8, 26.5, 23.0, 6.0, "summer", 1.5),
    "KwaZulu-Natal": (-29.0, 30.8, 24.5, 3.5, "summer", 2.6),
    "Western Cape": (-33.5, 19.5, 22.0, 5.0, "winter", 1.4),
    "Eastern Cape": (-32.3, 26.5, 22.5, 3.5, "summer", 1.6),
    "Northern Cape": (-29.5, 21.5, 26.0, 7.0, "summer", 0.5),
    "North West": (-26.0, 25.8, 25.0, 5.5, "summer", 1.4),
}

DEMAND_CENTRES = [  # name, province, lat, lon, typical peak share MW
    ("Johannesburg", "Gauteng", -26.20, 28.05, 8200), ("Pretoria", "Gauteng", -25.75, 28.19, 4000),
    ("Durban", "KwaZulu-Natal", -29.86, 31.02, 4100), ("Cape Town", "Western Cape", -33.92, 18.42, 4000),
    ("Richards Bay", "KwaZulu-Natal", -28.78, 32.04, 2100), ("Rustenburg", "North West", -25.67, 27.24, 1800),
    ("Gqeberha", "Eastern Cape", -33.96, 25.60, 1500), ("Bloemfontein", "Free State", -29.12, 26.21, 850),
    ("Polokwane", "Limpopo", -23.90, 29.45, 850), ("East London", "Eastern Cape", -33.02, 27.91, 750),
    ("Mbombela", "Mpumalanga", -25.47, 30.97, 650), ("Kimberley", "Northern Cape", -28.74, 24.77, 480),
]
DC = pd.DataFrame(DEMAND_CENTRES, columns=["centre", "province", "lat", "lon", "peak_mw"])

CORRIDORS = [  # name, from, to, capacity MW, base loading %, province (weather exposure), redundancy
    ("Highveld North – Gauteng", "Kendal", "Johannesburg", 9000, 76, "Mpumalanga", "N-1"),
    ("Highveld South – Gauteng", "Lethabo", "Johannesburg", 8000, 70, "Gauteng", "N-1"),
    ("Waterberg – Tshwane", "Medupi", "Pretoria", 6000, 72, "Limpopo", "N-1"),
    ("Highveld – KZN", "Majuba", "Durban", 4500, 78, "KwaZulu-Natal", "N-1"),
    ("Ingula – Richards Bay", "Ingula", "Richards Bay", 3500, 74, "KwaZulu-Natal", "N-1"),
    ("Cape Corridor", "Bloemfontein", "Cape Town", 3500, 84, "Western Cape", "N-0"),
    ("Koeberg – Cape Town", "Koeberg", "Cape Town", 2200, 66, "Western Cape", "N-1"),
    ("Gariep – Gqeberha", "Gariep", "Gqeberha", 2000, 79, "Eastern Cape", "N-0"),
    ("Southern Cape", "Gourikwa", "Gqeberha", 1500, 60, "Western Cape", "N-0"),
    ("Arnot – Lowveld", "Arnot", "Mbombela", 1500, 63, "Mpumalanga", "N-1"),
    ("Matimba – Polokwane", "Matimba", "Polokwane", 2000, 58, "Limpopo", "N-1"),
    ("Highveld – Rustenburg", "Matla", "Rustenburg", 3000, 71, "North West", "N-1"),
    ("Northern Cape link", "Vanderkloof", "Kimberley", 1200, 62, "Northern Cape", "N-0"),
]
COR = pd.DataFrame(CORRIDORS, columns=["corridor", "from", "to", "capacity_mw", "base_load",
                                       "province", "redundancy"])

# Story injected into the dummy data so the analytics has something to find
DETERIORATING = {"Kendal": 0.15, "Tutuka": 0.13, "Duvha": 0.11, "Camden": 0.05}
RECOVERING = {"Kusile": -0.07, "Medupi": -0.05}
COAL_SQUEEZE = {"Tutuka": 7.0, "Camden": 9.0}
HEAT_PROVINCES = ["Mpumalanga", "Gauteng", "Limpopo"]

SEASON = np.cos(2 * np.pi * (DATES.dayofyear.values - 15) / 365.25)  # +1 mid-Jan, -1 mid-Jul
DOW = DATES.dayofweek.values


def ar1(n, phi, sigma):
    e = rng.normal(0, sigma, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def ramp(n, days, amount):
    """Linear ramp over the last `days` of the series."""
    r = np.zeros(n)
    r[-days:] = np.linspace(0, amount, days)
    return r


# =============================================================================
# 1. DATA LAYER — acquisition (simulated)
# =============================================================================
def sim_weather():
    raw = []
    for prov, (lat, lon, tbase, amp, regime, rain_mu) in PROVINCES.items():
        tmax = tbase + amp * SEASON + ar1(N, 0.7, 1.6)
        if prov in HEAT_PROVINCES:  # recent spring heatwave
            tmax += np.r_[np.zeros(N - 12), np.linspace(2, 8.5, 12)]
        rain_season = (0.5 + 0.5 * SEASON) if regime == "summer" else (0.5 - 0.5 * SEASON)
        wet = rng.random(N) < 0.15 + 0.35 * rain_season
        rain = np.where(wet, rng.gamma(1.2, rain_mu * 2.2 * (0.4 + rain_season), N), 0.0)
        if prov in ("Mpumalanga", "Northern Cape"):  # dry spell, last 150 days
            rain[-150:] *= 0.35
        wind = np.clip(6 + 1.5 * (-SEASON if regime == "winter" else SEASON) + ar1(N, 0.6, 1.4), 0.5, None)
        raw.append(pd.DataFrame({"date": DATES, "province": prov, "tmax_c": tmax,
                                 "rain_mm": rain, "wind_ms": wind}))
    w = pd.concat(raw, ignore_index=True)
    # inject sensor faults so the cleaning pipeline has work to do
    nan_idx = rng.choice(len(w), int(0.02 * len(w)), replace=False)
    w.loc[nan_idx, "tmax_c"] = np.nan
    spike_idx = rng.choice(len(w), int(0.004 * len(w)), replace=False)
    w.loc[spike_idx, "tmax_c"] += rng.choice([-30, 28], len(spike_idx))
    return w


def clean_weather(w):
    """Validation → outlier removal → gap fill. Returns cleaned frame + QC report."""
    w = w.sort_values(["province", "date"]).copy()
    missing_before = int(w["tmax_c"].isna().sum())
    med = w.groupby("province")["tmax_c"].transform(lambda s: s.rolling(7, center=True, min_periods=3).median())
    outlier = (w["tmax_c"] - med).abs() > 12
    n_out = int(outlier.sum())
    w.loc[outlier, "tmax_c"] = np.nan
    w["tmax_c"] = w.groupby("province")["tmax_c"].transform(lambda s: s.interpolate(limit_direction="both"))
    w["heatwave"] = w["tmax_c"] >= 35
    # drought index: standardised 90-day rainfall anomaly vs same calendar window (z-score)
    w["rain_90"] = w.groupby("province")["rain_mm"].transform(lambda s: s.rolling(90, min_periods=30).sum())
    w["drought_idx"] = w.groupby("province")["rain_90"].transform(lambda s: (s - s.mean()) / s.std())
    qc = dict(dataset="Weather stations (provincial)", rows=len(w), missing_filled=missing_before + n_out,
              outliers_removed=n_out, passed_pct=round(100 * (1 - (missing_before + n_out) / len(w)), 1))
    return w.reset_index(drop=True), qc


def sim_stations(weather):
    """Daily per-station operations + outage event log."""
    causes_coal = ["Boiler tube leak", "Coal quality / mills", "Turbine vibration", "Auxiliary plant failure",
                   "Ash handling", "Water supply constraint", "Generator fault", "Draught group fault",
                   "Labour disruption", "Grid / protection trip"]
    causes_other = ["Generator fault", "Control system fault", "Grid / protection trip", "Auxiliary plant failure"]
    tmax_by_prov = weather.pivot(index="date", columns="province", values="tmax_c")
    rows, events = [], []
    for s in STN.itertuples():
        units, cap = s.units, s.capacity_mw
        unit_mw = cap / units
        is_coal = s.type == "Coal"
        # ---- unplanned capability loss (UCLF) ----
        if is_coal:
            base = 0.05 + 0.0042 * s.age + rng.uniform(-0.03, 0.03)
            if s.station == "Kusile":
                base += 0.10  # new-build defects
        elif s.type == "Nuclear":
            base = 0.05
        else:
            base = 0.03
        uclf = base + ar1(N, 0.9, 0.012)
        uclf += ramp(N, 110, DETERIORATING.get(s.station, 0.0))
        uclf += ramp(N, 300, RECOVERING.get(s.station, 0.0))
        p_trip = (0.0025 + 0.00012 * s.age) if is_coal else 0.0015
        p_vec = np.full(N, p_trip)
        if s.station in DETERIORATING:
            p_vec[-110:] *= 2.6
        heat = tmax_by_prov[s.province].values
        p_vec *= 1 + 0.06 * np.clip(heat - 32, 0, None)  # hot days stress plant
        starts = rng.random((N, units)) < p_vec[:, None]
        trip_loss = np.zeros(N)
        for d, u in zip(*np.nonzero(starts)):
            dur = int(rng.geometric(1 / 6)) + (int(rng.integers(10, 45)) if rng.random() < 0.08 else 0)
            frac = rng.uniform(0.5, 1.0)
            end = min(N, d + dur)
            trip_loss[d:end] += frac * unit_mw / cap
            pool = causes_coal if is_coal else causes_other
            if is_coal and s.station in COAL_SQUEEZE and d > N - 30:
                cause = "Coal quality / mills"
            elif heat[d] > 34 and rng.random() < 0.4:
                cause = "Water supply constraint" if is_coal and not s.dry_cooled else "Auxiliary plant failure"
            else:
                cause = pool[rng.integers(len(pool))]
            events.append(dict(station=s.station, province=s.province, start=DATES[d],
                               end=DATES[end - 1], days=end - d, unit=f"U{u + 1}",
                               mw_lost=round(frac * unit_mw), cause=cause))
        uclf = np.clip(uclf + trip_loss, 0, 0.85)
        # ---- planned loss (PCLF): more maintenance in summer for coal ----
        if is_coal:
            pclf = np.clip(0.085 + 0.05 * (SEASON + 1) / 2 + ar1(N, 0.95, 0.008), 0.02, 0.3)
        elif s.type == "Nuclear":
            pclf = np.full(N, 0.02)
            pclf[N - 330:N - 250] = 0.5  # unit refuelling outage
        else:
            pclf = np.clip(0.04 + ar1(N, 0.95, 0.006), 0, 0.2)
        oclf = np.clip(0.015 + ar1(N, 0.8, 0.004), 0, 0.06)
        eaf = np.clip(1 - uclf - pclf - oclf, 0.05, 0.98)
        avail = eaf * cap
        # ---- load factor / energy ----
        if s.type == "Wind":
            cf = np.clip(0.33 + 0.08 * -SEASON + ar1(N, 0.5, 0.09), 0.03, 0.75)
            gen = cap * cf * eaf * 24 / 1000
        elif s.type in ("Gas (OCGT)", "Pumped storage", "Hydro"):
            gen = np.zeros(N)  # dispatch filled in after the national balance
        else:
            gen = avail * rng.uniform(0.86, 0.93, N) * 24 / 1000
        # ---- thermal efficiency, fuel, water, emissions, cost ----
        if is_coal:
            eff = 36.5 - 0.11 * s.age - 18 * (uclf - uclf[:365].mean()).clip(0) + ar1(N, 0.8, 0.25)
            target = rng.uniform(22, 38)
            stock = np.clip(target + ar1(N, 0.97, 1.1), 4, 60)
            if s.station in COAL_SQUEEZE:
                stock[-30:] = np.linspace(stock[-31], COAL_SQUEEZE[s.station], 30) + rng.normal(0, 0.4, 30)
            water = (0.12 if s.dry_cooled else 1.85) + ar1(N, 0.7, 0.03)
            emis = 1.0 + 0.002 * s.age + rng.normal(0, 0.01, N)
            cost = 720 + 14 * s.age + 1100 * (stock < 10) + rng.normal(0, 40, N)
        else:
            eff = np.full(N, np.nan)
            stock = np.full(N, np.nan)
            water = np.full(N, 0.02 if s.type != "Nuclear" else 0.05)
            emis = np.zeros(N)
            cost = {"Nuclear": 420, "Gas (OCGT)": 7600, "Pumped storage": 650,
                    "Hydro": 160, "Wind": 210}[s.type] + rng.normal(0, 25, N)
        rows.append(pd.DataFrame({
            "date": DATES, "station": s.station, "type": s.type, "province": s.province,
            "capacity_mw": cap, "available_mw": avail, "eaf": eaf, "uclf": uclf, "pclf": pclf, "oclf": oclf,
            "energy_gwh": gen, "efficiency_pct": eff, "coal_stock_days": stock,
            "water_l_per_kwh": water, "co2_t_per_mwh": emis, "cost_r_per_mwh": cost,
            "tmax_c": heat}))
    ops = pd.concat(rows, ignore_index=True)
    ev = pd.DataFrame(events).sort_values("start", ascending=False).reset_index(drop=True)
    return ops, ev


def sim_economics():
    months = pd.date_range(DATES[0].to_period("M").to_timestamp(), TODAY, freq="MS")
    n = len(months)
    return pd.DataFrame({
        "month": months,
        "gdp_index": 100 * np.cumprod(1 + rng.normal(0.0012, 0.004, n)),
        "industrial_production_idx": 100 * np.cumprod(1 + rng.normal(0.0015, 0.008, n)) + np.linspace(0, 3, n),
        "zar_usd": 18.2 + np.cumsum(rng.normal(0.03, 0.25, n)),
        "coal_price_usd_t": 108 + np.cumsum(rng.normal(0.4, 3.5, n)),
        "cpi_yoy_pct": np.clip(4.6 + np.cumsum(rng.normal(0, 0.15, n)), 2.5, 7.5),
        "prime_rate_pct": np.clip(11.0 + np.cumsum(rng.normal(-0.02, 0.1, n)), 9.5, 12.5),
    })


def sim_demand_and_balance(ops, weather, econ):
    gp_t = weather[weather.province == "Gauteng"].set_index("date")["tmax_c"].reindex(DATES).values
    winter = np.clip(-SEASON, 0, None)
    growth = np.linspace(0, 0.022, N)
    ind = econ.set_index("month")["industrial_production_idx"].reindex(DATES, method="ffill").values
    peak = (30200 + 4000 * winter + 140 * np.clip(19 - gp_t, 0, None) + 110 * np.clip(gp_t - 30, 0, None)
            - 2500 * (DOW >= 5) - 900 * (DOW == 4)) * (1 + growth) + 25 * (ind - 100) + ar1(N, 0.6, 420)
    energy = peak * 0.80 * 24 / 1000  # GWh/day
    # IPP renewables (wind / PV / CSP) — capacity 6.4 GW
    wc_w = weather[weather.province == "Western Cape"].set_index("date")["wind_ms"].reindex(DATES).values
    wind_cf = np.clip(0.06 * wc_w - 0.03 + rng.normal(0, 0.04, N), 0.04, 0.7)
    pv_cf = np.clip(0.22 + 0.04 * SEASON + rng.normal(0, 0.03, N), 0.08, 0.32)
    ipp_peak = 3400 * wind_cf + 2500 * 0.04 + 500 * 0.55
    ipp_energy = (3400 * wind_cf + 2500 * pv_cf + 500 * 0.35) * 24 / 1000

    pv = ops.pivot_table(index="date", columns="type", values="available_mw", aggfunc="sum")
    eskom_avail = pv.sum(axis=1).values
    uclf_mw = (ops.assign(x=ops.uclf * ops.capacity_mw).groupby("date")["x"].sum()).values
    pclf_mw = (ops.assign(x=ops.pclf * ops.capacity_mw).groupby("date")["x"].sum()).values
    supply = eskom_avail + ipp_peak
    deficit = peak + OP_RESERVE_MW - supply
    stage = np.clip(np.ceil(deficit / 1000), 0, 8).astype(int)
    # OCGT dispatch rises when margin is tight
    gas_avail = pv["Gas (OCGT)"].values
    margin = (supply - peak) / peak
    gas_hours = np.clip(10 - 60 * margin, 0.3, 14)
    gas_gwh = gas_avail * gas_hours / 1000
    nat = pd.DataFrame({
        "date": DATES, "peak_demand_mw": peak, "energy_demand_gwh": energy,
        "eskom_available_mw": eskom_avail, "ipp_renewable_peak_mw": ipp_peak,
        "ipp_renewable_gwh": ipp_energy, "wind_cf": wind_cf, "uclf_mw": uclf_mw, "pclf_mw": pclf_mw,
        "reserve_margin": margin, "loadshed_stage": stage, "ocgt_gwh": gas_gwh,
        "gauteng_tmax": gp_t})
    # write dispatched energy back to the gas / storage stations
    for st in STN[STN.type.isin(["Gas (OCGT)", "Pumped storage", "Hydro"])].itertuples():
        m = ops.station == st.station
        a = ops.loc[m, "available_mw"].values
        hrs = gas_hours if st.type == "Gas (OCGT)" else (np.full(N, 5.5) if st.type == "Pumped storage" else 9 + 4 * SEASON)
        ops.loc[m, "energy_gwh"] = a * hrs / 1000
    return nat, ops


def sim_grid(weather, nat):
    rows = []
    dem_norm = (nat.peak_demand_mw / nat.peak_demand_mw.mean()).values
    for c in COR.itertuples():
        w = weather[weather.province == c.province].set_index("date").reindex(DATES)
        load = c.base_load * dem_norm + ar1(N, 0.8, 2.2)
        if c.corridor in ("Cape Corridor", "Gariep – Gqeberha"):
            load += ramp(N, 60, 9)
        load = np.clip(load, 20, 104)
        storm = (w.rain_mm.values > 18).astype(float)
        lam = 0.03 * (1 + 3 * np.clip(load - 85, 0, None) / 15 + 4 * storm + 0.6 * w.heatwave.values)
        faults = rng.poisson(lam)
        rows.append(pd.DataFrame({"date": DATES, "corridor": c.corridor, "loading_pct": load,
                                  "faults": faults, "flow_mw": load / 100 * c.capacity_mw}))
    return pd.concat(rows, ignore_index=True)


def sim_substations():
    out = []
    for d in DC.itertuples():
        for i in range(2):
            out.append(dict(substation=f"{d.centre} SS-{i + 1} (sim)", province=d.province,
                            lat=d.lat + rng.normal(0, 0.12), lon=d.lon + rng.normal(0, 0.12),
                            health_idx=int(rng.integers(52, 97)), transformer_age=int(rng.integers(6, 48)),
                            mva=int(rng.choice([500, 800, 1000, 1500, 2000]))))
    return pd.DataFrame(out)


SIGNAL_SEED = [  # days ago, category, domain, headline, impact 1-5, likelihood 0-1, source type
    (3, "Threat", "Fuel supply", "Rail-line disruption on Mpumalanga coal link reported by two haulage firms", 4, 0.7, "News / trade press"),
    (5, "Threat", "Fuel supply", "Truck-haul contractor dispute could affect coal deliveries to Tutuka", 4, 0.55, "Supplier disclosure"),
    (6, "Threat", "Climate", "Weather service extends heat advisory for Highveld through next week", 3, 0.8, "Weather service"),
    (8, "Uncertainty", "Regulatory", "Regulator opens consultation on multi-year tariff methodology", 3, 0.6, "Regulator notice"),
    (9, "Trend", "Demand", "Data-centre load applications in Gauteng up sharply year-on-year", 3, 0.75, "Grid-connection register"),
    (11, "Opportunity", "Renewables", "New bid window adds 2.1 GW of wind & PV preferred bidders", 4, 0.65, "Government announcement"),
    (12, "Threat", "Supply chain", "Boiler-tube spares lead times lengthen to 30+ weeks (OEM notice)", 3, 0.6, "Supplier disclosure"),
    (14, "Trend", "Storage", "Battery storage tender reaches financial close on three sites", 3, 0.7, "Tender portal"),
    (15, "Uncertainty", "Market reform", "Draft wholesale market code published for comment", 4, 0.5, "Policy publication"),
    (17, "Threat", "Labour", "Wage negotiations at two coal suppliers reach deadlock", 3, 0.45, "News / trade press"),
    (19, "Trend", "Demand", "Rooftop PV registrations continue to rise in metros, lowering midday demand", 2, 0.85, "Municipal data"),
    (21, "Threat", "Supply chain", "Second OEM flags shipping delays for turbine components", 3, 0.5, "Supplier disclosure"),
    (24, "Opportunity", "Transmission", "Private-sector transmission build programme shortlists partners", 4, 0.5, "Government announcement"),
    (27, "Uncertainty", "Macro", "Currency volatility increases imported spares and diesel costs", 3, 0.6, "Market data"),
    (31, "Threat", "Climate", "Seasonal outlook: below-normal rainfall for interior summer", 3, 0.55, "Seasonal forecast"),
    (35, "Trend", "EV adoption", "EV registrations double from a low base; charging corridors expand", 1, 0.8, "Industry body"),
    (38, "Opportunity", "Gas", "LNG import terminal reaches design milestone", 2, 0.4, "Company disclosure"),
    (42, "Uncertainty", "Regulatory", "Municipal debt resolution framework under review", 3, 0.5, "Policy publication"),
    (47, "Threat", "Water", "Vaal system dam levels fall below seasonal average", 3, 0.6, "Water department data"),
    (55, "Trend", "Industrial", "Smelter operators signal curtailment if tariffs rise above inflation", 3, 0.5, "Company disclosure"),
    (63, "Opportunity", "Efficiency", "Utility-scale demand-response pilot exceeds savings target", 2, 0.7, "Research publication"),
    (75, "Threat", "Security", "Cable-theft incidents on distribution network rise in two provinces", 2, 0.75, "Social media / news"),
    (90, "Trend", "International", "Peer utilities accelerate coal-fleet life-extension programmes", 2, 0.6, "Research publication"),
    (120, "Uncertainty", "Regulatory", "Emissions-limit compliance deadline extension still pending", 4, 0.5, "Regulator notice"),
]


def sim_signals():
    df = pd.DataFrame(SIGNAL_SEED, columns=["days_ago", "category", "domain", "headline", "impact",
                                            "likelihood", "source_type"])
    df["date"] = TODAY - pd.to_timedelta(df.days_ago, unit="D")
    df["reliability"] = rng.choice(["A – confirmed", "B – credible", "C – unverified"], len(df), p=[0.4, 0.45, 0.15])
    df["headline"] = "[Simulated] " + df["headline"]
    return df.drop(columns="days_ago")


# =============================================================================
# 2. ANALYTICS LAYER
# =============================================================================
def scale(x, good, bad):
    """Map x onto 0 (good) … 100 (bad)."""
    if bad == good:
        return 0.0
    return float(np.clip((x - good) / (bad - good) * 100, 0, 100))


def slope_per_30d(series):
    y = np.asarray(series, dtype=float)
    if len(y) < 5 or np.all(np.isnan(y)):
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0] * 30)


def risk_engine(asof, nat, ops, grid, weather, econ, signals):
    n = nat[nat.date <= asof]
    o = ops[ops.date <= asof]
    g = grid[grid.date <= asof]
    w = weather[weather.date <= asof]
    e = econ[econ.month <= asof]
    sg = signals[(signals.date <= asof) & (signals.date > asof - pd.Timedelta(days=60))]
    last7, last30 = n.tail(7), n.tail(30)
    coal = o[o.type == "Coal"]
    dims = {}

    def dim(name, drivers, sources):
        score = sum(wt * sub for _, _, wt, sub in drivers)
        dims[name] = dict(score=round(score, 1), sources=sources,
                          drivers=[dict(label=l, value=v, weight=wt, subscore=round(sub, 0),
                                        contribution=round(wt * sub, 1)) for l, v, wt, sub in drivers])

    rm = last7.reserve_margin.mean()
    fleet_uclf = (last7.uclf_mw / STN.capacity_mw.sum()).mean()
    ls_days = int((last30.loadshed_stage > 0).sum())
    dim("Generation adequacy", [
        ("Reserve margin (7-day avg)", f"{rm:.1%}", 0.5, scale(rm, 0.20, 0.0)),
        ("Fleet unplanned loss (UCLF)", f"{fleet_uclf:.1%}", 0.3, scale(fleet_uclf, 0.12, 0.32)),
        ("Load-shedding days (30d)", f"{ls_days}", 0.2, scale(ls_days, 0, 15))],
        ["Station operations", "National balance", "Outage event log"])

    g7 = g[g.date > asof - pd.Timedelta(days=7)].groupby("corridor").loading_pct.mean()
    f30 = g[g.date > asof - pd.Timedelta(days=30)].faults.sum()
    f_prev = g[(g.date <= asof - pd.Timedelta(days=30)) & (g.date > asof - pd.Timedelta(days=120))].faults.sum() / 3
    ratio = f30 / max(f_prev, 1)
    dim("Transmission vulnerability", [
        ("Highest corridor loading (7d)", f"{g7.max():.0f}% · {g7.idxmax()}", 0.4, scale(g7.max(), 70, 98)),
        ("Faults last 30d vs prior avg", f"{f30} vs {f_prev:.0f} ({ratio:.1f}×)", 0.35, scale(ratio, 0.8, 2.0)),
        ("Corridors above 85% loading", f"{int((g7 > 85).sum())}", 0.25, scale((g7 > 85).sum(), 0, 4))],
        ["Grid corridor telemetry", "Fault register", "Weather stations"])

    stock_now = coal[coal.date == coal.date.max()].set_index("station").coal_stock_days
    cp = e.coal_price_usd_t.values
    cp_chg = cp[-1] / cp[-4] - 1 if len(cp) > 4 else 0
    dim("Fuel security", [
        ("Lowest station coal stock", f"{stock_now.min():.1f} days · {stock_now.idxmin()}", 0.45, scale(stock_now.min(), 25, 5)),
        ("Stations below 12 days stock", f"{int((stock_now < 12).sum())}", 0.35, scale((stock_now < 12).sum(), 0, 4)),
        ("Coal price change (3m)", f"{cp_chg:+.1%}", 0.2, scale(cp_chg, -0.05, 0.25))],
        ["Fuel stockpile reports", "Market prices", "External signals"])

    w14 = w[w.date > asof - pd.Timedelta(days=14)]
    coal_provs = ["Mpumalanga", "Limpopo", "Free State"]
    dmin = w[(w.date == w.date.max()) & (w.province.isin(coal_provs))].drought_idx.min()
    hw_days = int(w14.groupby("date").heatwave.any().sum())
    dim("Environmental exposure", [
        ("Max provincial temperature (7d)", f"{w14.tail(63).tmax_c.max():.1f} °C", 0.35, scale(w14.tail(63).tmax_c.max(), 30, 40)),
        ("Heatwave days (14d, any province)", f"{hw_days}", 0.25, scale(hw_days, 0, 10)),
        ("Drought index – coal provinces", f"{dmin:+.2f} σ", 0.4, scale(dmin, 0, -2.0))],
        ["Weather stations", "Satellite rainfall (sim)", "Seasonal forecast"])

    wage = (STN.age * STN.capacity_mw).sum() / STN.capacity_mw.sum()
    daily_uclf = coal.groupby("date").apply(lambda d: (d.uclf * d.capacity_mw).sum() / d.capacity_mw.sum(), include_groups=False)
    sl = slope_per_30d(daily_uclf.tail(90).values * 100)
    eff_now = coal[coal.date > asof - pd.Timedelta(days=90)].efficiency_pct.mean()
    eff_prev = coal[(coal.date > asof - pd.Timedelta(days=455)) & (coal.date <= asof - pd.Timedelta(days=365))].efficiency_pct.mean()
    dim("Infrastructure condition", [
        ("Capacity-weighted fleet age", f"{wage:.0f} yrs", 0.3, scale(wage, 25, 45)),
        ("Coal UCLF trend (90d)", f"{sl:+.2f} pp / month", 0.4, scale(sl, -0.5, 1.5)),
        ("Thermal efficiency vs last year", f"{eff_now - eff_prev:+.2f} pp", 0.3, scale(eff_now - eff_prev, 0.5, -1.5))],
        ["Station operations", "Asset register", "Maintenance records"])

    yr_ago = nat[(nat.date > asof - pd.Timedelta(days=372)) & (nat.date <= asof - pd.Timedelta(days=365))]
    growth = last7.peak_demand_mw.mean() / yr_ago.peak_demand_mw.mean() - 1 if len(yr_ago) else 0
    ratio_pa = (last7.peak_demand_mw / (last7.eskom_available_mw + last7.ipp_renewable_peak_mw)).mean()
    ip = e.industrial_production_idx.values
    ip_tr = ip[-1] / ip[-7] - 1 if len(ip) > 7 else 0
    dim("Demand pressure", [
        ("Peak demand vs same week last year", f"{growth:+.1%}", 0.4, scale(growth, -0.02, 0.06)),
        ("Peak demand / available supply", f"{ratio_pa:.2f}", 0.4, scale(ratio_pa, 0.80, 1.0)),
        ("Industrial production trend (6m)", f"{ip_tr:+.1%}", 0.2, scale(ip_tr, -0.02, 0.04))],
        ["National balance", "Economic indicators", "Grid-connection register"])

    fx = e.zar_usd.values
    fx_chg = fx[-1] / fx[-7] - 1 if len(fx) > 7 else 0
    ocgt = last30.ocgt_gwh.sum()
    c_now = o[o.date > asof - pd.Timedelta(days=30)].cost_r_per_mwh.mean()
    c_prev = o[(o.date > asof - pd.Timedelta(days=395)) & (o.date <= asof - pd.Timedelta(days=365))].cost_r_per_mwh.mean()
    dim("Financial stress", [
        ("ZAR/USD change (6m)", f"{fx_chg:+.1%}", 0.35, scale(fx_chg, -0.05, 0.15)),
        ("OCGT diesel generation (30d)", f"{ocgt:,.0f} GWh", 0.4, scale(ocgt, 0, 300)),
        ("Avg cost per MWh vs last year", f"{c_now / c_prev - 1:+.1%}", 0.25, scale(c_now / c_prev - 1, 0, 0.2))],
        ["Economic indicators", "Station cost ledger", "National balance"])

    neg = sg[sg.category.isin(["Threat", "Uncertainty"])]
    wsum = float((neg.impact * neg.likelihood).sum())
    reg_open = int(((sg.category == "Uncertainty") & sg.domain.isin(["Regulatory", "Market reform"])).sum())
    dim("Regulatory & policy uncertainty", [
        ("Weighted threat/uncertainty signals (60d)", f"{wsum:.1f}", 0.7, scale(wsum, 0, 25)),
        ("Open regulatory / market consultations", f"{reg_open}", 0.3, scale(reg_open, 0, 4))],
        ["External signals", "Regulator notices", "Policy publications"])

    weights = {"Generation adequacy": 0.24, "Transmission vulnerability": 0.14, "Fuel security": 0.13,
               "Environmental exposure": 0.11, "Infrastructure condition": 0.14, "Demand pressure": 0.1,
               "Financial stress": 0.08, "Regulatory & policy uncertainty": 0.06}
    composite = sum(dims[k]["score"] * v for k, v in weights.items())
    return dims, round(composite, 1)


def risk_level(score):
    for t, lab in [(25, "Low"), (45, "Moderate"), (60, "Elevated"), (75, "High")]:
        if score < t:
            return lab
    return "Critical"


LEVEL_COLOR = {"Low": THEME["ok"], "Moderate": "#9ccc65", "Elevated": THEME["warn"],
               "High": THEME["bad"], "Critical": THEME["crit"]}


def station_analytics(ops, events):
    """Per-station failure probability (heuristic model) + 13-week EAF forecast."""
    out = {}
    for s in STN.itertuples():
        d = ops[ops.station == s.station].set_index("date")
        wk = d.resample("W").agg({"eaf": "mean", "uclf": "mean", "pclf": "mean", "energy_gwh": "sum",
                                  "efficiency_pct": "mean", "coal_stock_days": "mean", "tmax_c": "max",
                                  "cost_r_per_mwh": "mean", "water_l_per_kwh": "mean", "available_mw": "mean"})
        y = wk.eaf.values[-26:] * 100
        x = np.arange(len(y))
        b1, b0 = np.polyfit(x, y, 1)
        resid = np.std(y - (b0 + b1 * x))
        h = np.arange(1, 14)
        damp = np.cumsum(0.85 ** h)  # damped trend: recent slope fades over the horizon
        fc = np.clip(b0 + b1 * (len(y) - 1) + b1 * damp, 5, 98)
        band = 1.64 * resid * np.sqrt(1 + h / 6)
        # heuristic logistic risk model (NOT trained — illustrative)
        z_base = d.uclf.iloc[-380:-15]
        z = (d.uclf.tail(14).mean() - z_base.mean()) / max(z_base.std(), 1e-3)
        uclf_tr = slope_per_30d(d.uclf.tail(90).values * 100)
        trips90 = int(((events.station == s.station) & (events.start > TODAY - pd.Timedelta(days=90))).sum())
        stock = d.coal_stock_days.iloc[-1]
        heat = d.tmax_c.tail(7).max()
        feats = {
            "Outage anomaly (z-score)": (float(np.clip(z, -1, 5)), 0.45),
            "UCLF trend (pp/month)": (float(np.clip(uclf_tr, -2, 4)), 0.2),
            "Plant age (yrs/10)": (s.age / 10, 0.2),
            "Unit trips per unit (90d)": (trips90 / s.units, 0.35),
            "Low coal stock": (max(0, (15 - stock) / 5) if not math.isnan(stock) else 0, 0.6),
            "Heat stress (°C above 32)": (max(0, heat - 32) / 3, 0.25),
        }
        logit = -3.0 + sum(v * wgt for v, wgt in feats.values())
        prob = 1 / (1 + math.exp(-logit))
        contrib = sorted(((k, round(v * wgt, 2)) for k, (v, wgt) in feats.items()), key=lambda t: -t[1])
        out[s.station] = dict(weekly=wk, fc_dates=[(wk.index[-1] + pd.Timedelta(weeks=int(i))) for i in h],
                              fc=fc, fc_lo=np.clip(fc - band, 0, 100), fc_hi=np.clip(fc + band, 0, 100),
                              prob=prob, contrib=contrib, z=z, trips90=trips90, uclf_trend=uclf_tr,
                              stock=stock, heat=heat)
    return out


def build_alerts(ops, nat, grid, weather, signals, st_an, events):
    alerts = []

    def add(title, level, category, affected, evidence, horizon, action, impact, prob, urgency, mw=0, stations=()):
        alerts.append(dict(title=title, level=level, category=category, affected=affected, evidence=evidence,
                           horizon=horizon, action=action, impact=impact, prob=round(prob, 2), urgency=urgency,
                           exposure_mw=int(mw), stations=list(stations),
                           priority=round(impact * prob * urgency / 25 * 100, 0)))

    # 1. station forced-outage anomalies
    anom = [(k, v) for k, v in st_an.items() if v["z"] > 2.0]
    for name, v in anom:
        cap = int(STN.set_index("station").capacity_mw[name])
        add(f"Abnormal forced-outage rate at {name}", "High" if v["z"] > 3 else "Elevated", "Generation",
            name, [f"14-day UCLF is {v['z']:.1f}σ above its 12-month baseline",
                   f"{v['trips90']} unit trips in the last 90 days",
                   f"UCLF trend {v['uclf_trend']:+.2f} pp/month · model 90-day failure probability {v['prob']:.0%}",
                   "Sources: Station operations · Outage event log"],
            "0–6 weeks", "Review boiler/turbine defect backlog; confirm outage return-to-service dates",
            4, min(0.95, 0.4 + v["z"] / 10), 4, cap * 0.2, [name])

    # 2. compound provincial generation risk
    mp = [k for k, v in anom if STN.set_index("station").province[k] == "Mpumalanga"]
    mp_w = weather[(weather.province == "Mpumalanga")].tail(7)
    if len(mp) >= 2:
        mw = STN.set_index("station").loc[mp].capacity_mw.sum()
        add("Emerging Generation Risk: HIGH — Mpumalanga coal cluster", "Critical", "Compound", ", ".join(mp),
            [f"{len(mp)} stations show simultaneous outage anomalies ({', '.join(mp)})",
             f"Highveld max temperature {mp_w.tmax_c.max():.1f} °C, {int(mp_w.heatwave.sum())} heatwave days this week",
             f"Mpumalanga drought index {mp_w.drought_idx.iloc[-1]:+.2f}σ (water-cooled plant exposure)",
             "Coal-logistics threat signals in last 7 days (rail + road haulage)",
             "Sources: Station ops · Weather stations · External signals"],
            "1–4 weeks", "Convene cross-functional review: maintenance sequencing, coal logistics, water allocation",
            5, 0.75, 5, mw * 0.25, mp)

    # 3. coal stock
    last = ops[(ops.date == ops.date.max()) & (ops.type == "Coal")]
    for r in last[last.coal_stock_days < 12].itertuples():
        add(f"Coal stockpile below strategic level at {r.station}", "High" if r.coal_stock_days < 9 else "Elevated",
            "Fuel", r.station, [f"Stockpile at {r.coal_stock_days:.1f} days (strategic minimum 12–20 days)",
                                "Declining for 30 consecutive days",
                                "Related signals: haulage contractor dispute, rail disruption",
                                "Sources: Fuel stockpile reports · External signals"],
            "0–3 weeks", "Activate emergency road-haul allocation; verify coal-quality testing at mills",
            4, 0.7, 5, r.capacity_mw * 0.3, [r.station])

    # 4. corridor stress
    g7 = grid[grid.date > TODAY - pd.Timedelta(days=7)].groupby("corridor").agg(load=("loading_pct", "mean"),
                                                                                    faults=("faults", "sum"))
    f30 = grid[grid.date > TODAY - pd.Timedelta(days=30)].groupby("corridor").faults.sum()
    for c, r in g7[g7.load > 86].iterrows():
        cc = COR.set_index("corridor").loc[c]
        add(f"Transmission corridor under stress: {c}", "High" if cc.redundancy == "N-0" else "Elevated",
            "Transmission", c, [f"7-day average loading {r.load:.0f}% of {cc.capacity_mw:,} MW",
                                f"{int(f30.get(c, 0))} faults in last 30 days",
                                f"Redundancy: {cc.redundancy}" + (" — no contingency path" if cc.redundancy == "N-0" else ""),
                                "Sources: Grid corridor telemetry · Fault register"],
            "0–2 weeks", "Review contingency switching plan; defer non-critical line maintenance",
            4 if cc.redundancy == "N-0" else 3, 0.55, 4, cc.capacity_mw * 0.3)

    # 5. heat / demand
    hot = weather[(weather.date > TODAY - pd.Timedelta(days=5)) & weather.heatwave]
    if len(hot):
        provs = sorted(hot.province.unique())
        add("Heatwave affecting plant cooling and demand", "Elevated", "Environment", ", ".join(provs),
            [f"Tmax ≥ 35 °C recorded in {', '.join(provs)} in the last 5 days",
             "Hot days raise unit-trip probability in the model by ~6% per °C above 32",
             "Weather-service heat advisory extended (external signal)",
             "Sources: Weather stations · External signals"],
            "0–10 days", "Pre-position cooling-water contingency; tighten auxiliary plant inspections",
            3, 0.8, 4)

    # 6. supply chain cluster
    sc = signals[(signals.domain == "Supply chain") & (signals.date > TODAY - pd.Timedelta(days=30))]
    if len(sc) >= 2:
        add("Repeated supply-chain warnings on critical spares", "Moderate", "Supply chain", "Coal fleet",
            [f"{len(sc)} independent OEM / supplier notices in 30 days"] + [h.replace("[Simulated] ", "") for h in sc.headline]
            + ["Sources: Supplier disclosures"],
            "1–6 months", "Audit spares holding for boiler tubes and turbine parts; consider strategic stock",
            3, 0.5, 2)

    # 7. adequacy outlook
    if nat.tail(14).loadshed_stage.max() > 0:
        add("Supply shortfall risk over next fortnight", "High", "Adequacy", "National",
            [f"Load-shedding required on {int((nat.tail(14).loadshed_stage > 0).sum())} of last 14 days",
             f"Reserve margin 7-day avg {nat.tail(7).reserve_margin.mean():.1%}",
             "Sources: National balance"], "0–14 days",
            "Stress-test outlook in Scenario screen; confirm OCGT fuel cover", 5, 0.6, 5)

    df = pd.DataFrame(alerts).sort_values("priority", ascending=False).reset_index(drop=True)
    return df


def forward_outlook(nat, ops):
    """Simple 30-day baseline projection used by executive outlook + scenario simulator."""
    fut = pd.date_range(TODAY + pd.Timedelta(days=1), periods=HORIZON)
    hist = nat.set_index("date")
    ly = hist.peak_demand_mw.reindex(fut - pd.Timedelta(days=364)).values
    growth = hist.peak_demand_mw.tail(28).mean() / hist.peak_demand_mw.reindex(DATES[-28:] - pd.Timedelta(days=364)).mean()
    demand = ly * growth
    avail_trend = slope_per_30d(hist.eskom_available_mw.tail(60).values) / 30
    base_av = hist.eskom_available_mw.tail(14).mean()
    avail = base_av + 0.5 * avail_trend * np.arange(1, HORIZON + 1)  # half-weight the recent trend
    ipp = np.full(HORIZON, hist.ipp_renewable_peak_mw.tail(30).mean())
    last = ops[ops.date == ops.date.max()]
    low_coal_mw = float(last[(last.type == "Coal") & (last.coal_stock_days < 12)].available_mw.sum())
    gas_mw = float(last[last.type == "Gas (OCGT)"].available_mw.sum())
    return dict(dates=[d.strftime("%Y-%m-%d") for d in fut], demand=np.round(demand).tolist(),
                avail=np.round(avail).tolist(), ipp=np.round(ipp).tolist(), reserve=OP_RESERVE_MW,
                low_coal_mw=round(low_coal_mw), gas_mw=round(gas_mw),
                corridors={r.corridor: round(r.capacity_mw * 0.35) for r in COR.itertuples()})


def data_catalogue(qc_weather, frames):
    rows = [
        ("Station operations", "Plant historian (sim)", "Database / API", "Daily", "A", frames["ops"]),
        ("Outage event log", "Maintenance system (sim)", "Database", "Event-driven", "A", frames["events"]),
        ("National balance", "System operator (sim)", "API", "Hourly → daily", "A", frames["nat"]),
        ("Grid corridor telemetry", "SCADA (sim)", "Sensor stream", "15-min → daily", "A", frames["grid"]),
        ("Substation register", "Asset register (sim)", "Spreadsheet", "Quarterly", "B", frames["subs"]),
        ("Weather stations", "Weather service (sim)", "Open-data API", "Daily", "B", frames["weather"]),
        ("Satellite rainfall / drought", "Satellite product (sim)", "Satellite raster", "Daily", "B", frames["weather"]),
        ("Fuel stockpile reports", "Fuel desk (sim)", "Manual submission", "Daily", "B", frames["ops"]),
        ("Economic indicators", "Statistics office / markets (sim)", "Open-data portal", "Monthly", "A", frames["econ"]),
        ("External signals", "News, tenders, regulator notices (sim)", "Web / NLP scan", "Continuous", "C", frames["signals"]),
        ("Demand centres", "Distribution planning (sim)", "Spreadsheet", "Annual", "B", frames["dc"]),
    ]
    out = []
    for name, src, kind, freq, rel, df in rows:
        dcol = "date" if "date" in df.columns else ("month" if "month" in df.columns else ("start" if "start" in df.columns else None))
        last = df[dcol].max().strftime("%Y-%m-%d") if dcol else TODAY.strftime("%Y-%m-%d")
        nulls = float(df.isna().mean().mean() * 100)
        passed = qc_weather["passed_pct"] if name == "Weather stations" else round(100 - min(nulls, 15) - rng.uniform(0, 1.2), 1)
        out.append(dict(dataset=name, source=src, ingest=kind, frequency=freq, reliability=rel,
                        records=len(df), last_update=last, qc_pass_pct=passed))
    return pd.DataFrame(out)


# =============================================================================
# 3. INTELLIGENCE LAYER — visuals
# =============================================================================
def base_layout(fig, title=None, h=320, legend=True):
    fig.update_layout(
        template="plotly_dark", height=h, title=dict(text=title, font=dict(size=14)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=THEME["text"]),
        margin=dict(l=50, r=20, t=40 if title else 15, b=40), showlegend=legend,
        legend=dict(orientation="h", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=THEME["panel"], font_size=12))
    fig.update_xaxes(gridcolor=THEME["line"], zeroline=False)
    fig.update_yaxes(gridcolor=THEME["line"], zeroline=False)
    return fig


def fig_html(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"responsive": True, "displaylogo": False})


def hex_rgb(h, a=255):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def status_color(eaf):
    return hex_rgb(THEME["ok"]) if eaf >= 0.7 else (hex_rgb(THEME["warn"]) if eaf >= 0.5 else hex_rgb(THEME["bad"]))


def load_color(pct):
    t = np.clip((pct - 50) / 45, 0, 1)
    return [int(46 + t * 209), int(204 - t * 130), int(113 - t * 40), 220]


def deck_srcdoc(deck):
    return html.escape(deck.to_html(as_string=True, notebook_display=False, offline=OFFLINE), quote=True)


TOOLTIP = {"html": "<b>{tt_title}</b><br/>{tt_body}",
           "style": {"backgroundColor": THEME["panel"], "color": THEME["text"], "fontSize": "12px",
                     "border": f"1px solid {THEME['line']}", "borderRadius": "6px", "padding": "8px"}}


def build_system_map(ops, grid, subs, st_an, alert_stations):
    last = ops[ops.date == ops.date.max()].set_index("station")
    st = STN.copy()
    st["eaf"] = st.station.map(last.eaf)
    st["avail"] = st.station.map(last.available_mw)
    st["pos"] = st.apply(lambda r: [r.lon, r.lat], axis=1)
    st["color"] = st.eaf.apply(status_color)
    st["ring"] = st.station.apply(lambda s: hex_rgb(THEME["crit"]) if s in alert_stations else [0, 0, 0, 0])
    st["tt_title"] = st.station + " · " + st.type
    st["tt_body"] = st.apply(lambda r: (f"Installed {r.capacity_mw:,} MW · Available {r.avail:,.0f} MW<br/>"
                                         f"EAF {r.eaf:.0%} · Age {r.age} yrs · {r.province}<br/>"
                                         f"90-day failure prob (model) {st_an[r.station]['prob']:.0%}"
                                         + ("<br/><span style='color:#ff5c5c'>⚠ active alert</span>" if r.station in alert_stations else "")), axis=1)
    loc = {**{r.station: [r.lon, r.lat] for r in STN.itertuples()}, **{r.centre: [r.lon, r.lat] for r in DC.itertuples()}}
    g7 = grid[grid.date > TODAY - pd.Timedelta(days=7)].groupby("corridor").agg(load=("loading_pct", "mean"), faults=("faults", "sum"))
    cor = COR.copy()
    cor["path"] = cor.apply(lambda r: [loc[r["from"]], loc[r["to"]]], axis=1)
    cor["load"] = cor.corridor.map(g7.load)
    cor["color"] = cor.load.apply(load_color)
    cor["width"] = cor.capacity_mw * 1.6  # metres
    cor["tt_title"] = cor.corridor
    cor["tt_body"] = cor.apply(lambda r: f"Loading {r.load:.0f}% of {r.capacity_mw:,} MW · {r.redundancy}<br/>Faults (7d): {int(g7.faults[r.corridor])}", axis=1)
    dc = DC.copy()
    dc["pos"] = dc.apply(lambda r: [r.lon, r.lat], axis=1)
    dc["tt_title"] = dc.centre + " (demand centre)"
    dc["tt_body"] = dc.apply(lambda r: f"Typical peak ~{r.peak_mw:,} MW · {r.province}", axis=1)
    sb = subs.copy()
    sb["pos"] = sb.apply(lambda r: [r.lon, r.lat], axis=1)
    sb["color"] = sb.health_idx.apply(lambda h: hex_rgb(THEME["ok"], 200) if h > 75 else hex_rgb(THEME["warn"], 220) if h > 62 else hex_rgb(THEME["bad"], 230))
    sb["tt_title"] = sb.substation
    sb["tt_body"] = sb.apply(lambda r: f"Health index {r.health_idx} · {r.mva} MVA · transformer age {r.transformer_age} yrs", axis=1)

    layers = [
        pdk.Layer("PathLayer", cor, get_path="path", get_color="color", get_width="width",
                  width_min_pixels=2, width_max_pixels=9, cap_rounded=True, pickable=True, auto_highlight=True),
        pdk.Layer("ScatterplotLayer", dc, get_position="pos", get_radius="peak_mw * 9",
                  get_fill_color=hex_rgb(THEME["accent"], 70), get_line_color=hex_rgb(THEME["accent"], 220),
                  stroked=True, line_width_min_pixels=1, pickable=True),
        pdk.Layer("ScatterplotLayer", sb, get_position="pos", get_radius=4500, get_fill_color="color", pickable=True),
        pdk.Layer("ScatterplotLayer", st, get_position="pos", get_radius=26000, filled=False, stroked=True,
                  get_line_color="ring", line_width_min_pixels=3),
        pdk.Layer("ColumnLayer", st, get_position="pos", get_elevation="avail", elevation_scale=60,
                  radius=11000, get_fill_color="color", extruded=True, pickable=True, auto_highlight=True),
    ]
    view = pdk.ViewState(latitude=-28.9, longitude=25.6, zoom=4.55, pitch=42, bearing=-8)
    return pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="dark",
                    tooltip=TOOLTIP, height=560)


def build_env_map(weather, st_an):
    lastw = weather[weather.date > TODAY - pd.Timedelta(days=7)].groupby("province").agg(
        t=("tmax_c", "max"), d=("drought_idx", "last"))
    pts = []
    for prov, (lat, lon, *_rest) in PROVINCES.items():
        t, d = lastw.loc[prov, "t"], lastw.loc[prov, "d"]
        stress = max(0, t - 28) / 10 + max(0, -d) / 2
        for _ in range(140):
            pts.append(dict(pos=[lon + rng.normal(0, 1.3), lat + rng.normal(0, 1.0)], w=stress * rng.uniform(0.6, 1.2)))
    hm = pd.DataFrame(pts)
    pv = pd.DataFrame([dict(pos=[v[1], v[0]], tt_title=f"{k} weather", tt_body=f"Max temp 7d {lastw.loc[k, 't']:.1f} °C<br/>Drought index {lastw.loc[k, 'd']:+.2f}σ")
                       for k, v in PROVINCES.items()])
    st = STN.copy()
    st["pos"] = st.apply(lambda r: [r.lon, r.lat], axis=1)
    st["p"] = st.station.map(lambda s: st_an[s]["prob"])
    st["color"] = st.p.apply(lambda p: hex_rgb(THEME["bad"]) if p > 0.5 else hex_rgb(THEME["warn"]) if p > 0.25 else hex_rgb(THEME["ok"]))
    st["tt_title"] = st.station
    st["tt_body"] = st.p.apply(lambda p: f"90-day failure probability (model): {p:.0%}")
    layers = [
        pdk.Layer("HeatmapLayer", hm, get_position="pos", get_weight="w", radius_pixels=55, opacity=0.65,
                  color_range=[[255, 255, 178], [254, 217, 118], [254, 178, 76], [253, 141, 60], [240, 59, 32], [189, 0, 38]]),
        pdk.Layer("ScatterplotLayer", st, get_position="pos", get_radius="8000 + p * 30000", get_fill_color="color",
                  stroked=True, get_line_color=[255, 255, 255, 160], line_width_min_pixels=1, pickable=True),
        pdk.Layer("ScatterplotLayer", pv, get_position="pos", get_radius=9000, get_fill_color=hex_rgb(THEME["accent"], 230), pickable=True),
    ]
    view = pdk.ViewState(latitude=-28.9, longitude=25.0, zoom=4.5, pitch=0)
    return pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="dark",
                    tooltip=TOOLTIP, height=480)


def analytics_figs(nat, ops, events, grid, weather, econ, signals, risk_hist, dims, outlook):
    F = {}
    # executive outlook (14d)
    f = go.Figure()
    d = pd.to_datetime(outlook["dates"][:14])
    sup = (np.array(outlook["avail"]) + np.array(outlook["ipp"]))[:14]
    req = (np.array(outlook["demand"]) + OP_RESERVE_MW)[:14]
    f.add_trace(go.Scatter(x=d, y=sup, name="Available supply (forecast)", line=dict(color=THEME["ok"], width=2.5)))
    f.add_trace(go.Scatter(x=d, y=req, name="Peak demand + reserve (forecast)", line=dict(color=THEME["warn"], dash="dash")))
    F["outlook"] = fig_html(base_layout(f, h=250), "f-outlook")

    # national supply vs demand
    f = go.Figure()
    f.add_trace(go.Bar(x=nat.date, y=nat.loadshed_stage, name="Load-shedding stage", yaxis="y2",
                       marker_color="rgba(255,92,92,0.45)"))
    f.add_trace(go.Scatter(x=nat.date, y=nat.eskom_available_mw + nat.ipp_renewable_peak_mw, name="Available supply",
                           line=dict(color=THEME["ok"], width=1.6)))
    f.add_trace(go.Scatter(x=nat.date, y=nat.peak_demand_mw, name="Peak demand", line=dict(color=THEME["warn"], width=1.4)))
    f.update_layout(yaxis=dict(title="MW"), yaxis2=dict(overlaying="y", side="right", range=[0, 8], title="Stage", showgrid=False))
    base_layout(f, "National supply vs demand (daily)", 360)
    f.update_xaxes(rangeselector=dict(buttons=[dict(count=3, label="3m", step="month"), dict(count=1, label="1y", step="year"),
                                                dict(step="all", label="All")], bgcolor=THEME["panel"]))
    F["balance"] = fig_html(f, "f-balance")

    # weekly EAF heatmap by station, province dropdown
    wk = ops.assign(week=ops.date.dt.to_period("W").dt.start_time).groupby(["province", "station", "week"]).eaf.mean().reset_index()
    f = go.Figure()
    provs = ["All provinces"] + sorted(STN.province.unique())
    order = STN.sort_values(["province", "capacity_mw"]).station.tolist()
    for p in provs:
        sub = wk if p == "All provinces" else wk[wk.province == p]
        m = sub.pivot(index="station", columns="week", values="eaf")
        m = m.reindex([s for s in order if s in m.index])
        f.add_trace(go.Heatmap(z=m.values * 100, x=m.columns, y=m.index, colorscale="RdYlGn", zmin=30, zmax=95,
                               colorbar=dict(title="EAF %"), visible=(p == "All provinces"),
                               hovertemplate="%{y}<br>%{x|%d %b %Y}<br>EAF %{z:.0f}%<extra></extra>"))
    f.update_layout(updatemenus=[dict(buttons=[dict(label=p, method="update", args=[{"visible": [q == p for q in provs]}]) for p in provs],
                                      x=0, y=1.14, xanchor="left", bgcolor=THEME["panel"], font=dict(color=THEME["text"]))])
    F["eafheat"] = fig_html(base_layout(f, "Station availability (EAF) by week — filter by province", 520, False), "f-eafheat")

    # outages by cause
    ev = events[events.start > TODAY - pd.Timedelta(days=365)].assign(mwd=lambda d: d.mw_lost * d.days)
    bc = ev.groupby("cause").mwd.sum().sort_values() / 1000
    f = go.Figure(go.Bar(x=bc.values, y=bc.index, orientation="h", marker_color=THEME["accent"],
                         hovertemplate="%{y}: %{x:,.0f} GW-days<extra></extra>"))
    F["causes"] = fig_html(base_layout(f, "Unplanned outages by cause, last 12 months (GW-days lost)", 360, False), "f-causes")
    f = go.Figure(go.Histogram(x=ev.days, nbinsx=40, marker_color=THEME["warn"]))
    f.update_xaxes(title="Outage duration (days)")
    F["dur"] = fig_html(base_layout(f, "Outage duration distribution", 360, False), "f-dur")

    # typical demand curves
    hrs = np.arange(24)
    prof = lambda am, pm, base: base + am * np.exp(-((hrs - 7.5) ** 2) / 4) + pm * np.exp(-((hrs - 18.5) ** 2) / 5) - 0.12 * base * np.exp(-((hrs - 3) ** 2) / 8)
    f = go.Figure()
    f.add_trace(go.Scatter(x=hrs, y=prof(2500, 5500, 26500), name="Winter weekday", line=dict(color=THEME["accent"], width=2.5)))
    f.add_trace(go.Scatter(x=hrs, y=prof(1500, 2500, 24500), name="Summer weekday", line=dict(color=THEME["warn"], width=2.5)))
    f.add_trace(go.Scatter(x=hrs, y=prof(800, 3000, 22000), name="Weekend", line=dict(color=THEME["muted"], dash="dot")))
    f.update_xaxes(title="Hour of day", dtick=3)
    f.update_yaxes(title="MW")
    F["profile"] = fig_html(base_layout(f, "Typical demand curves (modelled)", 360), "f-profile")

    # correlation network → matrix
    coal_daily = ops[ops.type == "Coal"].groupby("date").agg(coal_stock=("coal_stock_days", "mean"))
    corr_df = pd.DataFrame({
        "Peak demand": nat.peak_demand_mw.values, "Gauteng temp": nat.gauteng_tmax.values,
        "Fleet UCLF": nat.uclf_mw.values, "Reserve margin": nat.reserve_margin.values,
        "Wind CF": nat.wind_cf.values, "Avg coal stock": coal_daily.coal_stock.values,
        "Grid faults": grid.groupby("date").faults.sum().values, "OCGT GWh": nat.ocgt_gwh.values})
    c = corr_df.corr().round(2)
    f = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, colorscale="RdBu", zmin=-1, zmax=1, text=c.values,
                             texttemplate="%{text}", hovertemplate="%{y} × %{x}: %{z}<extra></extra>"))
    F["corr"] = fig_html(base_layout(f, "Correlation matrix of daily system drivers", 420, False), "f-corr")

    # anomaly plot
    fu = (nat.uclf_mw / STN.capacity_mw.sum() * 100)
    mu, sd = fu.rolling(60).mean(), fu.rolling(60).std()
    z = (fu - mu) / sd
    f = go.Figure()
    f.add_trace(go.Scatter(x=nat.date, y=fu, name="Fleet UCLF %", line=dict(color=THEME["muted"], width=1.3)))
    f.add_trace(go.Scatter(x=nat.date, y=mu + 2 * sd, name="Normal band (+2σ)", line=dict(color=THEME["line"], dash="dot")))
    a = z > 2
    f.add_trace(go.Scatter(x=nat.date[a], y=fu[a], mode="markers", name="Anomaly", marker=dict(color=THEME["bad"], size=7)))
    F["anomaly"] = fig_html(base_layout(f, "Anomaly detection — fleet unplanned capability loss", 340), "f-anomaly")

    # fleet comparison
    last = ops[ops.date > TODAY - pd.Timedelta(days=90)].groupby("station").eaf.mean()
    s = STN.assign(eaf=STN.station.map(last) * 100)
    f = go.Figure()
    for t, g in s.groupby("type"):
        f.add_trace(go.Scatter(x=g.age, y=g.eaf, mode="markers+text", text=g.station, textposition="top center",
                               name=t, marker=dict(size=np.sqrt(g.capacity_mw) / 2.4, color=TYPE_COLORS[t], line=dict(width=1, color="#fff")),
                               textfont=dict(size=10, color=THEME["muted"])))
    f.update_xaxes(title="Plant age (years)")
    f.update_yaxes(title="EAF % (last 90 days)")
    F["fleet"] = fig_html(base_layout(f, "Asset performance — age vs availability (bubble = capacity)", 440), "f-fleet")

    # generation mix
    mix = ops.assign(m=ops.date.dt.to_period("M").dt.start_time).groupby(["m", "type"]).energy_gwh.sum().unstack()
    mix["IPP renewables"] = nat.assign(m=nat.date.dt.to_period("M").dt.start_time).groupby("m").ipp_renewable_gwh.sum()
    f = go.Figure()
    for t in ["Coal", "Nuclear", "Gas (OCGT)", "Pumped storage", "Hydro", "Wind", "IPP renewables"]:
        f.add_trace(go.Bar(x=mix.index, y=mix[t], name=t, marker_color=TYPE_COLORS[t]))
    f.update_layout(barmode="stack")
    f.update_yaxes(title="GWh / month")
    F["mix"] = fig_html(base_layout(f, "Energy sent out by technology (monthly)", 360), "f-mix")

    # risk radar + trend
    names = list(dims.keys())
    f = go.Figure(go.Scatterpolar(r=[dims[k]["score"] for k in names] + [dims[names[0]]["score"]], theta=names + [names[0]],
                                  fill="toself", line=dict(color=THEME["bad"]), fillcolor="rgba(255,92,92,0.25)"))
    f.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor=THEME["line"]),
                               angularaxis=dict(gridcolor=THEME["line"])))
    F["radar"] = fig_html(base_layout(f, "Risk profile (0 = benign, 100 = severe)", 400, False), "f-radar")
    f = go.Figure()
    f.add_hrect(y0=0, y1=25, fillcolor=THEME["ok"], opacity=0.06, line_width=0)
    f.add_hrect(y0=60, y1=100, fillcolor=THEME["bad"], opacity=0.06, line_width=0)
    f.add_trace(go.Scatter(x=risk_hist.date, y=risk_hist.composite, name="Composite", line=dict(color="#fff", width=3)))
    for k in names:
        f.add_trace(go.Scatter(x=risk_hist.date, y=risk_hist[k], name=k, line=dict(width=1.2), visible="legendonly"))
    f.update_yaxes(range=[0, 100])
    F["risktrend"] = fig_html(base_layout(f, "System risk over last 26 weeks (click legend to add dimensions)", 400), "f-risktrend")

    # corridors
    g30 = grid[grid.date > TODAY - pd.Timedelta(days=30)].groupby("corridor").agg(load=("loading_pct", "mean"), faults=("faults", "sum")).reset_index()
    g30 = g30.merge(COR[["corridor", "capacity_mw", "redundancy"]], on="corridor")
    f = go.Figure(go.Scatter(x=g30.load, y=g30.faults, mode="markers+text", text=g30.corridor, textposition="top center",
                             marker=dict(size=g30.capacity_mw / 280, color=[THEME["bad"] if r == "N-0" else THEME["accent"] for r in g30.redundancy],
                                         line=dict(width=1, color="#fff")), textfont=dict(size=10, color=THEME["muted"]),
                             hovertemplate="%{text}<br>Loading %{x:.0f}%<br>Faults %{y}<extra></extra>"))
    f.add_vline(x=85, line_dash="dot", line_color=THEME["warn"])
    f.update_xaxes(title="Avg loading % (30d)")
    f.update_yaxes(title="Faults (30d)")
    F["corridors"] = fig_html(base_layout(f, "Transmission corridors — loading vs faults (red = no N-1 redundancy)", 420, False), "f-corridors")

    # external signal matrix
    col = {"Threat": THEME["bad"], "Opportunity": THEME["ok"], "Trend": THEME["accent"], "Uncertainty": THEME["warn"]}
    f = go.Figure()
    for cat, g in signals.groupby("category"):
        f.add_trace(go.Scatter(x=g.likelihood + rng.normal(0, 0.01, len(g)), y=g.impact + rng.normal(0, 0.08, len(g)),
                               mode="markers", name=cat, marker=dict(size=16, color=col[cat], opacity=0.8, line=dict(width=1, color="#fff")),
                               text=g.headline.str.replace("[Simulated] ", ""), customdata=g.domain,
                               hovertemplate="<b>%{customdata}</b><br>%{text}<extra></extra>"))
    f.update_xaxes(title="Likelihood", range=[0.3, 0.95])
    f.update_yaxes(title="Impact (1–5)", range=[0.5, 5.5])
    F["signals"] = fig_html(base_layout(f, "Environmental scan — impact vs likelihood", 420), "f-signals")

    # economics small multiples
    from plotly.subplots import make_subplots
    f = make_subplots(rows=2, cols=3, subplot_titles=["GDP index", "Industrial production", "ZAR / USD",
                                                       "Coal price USD/t", "CPI % y/y", "Prime rate %"])
    for i, c_ in enumerate(["gdp_index", "industrial_production_idx", "zar_usd", "coal_price_usd_t", "cpi_yoy_pct", "prime_rate_pct"]):
        f.add_trace(go.Scatter(x=econ.month, y=econ[c_], line=dict(color=THEME["accent"], width=2)), row=i // 3 + 1, col=i % 3 + 1)
    F["econ"] = fig_html(base_layout(f, None, 420, False), "f-econ")
    return F


# =============================================================================
# 4. HTML ASSEMBLY
# =============================================================================
def esc(s):
    return html.escape(str(s))


def kpi(label, value, sub="", tone=""):
    return f'<div class="kpi {tone}"><div class="kl">{esc(label)}</div><div class="kv">{value}</div><div class="ks">{sub}</div></div>'


def chip(level):
    return f'<span class="chip" style="background:{LEVEL_COLOR.get(level, THEME["muted"])}">{esc(level)}</span>'


def alert_card(a, idx):
    btns = "".join(f'<button class="link" onclick="goStation(\'{esc(s)}\')">Open {esc(s)} →</button>' for s in a["stations"])
    ev = "".join(f"<li>{esc(e)}</li>" for e in a["evidence"])
    return f"""<div class="alert">
      <div class="ah"><span class="rank">#{idx}</span>{chip(a['level'])}<b>{esc(a['title'])}</b>
        <span class="pri">priority {a['priority']:.0f}</span></div>
      <div class="meta">{esc(a['category'])} · affected: {esc(a['affected'])} · horizon {esc(a['horizon'])} ·
        impact {a['impact']}/5 · probability {a['prob']:.0%} · urgency {a['urgency']}/5</div>
      <details><summary>Evidence & recommended investigation</summary><ul>{ev}</ul>
        <div class="rec">→ {esc(a['action'])}</div>{btns}</details></div>"""


def table(df, cls="tbl"):
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in df.astype(str).values)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def jsonable(x):
    if isinstance(x, (np.floating, float)):
        return None if (isinstance(x, float) and math.isnan(x)) or np.isnan(x) else round(float(x), 2)
    if isinstance(x, (np.integer,)):
        return int(x)
    return str(x)


def station_payload(ops, events, st_an):
    fleet = ops[ops.type == "Coal"].groupby("date").apply(lambda d: (d.eaf * d.capacity_mw).sum() / d.capacity_mw.sum(), include_groups=False)
    fleet_w = (fleet.resample("W").mean() * 100).round(1)
    type_w = ops.groupby(["type", "date"]).eaf.mean().unstack(0).resample("W").mean() * 100
    P = {}
    for s in STN.itertuples():
        a = st_an[s.station]
        wk = a["weekly"]
        ev = events[events.station == s.station].head(12)
        P[s.station] = dict(
            meta=dict(type=s.type, province=s.province, capacity=s.capacity_mw, units=s.units, age=s.age,
                      commissioned=s.commissioned, cooling="Dry" if s.dry_cooled else ("Wet" if s.type in ("Coal", "Nuclear") else "n/a")),
            weeks=[d.strftime("%Y-%m-%d") for d in wk.index],
            eaf=(wk.eaf * 100).round(1).tolist(), uclf=(wk.uclf * 100).round(1).tolist(), pclf=(wk.pclf * 100).round(1).tolist(),
            gen=wk.energy_gwh.round(1).tolist(), eff=[jsonable(v) for v in wk.efficiency_pct],
            coal=[jsonable(v) for v in wk.coal_stock_days], temp=wk.tmax_c.round(1).tolist(),
            cost=wk.cost_r_per_mwh.round(0).tolist(),
            fleet=fleet_w.reindex(wk.index).tolist(), peer=type_w[s.type].reindex(wk.index).round(1).tolist(),
            fc_dates=[d.strftime("%Y-%m-%d") for d in a["fc_dates"]], fc=np.round(a["fc"], 1).tolist(),
            fc_lo=np.round(a["fc_lo"], 1).tolist(), fc_hi=np.round(a["fc_hi"], 1).tolist(),
            prob=round(a["prob"], 3), contrib=a["contrib"], z=round(float(a["z"]), 2), trips90=a["trips90"],
            events=[dict(start=e.start.strftime("%d %b %Y"), unit=e.unit, days=int(e.days), mw=int(e.mw_lost), cause=e.cause) for e in ev.itertuples()],
            latest=dict(eaf=round(wk.eaf.iloc[-1] * 100, 1), avail=round(float(wk.available_mw.iloc[-1])),
                        stock=None if math.isnan(a["stock"]) else round(float(a["stock"]), 1), eff=jsonable(wk.efficiency_pct.iloc[-1]),
                        cost=round(float(wk.cost_r_per_mwh.iloc[-1])), water=round(float(wk.water_l_per_kwh.iloc[-1]), 2)))
    return P


CSS = """
:root{--bg:%(bg)s;--panel:%(panel)s;--line:%(line)s;--text:%(text)s;--muted:%(muted)s;--accent:%(accent)s;
--ok:%(ok)s;--warn:%(warn)s;--bad:%(bad)s;--crit:%(crit)s}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,"Segoe UI",system-ui,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:14px 22px;border-bottom:1px solid var(--line);background:#0d1527;position:sticky;top:0;z-index:5;flex-wrap:wrap}
header h1{font-size:17px;margin:0;letter-spacing:.3px}header .sub{color:var(--muted);font-size:12px}
.badge{margin-left:auto;display:flex;gap:10px;align-items:center}
.sim{background:#3a2a00;color:var(--warn);border:1px solid #6b4e00;padding:3px 9px;border-radius:12px;font-size:11px;font-weight:600}
nav{display:flex;gap:2px;padding:0 16px;border-bottom:1px solid var(--line);background:#0d1527;overflow-x:auto;position:sticky;top:57px;z-index:4}
nav button{background:none;border:0;color:var(--muted);padding:11px 14px;font:inherit;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
nav button.on{color:var(--text);border-color:var(--accent)}nav button:hover{color:var(--text)}
main{padding:18px 22px;max-width:1600px;margin:0 auto}.tab{display:none}.tab.on{display:block}
.grid{display:grid;gap:14px}.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:repeat(3,1fr)}.g32{grid-template-columns:2fr 1fr}
@media(max-width:1000px){.g2,.g3,.g32{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h3{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.kpis.k10{grid-template-columns:repeat(5,1fr)}@media(max-width:1000px){.kpis.k10{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px 13px;border-left:3px solid var(--accent)}
.kpi.ok{border-left-color:var(--ok)}.kpi.warn{border-left-color:var(--warn)}.kpi.bad{border-left-color:var(--bad)}
.kl{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px}.kv{font-size:22px;font-weight:700;margin:3px 0}.ks{color:var(--muted);font-size:11.5px}
iframe.map{width:100%%;height:560px;border:0;border-radius:8px;background:#000}iframe.map.sm{height:480px}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%%;margin-right:5px;vertical-align:middle}
.chip{display:inline-block;padding:1px 9px;border-radius:10px;font-size:11px;font-weight:700;color:#111;margin-right:8px}
.alert{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:9px;background:#0f1729}
.ah{display:flex;align-items:center;gap:4px;flex-wrap:wrap}.rank{color:var(--muted);margin-right:6px;font-weight:700}.pri{margin-left:auto;color:var(--muted);font-size:12px}
.meta{color:var(--muted);font-size:12px;margin:4px 0}details summary{cursor:pointer;color:var(--accent);font-size:12.5px}
details ul{margin:6px 0;padding-left:18px}.rec{color:var(--warn);margin:4px 0 6px}
button.link{background:none;border:1px solid var(--line);color:var(--accent);border-radius:6px;padding:3px 9px;margin:2px 6px 2px 0;cursor:pointer;font:inherit;font-size:12px}
.tbl{width:100%%;border-collapse:collapse;font-size:12.5px}.tbl th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);padding:6px}
.tbl td{border-bottom:1px solid #1a2540;padding:6px}.tbl tr:hover td{background:#16213a}
.clickrows tr{cursor:pointer}
select,input[type=range]{accent-color:var(--accent)}select{background:#0f1729;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font:inherit}
.ctrl{margin-bottom:14px}.ctrl label{display:flex;justify-content:space-between;font-size:12.5px;color:var(--muted);margin-bottom:3px}.ctrl b{color:var(--text)}
.ctrl input[type=range]{width:100%%}
.riskcard{border-left:4px solid var(--muted)}.bar{height:7px;background:#1f2b45;border-radius:4px;overflow:hidden;margin:6px 0}.bar span{display:block;height:100%%}
.note{color:var(--muted);font-size:12px}.obs{color:var(--ok)}.mod{color:var(--warn)}
.filters button{background:#0f1729;border:1px solid var(--line);color:var(--muted);padding:5px 11px;border-radius:14px;margin:0 4px 8px 0;cursor:pointer;font:inherit;font-size:12px}
.filters button.on{color:#111;background:var(--accent);border-color:var(--accent)}
.layers{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.layers div{background:#0f1729;border:1px solid var(--line);border-radius:8px;padding:10px;font-size:12.5px}
""" % THEME

JS = r"""
const STN=__STN__, SCN=__SCN__;
function showTab(id){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id===id));
  document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));
  document.querySelectorAll('#'+id+' iframe[data-srcdoc]').forEach(f=>{if(!f.srcdoc)f.srcdoc=f.dataset.srcdoc;});
  if(id==='station')renderStation(document.getElementById('stSel').value);
  if(id==='scenario')runScenario();
  setTimeout(()=>window.dispatchEvent(new Event('resize')),60);
}
function goStation(n){document.getElementById('stSel').value=n;showTab('station');window.scrollTo(0,0);}
const L={template:'plotly_dark',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Inter,Segoe UI,sans-serif',size:12,color:'#e6edf7'},
  margin:{l:48,r:16,t:36,b:36},legend:{orientation:'h',y:-0.2},xaxis:{gridcolor:'#22304d'},yaxis:{gridcolor:'#22304d'}};
const lay=(t,extra)=>JSON.parse(JSON.stringify(Object.assign({},L,{title:{text:t,font:{size:13}}},extra||{})));
const CFG={responsive:true,displaylogo:false};
function renderStation(n){
  const s=STN[n],m=s.meta,x=s.latest;
  const tone=x.eaf>=70?'ok':x.eaf>=50?'warn':'bad', pt=s.prob>0.5?'bad':s.prob>0.25?'warn':'ok';
  document.getElementById('stKpi').innerHTML=[
    k('Station',n,`${m.type} · ${m.province}`),k('Installed',m.capacity.toLocaleString()+' MW',`${m.units} units · since ${m.commissioned}`),
    k('Available now',x.avail.toLocaleString()+' MW','weekly average','',),k('EAF (latest week)',x.eaf+'%','observed',tone),
    k('Failure prob. 90d',Math.round(s.prob*100)+'%','model estimate',pt),k('Coal stock',x.stock==null?'n/a':x.stock+' days','observed',x.stock!=null&&x.stock<12?'bad':''),
    k('Efficiency',x.eff==null?'n/a':x.eff+'%','thermal'),k('Cost',('R '+x.cost.toLocaleString())+'/MWh',`water ${x.water} L/kWh · cooling ${m.cooling}`)].join('');
  Plotly.react('stEaf',[
    {x:s.weeks,y:s.eaf,name:'Observed EAF %',line:{color:'#2ecc71',width:2}},
    {x:s.fc_dates.concat([...s.fc_dates].reverse()),y:s.fc_hi.concat([...s.fc_lo].reverse()),fill:'toself',fillcolor:'rgba(245,166,35,0.18)',line:{width:0},name:'Forecast 90% band',hoverinfo:'skip'},
    {x:s.fc_dates,y:s.fc,name:'Model forecast',line:{color:'#f5a623',dash:'dash',width:2}}],
    lay('Availability timeline — observed vs model forecast (13 weeks)',{yaxis:{title:'EAF %',gridcolor:'#22304d'}}),CFG);
  Plotly.react('stCmp',[
    {x:s.weeks,y:s.eaf,name:n,line:{color:'#2ecc71',width:2.2}},
    {x:s.weeks,y:s.peer,name:m.type+' average',line:{color:'#36c5f0',dash:'dot'}},
    {x:s.weeks,y:s.fleet,name:'Coal fleet (cap-weighted)',line:{color:'#8da2c0',dash:'dash'}}],lay('Comparison vs peers & fleet (EAF %)'),CFG);
  Plotly.react('stLoss',[
    {x:s.weeks,y:s.uclf,name:'Unplanned (UCLF)',stackgroup:'a',line:{color:'#ff5c5c'}},
    {x:s.weeks,y:s.pclf,name:'Planned (PCLF)',stackgroup:'a',line:{color:'#36c5f0'}}],lay('Capability loss % — planned vs unplanned'),CFG);
  const t2=[{x:s.weeks,y:s.temp,name:'Max temp °C',line:{color:'#f5a623'}}];
  if(s.coal.some(v=>v!=null))t2.push({x:s.weeks,y:s.coal,name:'Coal stock (days)',yaxis:'y2',line:{color:'#9aa5b1'}});
  Plotly.react('stEnv',t2,lay('Environment & fuel',{yaxis2:{overlaying:'y',side:'right',showgrid:false}}),CFG);
  Plotly.react('stGen',[{x:s.weeks,y:s.gen,type:'bar',marker:{color:'#36c5f0'},name:'GWh'}],lay('Energy sent out (GWh / week)'),CFG);
  Plotly.react('stProb',[{type:'bar',orientation:'h',y:s.contrib.map(c=>c[0]).reverse(),x:s.contrib.map(c=>c[1]).reverse(),marker:{color:'#f5a623'}}],
    lay('What drives the failure estimate (log-odds contribution)',{margin:{l:190,r:16,t:36,b:30},showlegend:false,xaxis:{type:'linear',gridcolor:'#22304d'},yaxis:{type:'category',gridcolor:'#22304d'}}),CFG);
  document.getElementById('stEvents').innerHTML='<table class="tbl"><thead><tr><th>Start</th><th>Unit</th><th>Days</th><th>MW lost</th><th>Cause</th></tr></thead><tbody>'+
    s.events.map(e=>`<tr><td>${e.start}</td><td>${e.unit}</td><td>${e.days}</td><td>${e.mw}</td><td>${e.cause}</td></tr>`).join('')+'</tbody></table>';
}
function k(l,v,s,t){return `<div class="kpi ${t||''}"><div class="kl">${l}</div><div class="kv">${v}</div><div class="ks">${s||''}</div></div>`}
function val(id){return +document.getElementById(id).value}
function runScenario(){
  const units=val('scUnits'),dem=val('scDem')/100,ren=val('scRen')/100,cor=document.getElementById('scCor').value,
        coal=document.getElementById('scCoal').checked,gas=document.getElementById('scGas').checked;
  ['scUnits','scDem','scRen'].forEach(i=>document.getElementById(i+'V').textContent=document.getElementById(i).value);
  const corMW=cor?SCN.corridors[cor]:0, coalMW=coal?SCN.low_coal_mw*0.6:0, gasMW=gas?SCN.gas_mw*0.7:0;
  let sup=[],req=[],stage=[],bsup=[],ls=0,mx=0,use=0,mg=0;
  for(let i=0;i<SCN.dates.length;i++){
    const b=SCN.avail[i]+SCN.ipp[i]; bsup.push(b);
    const s=SCN.avail[i]-units-corMW-coalMW-gasMW+SCN.ipp[i]*(1-ren), r=SCN.demand[i]*(1+dem)+SCN.reserve;
    const st=Math.max(0,Math.min(8,Math.ceil((r-s)/1000)));
    sup.push(s);req.push(r);stage.push(st);if(st>0)ls++;mx=Math.max(mx,st);use+=Math.max(0,r-s)*4/1000;mg+=(s-SCN.demand[i]*(1+dem))/(SCN.demand[i]*(1+dem));
  }
  mg/=SCN.dates.length;
  document.getElementById('scKpi').innerHTML=[
    k('Avg reserve margin',(mg*100).toFixed(1)+'%','over next 30 days',mg<0.05?'bad':mg<0.12?'warn':'ok'),
    k('Days needing load-shedding',ls+' / 30','',ls>10?'bad':ls>0?'warn':'ok'),k('Worst stage',mx?'Stage '+mx:'None','',mx>=4?'bad':mx?'warn':'ok'),
    k('Unserved energy (est.)',use.toFixed(0)+' GWh','≈4 peak hours/day',use>50?'bad':use>0?'warn':'ok')].join('');
  Plotly.react('scChart',[
    {x:SCN.dates,y:stage,type:'bar',name:'Load-shedding stage',yaxis:'y2',marker:{color:'rgba(255,92,92,0.5)'}},
    {x:SCN.dates,y:bsup,name:'Baseline supply',line:{color:'#8da2c0',dash:'dot'}},
    {x:SCN.dates,y:sup,name:'Scenario supply',line:{color:'#2ecc71',width:2.5}},
    {x:SCN.dates,y:req,name:'Demand + operating reserve',line:{color:'#f5a623',width:2}}],
    lay('30-day adequacy under scenario (model projection)',{yaxis:{title:'MW',gridcolor:'#22304d'},yaxis2:{overlaying:'y',side:'right',range:[0,8],showgrid:false,title:'Stage'}}),CFG);
  const d0=SCN.demand[0], dD=d0*dem;
  Plotly.react('scWater',[{type:'waterfall',orientation:'v',
    x:['Baseline headroom','Units unavailable','Demand change','Renewable shortfall','Corridor outage','Coal disruption','OCGT fuel limit','Scenario headroom'],
    measure:['absolute','relative','relative','relative','relative','relative','relative','total'],
    y:[bsup[0]-d0-SCN.reserve,-units,-dD,-SCN.ipp[0]*ren,-corMW,-coalMW,-gasMW,0],
    decreasing:{marker:{color:'#ff5c5c'}},increasing:{marker:{color:'#2ecc71'}},totals:{marker:{color:'#36c5f0'}}}],
    lay('Day-1 headroom bridge (MW above demand + reserve)',{showlegend:false,xaxis:{type:'category',gridcolor:'#22304d'}}),CFG);
}
function filterSig(cat,btn){
  document.querySelectorAll('.filters button').forEach(b=>b.classList.toggle('on',b===btn));
  document.querySelectorAll('#sigTable tbody tr').forEach(r=>{r.style.display=(cat==='All'||r.dataset.cat===cat)?'':'none'});
}
document.addEventListener('DOMContentLoaded',()=>showTab('overview'));
"""


def build_html(ctx):
    nat, dims, comp, alerts, F = ctx["nat"], ctx["dims"], ctx["composite"], ctx["alerts"], ctx["F"]
    lvl = risk_level(comp)
    last = nat.iloc[-1]
    l7 = nat.tail(7)
    supply = last.eskom_available_mw + last.ipp_renewable_peak_mw
    rm = (supply - last.peak_demand_mw) / last.peak_demand_mw
    ren_share = nat.tail(7).ipp_renewable_gwh.sum() / nat.tail(7).energy_demand_gwh.sum()
    emis = ctx["ops"][ctx["ops"].date > TODAY - pd.Timedelta(days=7)].eval("energy_gwh * co2_t_per_mwh").sum()  # kt
    water_idx = scale(ctx["weather"][(ctx["weather"].date == TODAY) & ctx["weather"].province.isin(["Mpumalanga", "Limpopo", "Free State"])].drought_idx.min(), 0.5, -2)
    out_stage = max(ctx["outlook_stage"][:7]) if ctx["outlook_stage"] else 0

    kp = "".join([
        kpi("Available capacity", f"{supply:,.0f} MW", f"Eskom {last.eskom_available_mw:,.0f} + IPP {last.ipp_renewable_peak_mw:,.0f}"),
        kpi("Peak demand (today)", f"{last.peak_demand_mw:,.0f} MW", f"7-day max {l7.peak_demand_mw.max():,.0f}"),
        kpi("Reserve margin", f"{rm:.1%}", "target ≥ 15%", "ok" if rm > 0.15 else "warn" if rm > 0.06 else "bad"),
        kpi("Unplanned loss (UCLF)", f"{last.uclf_mw:,.0f} MW", f"{last.uclf_mw / STN.capacity_mw.sum():.1%} of installed", "bad" if last.uclf_mw > 13000 else "warn"),
        kpi("Planned maintenance", f"{last.pclf_mw:,.0f} MW", f"{last.pclf_mw / STN.capacity_mw.sum():.1%} of installed"),
        kpi("System risk level", f'<span style="color:{LEVEL_COLOR[lvl]}">{lvl}</span>', f"composite score {comp:.0f}/100", "bad" if comp >= 60 else "warn"),
        kpi("Renewable contribution", f"{ren_share:.1%}", "IPP share of energy, 7d"),
        kpi("CO₂ emissions (7d)", f"{emis / 1000:,.2f} Mt", "thermal fleet"),
        kpi("Water stress index", f"{water_idx:.0f}/100", "coal-province drought", "bad" if water_idx > 60 else "warn" if water_idx > 35 else "ok"),
        kpi("Load-shedding outlook", f"Stage {out_stage}" if out_stage else "None", "max, next 7 days (model)", "bad" if out_stage >= 3 else "warn" if out_stage else "ok"),
    ])
    top5 = "".join(alert_card(a, i + 1) for i, a in enumerate(alerts.head(5).to_dict("records")))
    all_alerts = "".join(alert_card(a, i + 1) for i, a in enumerate(alerts.to_dict("records")))
    ov = ctx["events"][ctx["events"].start >= TODAY - pd.Timedelta(days=2)][["station", "unit", "mw_lost", "cause"]].head(10)
    ov_html = table(ov.rename(columns={"station": "Station", "unit": "Unit", "mw_lost": "MW", "cause": "Cause"})) if len(ov) else '<p class="note">No new trips in the last 48 hours.</p>'

    legend = (f'<div class="legend"><span><i style="background:{THEME["ok"]}"></i>EAF ≥ 70%</span><span><i style="background:{THEME["warn"]}"></i>50–70%</span>'
              f'<span><i style="background:{THEME["bad"]}"></i>&lt; 50%</span><span><i style="border:2px solid {THEME["crit"]};background:none"></i>active alert</span>'
              f'<span><i style="background:{THEME["accent"]}"></i>demand centre</span><span>column height = available MW · line colour = corridor loading</span></div>')

    # risk cards
    rc = []
    for name, d in sorted(dims.items(), key=lambda kv: -kv[1]["score"]):
        lv = risk_level(d["score"])
        drv = "".join(f"<tr><td>{esc(x['label'])}</td><td>{esc(x['value'])}</td><td>{x['weight']:.0%}</td><td>{x['contribution']:.0f}</td></tr>" for x in d["drivers"])
        rc.append(f"""<div class="card riskcard" style="border-left-color:{LEVEL_COLOR[lv]}"><div class="ah">{chip(lv)}<b>{esc(name)}</b>
          <span class="pri" style="font-size:20px;color:var(--text)">{d['score']:.0f}</span></div>
          <div class="bar"><span style="width:{d['score']}%;background:{LEVEL_COLOR[lv]}"></span></div>
          <details><summary>What is driving this score?</summary><table class="tbl"><thead><tr><th>Driver</th><th>Current value</th><th>Weight</th><th>Points</th></tr></thead><tbody>{drv}</tbody></table>
          <div class="note">Evidence sources: {esc(', '.join(d['sources']))}</div></details></div>""")

    # station league
    lg = ctx["league"]
    league = "".join(f'<tr onclick="goStation(\'{esc(r.station)}\')"><td>{esc(r.station)}</td><td>{esc(r.type)}</td><td>{esc(r.province)}</td>'
                     f'<td>{r.capacity_mw:,}</td><td>{r.eaf:.0f}%</td><td style="color:{THEME["bad"] if r.prob > 0.5 else THEME["warn"] if r.prob > 0.25 else THEME["ok"]}">{r.prob:.0%}</td></tr>'
                     for r in lg.itertuples())
    opts = "".join(f'<optgroup label="{esc(p)}">' + "".join(f'<option{" selected" if s == ctx["default_station"] else ""}>{esc(s)}</option>' for s in g.station) + "</optgroup>"
                   for p, g in STN.sort_values("station").groupby("province"))
    cor_opts = '<option value="">None</option>' + "".join(f'<option value="{esc(c)}">{esc(c)}</option>' for c in COR.corridor)

    sig = ctx["signals"].sort_values("date", ascending=False)
    sig_rows = "".join(f'<tr data-cat="{esc(r.category)}"><td>{r.date:%d %b}</td><td>{esc(r.category)}</td><td>{esc(r.domain)}</td><td>{esc(r.headline)}</td>'
                       f'<td>{r.impact}</td><td>{r.likelihood:.0%}</td><td>{esc(r.source_type)}</td><td>{esc(r.reliability)}</td></tr>' for r in sig.itertuples())
    cat_btns = "".join(f'<button class="{"on" if c == "All" else ""}" onclick="filterSig(\'{c}\',this)">{c}</button>' for c in ["All", "Threat", "Opportunity", "Trend", "Uncertainty"])
    counts = sig.category.value_counts()
    sig_k = "".join(kpi(c, int(counts.get(c, 0)), "signals tracked", t) for c, t in [("Threat", "bad"), ("Opportunity", "ok"), ("Trend", ""), ("Uncertainty", "warn")])

    cat = ctx["catalogue"].rename(columns={"dataset": "Dataset", "source": "Source system", "ingest": "Ingestion", "frequency": "Frequency",
                                           "reliability": "Reliability", "records": "Records", "last_update": "Last update", "qc_pass_pct": "QC pass %"})
    qc = ctx["qc"]

    body = f"""
<header><div><h1>⚡ Eskom System Intelligence</h1><div class="sub">National electricity-system intelligence · prototype v1 · as of {TODAY:%A %d %B %Y}</div></div>
<div class="badge"><span class="sim">SIMULATED DATA — NOT REAL ESKOM FIGURES</span>{chip(lvl)}<span class="note">risk {comp:.0f}/100</span></div></header>
<nav>{''.join(f'<button data-t="{t}" onclick="showTab(\'{t}\')">{lab}</button>' for t, lab in [("overview", "Executive view"), ("map", "System map"), ("station", "Station intelligence"), ("risk", "Risk engine"), ("scenario", "Scenarios"), ("warning", "Early warning"), ("analytics", "Analytics"), ("external", "External scan"), ("lineage", "Data & lineage")])}</nav>
<main>
<section id="overview" class="tab">
  <div class="kpis k10">{kp}</div>
  <div class="grid g32">
    <div class="card"><h3>National electricity system — live view</h3><iframe class="map" data-srcdoc="{ctx['map1']}"></iframe>{legend}</div>
    <div><div class="card"><h3>Five issues requiring attention</h3>{top5}</div></div>
  </div>
  <div class="grid g2" style="margin-top:14px">
    <div class="card"><h3>Generation outlook — next 14 days</h3>{F['outlook']}<div class="note"><span class="mod">Model projection</span> — see Scenarios to stress-test.</div></div>
    <div class="card"><h3>Significant events in the last 48 hours</h3>{ov_html}</div>
  </div>
</section>

<section id="map" class="tab">
  <div class="grid g2">
    <div class="card"><h3>Generation, transmission & demand</h3><iframe class="map sm" data-srcdoc="{ctx['map1']}"></iframe>{legend}</div>
    <div class="card"><h3>Environmental exposure & station failure risk</h3><iframe class="map sm" data-srcdoc="{ctx['map2']}"></iframe>
      <div class="legend"><span>Heat = combined heat + drought stress</span><span><i style="background:{THEME['bad']}"></i>failure prob &gt; 50%</span><span><i style="background:{THEME['warn']}"></i>25–50%</span><span><i style="background:{THEME['accent']}"></i>provincial weather node</span></div></div>
  </div>
  <div class="card" style="margin-top:14px">{F['corridors']}</div>
</section>

<section id="station" class="tab">
  <div class="card" style="margin-bottom:14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap"><b>Select power station</b><select id="stSel" onchange="renderStation(this.value)">{opts}</select>
    <span class="note"><span class="obs">■ observed data</span> &nbsp; <span class="mod">■ model-generated estimates</span></span></div>
  <div class="kpis" id="stKpi"></div>
  <div class="grid g2"><div class="card"><div id="stEaf" style="height:330px"></div></div><div class="card"><div id="stCmp" style="height:330px"></div></div>
  <div class="card"><div id="stLoss" style="height:300px"></div></div><div class="card"><div id="stEnv" style="height:300px"></div></div>
  <div class="card"><div id="stGen" style="height:280px"></div></div><div class="card"><div id="stProb" style="height:280px"></div><div class="note">Heuristic logistic model — illustrative, not trained on real outcomes.</div></div></div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>Recent operational events</h3><div id="stEvents"></div></div>
  <div class="card"><h3>Fleet league table (click a row)</h3><table class="tbl clickrows"><thead><tr><th>Station</th><th>Type</th><th>Province</th><th>MW</th><th>EAF 90d</th><th>Fail prob</th></tr></thead><tbody>{league}</tbody></table></div></div>
</section>

<section id="risk" class="tab">
  <div class="grid g2"><div class="card">{F['radar']}</div><div class="card">{F['risktrend']}</div></div>
  <div class="grid g2" style="margin-top:14px">{''.join(rc)}</div>
  <p class="note">Each dimension score = Σ (weight × driver sub-score). Sub-scores map the current value between a benign and a severe threshold. Composite = weighted average of dimensions. Scores are recomputed weekly as new data arrives.</p>
</section>

<section id="scenario" class="tab">
  <div class="grid g32">
    <div><div class="kpis" id="scKpi"></div><div class="card"><div id="scChart" style="height:380px"></div></div><div class="card" style="margin-top:14px"><div id="scWater" style="height:330px"></div></div></div>
    <div class="card"><h3>Scenario levers</h3>
      <div class="ctrl"><label>Additional units unavailable <b><span id="scUnitsV">0</span> MW</b></label><input id="scUnits" type="range" min="0" max="8000" step="250" value="0" oninput="runScenario()"></div>
      <div class="ctrl"><label>Demand change <b><span id="scDemV">0</span> %</b></label><input id="scDem" type="range" min="-8" max="15" step="1" value="0" oninput="runScenario()"></div>
      <div class="ctrl"><label>Renewables below forecast <b><span id="scRenV">0</span> %</b></label><input id="scRen" type="range" min="0" max="100" step="5" value="0" oninput="runScenario()"></div>
      <div class="ctrl"><label>Transmission corridor unavailable</label><select id="scCor" onchange="runScenario()" style="width:100%">{cor_opts}</select></div>
      <div class="ctrl"><label><span><input id="scCoal" type="checkbox" onchange="runScenario()"> Coal-supply disruption at low-stock stations</span></label></div>
      <div class="ctrl"><label><span><input id="scGas" type="checkbox" onchange="runScenario()"> OCGT diesel supply constrained</span></label></div>
      <p class="note">Baseline = 30-day model projection of demand (seasonal, last-year shape × recent growth) and available capacity (recent level + trend). Stage = ⌈shortfall / 1 000 MW⌉. A structured view of consequences, not a precise forecast.</p></div>
  </div>
</section>

<section id="warning" class="tab">
  <div class="grid g32"><div class="card"><h3>Ranked alerts (impact × probability × urgency)</h3>{all_alerts}</div>
  <div><div class="card">{F['anomaly']}</div><div class="card" style="margin-top:14px"><h3>How alerts are generated</h3><ul class="note">
    <li>Station UCLF z-score vs 12-month baseline &gt; 2σ</li><li>Coal stockpile below 12 days</li><li>Corridor loading &gt; 86% with fault history</li>
    <li>Heatwave (Tmax ≥ 35 °C) in last 5 days</li><li>Clusters of related external signals</li><li>Compound rule: ≥2 anomalous stations in a province + heat/drought</li></ul></div></div></div>
</section>

<section id="analytics" class="tab">
  <div class="card">{F['balance']}</div>
  <div class="grid g2" style="margin-top:14px"><div class="card">{F['mix']}</div><div class="card">{F['profile']}</div>
  <div class="card">{F['causes']}</div><div class="card">{F['dur']}</div></div>
  <div class="card" style="margin-top:14px">{F['eafheat']}</div>
  <div class="grid g2" style="margin-top:14px"><div class="card">{F['fleet']}</div><div class="card">{F['corr']}</div></div>
</section>

<section id="external" class="tab">
  <div class="kpis">{sig_k}</div>
  <div class="grid g2"><div class="card">{F['signals']}</div><div class="card"><h3>Macro & market indicators</h3>{F['econ']}</div></div>
  <div class="card" style="margin-top:14px"><h3>Environmental scanning feed</h3><div class="filters">{cat_btns}</div>
    <table class="tbl" id="sigTable"><thead><tr><th>Date</th><th>Category</th><th>Domain</th><th>Signal</th><th>Impact</th><th>Likelihood</th><th>Source</th><th>Reliability</th></tr></thead><tbody>{sig_rows}</tbody></table></div>
</section>

<section id="lineage" class="tab">
  <div class="card"><h3>Platform architecture</h3><div class="layers">
    <div><b>1 · Data layer</b><br/>APIs, SCADA, historians, satellite, spreadsheets, manual submissions, web scan → validation → cleaning → standardisation → common store with source, date, location, reliability & frequency on every record.</div>
    <div><b>2 · Analytics layer</b><br/>KPIs, anomaly detection (z-scores), trend & seasonal forecasts, heuristic failure model, 8-dimension risk engine, correlation analysis.</div>
    <div><b>3 · Intelligence layer</b><br/>Ranked early warnings, executive issues, scenario testing, drill-down from nation → province → station → unit → event, with evidence links back to source datasets.</div></div></div>
  <div class="card" style="margin-top:14px"><h3>Source catalogue</h3>{table(cat)}</div>
  <div class="card" style="margin-top:14px"><h3>Data-quality pipeline — example run (weather feed)</h3>
    <div class="kpis">{kpi("Rows ingested", f"{qc['rows']:,}")}{kpi("Outliers removed", qc['outliers_removed'], "rolling-median rule (±12 °C)", "warn")}{kpi("Values gap-filled", qc['missing_filled'], "linear interpolation")}{kpi("QC pass rate", f"{qc['passed_pct']}%", "", "ok")}</div>
    <p class="note">All datasets are also exported as CSV to <code>output/data/</code> for inspection in VS Code, Excel or pandas.</p></div>
</section>
</main>"""
    if OFFLINE:
        import plotly as _pl
        plotly_tag = "<script>" + (Path(_pl.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8") + "</script>"
    else:
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    js = JS.replace("__STN__", json.dumps(ctx["stn_payload"])).replace("__SCN__", json.dumps(ctx["outlook"]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eskom System Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
{plotly_tag}<style>{CSS}</style></head>
<body>{body}<script>{js}</script></body></html>"""


# =============================================================================
# 5. PIPELINE
# =============================================================================
def main(open_browser=True):
    print("① Data layer — simulating & cleaning sources …")
    weather, qc = clean_weather(sim_weather())
    ops, events = sim_stations(weather)
    econ = sim_economics()
    nat, ops = sim_demand_and_balance(ops, weather, econ)
    grid = sim_grid(weather, nat)
    subs = sim_substations()
    signals = sim_signals()

    print("② Analytics layer — risk engine, forecasts, alerts …")
    dims, comp = risk_engine(TODAY, nat, ops, grid, weather, econ, signals)
    hist = []
    for d in pd.date_range(TODAY - pd.Timedelta(weeks=25), TODAY, freq="7D"):
        dd, cc = risk_engine(d, nat, ops, grid, weather, econ, signals)
        hist.append({"date": d, "composite": cc, **{k: v["score"] for k, v in dd.items()}})
    risk_hist = pd.DataFrame(hist)
    st_an = station_analytics(ops, events)
    alerts = build_alerts(ops, nat, grid, weather, signals, st_an, events)
    outlook = forward_outlook(nat, ops)
    o_sup = np.array(outlook["avail"]) + np.array(outlook["ipp"])
    o_stage = np.clip(np.ceil((np.array(outlook["demand"]) + OP_RESERVE_MW - o_sup) / 1000), 0, 8).astype(int).tolist()
    catalogue = data_catalogue(qc, dict(ops=ops, events=events, nat=nat, grid=grid, subs=subs, weather=weather,
                                        econ=econ, signals=signals, dc=DC))

    print("③ Intelligence layer — maps, charts, dashboard …")
    alert_st = set(s for lst in alerts.stations for s in lst)
    map1 = deck_srcdoc(build_system_map(ops, grid, subs, st_an, alert_st))
    map2 = deck_srcdoc(build_env_map(weather, st_an))
    F = analytics_figs(nat, ops, events, grid, weather, econ, signals, risk_hist, dims, outlook)
    eaf90 = ops[ops.date > TODAY - pd.Timedelta(days=90)].groupby("station").eaf.mean() * 100
    league = STN.assign(eaf=STN.station.map(eaf90), prob=STN.station.map(lambda s: st_an[s]["prob"])).sort_values("prob", ascending=False)
    ctx = dict(nat=nat, ops=ops, events=events, weather=weather, signals=signals, dims=dims, composite=comp, alerts=alerts,
               F=F, map1=map1, map2=map2, outlook=outlook, outlook_stage=o_stage, league=league, catalogue=catalogue, qc=qc,
               stn_payload=station_payload(ops, events, st_an),
               default_station=alerts.stations.explode().dropna().iloc[0] if len(alert_st) else "Kendal")

    OUT.mkdir(exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    for name, df in dict(station_operations=ops, outage_events=events, national_balance=nat, grid_corridors=grid,
                         substations=subs, weather=weather, economics=econ, external_signals=signals,
                         risk_history=risk_hist, alerts=alerts.drop(columns=["evidence", "stations"]),
                         data_catalogue=catalogue, stations=STN).items():
        df.to_csv(OUT / "data" / f"{name}.csv", index=False)
    path = OUT / "eskom_intelligence_v1.html"
    path.write_text(build_html(ctx), encoding="utf-8")
    print(f"✔ Dashboard: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  System risk {comp:.0f}/100 ({risk_level(comp)}) · {len(alerts)} alerts · "
          f"load-shedding days (30d): {int((nat.tail(30).loadshed_stage > 0).sum())}")
    if open_browser:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the Eskom System Intelligence prototype dashboard.")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    ap.add_argument("--offline", action="store_true",
                    help="embed plotly.js and deck.gl in the HTML (bigger file; map tiles still need internet)")
    args = ap.parse_args()
    OFFLINE = args.offline
    main(open_browser=not args.no_open)
