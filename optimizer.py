"""
optimizer.py
============
Core optimization engine for battery energy storage system (BESS) arbitrage,
simulating a realistic day-ahead-market trader.

WHY DAY-AHEAD MATTERS (read this first)
----------------------------------------
A real BESS trader never sees the full future price curve. Day-ahead markets
auction off exactly one day's worth of prices at a time (published as a
single block, once, ahead of the delivery day). So a trader standing at the
start of day D only ever knows day D's own price curve -- never day D+1's,
D+2's, etc. This module has two solving modes that reflect that:

    optimize(...)          Solves ONE horizon in a single LP shot, given
                            perfect knowledge of every price in that horizon.
                            Correct and exact when you feed it a single
                            day's data (that IS what day-ahead knowledge
                            looks like). If you feed it many days at once,
                            it becomes a THEORETICAL UPPER BOUND -- useful
                            as a reference number, but not an achievable
                            trading strategy, because it silently assumes
                            foresight the trader would not actually have.

    simulate_day_ahead(...) The realistic mode. Solves one LP per calendar
                            day, using ONLY that day's own data, then carries
                            the battery's ACTUAL resulting state-of-charge
                            and net power into the next day's solve. This is
                            what "simulate a battery trader operating on the
                            day-ahead market" means, and it is the function
                            the GUI uses by default.

Within each single day, the schedule produced is still the exact, certified
LP optimum (HiGHS solver, not a heuristic) -- "true optimal" now means
"optimal given what was actually knowable at the time", which is the
correct standard for a realistic trader simulation.

THE LP ITSELF (applies identically inside each day-ahead block or in a
single-shot optimize() call)
------------------------------------------------------------------------
Decision variables, one triple per timestep t = 0..N-1:
    charge[t]     kW drawn from the grid to charge the battery
    discharge[t]  kW delivered to the grid by discharging the battery
    soc[t]        kWh stored in the battery at the END of timestep t

Objective (maximized, i.e. we minimize its negative):
    profit = sum_t ( discharge[t]*price[t]*dt_h
                      - charge[t]*price[t]*dt_h
                      - charge[t]*fee_per_kwh*dt_h )
             - fixed_cost_allocated

Constraints:
    1. State-of-charge balance (equality), with charge/discharge efficiency
       losses applied:
           soc[t] = soc[t-1] + charge[t]*eff_c*dt_h - discharge[t]/eff_d*dt_h
       (soc[-1] = the battery's actual carried-in state of charge)
    2. soc bounded by DoD:  capacity*(1-DoD) <= soc[t] <= capacity
    3. 0 <= charge[t] <= min(max_charge_kw, available_power[t])
       0 <= discharge[t] <= min(max_discharge_kw, available_power[t])
       (available_power[t] also only known day-by-day, same as price)
    4. Ramp-rate limit (same % for charging and discharging, applied to the
       NET power discharge[t]-charge[t], continuous across day boundaries
       via the carried-in net power):
           |netP[t] - netP[t-1]| <= ramp_pct_per_min * dt_min * max_power_ref
    5. (optional) throughput cap for degradation, tracked CUMULATIVELY
       across the whole simulation so the annual budget is honored in
       aggregate rather than rigidly per day.

No binary/integer variables are needed anywhere: if the charging fee plus
round-trip losses exceed the achievable price spread at a given time, the
LP simply leaves charge=discharge=0 there on its own. This is what
implicitly enforces a "minimum trade threshold" without sacrificing
exactness or requiring a slower Mixed-Integer LP.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog


# --------------------------------------------------------------------------
# Input data loading
# --------------------------------------------------------------------------

def load_timeseries_csv(path: str, value_col: Optional[str] = None) -> pd.DataFrame:
    """
    Load a CSV with a timestamp column and one value column.
    Handles both international (comma-separated, decimal point) and German
    (semicolon-separated, decimal comma, DD.MM.YYYY timestamps) files; the
    separator is detected automatically.
    Returns a DataFrame with columns ['timestamp', 'value'], sorted by time.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    if ";" in header:
        sep = ";"
    elif "\t" in header:
        sep = "\t"
    else:
        sep = ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    ts_col = None
    for key in ("timestamp", "zeitstempel", "time", "zeit", "date", "datum"):
        if key in cols_lower:
            ts_col = cols_lower[key]
            break
    if ts_col is None:
        ts_col = df.columns[0]

    if value_col is None:
        remaining = [c for c in df.columns if c != ts_col]
        if not remaining:
            raise ValueError(f"CSV '{path}' has no value column besides '{ts_col}'.")
        value_col = remaining[0]

    values = df[value_col]
    if not pd.api.types.is_numeric_dtype(values):
        s = values.astype(str).str.strip()
        if s.str.contains(",").any():
            # German number format: "1.234,50" -> 1234.50
            s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        values = s

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col], dayfirst=True),
        "value": pd.to_numeric(values, errors="coerce"),
    }).dropna()
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Parameters / results
# --------------------------------------------------------------------------

@dataclass
class BatteryParams:
    capacity_kwh: float            # nameplate energy capacity
    dod_pct: float                 # depth of discharge, 0-1 (e.g. 0.8 = 80%)
    max_charge_kw: float           # max charging power
    max_discharge_kw: float        # max discharging power
    charge_eff: float = 0.95       # fraction of drawn energy actually stored
    discharge_eff: float = 0.95    # fraction of stored energy actually delivered
    ramp_pct_per_min: float = 0.10 # fraction of reference max power per minute
    fee_per_kwh_charged: float = 0.0   # extra fee per kWh charged (grid side)
    fixed_annual_cost: float = 0.0     # e.g. O&M, amortized capex, etc.
    initial_soc_kwh: Optional[float] = None  # defaults to SoC_min if None
    max_annual_throughput_kwh: Optional[float] = None  # degradation cap, optional
    price_currency: str = "EUR"

    def soc_bounds(self):
        soc_min = self.capacity_kwh * (1.0 - self.dod_pct)
        soc_max = self.capacity_kwh
        return soc_min, soc_max


@dataclass
class OptimizationResult:
    schedule: pd.DataFrame     # timestamp, price, charge_kw, discharge_kw, soc_kwh, net_kw
    total_profit: float
    total_revenue: float
    total_charge_cost: float
    total_fees: float
    fixed_cost_allocated: float
    total_energy_charged_kwh: float
    total_energy_discharged_kwh: float
    status: str
    solver_message: str


@dataclass
class DayAheadSimulationResult:
    schedule: pd.DataFrame          # full concatenated realized schedule, all days
    daily_breakdown: pd.DataFrame   # one row per simulated day
    total_profit: float
    total_revenue: float
    total_charge_cost: float
    total_fees: float
    fixed_cost_allocated: float
    total_energy_charged_kwh: float
    total_energy_discharged_kwh: float
    n_days: int
    status: str
    messages: List[str]             # warnings, e.g. days skipped for insufficient data


# --------------------------------------------------------------------------
# Core LP builder / solver -- one horizon, one shot
# --------------------------------------------------------------------------

def optimize(
    prices: pd.DataFrame,
    params: BatteryParams,
    grid_limit: Optional[pd.DataFrame] = None,
    initial_soc_kwh_override: Optional[float] = None,
    initial_net_power_kw: float = 0.0,
    throughput_cap_kwh_override: Optional[float] = None,
    fixed_cost_override: Optional[float] = None,
) -> OptimizationResult:
    """
    Solve the LP over exactly the timesteps given in `prices`, assuming full
    knowledge of that whole horizon (this is the day-ahead-correct thing to
    do IF `prices` covers a single day-ahead block; it becomes a theoretical
    upper bound if you pass it many days at once -- see module docstring).

    prices: DataFrame with columns ['timestamp', 'value'] (value = price per kWh)
    grid_limit: optional DataFrame with columns ['timestamp', 'value']
                (value = max available power in kW at that timestamp). Safe
                to pass a multi-day DataFrame even for a single-day solve --
                only this call's own timestamps are used (reindexed), so no
                future information leaks into the LP.
    initial_soc_kwh_override: state of charge (kWh) the battery actually has
                at the start of this horizon. Falls back to
                params.initial_soc_kwh (or SoC_min) if not given.
    initial_net_power_kw: the battery's actual net power (discharge-charge)
                immediately before this horizon starts, so the ramp-rate
                constraint at t=0 is continuous across a day boundary
                instead of assuming the battery was idle.
    throughput_cap_kwh_override: an explicit kWh cap for this horizon's
                total charge+discharge throughput. Falls back to
                params.max_annual_throughput_kwh prorated by horizon length
                if not given.
    fixed_cost_override: an explicit fixed-cost amount to allocate to this
                horizon. Falls back to params.fixed_annual_cost prorated by
                horizon length if not given. Rolling-horizon simulation
                passes 0.0 here and allocates the fixed cost once, at the
                end, over the full simulated period instead.
    """
    n = len(prices)
    if n < 2:
        raise ValueError("Need at least 2 price timesteps to optimize.")

    prices = prices.sort_values("timestamp").reset_index(drop=True)
    timestamps = prices["timestamp"].to_numpy()
    price = prices["value"].to_numpy(dtype=float)

    # timestep length in hours (assume roughly uniform spacing; use median gap)
    deltas = np.diff(timestamps).astype("timedelta64[s]").astype(float)
    dt_seconds = float(np.median(deltas))
    dt_hours = dt_seconds / 3600.0
    dt_minutes = dt_seconds / 60.0

    horizon_days = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, "D") \
                   + dt_hours / 24.0  # include the final step's own span

    # --- per-timestep power caps (apply optional available-power limit) ---
    max_c = np.full(n, params.max_charge_kw, dtype=float)
    max_d = np.full(n, params.max_discharge_kw, dtype=float)
    if grid_limit is not None and len(grid_limit) > 0:
        gl = grid_limit.set_index("timestamp")["value"].reindex(
            prices["timestamp"]).ffill().bfill().to_numpy(dtype=float)
        max_c = np.minimum(max_c, gl)
        max_d = np.minimum(max_d, gl)

    soc_min, soc_max = params.soc_bounds()
    if initial_soc_kwh_override is not None:
        initial_soc = initial_soc_kwh_override
    elif params.initial_soc_kwh is not None:
        initial_soc = params.initial_soc_kwh
    else:
        initial_soc = soc_min
    initial_soc = float(np.clip(initial_soc, soc_min, soc_max))

    # --- variable layout: [charge(0..n-1), discharge(0..n-1), soc(0..n-1)] ---
    n_vars = 3 * n
    idx_c = np.arange(0, n)
    idx_d = np.arange(n, 2 * n)
    idx_s = np.arange(2 * n, 3 * n)

    # --- objective: minimize (cost - revenue) ---
    c_obj = np.zeros(n_vars)
    c_obj[idx_c] = dt_hours * (price + params.fee_per_kwh_charged)
    c_obj[idx_d] = -dt_hours * price
    # soc has no direct objective coefficient

    # --- bounds ---
    bounds = [(0.0, None)] * n_vars
    for t in range(n):
        bounds[idx_c[t]] = (0.0, max(0.0, max_c[t]))
        bounds[idx_d[t]] = (0.0, max(0.0, max_d[t]))
        bounds[idx_s[t]] = (soc_min, soc_max)

    # --- equality constraints: SoC balance ---
    eq_rows, eq_cols, eq_data = [], [], []
    b_eq = np.zeros(n)
    row = 0
    for t in range(n):
        eq_rows += [row, row, row]
        eq_cols += [idx_s[t], idx_c[t], idx_d[t]]
        eq_data += [1.0, -params.charge_eff * dt_hours, dt_hours / params.discharge_eff]
        if t == 0:
            b_eq[row] = initial_soc
        else:
            eq_rows.append(row)
            eq_cols.append(idx_s[t - 1])
            eq_data.append(-1.0)
            b_eq[row] = 0.0
        row += 1
    A_eq = sparse.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(n, n_vars))

    # --- inequality constraints: ramp rate on net power ---
    max_power_ref = max(params.max_charge_kw, params.max_discharge_kw)
    ramp_limit = max(params.ramp_pct_per_min * dt_minutes * max_power_ref, 0.0)

    ub_rows, ub_cols, ub_data = [], [], []
    b_ub = []
    row = 0
    for t in range(n):
        # netP[t] - netP[t-1] <= ramp_limit   and   -(netP[t]-netP[t-1]) <= ramp_limit
        # netP[t] = discharge[t] - charge[t]
        if t == 0:
            # previous net power is a KNOWN CONSTANT (carried in from the
            # previous day's actual realized schedule), not a variable
            ub_rows += [row]; ub_cols += [idx_d[t]]; ub_data += [1.0]
            ub_rows += [row]; ub_cols += [idx_c[t]]; ub_data += [-1.0]
            b_ub.append(ramp_limit + initial_net_power_kw)
            row += 1
            ub_rows += [row]; ub_cols += [idx_d[t]]; ub_data += [-1.0]
            ub_rows += [row]; ub_cols += [idx_c[t]]; ub_data += [1.0]
            b_ub.append(ramp_limit - initial_net_power_kw)
            row += 1
        else:
            ub_rows += [row, row, row, row]
            ub_cols += [idx_d[t], idx_c[t], idx_d[t - 1], idx_c[t - 1]]
            ub_data += [1.0, -1.0, -1.0, 1.0]
            b_ub.append(ramp_limit)
            row += 1
            ub_rows += [row, row, row, row]
            ub_cols += [idx_d[t], idx_c[t], idx_d[t - 1], idx_c[t - 1]]
            ub_data += [-1.0, 1.0, 1.0, -1.0]
            b_ub.append(ramp_limit)
            row += 1

    # --- optional: throughput cap (degradation) ---
    if throughput_cap_kwh_override is not None:
        cap = max(0.0, throughput_cap_kwh_override)
        for t in range(n):
            ub_rows += [row, row]
            ub_cols += [idx_c[t], idx_d[t]]
            ub_data += [dt_hours, dt_hours]
        b_ub.append(cap)
        row += 1
    elif params.max_annual_throughput_kwh is not None:
        cap = params.max_annual_throughput_kwh * (horizon_days / 365.25)
        for t in range(n):
            ub_rows += [row, row]
            ub_cols += [idx_c[t], idx_d[t]]
            ub_data += [dt_hours, dt_hours]
        b_ub.append(cap)
        row += 1

    n_ub_rows = row
    A_ub = sparse.csr_matrix((ub_data, (ub_rows, ub_cols)), shape=(n_ub_rows, n_vars))
    b_ub = np.array(b_ub, dtype=float)

    res = linprog(
        c_obj,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return OptimizationResult(
            schedule=pd.DataFrame(),
            total_profit=float("nan"),
            total_revenue=float("nan"),
            total_charge_cost=float("nan"),
            total_fees=float("nan"),
            fixed_cost_allocated=float("nan"),
            total_energy_charged_kwh=float("nan"),
            total_energy_discharged_kwh=float("nan"),
            status="FAILED",
            solver_message=res.message,
        )

    x = res.x
    charge = x[idx_c]
    discharge = x[idx_d]
    soc = x[idx_s]

    schedule = pd.DataFrame({
        "timestamp": prices["timestamp"].to_numpy(),
        "price": price,
        "charge_kw": charge,
        "discharge_kw": discharge,
        "net_kw": discharge - charge,
        "soc_kwh": soc,
    })

    revenue = float(np.sum(discharge * price * dt_hours))
    charge_cost = float(np.sum(charge * price * dt_hours))
    fees = float(np.sum(charge * params.fee_per_kwh_charged * dt_hours))
    if fixed_cost_override is not None:
        fixed_cost_allocated = fixed_cost_override
    else:
        fixed_cost_allocated = params.fixed_annual_cost * (horizon_days / 365.25)
    total_profit = revenue - charge_cost - fees - fixed_cost_allocated

    return OptimizationResult(
        schedule=schedule,
        total_profit=total_profit,
        total_revenue=revenue,
        total_charge_cost=charge_cost,
        total_fees=fees,
        fixed_cost_allocated=fixed_cost_allocated,
        total_energy_charged_kwh=float(np.sum(charge * dt_hours)),
        total_energy_discharged_kwh=float(np.sum(discharge * dt_hours)),
        status="OPTIMAL",
        solver_message=res.message,
    )


# --------------------------------------------------------------------------
# Realistic day-ahead rolling-horizon simulation
# --------------------------------------------------------------------------

def simulate_day_ahead(
    prices: pd.DataFrame,
    params: BatteryParams,
    grid_limit: Optional[pd.DataFrame] = None,
) -> DayAheadSimulationResult:
    """
    Realistically simulate a battery trader operating on the day-ahead
    market: solve one exact LP per calendar day, using ONLY that day's own
    price (and available-power) data, carrying the battery's ACTUAL
    resulting state-of-charge and net power forward into the next day.

    This is the mode that matches how day-ahead trading really works and is
    what the GUI runs by default. See the module docstring for why this
    differs from optimize() run across many days at once.
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 price timesteps to simulate.")

    prices = prices.sort_values("timestamp").reset_index(drop=True)
    day_key = prices["timestamp"].dt.floor("D")
    unique_days = sorted(day_key.unique())

    soc_min, soc_max = params.soc_bounds()
    carry_soc = params.initial_soc_kwh if params.initial_soc_kwh is not None else soc_min
    carry_soc = float(np.clip(carry_soc, soc_min, soc_max))
    carry_net_power = 0.0  # battery assumed idle before the simulation starts

    cum_used_kwh = 0.0
    sim_start = prices["timestamp"].iloc[0]

    schedules = []
    daily_rows = []
    messages = []

    for day in unique_days:
        day_prices = prices.loc[day_key == day].reset_index(drop=True)

        if len(day_prices) < 2:
            messages.append(
                f"Skipped {pd.Timestamp(day).date()}: fewer than 2 price points that day."
            )
            continue

        throughput_cap_kwh = None
        if params.max_annual_throughput_kwh is not None:
            elapsed_days = (day_prices["timestamp"].iloc[-1] - sim_start) \
                           / np.timedelta64(1, "D")
            elapsed_days = max(elapsed_days, 0.0) + 1e-9
            allowed_cumulative = params.max_annual_throughput_kwh * (elapsed_days / 365.25)
            throughput_cap_kwh = max(0.0, allowed_cumulative - cum_used_kwh)

        # grid_limit is passed in full (if given) -- optimize() reindexes it
        # to this day's own timestamps only, so no future info leaks in.
        day_result = optimize(
            day_prices,
            params,
            grid_limit=grid_limit,
            initial_soc_kwh_override=carry_soc,
            initial_net_power_kw=carry_net_power,
            throughput_cap_kwh_override=throughput_cap_kwh,
            fixed_cost_override=0.0,
        )

        if day_result.status != "OPTIMAL":
            messages.append(f"{pd.Timestamp(day).date()} solve failed: {day_result.solver_message}")
            continue

        schedules.append(day_result.schedule)
        daily_rows.append({
            "date": pd.Timestamp(day).date(),
            "profit": day_result.total_profit,
            "revenue": day_result.total_revenue,
            "charge_cost": day_result.total_charge_cost,
            "fees": day_result.total_fees,
            "energy_charged_kwh": day_result.total_energy_charged_kwh,
            "energy_discharged_kwh": day_result.total_energy_discharged_kwh,
            "ending_soc_kwh": day_result.schedule["soc_kwh"].iloc[-1],
        })

        carry_soc = float(day_result.schedule["soc_kwh"].iloc[-1])
        carry_net_power = float(day_result.schedule["net_kw"].iloc[-1])
        cum_used_kwh += day_result.total_energy_charged_kwh + day_result.total_energy_discharged_kwh

    if not schedules:
        raise ValueError("No day produced a valid schedule -- check your input data.")

    full_schedule = pd.concat(schedules, ignore_index=True)
    daily_breakdown = pd.DataFrame(daily_rows)

    total_revenue = float(daily_breakdown["revenue"].sum())
    total_charge_cost = float(daily_breakdown["charge_cost"].sum())
    total_fees = float(daily_breakdown["fees"].sum())

    ts = full_schedule["timestamp"].to_numpy()
    dt_seconds = float(np.median(np.diff(ts).astype("timedelta64[s]").astype(float)))
    total_horizon_days = (ts[-1] - ts[0]) / np.timedelta64(1, "D") + (dt_seconds / 86400.0)
    fixed_cost_allocated = params.fixed_annual_cost * (total_horizon_days / 365.25)

    total_profit = total_revenue - total_charge_cost - total_fees - fixed_cost_allocated

    return DayAheadSimulationResult(
        schedule=full_schedule,
        daily_breakdown=daily_breakdown,
        total_profit=total_profit,
        total_revenue=total_revenue,
        total_charge_cost=total_charge_cost,
        total_fees=total_fees,
        fixed_cost_allocated=fixed_cost_allocated,
        total_energy_charged_kwh=float(daily_breakdown["energy_charged_kwh"].sum()),
        total_energy_discharged_kwh=float(daily_breakdown["energy_discharged_kwh"].sum()),
        n_days=len(daily_breakdown),
        status="OPTIMAL",
        messages=messages,
    )
