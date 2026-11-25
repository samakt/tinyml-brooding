"""
Brooding pen simulator v2: THREE parameters only -> temperature, sound, flood.
One row per minute per flock for 28 days. All sources and assumptions are listed in
SIM_ASSUMPTIONS in this file and in the README that ships with the data.

Sources (checked):
 [A] Texas A&M broiler management guide (radiant brooder table: 33/30/28/26 C under the brooder at day 0/7/14/21)
 [B] Manitoba Agriculture, brooding temperatures (32-35 C at the heat-source edge in week 1, -3 C per week; comfort/cold/heat behaviour)
 [C] Broiler vocalisation study, PMC12173062 (call types in seconds per minute; age and time-of-day patterns)
 [D] climate-zone.com Kumasi monthly highs/lows (NOTE: its temperature/rain station is ~198 km away; replace with GMet data)
Everything else is an ASSUMPTION to be calibrated with real farm data.
"""
import numpy as np, pandas as pd, argparse

DAYS, MIN = 28, 1440
# [A] chick-level (under-brooder) target by age in days; day 28 taken as 25 C (table prints 23 C / 77 F, inconsistent)
_AGE = [0, 7, 14, 21, 28]; _SP = [33.0, 30.0, 28.0, 26.0, 25.0]
def setpoint(age_day): return np.interp(np.asarray(age_day, float)-1, _AGE, _SP)
# [D] Kumasi monthly mean high/low (C) and rainfall (mm) from the table (deg F / inches converted)
HI = [34.4, 35.6, 35.0, 33.3, 32.2, 30.0, 28.3, 27.8, 29.4, 31.1, 32.8, 33.3]
LO = [20.6, 22.8, 23.3, 23.3, 22.8, 22.2, 21.7, 21.7, 21.7, 21.7, 22.2, 20.6]
RAIN = [5, 25, 66, 107, 122, 132, 84, 69, 127, 124, 28, 13]
# [C] baseline call time, seconds per minute, starter phase (weeks 1-3): short peeps, pleasure notes, distress calls, warbles
# time-of-day multipliers (night/evening/midday) follow the pattern reported in [C]

def sim_flock(rng, farm, pen, flock, start, tau_pen, n_chicks):
    N = DAYS*MIN; t = np.arange(N); hour = (t % MIN)/60.0; age = t//MIN + 1
    month = (start + pd.to_timedelta(t, unit='min')).month.values-1
    hi = np.array(HI)[month]; lo = np.array(LO)[month]
    mean = (hi+lo)/2; amp = (hi-lo)/2
    dshift = np.interp(t/MIN, np.arange(DAYS+1), rng.normal(0, 1.0, DAYS+1))
    out = mean + amp*np.sin(2*np.pi*(hour-9)/24) + dshift                   # outdoor temperature
    dmean = pd.Series(out).rolling(1440, min_periods=1, center=True).mean().values
    amb = dmean + 0.6*(out-dmean)                                             # ASSUMPTION: house damps outdoor swing to 60%
    events = np.full(N, '', dtype=object)
    m = {k: np.zeros(N, bool) for k in ['heater_fail', 'cold_night', 'heat_wave', 'flood', 'disturbance', 'noise']}
    def add(kind, s0, dur):
        s0 = int(s0) % N; e = min(N, s0+int(dur)); m[kind][s0:e] = True
        events[s0:e] = np.where(events[s0:e] == '', kind, events[s0:e])
    rain_f = np.array(RAIN)[month[0]]/60.0                                    # rainy-season weight for this flock
    for _ in range(rng.poisson(5)):
        d = rng.integers(0, DAYS); h = rng.uniform(19, 29) if rng.random() < .7 else rng.uniform(5, 18)
        add('heater_fail', d*MIN+h*60, rng.uniform(60, 360))                  # ASSUMPTION: power/lamp failures, mostly at night
    for _ in range(rng.poisson(1+2*rain_f)):
        d = rng.integers(0, DAYS); add('cold_night', d*MIN+rng.uniform(0, 6)*60+18*60*(rng.random() < .5), rng.uniform(120, 420))
    for _ in range(rng.poisson(1+3*(1.6-min(rain_f, 1.6)))):
        d = rng.integers(0, DAYS); add('heat_wave', d*MIN+rng.uniform(10, 13)*60, rng.uniform(180, 420))
    for _ in range(rng.poisson(0.3+1.6*rain_f)):                              # ASSUMPTION: flood odds scale with monthly rainfall
        d = rng.integers(2, DAYS); add('flood', d*MIN+rng.uniform(0, 24)*60, rng.uniform(30, 240))
    for _ in range(rng.poisson(5)):
        d = rng.integers(0, DAYS); add('disturbance', d*MIN+rng.uniform(21, 28)*60, rng.uniform(5, 40))   # snake/rat at night
    for _ in range(rng.poisson(6)):
        d = rng.integers(0, DAYS); add('noise', d*MIN+rng.uniform(0, 24)*60, rng.uniform(10, 90))        # generator, workers, roof rain
    amb = amb - 5.0*m['cold_night'] + 5.0*m['heat_wave']
    # ---- thermal model: first-order, farmer-style on/off lamp around the age setpoint [A]
    q_chick = 0.15 + 1.6*(1-np.exp(-age/12.0))*(n_chicks/300.0)
    Tp = np.zeros(N); Tp[0] = setpoint(1); lamp = np.zeros(N, bool); on = True
    sp = setpoint(age); wet = np.zeros(N)
    for i in range(1, N):
        if on and Tp[i-1] > sp[i]+0.8: on = False
        elif (not on) and Tp[i-1] < sp[i]-0.8: on = True
        if m['heater_fail'][i]: on = False
        lamp[i] = on
        teq = amb[i] + q_chick[i] + (12.0 if on else 0.0) - (1.0 if m['flood'][i] else 0.0)   # wet floor chills (ASSUMPTION)
        Tp[i] = Tp[i-1] + (teq-Tp[i-1])/tau_pen + rng.normal(0, 0.03)
    def probe(x, s, drift=0):
        return x + rng.normal(0, s, N) + drift
    drift = np.cumsum(rng.normal(0, 0.0004, N))
    T1 = probe(Tp, 0.15, drift); T2 = probe(amb+(Tp-amb)*0.35+0.3, 0.15); T3 = probe(amb+0.5, 0.15)
    for arr in (T1, T2, T3):                                                  # dropouts / stuck values
        for _ in range(rng.poisson(3)):
            s0 = rng.integers(0, N-200); L = int(rng.uniform(5, 120))
            if rng.random() < .6: arr[s0:s0+L] = np.nan
            else: arr[s0:s0+L] = arr[s0]
    flt = m['flood'].astype(int)
    # ---- sound: call time per minute by type [C] + ASSUMED stress effects
    devc = pd.Series(Tp-sp).rolling(10, min_periods=1).median().values
    cold = np.clip((-devc-1.0)/3.0, 0, 2.0); hot = np.clip((devc-1.5)/3.0, 0, 2.0)
    tod_sp = np.where((hour >= 17) & (hour < 21), 1.25, np.where((hour >= 22) | (hour < 5), 0.65, 1.0))   # evening peak, night low [C]
    tod_dc = np.where((hour >= 11) & (hour < 14), 1.5, np.where((hour >= 22) | (hour < 5), 0.4, 1.0))
    tod_pn = np.where((hour >= 22) | (hour < 5), 1.6, 0.7)
    SP = 24*tod_sp*np.where(age <= 21, 1.0, 0.75)*(1+0.3*cold)*(1-0.25*hot)
    PN = 4.5*tod_pn*np.where(age <= 21, 1.0, 0.4)*(1-0.5*np.minimum(cold, 1))
    DC = 2.5*tod_dc*(0.6 if True else 1)*(1+0.08*age/7) + 14*cold + 2*hot + 18*m['disturbance'] + rng.gamma(2, 0.4, N)   # ASSUMPTION: cold and fright raise distress calls
    WB = 0.15*np.where((hour >= 22) | (hour < 5), 1.1, 0.5)
    sp_s = np.clip(SP+rng.normal(0, 2.5, N), 0, 60); pn_s = np.clip(PN+rng.normal(0, 1.0, N), 0, 60)
    dc_s = np.clip(DC+rng.normal(0, 1.2, N), 0, 60); wb_s = np.clip(WB+rng.normal(0, 0.05, N), 0, 60)
    oth_s = np.clip(rng.gamma(1.5, 0.5, N)+25*m['noise']+10*m['flood'], 0, 60)       # other noise: machines, workers, rain on roof
    dcs = pd.Series(dc_s).rolling(5, min_periods=1).mean().values
    scls = np.zeros(N, int); scls[dcs > 8] = 1; scls[dcs > 20] = 2                  # ASSUMPTION thresholds (s/min of distress calls)
    scls[(oth_s > 20) & (scls == 0)] = 3
    # ---- labels: 0 normal / 1 warning / 2 critical
    sp_hot = np.maximum(sp, 28.0)
    devh = pd.Series(Tp).rolling(10, min_periods=1).median().values - sp_hot
    h_temp = np.zeros(N, int); h_temp[(devc < -2.0) | (devh > 2.5)] = 1; h_temp[(devc < -4.0) | (devh > 4.5)] = 2
    t_dir = np.where(h_temp > 0, np.where(devc < -2.0, -1, 1), 0)
    h_snd = np.zeros(N, int); h_snd[scls == 1] = 1; h_snd[scls == 2] = 2
    h_fl = np.where(flt == 1, 2, 0)
    ts = pd.date_range(start, periods=N, freq='min')
    return pd.DataFrame(dict(
        timestamp=ts, farm=farm, pen=pen, flock=flock, age_day=age, month=month+1,
        t_chick=T1, t_edge=T2, t_amb=T3, float_switch=flt,
        peep_s_per_min=sp_s, pleasure_s_per_min=pn_s, distress_s_per_min=dc_s, warble_s_per_min=wb_s, other_noise_s_per_min=oth_s,
        setpoint=sp, hazard_temp=h_temp, temp_direction=t_dir, sound_class=scls, hazard_sound=h_snd, hazard_flood=h_fl,
        event=events, lamp_on=lamp.astype(int)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='synthetic_brooding_3param.csv.gz'); ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    farms = {'F1': 22, 'F2': 30, 'F3': 18}                       # ASSUMPTION: pen thermal time constant (min) = insulation differences
    months = rng.permutation(12); parts = []; k = 0
    for fm, tau in farms.items():
        for pen in ('P1', 'P2'):
            for fl in (1, 2):
                start = pd.Timestamp(2026, int(months[k])+1, 3) + pd.Timedelta(days=int(rng.integers(0, 20)))
                parts.append(sim_flock(rng, fm, pen, f'{fm}-{pen}-{fl}', start, tau*rng.uniform(.9, 1.1), int(rng.integers(250, 450)))); k += 1
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(a.out, index=False, float_format='%.3f'); print(len(df), 'rows ->', a.out)
