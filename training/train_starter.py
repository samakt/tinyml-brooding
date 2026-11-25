"""
Training starter for the RAIL poultry brooding project (temperature, sound, flood).

Data : 12-flock synthetic set from the pen simulator (3 farms x 2 pens x 2 flocks, 1 reading/min, 28 days).
Split: leave-one-farm-out. The model never sees the farm it is tested on.
Tasks:
  A. Climate forecast : chick-level temperature 30 min ahead (regression).
  B. Temperature early warning : hazard level (normal/warning/critical) 30 min ahead (3 classes).
  C. Sound classifier : contented / distress / panic / other noise (4 classes).
  D. Flood : float switch rule (no learning needed).
Models: a small MLP (the TinyML candidate, exported to int8 TFLite) and a gradient-boosting
        model (reference, not meant for the microcontroller).
Inputs use only what the node can measure: three temperatures, call time per minute by type, the
float switch, the pen age and the time of day. Hazard labels, events, setpoint column and lamp state are never inputs.
"""
import sys, json, os, numpy as np, pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, confusion_matrix

CSV = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else 'out'; os.makedirs(OUT, exist_ok=True)
STRIDE, HOR, WIN = 5, 6, 12            # one sample every 5 min; horizon 6 steps = 30 min; window 12 steps = 60 min
SEED = 7; rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- data
d = pd.read_csv(CSV, parse_dates=['timestamp']); d['event'] = d['event'].fillna('')
SENS = ['t_chick', 't_edge', 't_amb']; CALL = ['peep_s_per_min', 'pleasure_s_per_min', 'distress_s_per_min', 'warble_s_per_min', 'other_noise_s_per_min']
def prep(g):
    g = g.sort_values('timestamp').reset_index(drop=True)
    miss = g[SENS].isna().any(axis=1).astype(float).values                  # sensor dropout flag
    g[SENS] = g[SENS].ffill(limit=180).bfill()                              # hold last value; a real node would do the same
    g['miss'] = miss; return g
flocks = {k: prep(g) for k, g in d.groupby('flock')}

def rule_level(tc, sp):
    """Hazard rule on a 10-min median of the chick probe (same limits as the simulator labels)."""
    devc = tc - sp; devh = tc - np.maximum(sp, 28.0)
    lv = np.zeros(len(tc), int); lv[(devc < -2.0) | (devh > 2.5)] = 1; lv[(devc < -4.0) | (devh > 4.5)] = 2; return lv

def build(g):
    n = len(g); tc, te, ta = g.t_chick.values, g.t_edge.values, g.t_amb.values
    sp = g.setpoint.values; age = g.age_day.values; hr = g.timestamp.dt.hour.values + g.timestamp.dt.minute.values / 60
    med = pd.Series(tc).rolling(10, min_periods=1).median().values
    idx = np.arange(WIN * STRIDE, n - HOR * STRIDE, STRIDE)                 # sample times t
    lag = idx[:, None] - np.arange(WIN)[None, ::-1] * STRIDE                # 12 points, 5 min apart, oldest first
    now = tc[idx]
    X = np.hstack([tc[lag] - now[:, None], te[lag] - now[:, None], ta[lag] - now[:, None],
                   now[:, None], (now - sp[idx])[:, None], (ta[idx] - now)[:, None], age[idx][:, None] / 28,
                   np.sin(2 * np.pi * hr[idx])[:, None], np.cos(2 * np.pi * hr[idx])[:, None], g.miss.values[idx][:, None]])
    X = np.hstack([X[:, :36], X[:, 36:]]).astype(np.float32)
    y_f = (tc[idx + HOR * STRIDE] - now).astype(np.float32)                 # change in chick temp over 30 min
    y_h = g.hazard_temp.values[idx + HOR * STRIDE]                          # hazard 30 min ahead
    rule_now = rule_level(med, sp)[idx]                                     # persistence baseline: label from today's reading
    cm = [g[c].rolling(5, min_periods=1).mean().values for c in CALL]
    Xs = np.column_stack([c[idx] for c in cm] + [np.sin(2 * np.pi * hr[idx]), np.cos(2 * np.pi * hr[idx]), age[idx] / 28, g.float_switch.values[idx]]).astype(np.float32)
    y_s = g.sound_class.values[idx]; y_fl = g.hazard_flood.values[idx]; fs = g.float_switch.values[idx]
    return dict(X=X, y_f=y_f, y_h=y_h, rule=rule_now, Xs=Xs, y_s=y_s, y_fl=y_fl, fs=fs, now=now, idx=idx, truth_h=g.hazard_temp.values[idx + HOR * STRIDE], ev=g.event.values[idx + HOR * STRIDE])
data = {k: build(g) for k, g in flocks.items()}; farm = {k: k.split('-')[0] for k in flocks}
def stack(keys, f): return np.concatenate([data[k][f] for k in keys])
print('samples per flock', len(next(iter(data.values()))['X']), 'features', data[next(iter(data))]['X'].shape[1], '/ sound', data[next(iter(data))]['Xs'].shape[1])

# ---------------------------------------------------------------- tiny MLPs
import tensorflow as tf; tf.random.set_seed(SEED); tf.get_logger().setLevel('ERROR')
def mlp(nin, nout, kind, units=(32, 16)):
    L = tf.keras.layers; inp = tf.keras.Input((nin,)); x = inp
    for u in units: x = L.Dense(u, activation='relu')(x)
    out = L.Dense(nout, activation=None if kind == 'reg' else 'softmax')(x); m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss='mae' if kind == 'reg' else 'sparse_categorical_crossentropy'); return m
def fit_mlp(Xtr, ytr, Xva, yva, kind, nout, units=(32, 16), cw=None, epochs=30):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    m = mlp(Xtr.shape[1], nout, kind, units); cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    m.fit((Xtr - mu) / sd, ytr, validation_data=((Xva - mu) / sd, yva), epochs=epochs, batch_size=256, verbose=0, callbacks=cb, class_weight=cw)
    return m, mu, sd
def cweights(y, power=.5):
    c = np.bincount(y, minlength=y.max() + 1).astype(float); w = (c.sum() / (len(c) * np.maximum(c, 1))) ** power; return {i: float(w[i]) for i in range(len(c))}

# ---------------------------------------------------------------- event-level scoring
def episodes(mask, minlen=3):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            if j - i >= minlen: out.append((i, j))
            i = j
        else: i += 1
    return out
def event_scores(keys, pred_by_flock, truth_key='truth_h', k=3):
    """Alert = predicted level >= 1 for k samples in a row (15 min). Prediction made at time t refers to t+30 min."""
    hit = tot = fa = 0; leads = []; days = 0
    for key in keys:
        p = pred_by_flock[key]; t = data[key][truth_key] > 0; a = (p >= 1)
        run = np.convolve(a.astype(int), np.ones(k, int), 'full')[:len(a)] >= k           # sustained for k samples
        for s, e in episodes(t):
            tot += 1; lo = max(0, s - 3)
            w = np.where(run[lo:e])[0]
            if len(w): hit += 1; leads.append((s - (lo + w[0])) * STRIDE + HOR * STRIDE)   # alert is issued 30 min before the time it refers to
        td = np.convolve(t.astype(int), np.ones(7, int), 'same') > 0                        # truth dilated by +/- 15 min
        for s, e in episodes(run, 1):
            if not td[s:e].any(): fa += 1
        days += len(a) * STRIDE / 1440
    return dict(episodes=tot, detected=hit, sensitivity=hit / max(tot, 1), false_alerts_per_pen_day=fa / days, median_lead_min=float(np.median(leads)) if leads else None)

# ---------------------------------------------------------------- leave-one-farm-out
res = {'A_forecast': {}, 'B_early_warning': {}, 'C_sound': {}, 'D_flood': {}}; last = {}
for test_farm in ['F1', 'F2', 'F3']:
    te_k = [k for k in flocks if farm[k] == test_farm]; tr_all = [k for k in flocks if farm[k] != test_farm]
    va_k = [tr_all[-1]]; tr_k = tr_all[:-1]                                   # one training flock held out for early stopping
    # ---- A. forecast
    Xtr, ytr, Xva, yva, Xte, yte = stack(tr_k, 'X'), stack(tr_k, 'y_f'), stack(va_k, 'X'), stack(va_k, 'y_f'), stack(te_k, 'X'), stack(te_k, 'y_f')
    m, mu, sd = fit_mlp(Xtr, ytr, Xva, yva, 'reg', 1); p_mlp = m.predict((Xte - mu) / sd, verbose=0)[:, 0]
    g = HistGradientBoostingRegressor(max_iter=200, learning_rate=.08, random_state=SEED).fit(Xtr, ytr); p_gb = g.predict(Xte)
    p_lin = -Xte[:, 8] * 2.0                                                  # linear extrapolation: last 15 min of change, doubled to cover 30 min
    mae = lambda p: float(mean_absolute_error(yte, p)); rmse = lambda p: float(np.sqrt(mean_squared_error(yte, p)))
    res['A_forecast'][test_farm] = {'persistence_MAE': mae(np.zeros_like(yte)), 'linear_MAE': mae(p_lin), 'MLP_MAE': mae(p_mlp), 'GBM_MAE': mae(p_gb), 'MLP_RMSE': rmse(p_mlp), 'persistence_RMSE': rmse(np.zeros_like(yte))}
    # ---- B. early warning (hazard level 30 min ahead)
    ytr_h, yva_h, yte_h = stack(tr_k, 'y_h'), stack(va_k, 'y_h'), stack(te_k, 'y_h')
    mh, muh, sdh = fit_mlp(Xtr, ytr_h, Xva, yva_h, 'clf', 3, cw=cweights(ytr_h))
    ph = {k: mh.predict((data[k]['X'] - muh) / sdh, verbose=0).argmax(1) for k in te_k}
    gh = HistGradientBoostingClassifier(max_iter=200, learning_rate=.08, random_state=SEED, class_weight='balanced').fit(Xtr, ytr_h); pg = {k: gh.predict(data[k]['X']) for k in te_k}
    pr = {k: data[k]['rule'] for k in te_k}
    cat = lambda P: np.concatenate([P[k] for k in te_k])
    res['B_early_warning'][test_farm] = {n: dict(macroF1=float(f1_score(yte_h, cat(P), average='macro')), **event_scores(te_k, P)) for n, P in [('persistence_rule', pr), ('MLP', ph), ('GBM', pg)]}
    # ---- C. sound
    Xs_tr, ys_tr, Xs_va, ys_va, Xs_te, ys_te = stack(tr_k, 'Xs'), stack(tr_k, 'y_s'), stack(va_k, 'Xs'), stack(va_k, 'y_s'), stack(te_k, 'Xs'), stack(te_k, 'y_s')
    ms, mus, sds = fit_mlp(Xs_tr, ys_tr, Xs_va, ys_va, 'clf', 4, units=(16, 8), cw=cweights(ys_tr))
    ps = ms.predict((Xs_te - mus) / sds, verbose=0).argmax(1)
    gs = HistGradientBoostingClassifier(max_iter=150, learning_rate=.1, random_state=SEED, class_weight='balanced').fit(Xs_tr, ys_tr); pgs = gs.predict(Xs_te)
    res['C_sound'][test_farm] = {n: dict(macroF1=float(f1_score(ys_te, P, average='macro', labels=[0, 1, 2, 3])), per_class_F1=[float(x) for x in f1_score(ys_te, P, average=None, labels=[0, 1, 2, 3])],
                                         recall=[float(x) for x in (confusion_matrix(ys_te, P, labels=[0, 1, 2, 3]).diagonal() / np.maximum(np.bincount(ys_te, minlength=4), 1))]) for n, P in [('MLP', ps), ('GBM', pgs)]}
    res['C_sound'][test_farm]['support'] = np.bincount(ys_te, minlength=4).tolist()
    # ---- D. flood rule
    res['D_flood'][test_farm] = {'rule_accuracy': float((stack(te_k, 'fs') * 2 == stack(te_k, 'y_fl')).mean()), 'flood_samples': int((stack(te_k, 'y_fl') > 0).sum())}
    if test_farm == 'F3': last = dict(mA=m, muA=mu, sdA=sd, mB=mh, muB=muh, sdB=sdh, mS=ms, muS=mus, sdS=sds, te=te_k, p_mlp=p_mlp, yte=yte, Xtr=Xtr, Xs_tr=Xs_tr, Xtr_h=Xtr)
    print('done fold', test_farm, flush=True)
json.dump(res, open(f'{OUT}/metrics.json', 'w'), indent=1)

# ---------------------------------------------------------------- export the F3-held-out models as int8 TFLite
def export(model, mu, sd, Xcal, name):
    cal = ((Xcal - mu) / sd).astype(np.float32)
    def rep():
        for i in rng.integers(0, len(cal), 400): yield [cal[i:i + 1]]
    conv = tf.lite.TFLiteConverter.from_keras_model(model); conv.optimizations = [tf.lite.Optimize.DEFAULT]; conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
    tfl = conv.convert(); open(f'{OUT}/{name}.tflite', 'wb').write(tfl); json.dump({'mean': mu.tolist(), 'std': sd.tolist()}, open(f'{OUT}/{name}_scaler.json', 'w')); return tfl
def run_tfl(tfl, X):
    it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors(); i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i['quantization']; so, zo = o['quantization']; out = []
    for r in X:
        it.set_tensor(i['index'], np.clip(np.round(r / s + z), -128, 127).astype(np.int8)[None]); it.invoke(); out.append((it.get_tensor(o['index'])[0].astype(np.float32) - zo) * so)
    return np.array(out)
te_k = last['te']; size = {}; q = {}
Xte = stack(te_k, 'X'); yte = stack(te_k, 'y_f'); sub = np.arange(0, len(Xte), 6)
tA = export(last['mA'], last['muA'], last['sdA'], last['Xtr'], 'forecast_30min'); pA = run_tfl(tA, (Xte[sub] - last['muA']) / last['sdA'])[:, 0]
q['forecast_float_MAE'] = float(mean_absolute_error(yte[sub], last['p_mlp'][sub])); q['forecast_int8_MAE'] = float(mean_absolute_error(yte[sub], pA))
tB = export(last['mB'], last['muB'], last['sdB'], last['Xtr'], 'hazard_early_warning'); yh = stack(te_k, 'y_h'); pB = run_tfl(tB, (Xte[sub] - last['muB']) / last['sdB']).argmax(1)
q['hazard_float_acc'] = float((last['mB'].predict((Xte[sub] - last['muB']) / last['sdB'], verbose=0).argmax(1) == yh[sub]).mean()); q['hazard_int8_acc'] = float((pB == yh[sub]).mean())
Xs_te = stack(te_k, 'Xs'); ys = stack(te_k, 'y_s'); sub2 = np.arange(0, len(Xs_te), 6)
tS = export(last['mS'], last['muS'], last['sdS'], last['Xs_tr'], 'sound_class'); pS = run_tfl(tS, (Xs_te[sub2] - last['muS']) / last['sdS']).argmax(1)
q['sound_float_acc'] = float((last['mS'].predict((Xs_te[sub2] - last['muS']) / last['sdS'], verbose=0).argmax(1) == ys[sub2]).mean()); q['sound_int8_acc'] = float((pS == ys[sub2]).mean())
q['sizes_KB'] = {n: round(os.path.getsize(f'{OUT}/{n}.tflite') / 1024, 1) for n in ['forecast_30min', 'hazard_early_warning', 'sound_class']}
json.dump(q, open(f'{OUT}/export_check.json', 'w'), indent=1); print(json.dumps(q, indent=1))

# ---------------------------------------------------------------- figure: one held-out flock
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
k = te_k[0]; D = data[k]; pk = last['mA'].predict((D['X'] - last['muA']) / last['sdA'], verbose=0)[:, 0]
eps = episodes(D['truth_h'] > 0, 6); s0, e0 = max(eps, key=lambda e: e[1] - e[0]) if eps else (0, 150)   # longest hazard episode in this flock
lo = max(0, s0 - 100); sl = slice(lo, min(len(D['now']), e0 + 60)); t = (np.arange(len(D['now']))[sl] * STRIDE + HOR * STRIDE) / 60   # time of the value being predicted
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t, D['now'][sl] + D['y_f'][sl], label='Actual chick temp, 30 min later', color='#15524d'); ax[0].plot(t, D['now'][sl] + pk[sl], label='Forecast made 30 min earlier', color='#d21f2b', lw=1); ax[0].set_ylabel('°C'); ax[0].legend(); ax[0].set_title(f'Held-out flock {k}: longest temperature hazard (farm never seen in training)')
ax[1].fill_between(t, 0, D['truth_h'][sl], step='mid', color='#c98a00', alpha=.4, label='True hazard level'); ax[1].plot(t, last['mB'].predict((D['X'][sl] - last['muB']) / last['sdB'], verbose=0).argmax(1), color='#15524d', lw=1, label='MLP prediction'); ax[1].set_xlabel('hours from start of flock (time being predicted)'); ax[1].set_yticks([0, 1, 2]); ax[1].set_yticklabels(['normal', 'warning', 'critical']); ax[1].legend()
plt.tight_layout(); plt.savefig(f'{OUT}/held_out_flock.png', dpi=130)
print('OK')
