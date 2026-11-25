# Synthetic brooding data (temperature, sound, flood): what it is built on

File: `synthetic_brooding_3param.csv.gz` (483,840 rows = 12 flocks x 28 days x 1 row per minute; 3 farms x 2 pens x 2 flocks, start months spread across the year).
Made by `pen_simulator_3param.py` (seed 11). Every number marked ASSUMPTION is a placeholder to calibrate with real farm data.

## Columns
| Column | Meaning |
|---|---|
| timestamp, farm, pen, flock, age_day, month | identifiers; age 1-28 |
| t_chick, t_edge, t_amb | three probes (chick level, pen edge, ambient), with noise, drift, dropouts (NaN) and stuck values |
| float_switch | 1 = water on the floor |
| peep, pleasure, distress, warble, other_noise _s_per_min | seconds of each call type per minute (not audio) |
| setpoint | target chick-level temperature for that age |
| hazard_temp (0/1/2), temp_direction (-1 cold, +1 hot) | label: normal / warning / critical |
| sound_class (0 contented, 1 distress, 2 panic, 3 other noise), hazard_sound | label from distress-call time |
| hazard_flood (0/2) | label: float switch triggered |
| event | injected cause (heater_fail, cold_night, heat_wave, flood, disturbance, noise; empty = none) |
| lamp_on | simulator internal state (do not use as a model input) |

## What each choice is based on
| Item | Value used | Basis | Status |
|---|---|---|---|
| Target chick-level temperature | 33, 30, 28, 26, 25 C at day 0, 7, 14, 21, 28 (linear between) | Texas A&M broiler management guide, radiant brooder table (33/30/28/26 under the brooder). Manitoba Agriculture: 32-35 C at the heat-source edge in week 1, then -3 C per week | Sourced. The guide's day-28 entry is inconsistent (23 C vs 77 F = 25 C); I used 25 C |
| Comfort / cold / heat behaviour | cold: noisy, huddled; hot: drowsy, spread out | Texas A&M guide; Manitoba Agriculture | Sourced (qualitative) |
| Warning / critical temperature limits | warning at 2 C below or 2.5 C above target; critical at 4 C below or 4.5 C above, sustained 10 min; heat limit never below 28 C | Engineering choice. The guides give targets and say to follow bird behaviour; they give no numeric alarm limits | ASSUMPTION: needs a poultry specialist |
| Outdoor temperature | Kumasi monthly mean high/low, sinusoidal day | climate-zone.com Kumasi table (1991-2020) | Sourced, but its temperature/rain station is ~198 km from Kumasi: replace with GMet data |
| House damping | indoor swing = 60% of outdoor | none | ASSUMPTION |
| Pen thermal response | first-order, time constant 18-33 min, lamp adds up to 12 C, chick body heat up to ~1.8 C | none | ASSUMPTION: calibrate with a lamp on/off test on the farm |
| Call types and baseline rates | short peeps ~24, pleasure notes ~4.5, distress ~2.5, warbles ~0.15 s/min in weeks 1-3; peeps peak in the evening, pleasure notes at night, distress at midday; distress rises after week 3 | Broiler vocalisation study (PMC12173062): call types in s/min, age and time-of-day patterns | Sourced for pattern and rough size |
| Effect of cold/heat/fright on calls | cold and fright raise distress calls; heat lowers peeps slightly | guides say cold chicks are noisy; the study found heat stress did not significantly change vocalisation | ASSUMPTION for size; the heat effect is weak on purpose |
| Sound classes | distress when 5-min mean distress call time > 8 s/min; panic > 20 s/min | none | ASSUMPTION |
| Flood | rate scales with monthly rainfall; float switch = flood; wet floor chills by 1 C | rainfall pattern from the Kumasi table; no flood standard exists | ASSUMPTION |
| Event rates (heater failures ~5, cold nights, heat waves, disturbances ~5 and noise ~6 per flock) | | none | ASSUMPTION: ask farmers how often each happens |

## What this data can and cannot do
It lets us build and test the pipeline, the rules and the hazard engine, and pretrain the climate forecast. It cannot prove detection targets, because a model only learns the simulator's assumptions. Real farm data must be used for the final test, and the real audio for the sound model.
