# Setting up the brooding-pen simulation (AnyLogic or alternatives)

## My recommendation
For this project, keep **Python as the data generator** (already working: `pen_simulator_3param.py`). Reasons: it is scripted and versioned (one command regenerates the data), it runs thousands of runs quickly, it writes the CSVs our training code reads, and your team can review every assumption in plain text.
Use **AnyLogic only if** Prof. Kponyo or the team wants a visual, interactive model to show stakeholders, or you want to practise agent-based modelling. In that case build it as a second model and check that it reproduces the Python numbers.
Other options: **SimPy** (Python event simulation, same language as the generator) and **OpenModelica** (equation-based physics, good if we later model the heat lamp and airflow in detail).
Before committing to AnyLogic, check which edition KNUST can use: the free Personal Learning Edition has limits (check them on the AnyLogic site) and may not allow everything we need.

## If we build it in AnyLogic
Model time unit: **minutes**. Stop time: **40,320** (28 days).

1. **Agent `Pen`** (one agent per flock). Parameters: `tauPen`, `nChicks`, `startMonth`, event rates, random seed.
2. **Weather:** a function `outdoorT(t)` using the Kumasi monthly highs and lows (a table) and a sine for the day. Indoor `amb = dayMean + 0.6*(outdoor - dayMean)` (assumption).
3. **Temperature (system dynamics):** one stock `Tpen` with flow `dT/dt = (Teq - Tpen)/tauPen`, where `Teq = amb + qChick(age) + (lampOn ? 12 : 0) - (flood ? 1 : 0)`.
4. **Lamp (statechart):** states `On`, `Off`, `Failed`. Transitions: `On -> Off` when `Tpen > setpoint + 0.8`; `Off -> On` when `Tpen < setpoint - 0.8`; any state -> `Failed` on a heater-failure event, and back after its duration.
5. **Setpoint:** function of age using 33/30/28/26/25 C at days 0/7/14/21/28.
6. **Events:** one Event object per hazard with exponential inter-arrival times and random durations: heater failure, cold night, heat wave, flood, disturbance (snake/rat), other noise. Rates in the README table.
7. **Sensors:** a function that adds noise, slow drift, dropouts (NaN) and stuck values to the three probes.
8. **Sound:** a function that gives seconds per minute for each call type from age, time of day, cold stress and disturbance (rules in the README).
9. **Labels:** compute warning/critical flags with the same rules as the README.
10. **Output:** each simulated minute, insert a row into an AnyLogic database table (or write a CSV with a Java `FileWriter`). Use a **Parameter Variation** or **Monte Carlo** experiment to run 12 or more flocks with different seeds, farms and start months.
11. **Check it:** compare means, event curves and label shares against `synthetic_brooding_3param.csv.gz`. They should agree within sampling noise.

## Calibration to do with the farm (this is what makes the data believable)
- Lamp on/off test: how fast the pen cools and warms (gives `tauPen` and the lamp's heating).
- Typical night and day temperatures in the pen, and how often heater or power failures happen.
- How often flooding happens, in which months, and how deep.
- A few hours of real chick recordings in comfortable, cold and frightened conditions.
- Replace the Kumasi table with weather data from the Ghana Meteorological Agency.
