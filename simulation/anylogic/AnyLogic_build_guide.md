# AnyLogic build guide: brooding pen (temperature, sound, flood)

**How this is organised.** The simulation logic is already written and tested in one plain Java file, `PenEngine.java`. AnyLogic runs on Java, so the model is a thin shell: it creates the engine, advances it one minute at a time, shows live charts, and writes the CSV. You do not retype the physics in AnyLogic blocks.

**Tested here:** `PenEngine.java` compiles (Java 11 target) and writes 483,840 rows in 3 seconds. Across 4 random seeds on each side, its label shares, baseline call times, temperature by age and event rates agree with the Python simulator within random variation. The only gap is flood time (0.69% vs 0.95% of minutes), which is too rare to judge from 4 runs (rainfall table implies about 0.77%).
**Not tested:** the AnyLogic wiring below. I cannot run AnyLogic here, and the menu names are from AnyLogic 8 and may differ slightly in your version. If a step does not match, tell me what you see.

## A. One-time setup
1. Install AnyLogic (check which edition KNUST provides; the free Personal Learning Edition has size limits).
2. **File > New > Model.** Name it `BroodingPen`. Choose **minutes** as the model time unit.
3. In the Projects tree, right-click the model > **New > Java Class**. Name it `PenEngine`.
4. The new class opens as a template (`public class PenEngine implements Serializable { ... }`, with a default constructor, `toString` and `serialVersionUID`). Click inside the editor, press **Ctrl+A** to select everything, and **paste the whole of `PenEngine.java`** over it. The downloaded file already starts with the imports and declares `implements Serializable`, so no other edits are needed. (If your version shows a `package ...;` line in the template, keep that line at the top.) Save with **Ctrl+S**. If the editor shows red errors, check that nothing from the template is left over; if the compiler complains about the `main` method, delete `main`.

## B. Live single-pen model (for demos)
1. Open `Main`. Add **Parameters** (Agent palette > Parameter):
   `farm` (String, "F1"), `pen` (String, "P1"), `flockId` (String, "F1-P1-1"), `startMonth` (int, 10), `startDay` (int, 5), `tauPen` (double, 22), `nChicks` (int, 350), `seed` (int, 11), `outFile` (String, "flock.csv").
2. Add two **Variables** (type Other): `engine` of class `PenEngine`, and `writer` of class `java.io.PrintWriter`.
3. In `Main` properties, **On startup**:
   ```java
   engine = new PenEngine(farm, pen, flockId, java.time.LocalDateTime.of(2026, startMonth, startDay, 0, 0), tauPen, nChicks, seed);
   try { writer = new java.io.PrintWriter(outFile); } catch (Exception e) { throw new RuntimeException(e); }
   writer.println(PenEngine.HEADER);
   ```
4. Add an **Event** named `tick`: Trigger type **Timeout**, **Mode: Cyclic**, first occurrence 0, **Recurrence time 1 minute**. Action:
   ```java
   if (engine.hasNext()) { writer.println(engine.step()); }
   else { writer.close(); finishSimulation(); }
   ```
5. **On destroy** (Main): `if (writer != null) writer.close();`
6. Charts (Analysis palette > Time plot), each updated every 1 minute (use the Time plot's update setting):
   - Temperature: `engine.getTempChick()`, `engine.getTempAmbient()`, `engine.getSetpoint()`.
   - Sound: `engine.getDistressSecPerMin()` (and the class `engine.getSoundClass()` as a second plot).
   - Flood and lamp: `engine.isFlood() ? 1 : 0`, `engine.isLampOn() ? 1 : 0`.
   - Optional text boxes: `"Day " + engine.getAge()` and `engine.getEvent()`.
7. Run the **Simulation** experiment: **Model time: stop at 40320 minutes** (28 days). Use **virtual time** (fastest) to write the CSV, or real time scaled up to watch the charts.

## C. Batch: 12 flocks like the Python dataset
Easiest, no AnyLogic experiment needed: in the Java class run `PenEngine.main(new String[]{"synthetic_brooding_java.csv", "11"});` from a **Custom experiment** (or any button). It writes one CSV with 12 flocks.
AnyLogic-native alternative: a **Parameter Variation** experiment over a parameter `flockIndex` (fixed range 0 to 11, step 1) and in `Main` compute `farm`, `pen`, flock number, `tauPen` (22/30/18 by farm) and `startMonth` (a shuffled 1 to 12 list) from `flockIndex`, with `outFile = "flock_" + flockIndex + ".csv"`. Merge the 12 files afterwards.

## D. Check your build
- Open the CSV: the header must equal `PenEngine.HEADER`, and a full flock has 40,320 rows.
- Temperature should follow the target in the live chart (33 C on day 1, falling to about 25 C) and drop sharply during a `heater_fail` event.
- Compare your output with `java_engine_sample_output.csv.gz` (same seed 11 gives the same rows) and with the Python file.

## E. Changing the assumptions
All assumptions are plain numbers inside `PenEngine.java` (marked ASSUMPTION): lamp heating (12 C), hysteresis (0.8 C), warning/critical limits, event rates, sound thresholds, house damping (0.6), flood chill (1 C). Change them in one place and regenerate. Replace the Kumasi `HI`, `LO` and `RAIN` tables with Ghana Meteorological Agency data when you have it.
After the farm visit, calibrate `tauPen`, the lamp heating and the event rates from real measurements (see the README calibration list).
