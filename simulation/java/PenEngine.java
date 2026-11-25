import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.*;

/**
 * PenEngine: one brooding flock (28 days, 1 step = 1 minute). Temperature, sound, flood only.
 * Plain Java (no AnyLogic imports) so it can be unit-tested and called from an AnyLogic agent.
 * Every parameter marked ASSUMPTION is a placeholder to calibrate with real farm data.
 * Sources: Texas A&M broiler guide + Manitoba Agriculture (targets), PMC12173062 (call types), climate-zone.com Kumasi table.
 */
public class PenEngine implements Serializable {
    private static final long serialVersionUID = 1L;
    public static final int DAYS = 28, MIN = 1440, N = DAYS * MIN;
    static final double[] AGE_PTS = {0, 7, 14, 21, 28};
    static final double[] SP_PTS = {33, 30, 28, 26, 25};
    static final double[] HI = {34.4, 35.6, 35.0, 33.3, 32.2, 30.0, 28.3, 27.8, 29.4, 31.1, 32.8, 33.3};
    static final double[] LO = {20.6, 22.8, 23.3, 23.3, 22.8, 22.2, 21.7, 21.7, 21.7, 21.7, 22.2, 20.6};
    static final double[] RAIN = {5, 25, 66, 107, 122, 132, 84, 69, 127, 124, 28, 13};
    public static final String[] EVENTS = {"heater_fail", "cold_night", "heat_wave", "flood", "disturbance", "noise"};
    public static final String HEADER = "timestamp,farm,pen,flock,age_day,month,t_chick,t_edge,t_amb,float_switch,"
        + "peep_s_per_min,pleasure_s_per_min,distress_s_per_min,warble_s_per_min,other_noise_s_per_min,setpoint,"
        + "hazard_temp,temp_direction,sound_class,hazard_sound,hazard_flood,event,lamp_on";

    // ---- configuration (AnyLogic agent parameters map onto these)
    public final String farm, pen, flock;
    public final LocalDateTime start;
    public final double tauPen;        // pen thermal time constant in minutes (ASSUMPTION 18-33)
    public final int nChicks;
    final Random rng;

    // ---- precomputed schedules
    final boolean[][] ev = new boolean[EVENTS.length][N];
    final String[] evLabel = new String[N];
    final double[] amb = new double[N];
    final int[] monthIdx = new int[N];
    final int[][] fStart = new int[3][]; final int[][] fLen = new int[3][]; final boolean[][] fNaN = new boolean[3][];

    // ---- state
    int i = -1; double tp, drift; boolean lampOn = true;
    final double[] bufDev = new double[10], bufTp = new double[10], bufDc = new double[5];
    double[] stuck = new double[3];

    public PenEngine(String farm, String pen, String flock, LocalDateTime start, double tauPen, int nChicks, long seed) {
        this.farm = farm; this.pen = pen; this.flock = flock; this.start = start; this.tauPen = tauPen; this.nChicks = nChicks;
        rng = new Random(seed);
        double[] hi = new double[N], lo = new double[N], out = new double[N];
        double[] node = new double[DAYS + 1]; for (int k = 0; k <= DAYS; k++) node[k] = rng.nextGaussian();
        for (int t = 0; t < N; t++) {
            monthIdx[t] = start.plusMinutes(t).getMonthValue() - 1;
            double hour = (t % MIN) / 60.0, x = t / (double) MIN; int k0 = Math.min((int) x, DAYS - 1);
            double dshift = node[k0] + (node[k0 + 1] - node[k0]) * (x - k0);
            double mean = (HI[monthIdx[t]] + LO[monthIdx[t]]) / 2, amp = (HI[monthIdx[t]] - LO[monthIdx[t]]) / 2;
            out[t] = mean + amp * Math.sin(2 * Math.PI * (hour - 9) / 24) + dshift;
        }
        // centred 24 h rolling mean, then ASSUMPTION: house damps the outdoor swing to 60%
        double[] dm = new double[N]; double[] cs = new double[N + 1];
        for (int t = 0; t < N; t++) cs[t + 1] = cs[t] + out[t];
        for (int t = 0; t < N; t++) { int a = Math.max(0, t - 720), b = Math.min(N, t + 720); dm[t] = (cs[b] - cs[a]) / (b - a); amb[t] = dm[t] + 0.6 * (out[t] - dm[t]); }
        double rainF = RAIN[monthIdx[0]] / 60.0;
        Arrays.fill(evLabel, "");
        for (int n = poisson(5); n > 0; n--) { int d = rng.nextInt(DAYS); double h = rng.nextDouble() < .7 ? u(19, 29) : u(5, 18); add(0, d * MIN + h * 60, u(60, 360)); }
        for (int n = poisson(1 + 2 * rainF); n > 0; n--) { int d = rng.nextInt(DAYS); add(1, d * MIN + u(0, 6) * 60 + (rng.nextDouble() < .5 ? 1080 : 0), u(120, 420)); }
        for (int n = poisson(1 + 3 * (1.6 - Math.min(rainF, 1.6))); n > 0; n--) { int d = rng.nextInt(DAYS); add(2, d * MIN + u(10, 13) * 60, u(180, 420)); }
        for (int n = poisson(0.3 + 1.6 * rainF); n > 0; n--) { int d = 2 + rng.nextInt(DAYS - 2); add(3, d * MIN + u(0, 24) * 60, u(30, 240)); }
        for (int n = poisson(5); n > 0; n--) { int d = rng.nextInt(DAYS); add(4, d * MIN + u(21, 28) * 60, u(5, 40)); }
        for (int n = poisson(6); n > 0; n--) { int d = rng.nextInt(DAYS); add(5, d * MIN + u(0, 24) * 60, u(10, 90)); }
        for (int t = 0; t < N; t++) { if (ev[1][t]) amb[t] -= 5; if (ev[2][t]) amb[t] += 5; }
        for (int p = 0; p < 3; p++) {                       // probe faults: dropouts (NaN) or stuck values
            int c = poisson(3); fStart[p] = new int[c]; fLen[p] = new int[c]; fNaN[p] = new boolean[c];
            for (int k = 0; k < c; k++) { fStart[p][k] = rng.nextInt(N - 200); fLen[p][k] = (int) u(5, 120); fNaN[p][k] = rng.nextDouble() < .6; }
        }
        tp = setpoint(1);
    }

    // ---- helpers
    double u(double a, double b) { return a + (b - a) * rng.nextDouble(); }
    int poisson(double lam) { double L = Math.exp(-lam), p = 1; int k = 0; do { k++; p *= rng.nextDouble(); } while (p > L); return k - 1; }
    double gamma(double k, double scale) {  // Marsaglia-Tsang, k >= 1
        double d = k - 1.0 / 3, c = 1 / Math.sqrt(9 * d);
        while (true) { double x, v; do { x = rng.nextGaussian(); v = 1 + c * x; } while (v <= 0); v = v * v * v; double uu = rng.nextDouble();
            if (Math.log(uu) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v * scale; }
    }
    void add(int kind, double s0, double dur) {
        int s = ((int) s0) % N, e = Math.min(N, s + (int) dur);
        for (int t = s; t < e; t++) { ev[kind][t] = true; if (evLabel[t].isEmpty()) evLabel[t] = EVENTS[kind]; }
    }
    public static double setpoint(double ageDay) {  // piecewise-linear target from the guide
        double x = ageDay - 1; if (x <= AGE_PTS[0]) return SP_PTS[0];
        for (int k = 1; k < AGE_PTS.length; k++) if (x <= AGE_PTS[k]) return SP_PTS[k - 1] + (SP_PTS[k] - SP_PTS[k - 1]) * (x - AGE_PTS[k - 1]) / (AGE_PTS[k] - AGE_PTS[k - 1]);
        return SP_PTS[SP_PTS.length - 1];
    }
    static double median(double[] buf, int n) { double[] c = Arrays.copyOf(buf, n); Arrays.sort(c); return n % 2 == 1 ? c[n / 2] : (c[n / 2 - 1] + c[n / 2]) / 2; }
    static double clip(double x, double a, double b) { return Math.max(a, Math.min(b, x)); }
    public boolean hasNext() { return i + 1 < N; }
    // ---- live readouts for AnyLogic charts and animation (valid after the first step())
    public double getTempChick() { return tp; }
    public double getTempAmbient() { return i >= 0 ? amb[i] : amb[0]; }
    public double getSetpoint() { return setpoint(Math.max(1, getAge())); }
    public boolean isLampOn() { return lampOn; }
    public int getAge() { return i < 0 ? 1 : i / MIN + 1; }
    public String getEvent() { return i < 0 ? "" : evLabel[i]; }
    public boolean isFlood() { return i >= 0 && ev[3][i]; }
    public double getDistressSecPerMin() { return lastDc; }
    public int getSoundClass() { return lastCls; }
    double lastDc; int lastCls;

    /** Advance one minute and return one CSV row. In AnyLogic: call this from a cyclic event with period 1 minute. */
    public String step() {
        i++; int age = i / MIN + 1; double hour = (i % MIN) / 60.0; double sp = setpoint(age);
        double qChick = 0.15 + 1.6 * (1 - Math.exp(-age / 12.0)) * (nChicks / 300.0);
        if (i > 0) {                                           // lamp thermostat (+/-0.8 C) and thermal response
            if (lampOn && tp > sp + 0.8) lampOn = false; else if (!lampOn && tp < sp - 0.8) lampOn = true;
            if (ev[0][i]) lampOn = false;
            double teq = amb[i] + qChick + (lampOn ? 12.0 : 0.0) - (ev[3][i] ? 1.0 : 0.0);   // wet floor chills by 1 C (ASSUMPTION)
            tp = tp + (teq - tp) / tauPen + 0.03 * rng.nextGaussian();
        }
        drift += 0.0004 * rng.nextGaussian();
        double[] raw = {tp + 0.15 * rng.nextGaussian() + drift, amb[i] + (tp - amb[i]) * 0.35 + 0.3 + 0.15 * rng.nextGaussian(), amb[i] + 0.5 + 0.15 * rng.nextGaussian()};
        double[] sens = new double[3];
        for (int p = 0; p < 3; p++) {
            sens[p] = raw[p];
            for (int k = 0; k < fStart[p].length; k++) if (i >= fStart[p][k] && i < fStart[p][k] + fLen[p][k]) {
                if (i == fStart[p][k]) stuck[p] = raw[p];
                sens[p] = fNaN[p][k] ? Double.NaN : stuck[p];
            }
        }
        // rolling windows
        bufDev[i % 10] = tp - sp; bufTp[i % 10] = tp; int n10 = Math.min(10, i + 1);
        double devc = median(bufDev, n10), devh = median(bufTp, n10) - Math.max(sp, 28.0);
        double cold = clip((-devc - 1.0) / 3.0, 0, 2), hot = clip((devc - 1.5) / 3.0, 0, 2);
        // sound: call time per minute by type, time-of-day pattern, assumed stress effects
        boolean night = hour >= 22 || hour < 5;
        double todSp = (hour >= 17 && hour < 21) ? 1.25 : (night ? 0.65 : 1.0);
        double todDc = (hour >= 11 && hour < 14) ? 1.5 : (night ? 0.4 : 1.0);
        double todPn = night ? 1.6 : 0.7;
        double SP = 24 * todSp * (age <= 21 ? 1.0 : 0.75) * (1 + 0.3 * cold) * (1 - 0.25 * hot);
        double PN = 4.5 * todPn * (age <= 21 ? 1.0 : 0.4) * (1 - 0.5 * Math.min(cold, 1));
        double DC = 2.5 * todDc * 0.6 * (1 + 0.08 * age / 7) + 14 * cold + 2 * hot + (ev[4][i] ? 18 : 0) + gamma(2, 0.4);
        double WB = 0.15 * (night ? 1.1 : 0.5);
        double sp_s = clip(SP + 2.5 * rng.nextGaussian(), 0, 60), pn_s = clip(PN + rng.nextGaussian(), 0, 60);
        double dc_s = clip(DC + 1.2 * rng.nextGaussian(), 0, 60), wb_s = clip(WB + 0.05 * rng.nextGaussian(), 0, 60);
        double oth = clip(gamma(1.5, 0.5) + (ev[5][i] ? 25 : 0) + (ev[3][i] ? 10 : 0), 0, 60);
        bufDc[i % 5] = dc_s; double s5 = 0; int n5 = Math.min(5, i + 1); for (int k = 0; k < n5; k++) s5 += bufDc[k]; double dcs = s5 / n5;
        int scls = dcs > 20 ? 2 : (dcs > 8 ? 1 : 0); if (oth > 20 && scls == 0) scls = 3; lastDc = dc_s; lastCls = scls;
        // labels (thresholds are ASSUMPTIONS for a poultry specialist to set)
        int hTemp = 0; if (devc < -2.0 || devh > 2.5) hTemp = 1; if (devc < -4.0 || devh > 4.5) hTemp = 2;
        int tDir = hTemp > 0 ? (devc < -2.0 ? -1 : 1) : 0;
        int hSnd = scls == 1 ? 1 : (scls == 2 ? 2 : 0); int flood = ev[3][i] ? 1 : 0;
        StringBuilder sb = new StringBuilder();
        sb.append(start.plusMinutes(i).toString().replace('T', ' ')).append(':').append("00").append(',').append(farm).append(',').append(pen).append(',').append(flock).append(',')
          .append(age).append(',').append(monthIdx[i] + 1).append(',');
        for (int p = 0; p < 3; p++) sb.append(Double.isNaN(sens[p]) ? "" : String.format(Locale.ROOT, "%.3f", sens[p])).append(',');
        sb.append(flood).append(',');
        sb.append(String.format(Locale.ROOT, "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,", sp_s, pn_s, dc_s, wb_s, oth, sp));
        sb.append(hTemp).append(',').append(tDir).append(',').append(scls).append(',').append(hSnd).append(',').append(flood == 1 ? 2 : 0).append(',').append(evLabel[i]).append(',').append(lampOn ? 1 : 0);
        return sb.toString();
    }

    /** Stand-alone test driver: 12 flocks like the Python version. */
    public static void main(String[] a) throws Exception {
        String out = a.length > 0 ? a[0] : "java_synthetic.csv"; long seed = a.length > 1 ? Long.parseLong(a[1]) : 11;
        Random r = new Random(seed); List<Integer> months = new ArrayList<>(); for (int m = 1; m <= 12; m++) months.add(m); Collections.shuffle(months, r);
        String[] farms = {"F1", "F2", "F3"}; double[] tau = {22, 30, 18}; int k = 0;
        try (java.io.PrintWriter w = new java.io.PrintWriter(out)) {
            w.println(HEADER);
            for (int f = 0; f < 3; f++) for (String pen : new String[]{"P1", "P2"}) for (int fl = 1; fl <= 2; fl++) {
                LocalDateTime st = LocalDateTime.of(2026, months.get(k), 3, 0, 0).plusDays(r.nextInt(20));
                PenEngine e = new PenEngine(farms[f], pen, farms[f] + "-" + pen + "-" + fl, st, tau[f] * (0.9 + 0.2 * r.nextDouble()), 250 + r.nextInt(200), seed * 100 + k);
                while (e.hasNext()) w.println(e.step()); k++;
            }
        }
        System.out.println("wrote " + out);
    }
}
