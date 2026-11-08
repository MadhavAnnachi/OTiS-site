/* OTiS hosted demo: answers the same /api/* calls as otis/server.py, in the
   browser, from data recorded on the GPU by otis/record_web_demo.py.

   - /api/plan    classical planners from plant.js, plus the trained network,
                  run here from its exported weights (demo/<vehicle>.bin)
   - /api/train   replays the recorded GPU training run for the preset plant
                  at its recorded speed: loss curves, log rows, showcase plan
   - /api/state, /api/evolution, /api/pause, /api/resume, /api/stop

   app.js routes api() here only when index.html carries
   <meta name="otis-deploy" content="static"> (added by build_site.py). */
(() => {
  "use strict";
  const BASE = "demo/";
  const SQ2PI = Math.sqrt(2 / Math.PI);

  // ------------------------------------------------------------ network
  // Exact mirror of otis/network.py PolicyNet (Flax defaults: LayerNorm
  // eps 1e-6, tanh-approximate GELU, Dense kernels stored [in, out]).
  function makeNet(net, buffer) {
    const all = new Float32Array(buffer);
    const W = {};
    let off = 0;
    for (const [name, shape] of net.layers) {
      const n = shape.reduce((a, b) => a * b, 1);
      W[name] = { data: all.subarray(off, off + n), shape };
      off += n;
    }
    if (off !== all.length) throw new Error(`weights: expected ${off} floats, got ${all.length}`);
    const dense = (x, name) => {
      const K = W[name + "/kernel"], b = W[name + "/bias"].data;
      const [nin, nout] = K.shape, k = K.data;
      const y = Float32Array.from(b);
      for (let i = 0; i < nin; i++) {
        const xi = x[i];
        if (xi === 0) continue;
        const row = i * nout;
        for (let j = 0; j < nout; j++) y[j] += xi * k[row + j];
      }
      return y;
    };
    const norm = (x, name) => {
      const s = W[name + "/scale"].data, b = W[name + "/bias"].data, n = x.length;
      let mean = 0;
      for (let i = 0; i < n; i++) mean += x[i];
      mean /= n;
      let v = 0;
      for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
      const inv = 1 / Math.sqrt(v / n + 1e-6);
      const y = new Float32Array(n);
      for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * s[i] + b[i];
      return y;
    };
    const gelu = (x) => { for (let i = 0; i < x.length; i++) { const t = x[i]; x[i] = 0.5 * t * (1 + Math.tanh(SQ2PI * (t + 0.044715 * t * t * t))); } return x; };

    return function forward(x0, goal) {
      const PL = net.position_limit, VS = Math.max(net.velocity_limit, 1e-3), H = net.horizon;
      const [px, py, vx, vy] = x0, [gx, gy] = goal;
      let dx = px + vx * H, dy = py + vy * H;
      if (net.gravity) dy -= 0.5 * net.gravity * H * H;
      const f = [px / PL, py / PL, vx / VS, vy / VS, gx / PL, gy / PL, (gx - px) / PL, (gy - py) / PL, (gx - dx) / PL, (gy - dy) / PL];
      let x = dense(f, "Dense_0");
      for (let i = 0; i < net.depth; i++) {
        const p = `ResidualBlock_${i}`;
        const h = dense(gelu(dense(norm(x, `${p}/LayerNorm_0`), `${p}/Dense_0`)), `${p}/Dense_1`);
        for (let j = 0; j < x.length; j++) x[j] += h[j];
      }
      const raw = dense(norm(x, "LayerNorm_0"), "Dense_1");
      const m = net.max_thrust, u = [];
      for (let k = 0; k < net.n_steps; k++) {
        const a = Math.tanh(raw[2 * k]) * m, b = Math.tanh(raw[2 * k + 1]) * m;
        u.push([Math.min(m, Math.max(-m, a)), Math.min(m, Math.max(-m, b))]);
      }
      return u;
    };
  }

  if (typeof window === "undefined") { module.exports = { makeNet }; return; }

  // ------------------------------------------------------------ data
  const runs = {};        // vehicle -> recorded run
  const bySig = {};       // signature -> vehicle
  const nets = {};        // vehicle -> Promise<forward>
  let catalog = null, defaults = {}, ready = null;

  const getJSON = async (path) => {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
  };
  function boot() {
    if (!ready) {
      ready = (async () => {
        const [cat, index] = await Promise.all([getJSON("vehicles.json"), getJSON(BASE + "index.json")]);
        catalog = cat; defaults = index.defaults || {};
        window.OTiSPlant.setCatalog(cat);
        await Promise.all(Object.keys(index.runs).map(async (vid) => {
          runs[vid] = await getJSON(`${BASE}${vid}.json`);
          bySig[runs[vid].signature] = vid;
        }));
      })();
    }
    return ready;
  }
  function netFor(vid) {
    if (!nets[vid]) {
      const run = runs[vid];
      nets[vid] = fetch(BASE + run.network.file)
        .then((r) => { if (!r.ok) throw new Error(`weights: HTTP ${r.status}`); return r.arrayBuffer(); })
        .then((buf) => makeNet(run.network, buf));
      nets[vid].catch(() => { delete nets[vid]; });
    }
    return nets[vid];
  }

  // ------------------------------------------------------------ training replay
  let shown = "rocket";   // which run /api/state describes when idle
  let replay = null;      // {vid, start, pausedAt, pausedFor, done}
  let version = 0, lastKey = "";

  const now = () => performance.now() / 1000;
  const elapsed = (r) => ((r.pausedAt ?? now()) - r.start - r.pausedFor);
  const sliceHistory = (h, n) => Object.fromEntries(Object.entries(h).map(([k, v]) => [k, v.slice(0, n)]));

  function frameAt(run, t) {
    const fr = run.frames;
    let lo = 0, hi = fr.length - 1;
    if (t >= fr[hi].t) return null;  // finished
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (fr[mid].t <= t) lo = mid; else hi = mid - 1; }
    return fr[lo];
  }

  function config() {
    return {
      ...defaults,
      steps: runs[shown]?.frames.at(-1)?.total_steps ?? defaults.steps,
      n_steps: 50, dt: 0.08, max_thrust: 1,
      sizes: { [defaults.size || "small"]: [runs.rocket?.network.width ?? 512, runs.rocket?.network.depth ?? 1] },
      device: `${runs.rocket?.device || "GPU"} (cuda)`,
      app_name: "OTiS", app_full_name: "Optimized Trajectory Simulator", catalog,
    };
  }
  const models = () => Object.values(runs).map((r) => ({ ...r.model }));

  function doneState(run) {
    // A stopped replay ends where it was, like a stopped run on the server.
    const cut = replay?.stoppedAt;
    return {
      phase: replay ? "done" : "pretrained",
      step: cut ? cut.step : run.frames.at(-1).step, total_steps: run.frames.at(-1).total_steps,
      history: cut ? sliceHistory(run.history, cut.hlen) : run.history,
      message: cut ? `Stopped at step ${cut.step}` : run.message, plant: run.plant,
      n_params: run.model.n_params, size: run.model.size,
      best_step: run.best_step, train_seconds: run.train_seconds,
      rate: run.frames.at(-1).rate,
      model: { ...run.model }, snapshots: cut ? cut.n_snap : run.snapshots.length,
    };
  }

  function state() {
    let st;
    if (replay && !replay.done) {
      const run = runs[replay.vid];
      const f = frameAt(run, elapsed(replay));
      if (!f) { replay.done = true; st = doneState(run); }
      else {
        const { t, n_snap, hlen, ...fields } = f;
        st = {
          ...fields,
          phase: replay.pausedAt != null && fields.phase === "training" ? "paused" : fields.phase,
          history: sliceHistory(run.history, hlen), message: "", plant: run.plant,
          model: { size: fields.size, n_params: fields.n_params, source: "training", step: fields.step, signature: run.signature },
          snapshots: n_snap,
        };
      }
    } else {
      st = doneState(runs[shown]);
    }
    const key = `${st.phase}:${st.step}:${st.snapshots}`;
    if (key !== lastKey) { lastKey = key; version++; }
    return { ...st, has_model: true, version, config: config(), models: models() };
  }

  function evolution() {
    const run = runs[replay ? replay.vid : shown];
    const st = state();
    return { snapshots: run.snapshots.slice(0, st.snapshots) };
  }

  function bad(message, http = 400) {
    const err = new Error(message);
    err.http = http; err.kind = "invalid"; err.data = { error: message, kind: "invalid" };
    return err;
  }

  function train(body) {
    if (replay && !replay.done) return { ok: false, message: "Training already running" };
    const spec = window.OTiSPlant.resolve(body.vehicle || "rocket", body.params);
    if (spec.error) throw bad(spec.error, 422);
    const vid = bySig[spec.signature];
    if (!vid) {
      return { ok: false, message: "Simulated training replays the GPU runs recorded for each preset vehicle, and these parameters differ from the preset. Press Reset in Parameters to train." };
    }
    shown = vid;
    replay = { vid, start: now(), pausedAt: null, pausedFor: 0, done: false };
    return { ok: true, message: "started" };
  }

  // ------------------------------------------------------------ planning
  async function plan(body) {
    const vehicle = body.vehicle || "rocket";
    const res = window.OTiSPlant.planLocal(vehicle, body.x0, body.goal, body.params);
    if (res.error) throw bad(res.error, 422);
    const spec = window.OTiSPlant.resolve(vehicle, body.params);
    const vid = bySig[res.spec.signature];
    const wantsNeural = (catalog.vehicles[vehicle]?.methods || []).includes("neural") && (!body.methods || body.methods.includes("neural"));
    if (!vid || !wantsNeural) {
      res.neural_status = { available: false, trainable: false,
        reason: "The networks in this demo were trained on the GPU for each preset vehicle. These parameters differ from the preset, so only the classical planners run. Press Reset to compare against the network." };
      return res;
    }
    const forward = await netFor(vid);
    forward(res.x0, res.goal);  // warm
    let best = Infinity, u = null;
    for (let i = 0; i < 3; i++) { const t0 = performance.now(); u = forward(res.x0, res.goal); best = Math.min(best, performance.now() - t0); }
    const plans = { neural: { ...window.OTiSPlant.score(u, res.x0, res.goal, spec), ms: best }, ...res.plans };
    res.plans = plans;
    const run = runs[vid];
    const status = { available: true, trainable: true, model: { ...run.model }, reason: "" };
    const ws = run.model.workspace, vs = run.model.start_speed;
    const maxPos = Math.max(...[res.x0[0], res.x0[1], res.goal[0], res.goal[1]].map(Math.abs));
    if (ws && maxPos > ws * 1.001) status.warning = `Outside the network's training workspace (±${ws} m): expect extrapolation error.`;
    else if (vs != null && Math.max(Math.abs(res.x0[2]), Math.abs(res.x0[3])) > vs * 1.001) status.warning = `Start velocity beyond the training range (±${vs} m/s): expect extrapolation error.`;
    res.neural_status = status;
    return res;
  }

  // ------------------------------------------------------------ router
  async function api(path, body) {
    await boot();
    switch (path) {
      case "/api/state": return state();
      case "/api/evolution": return evolution();
      case "/api/plan": return plan(body || {});
      case "/api/train": return train(body || {});
      case "/api/pause":
        if (replay && !replay.done && replay.pausedAt == null) replay.pausedAt = now();
        return { ok: true };
      case "/api/resume":
        if (replay && replay.pausedAt != null) { replay.pausedFor += now() - replay.pausedAt; replay.pausedAt = null; }
        return { ok: true };
      case "/api/stop":
        if (replay && !replay.done) { replay.stoppedAt = frameAt(runs[replay.vid], elapsed(replay)); replay.done = true; }
        return { ok: true };
      default: throw bad(`unknown endpoint ${path}`, 404);
    }
  }

  window.OTiSDemo = { api, makeNet };
})();
