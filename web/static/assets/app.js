(function () {
  const $ = (id) => document.getElementById(id);
  const IS_COMPACT = document.documentElement.classList.contains("ui-compact");

  /* ─── Property definitions ─── */
  const PROPS = [
    { key: "Density",                zh: "Density",    unit: "kg·m⁻³"         },
    { key: "ElectricalConductivity", zh: "Conductivity",  unit: "S·m⁻¹"          },
    { key: "HeatCapacity",           zh: "Heat Capacity",    unit: "J·mol⁻¹·K⁻¹"   },
    { key: "SurfaceTension",         zh: "Surface Tension", unit: "mN·m⁻¹"         },
    { key: "ThermalConductivity",    zh: "Thermal Cond.",  unit: "W·m⁻¹·K⁻¹"     },
    { key: "Viscosity",              zh: "Viscosity",    unit: "Pa·s"            },
  ];

  /* ─── Sample data ─── */
  const SAMPLES = [
    {
      name: "[BMIM][NTf₂]",
      cond: "298.15 K · 101.325 kPa",
      smiles: "CCCC[n+]1ccn(C)c1.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
      T: 298.15, P: 101.325,
    },
    {
      name: "[EMIM][NTf₂]",
      cond: "298.15 K · 101.325 kPa",
      smiles: "CC[n+]1ccn(C)c1.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
      T: 298.15, P: 101.325,
    },
    {
      name: "[HMIM][NTf₂]",
      cond: "298.15 K · 101.325 kPa",
      smiles: "CCCCCC[n+]1ccn(C)c1.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
      T: 298.15, P: 101.325,
    },
    {
      name: "[BMIM][BF₄]",
      cond: "298.15 K · 101.325 kPa",
      smiles: "CCCC[n+]1ccn(C)c1.[B-](F)(F)(F)F",
      T: 298.15, P: 101.325,
    },
    {
      name: "BMIM Amino Acid Salt",
      cond: "298.15 K · 101 kPa",
      smiles: "CCCC[n+]1ccn(C)c1.N[C@@H](CCC(=O)O)C(=O)[O-]",
      T: 298.15, P: 101,
    },
    {
      name: "Ether-chain Im. TFSI",
      cond: "298.15 K · 101.325 kPa",
      smiles: "COCCOCC[n+]1ccn(C)c1.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
      T: 298.15, P: 101.325,
    },
  ];

  /* ─── Normalisation ranges (typical IL values) ─── */
  const RANGES = [
    { log: false, min: 900,    max: 1700  },   // Density
    { log: true,  min: 0.001,  max: 5     },   // ElectricalConductivity (log)
    { log: false, min: 300,    max: 900   },   // HeatCapacity
    { log: false, min: 10,     max: 65    },   // SurfaceTension
    { log: false, min: 0.05,   max: 0.30  },   // ThermalConductivity
    { log: true,  min: 0.001,  max: 5     },   // Viscosity (log)
  ];

  const RADAR_COLORS = ['#3b82f6','#06b6d4','#7c3aed','#10b981','#f59e0b','#ef4444'];

  let lastValues = null;
  let apiBase = null;

  /* ─── Number formatting ─── */
  function fmt(x) {
    if (!isFinite(x)) return "—";
    const a = Math.abs(x);
    if (a >= 1000) return x.toFixed(2);
    if (a >= 10)   return x.toFixed(3);
    if (a >= 1)    return x.toFixed(4);
    if (a >= 0.01) return x.toFixed(5);
    return x.toExponential(3);
  }

  /* ─── API base detection ─── */
  async function resolveApiBase() {
    const forced = (new URLSearchParams(location.search).get("api") || "").replace(/\/$/, "");
    if (forced) return forced;
    if (location.protocol === "http:" || location.protocol === "https:") return "";
    try {
      const r = await fetch("http://127.0.0.1:8765/api/health");
      if (r.ok) return "http://127.0.0.1:8765";
    } catch { /* ignore */ }
    return null;
  }

  /* ─── Status badge ─── */
  function setBadge() {
    if (IS_COMPACT) {
      const b = $("modeBadge");
      if (!b) return;
      b.textContent = apiBase !== null ? "Connected" : "Offline";
      b.className   = "app-badge " + (apiBase !== null ? "live" : "warn");
      return;
    }
    const b = $("statusBadge");
    if (!b) return;
    const textEl = b.querySelector(".status-text");
    if (textEl) textEl.textContent = apiBase !== null ? "Model Ready" : "Offline";
    b.className = "status-dot " + (apiBase !== null ? "live" : "warn");
  }

  /* ─── Render result cards ─── */
  function renderGrid(values) {
    const grid = $("grid");
    if (!grid) return;
    grid.innerHTML = "";

    PROPS.forEach((p, i) => {
      const v = values[p.key];
      const el = document.createElement("div");

      if (IS_COMPACT) {
        el.className = "card";
        el.innerHTML =
          `<div class="card-n">${p.zh}</div>` +
          `<div class="card-v">${fmt(v)}</div>` +
          `<div class="card-u">${p.unit}</div>`;
      } else {
        el.className = "prop-card";
        el.style.animationDelay = (i * 0.05) + "s";
        el.innerHTML =
          `<div class="prop-accent"></div>` +
          `<div class="prop-name">${p.zh}</div>` +
          `<div class="prop-value">${fmt(v)}</div>` +
          `<div class="prop-unit">${p.unit}</div>`;
      }
      grid.appendChild(el);
    });

    const emptyEl = $("emptyState");
    if (emptyEl) emptyEl.classList.add("hidden");
    const outEl = $("out");
    if (outEl) outEl.classList.remove("hidden");

    /* Radar chart (website version) */
    if (!IS_COMPACT) drawRadar(values);
  }

  /* ─── Radar chart drawing ─── */
  function normalize(v, range) {
    if (range.log) {
      const lv   = Math.log10(Math.max(Math.abs(v), 1e-12));
      const lmin = Math.log10(range.min);
      const lmax = Math.log10(range.max);
      return Math.max(0.06, Math.min(1, (lv - lmin) / (lmax - lmin)));
    }
    return Math.max(0.06, Math.min(1, (v - range.min) / (range.max - range.min)));
  }

  function drawRadar(values) {
    const svg = $("radarSvg");
    if (!svg) return;

    const cx = 180, cy = 158, maxR = 108;
    const n  = PROPS.length;
    /* Clockwise from top (-90°) */
    const angles = Array.from({ length: n }, (_, i) =>
      (i * 2 * Math.PI / n) - Math.PI / 2);

    const pt = (r, i) => ({
      x: cx + r * Math.cos(angles[i]),
      y: cy + r * Math.sin(angles[i]),
    });

    let html = '';

    /* ── Grid hexagons ── */
    [0.2, 0.4, 0.6, 0.8, 1.0].forEach(lv => {
      const r = lv * maxR;
      const d = angles.map((_, i) => {
        const p = pt(r, i);
        return `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      }).join(' ') + ' Z';
      const alpha = lv === 1 ? 0.12 : 0.06;
      html += `<path d="${d}" fill="none" stroke="rgba(71,85,105,${alpha})" stroke-width="1"/>`;
    });

    /* ── Axis lines ── */
    for (let i = 0; i < n; i++) {
      const e = pt(maxR, i);
      html += `<line x1="${cx}" y1="${cy}" x2="${e.x.toFixed(1)}" y2="${e.y.toFixed(1)}"
        stroke="rgba(71,85,105,0.1)" stroke-width="1"/>`;
    }

    /* ── Data polygon ── */
    const normVals = PROPS.map((p, i) => normalize(values[p.key] || 0, RANGES[i]));
    const dataPts  = normVals.map((v, i) => pt(v * maxR, i));
    const dataPath = dataPts.map((p, i) =>
      `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`
    ).join(' ') + ' Z';

    html += `<path d="${dataPath}"
      fill="rgba(37,99,235,0.10)" stroke="#2563eb" stroke-width="2"
      stroke-linejoin="round" stroke-linecap="round"/>`;

    /* ── Data nodes ── */
    dataPts.forEach((p, i) => {
      html += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}"
        r="5" fill="${RADAR_COLORS[i]}" stroke="#fff" stroke-width="2"/>`;
    });

    /* ── Axis labels ── */
    const LBL_OFF = 18;
    PROPS.forEach((prop, i) => {
      const lp = pt(maxR + LBL_OFF, i);
      /* Text anchor alignment */
      const anchor = lp.x < cx - 6 ? 'end' : lp.x > cx + 6 ? 'start' : 'middle';
      /* Vertical nudge: up for top labels, down for bottom */
      const dy = lp.y < cy - 6 ? -4 : lp.y > cy + 6 ? 10 : 4;
      html += `<text x="${lp.x.toFixed(1)}" y="${(lp.y + dy).toFixed(1)}"
        text-anchor="${anchor}"
        font-size="11" font-weight="600" fill="#475569"
        font-family="Noto Sans SC,Inter,sans-serif">${prop.zh}</text>`;
    });

    svg.innerHTML = html;
  }

  /* ─── Call prediction API ─── */
  async function predictLive(body) {
    const res = await fetch((apiBase || "") + "/api/predict/one", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; }
    catch { throw new Error(text || "Invalid response"); }
    if (!res.ok) {
      const d = data.detail;
      throw new Error(typeof d === "string" ? d : JSON.stringify(data));
    }
    const row = (data.results && data.results[0]) || {};
    return { values: row.values || {}, meta: row };
  }

  /* ─── Form submit ─── */
  $("f").addEventListener("submit", async (e) => {
    e.preventDefault();
    const errEl = $("err");
    if (errEl) errEl.textContent = "";

    const smiles = $("smiles").value.trim();
    const T = parseFloat($("temp").value);
    const pressEl = $("press");
    const Praw = pressEl ? pressEl.value.trim() : "";
    const P = Praw === "" ? 101.325 : parseFloat(Praw);

    if (!smiles) {
      if (errEl) errEl.textContent = "Please enter the ionic liquid SMILES.";
      return;
    }
    if (apiBase === null) {
      if (errEl) errEl.textContent = "Backend not connected. Start serve_screening_ui.py and refresh the page.";
      return;
    }

    const go = $("go"), spin = $("spin"), lbl = $("btnLabel");
    go.disabled = true;
    if (spin) spin.classList.remove("hidden");
    if (lbl)  lbl.classList.add("hidden");

    try {
      const r = await predictLive({
        IL_SMILES: smiles,
        Temperature_K: T,
        Pressure_kPa: Praw === "" ? null : P,
      });

      if (r.meta.graph_error && errEl) {
        errEl.textContent = "Graph warning: " + r.meta.graph_error;
      }

      const pLabel = Praw === "" ? "101.325" : String(P);
      const metaEl = $("meta");
      if (metaEl) {
        metaEl.textContent = `T = ${T} K  ·  P = ${pLabel} kPa  ·  MIPGraph`;
      }

      lastValues = r.values;
      renderGrid(r.values);
    } catch (err) {
      if (errEl) errEl.textContent = err.message || String(err);
      const emptyEl = $("emptyState");
      if (emptyEl) emptyEl.classList.remove("hidden");
      const outEl = $("out");
      if (outEl) outEl.classList.add("hidden");
    } finally {
      go.disabled = false;
      if (spin) spin.classList.add("hidden");
      if (lbl)  lbl.classList.remove("hidden");
    }
  });

  /* ─── Sample list ─── */
  const chipsEl = $("chips");
  if (chipsEl) {
    SAMPLES.forEach((x, i) => {
      const btn = document.createElement("button");
      btn.type = "button";

      if (IS_COMPACT) {
        btn.className = "chip";
        btn.textContent = x.name;
      } else {
        btn.className = "sample-item";
        btn.innerHTML =
          `<span class="sample-dot sample-dot-${i}"></span>` +
          `<span class="sample-name">${x.name}</span>` +
          `<span class="sample-cond">${x.cond}</span>`;
      }

      btn.addEventListener("click", () => {
        const smilesEl = $("smiles"), tempEl = $("temp"), pressEl2 = $("press");
        if (smilesEl) smilesEl.value = x.smiles;
        if (tempEl)   tempEl.value   = x.T;
        if (pressEl2) pressEl2.value = x.P;

        const errEl = $("err");
        if (errEl) errEl.textContent = "";
        const outEl = $("out"), emptyEl = $("emptyState");
        if (outEl)   outEl.classList.add("hidden");
        if (emptyEl) emptyEl.classList.remove("hidden");

        if (!IS_COMPACT && smilesEl) {
          smilesEl.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });

      chipsEl.appendChild(btn);
    });
  }

  /* ─── Copy All ─── */
  const copyAllBtn = $("copyAll");
  if (copyAllBtn) {
    copyAllBtn.addEventListener("click", () => {
      if (!lastValues) return;
      const text = PROPS.map(p => `${p.zh}: ${fmt(lastValues[p.key])} ${p.unit}`).join('\n');
      navigator.clipboard.writeText(text).then(() => {
        copyAllBtn.textContent = "Copied ✓";
        copyAllBtn.classList.add("copied");
        setTimeout(() => {
          copyAllBtn.innerHTML =
            `<svg viewBox="0 0 16 16" fill="none" width="13" height="13">` +
            `<rect x="5" y="5" width="9" height="10" rx="1.5" stroke="currentColor" stroke-width="1.4"/>` +
            `<path d="M3 11V3a1 1 0 0 1 1-1h8" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>` +
            `</svg> Copy All`;
          copyAllBtn.classList.remove("copied");
        }, 2200);
      }).catch(() => {});
    });
  }

  /* ─── Init ─── */
  (async function init() {
    if (!IS_COMPACT) {
      const b = $("statusBadge");
      if (b) {
        const textEl = b.querySelector(".status-text");
        if (textEl) textEl.textContent = "Checking…";
        b.className = "status-dot";
      }
    }
    apiBase = await resolveApiBase();
    setBadge();
  })();
})();
