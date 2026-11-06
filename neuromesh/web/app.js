/* NeuroMesh web interface: a live view built on the daemon's JSON API (see server.py / web.py).
   No framework and no external requests: plain DOM, one poll of /dashboard every 1.5 s, and the
   activity log fetched by sequence number so only new entries travel. */
"use strict";
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const XLINK = "http://www.w3.org/1999/xlink";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ DOM helpers
  function setAttrs(el, attrs) {
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "text") el.textContent = v;
      else el.setAttribute(k, v === true ? "" : String(v));
    }
  }
  function addKids(el, kids) {
    for (const k of kids.flat(Infinity)) {
      if (k === null || k === undefined || k === false) continue;
      el.append(k instanceof Node ? k : document.createTextNode(String(k)));
    }
  }
  function h(tag, attrs, ...kids) { const el = document.createElement(tag); setAttrs(el, attrs); addKids(el, kids); return el; }
  function sv(tag, attrs, ...kids) { const el = document.createElementNS(NS, tag); setAttrs(el, attrs); addKids(el, kids); return el; }
  function fill(el, ...kids) { el.replaceChildren(); addKids(el, kids); return el; }
  // a panel with something typed into it is left alone until it is sent or cleared
  function busy(el) {
    if (!el) return false;
    if (el.contains(document.activeElement) && document.activeElement !== document.body) return true;
    return $$("input[type=text], input:not([type]), textarea", el).some(i => i.value.trim() !== "" && !i.dataset.free);
  }

  // ------------------------------------------------------------------ storage (may be unavailable)
  const store = {
    get(k, d = null) { try { const v = localStorage.getItem("neuromesh." + k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("neuromesh." + k, v); } catch { /* private window */ } },
  };

  // ------------------------------------------------------------------ formatting
  const pad = n => String(n).padStart(2, "0");
  function clock(ts) { const d = new Date(ts * 1000); return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; }
  function dateTime(ts) { const d = new Date(ts * 1000); return `${d.toLocaleDateString()} ${clock(ts)}`; }
  function span(sec) {
    sec = Math.max(0, Math.round(sec));
    if (sec < 90) return `${sec}s`;
    if (sec < 5400) return `${Math.round(sec / 60)}m`;
    if (sec < 172800) return `${Math.floor(sec / 3600)}h ${pad(Math.round((sec % 3600) / 60))}m`;
    return `${Math.floor(sec / 86400)}d ${Math.floor((sec % 86400) / 3600)}h`;
  }
  function took(sec) { return sec < 1 ? `${Math.round(sec * 1000)}ms` : `${sec.toFixed(1)}s`; }
  function bytes(n) {
    if (!n) return "0 B";
    if (n < 10000) return `${n.toLocaleString()} B`;
    if (n < 1e6) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1048576).toFixed(2)} MB`;
  }
  const pct = x => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
  const num = n => (n ?? 0).toLocaleString();
  const cut = (t, n) => { t = String(t ?? ""); return t.length > n ? t.slice(0, n - 1) + "…" : t; };
  const shape = arch => String(arch || "").replace(/x/g, "×");
  const goalName = id => { const s = String(id).replace(/^G\d+-/, "").replace(/-/g, " "); return s.charAt(0).toUpperCase() + s.slice(1); };

  // ------------------------------------------------------------------ state
  const S = {
    token: store.get("token", ""),
    after: 0, log: [], logVersion: 0, rendered: {},
    dash: null, lastOrderTs: 0, failures: 0,
    view: "live", logsTab: "activity", lineageCap: "", memNode: null, memDetail: null, memTests: null,
    filter: { q: "", levels: new Set(), routine: true },
    extra: {}, extraAt: {},
    haltArmed: 0, meshKey: "", layerKey: "", geometry: null,
    history: JSON.parse(store.get("history", "[]") || "[]"), histPos: -1,
  };

  // ------------------------------------------------------------------ API
  async function api(path, body) {
    const opt = { headers: {} };
    if (S.token) opt.headers.Authorization = "Bearer " + S.token;
    if (body !== undefined) {
      opt.method = "POST";
      opt.headers["Content-Type"] = "application/json";
      opt.body = JSON.stringify(body);
    }
    const r = await fetch(path, opt);
    if (r.status === 401) { askToken(); throw Object.assign(new Error("NeuroMesh needs its access token"), { auth: true }); }
    let data = {};
    try { data = await r.json(); } catch { /* empty body */ }
    if (!r.ok) throw Object.assign(new Error(data.message || data.error || `${r.status} ${r.statusText}`), { data, status: r.status });
    return data;
  }

  function askToken() {
    const dlg = $("#token");
    if (dlg.open) return;
    $("#token-input").value = "";
    try { dlg.showModal(); } catch { dlg.setAttribute("open", ""); }
  }
  $("#token-form").addEventListener("submit", () => {
    S.token = $("#token-input").value.trim();
    store.set("token", S.token);
    S.failures = 0;
    setTimeout(start, 50);
  });

  let toastTimer = 0;
  function toast(text) {
    let t = $(".toast");
    if (!t) { t = h("div", { class: "toast", role: "status" }); document.body.append(t); }
    t.textContent = text;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.remove(), 4200);
  }

  // run an action from a button: disable it while it works, report the outcome
  async function act(button, fn, done) {
    if (button) button.disabled = true;
    try {
      const r = await fn();
      if (done) toast(typeof done === "function" ? done(r) : done);
      await tick(true);
      return r;
    } catch (e) {
      toast(e.message);
    } finally {
      if (button) button.disabled = false;
    }
  }

  // ------------------------------------------------------------------ theme
  function applyTheme() {
    const pref = store.get("theme", "");
    if (pref) document.documentElement.dataset.theme = pref; else delete document.documentElement.dataset.theme;
    const dark = pref ? pref === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    $("#theme").textContent = dark ? "Light" : "Dark";
  }
  $("#theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    store.set("theme", dark ? "light" : "dark");
    applyTheme();
  });
  applyTheme();

  // ------------------------------------------------------------------ header
  function renderHeader(st) {
    document.title = `${st.name}${st.halted ? " (halted)" : ""}`;
    $("#name").textContent = st.name;
    const chip = $("#state");
    if (st.halted) { chip.textContent = `Halted: ${cut(st.halt_reason, 60)}`; chip.className = "state halted"; }
    else if (st.dreams && st.dreams.asleep) { chip.textContent = `Asleep, dreaming (${st.dreams.dreams_tonight})`; chip.className = "state asleep"; }
    else if (st.background_paused) { chip.textContent = "Background paused"; chip.className = "state paused"; }
    else { chip.textContent = "Running"; chip.className = "state running"; }

    const controls = $("#controls");
    const btns = [];
    if (st.halted) {
      btns.push(h("button", { type: "button", class: "primary", onclick: e => act(e.target, () => api("/resume", { by: "web" }), "Resumed") }, "Resume"));
    } else {
      btns.push(st.background_paused
        ? h("button", { type: "button", onclick: e => act(e.target, () => api("/pause", { paused: false, by: "web" }), "Background work resumed") }, "Resume background")
        : h("button", { type: "button", onclick: e => act(e.target, () => api("/pause", { paused: true, by: "web" }), "Background work paused") }, "Pause background"));
      const armed = Date.now() < S.haltArmed;
      btns.push(h("button", {
        type: "button", class: "danger" + (armed ? " armed" : ""),
        title: "Stops all orders and background work until you resume",
        onclick: e => {
          if (Date.now() < S.haltArmed) { S.haltArmed = 0; act(e.target, () => api("/halt", { reason: "halted from the web interface", by: "web" }), "Halted"); }
          else { S.haltArmed = Date.now() + 4000; renderHeader(st); setTimeout(() => S.dash && renderHeader(S.dash.status), 4100); }
        },
      }, armed ? "Click again to halt" : "Halt"));
    }
    fill(controls, btns);

    const n = st.nodes, mem = st.memory;
    const items = [
      ["Uptime", span(st.uptime_s)],
      ["Orders", num(st.orders) + (st.inflight ? ` (${st.inflight} in flight)` : "")],
      ["Nodes", `${n.live} live, ${n.ever} built`],
      mem && mem.enabled && mem.nodes.length ? ["Memory", `${mem.facts} facts in ${bytes(mem.bytes)}`] : null,
      ["Background", pct(st.background_duty_cycle)],
      ["Compute", st.compute],
    ].filter(Boolean);
    fill($("#stats"), items.map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, v))));

    const asks = (st.mind?.inquiries?.length || 0) + (st.mind?.proposals?.length || 0);
    $("#badge-mind").textContent = asks ? String(asks) : "";
    const pending = (st.approvals?.length || 0) + (st.questions?.length || 0);
    $("#badge-status").textContent = pending ? String(pending) : "";
  }

  function setConn(ok, text) {
    const c = $("#conn");
    c.textContent = text;
    c.className = "conn" + (ok ? "" : " down");
    if (!ok) { const chip = $("#state"); chip.textContent = "Not reachable"; chip.className = "state down"; }
  }

  // ------------------------------------------------------------------ command bar and answers
  const cmd = $("#cmd");
  $("#command").addEventListener("submit", async e => {
    e.preventDefault();
    const text = cmd.value.trim();
    if (!text) return;
    S.history = [text, ...S.history.filter(x => x !== text)].slice(0, 50);
    store.set("history", JSON.stringify(S.history));
    S.histPos = -1;
    const button = $("#send");
    button.disabled = true;
    showAnswer({ pending: text });
    try {
      if (text.startsWith("/")) {
        const r = await api("/command", { line: text });
        showAnswer({ text, lines: r.lines });
      } else {
        const r = await api("/order", { text, source: "web" });
        showAnswer({ text, order: r });
      }
      cmd.value = "";
      tick(true);
    } catch (err) {
      showAnswer({ text, error: err.message });
    } finally {
      button.disabled = false;
      cmd.focus();
    }
  });
  cmd.addEventListener("keydown", e => {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    if (!S.history.length) return;
    e.preventDefault();
    S.histPos = Math.max(-1, Math.min(S.history.length - 1, S.histPos + (e.key === "ArrowUp" ? 1 : -1)));
    cmd.value = S.histPos < 0 ? "" : S.history[S.histPos];
  });

  function showAnswer(a) {
    const box = $("#answer");
    box.hidden = false;
    box.className = "answer";
    if (a.pending) { fill(box, h("div", { class: "src" }, `Working on “${cut(a.pending, 120)}”…`)); return; }
    const close = h("button", { type: "button", class: "quiet small", onclick: () => { box.hidden = true; } }, "Close");
    if (a.error) {
      box.classList.add("failed");
      fill(box, h("div", { class: "lead" }, a.error), h("div", { class: "meta" }, cut(a.text, 100), close));
      return;
    }
    if (a.lines) {
      fill(box, h("pre", { class: "mono" }, a.lines.join("\n") || "(no output)"), h("div", { class: "meta" }, a.text, close));
      return;
    }
    const r = a.order;
    if (!r.ok) box.classList.add("failed");
    const text = r.answer || String(r.output ?? r.note ?? "");
    const m = text.match(/\n(Sources?: [\s\S]*)$/);
    const lead = m ? text.slice(0, m.index) : text;
    const kids = [h("div", { class: "lead" }, lead)];
    if (m) kids.push(h("div", { class: "src" }, m[1]));
    if (r.inquiry) {
      kids.push(h("div", { class: "ask" }, h("p", {}, r.inquiry.question), replyForm(r.inquiry.id, [])));
    }
    const meta = [
      h("span", {}, `route ${r.route || "–"}`), h("span", {}, `confidence ${pct(r.confidence)}`),
      h("span", {}, `served by ${r.served_by}`), h("span", {}, r.order_id),
    ];
    if (r.disposition) meta.push(h("span", {}, `temperament: ${r.disposition}`));
    kids.push(h("div", { class: "meta" }, meta, close));
    if (r.trace?.length) kids.push(h("details", {}, h("summary", {}, "How it got there"), h("ul", {}, r.trace.map(t => h("li", {}, t)))));
    fill(box, kids);
  }

  // ------------------------------------------------------------------ the mesh, drawn like a specimen
  const radius = params => Math.max(7, Math.min(17, 5 + 2.4 * Math.log2(Math.max(1, (params || 4000) / 4000))));
  const hexR = b => Math.max(6.5, Math.min(12, 4.5 + 1.4 * Math.log2(Math.max(1, (b || 256) / 256))));
  function hexagon(cx, cy, r) {
    const pts = [];
    for (let k = 0; k < 6; k++) { const a = Math.PI / 6 + (k * Math.PI) / 3; pts.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`); }
    return pts.join(" ");
  }
  function nodeTip(n) {
    return `${n.id} (${n.status}, ${n.origin}, generation ${n.generation})\nshape ${shape(n.arch)}, ${num(n.params)} weights\n` +
      `held-out ${pct(n.holdout)}, ${num(n.steps)} training slices, served ${num(n.serves)}`;
  }

  function meshGeometry(m, width) {
    const n = Math.max(m.routes.length, 1);
    const rowH = 50, top = m.director ? 92 : 22;
    const routerR = radius(m.router.champion?.params);
    const yMid = top + (n * rowH) / 2;
    const lane = top + n * rowH + 42;
    const longest = Math.max(10, ...m.routes.map(r => Math.min(26, r.route.length)));
    const label = 300, target = label + Math.round(longest * 7.8) + 30;
    return { W: width, H: lane + 34 + (m.dreams ? 56 : 0), rowH, top, yMid, lane, routerR, X: { inlet: 16, router: 160, label, target },
             y: i => top + i * rowH + rowH / 2, chars: x => Math.max(12, Math.floor((width - x - 12) / 6.9)) };
  }
  function routePath(g, i) {
    const y = g.y(i), X = g.X;
    return `M ${X.router} ${g.yMid} C ${X.router + 95} ${g.yMid}, ${X.label - 95} ${y}, ${X.label - 10} ${y}`;
  }

  function directorGlyph(x, y, r, n, ds) {  // a diamond: the directors, at both ends of every order
    const cls = "m-director" + (n.role === "primary" ? "" : " dup") + (n.busy ? " busy" : "") + (ds && ds.wary ? " wary" : "");
    const tip = `${n.id}${n.role === "duplicate" ? ` (a duplicate of ${n.parent})` : n.role === "primary" ? " (the primary director)" : ""}: ${n.doing}` +
      (n.handled ? `\n${n.handled.intake || 0} taken, ${n.handled.review || 0} reviewed, ${n.handled.consult || 0} second opinions` : "") +
      (ds ? `\nstance: ${ds.mood}${ds.wary ? " (wary)" : ""}` : "");
    return sv("polygon", { points: `${x},${y - r} ${x + r},${y} ${x},${y + r} ${x - r},${y}`, class: cls }, sv("title", {}, tip));
  }

  function drawMesh(m, width) {
    const g = meshGeometry(m, width);
    const { X } = g;
    const svg = sv("svg", { viewBox: `0 0 ${g.W} ${g.H}`, width: g.W, height: g.H, role: "img",
      "aria-label": `Node network: the router sends orders to ${m.routes.length} routes; every answer passes the emotional engines` });
    // its own initiative: the agency node in front of the director
    const dr = m.director;
    if (dr) {
      const y = 36, an = dr.node;
      svg.append(sv("text", { x: X.inlet, y: y + 4, class: "m-sub" }, "initiative"));
      if (an) {
        const rr = Math.min(13, radius(an.params));
        if (dr.trusted) svg.append(sv("circle", { cx: X.router, cy: y, r: rr + 4, class: "m-champ-ring" }));
        svg.append(sv("circle", { cx: X.router, cy: y, r: rr, class: dr.trusted ? "m-champ" : "m-chall" },
          sv("title", {}, `agency node: ${nodeTip(an)}${dr.trusted ? "\nit chooses the director's mode" : "\nstill learning; the drive policy chooses"}`)));
      } else {
        svg.append(sv("circle", { cx: X.router, cy: y, r: 8, class: "m-none" }, sv("title", {}, "agency node: not built yet; the drive policy chooses")));
      }
      svg.append(sv("text", { x: X.router + 20, y: y - 14, class: "m-dim" }, an ? (dr.trusted ? "agency node" : "agency node (learning)") : "drive policy"));
      svg.append(sv("path", { d: `M ${X.router + 16} ${y} L ${X.label - 10} ${y}`, class: "m-edge" + (dr.enabled ? "" : " inactive") }));
      const ds = m.directors;
      const head = sv("text", { x: X.label, y: y + 4.5, class: "m-route" }, ds && ds.nodes.length > 1 ? "directors" : "director");
      svg.append(head);
      if (ds) {  // the director nodes: director-0 and the duplicates it made
        let x = X.label + (ds.nodes.length > 1 ? 84 : 74);
        ds.nodes.slice(0, 6).forEach(n => {
          const r = n.role === "primary" ? 7 : 5.5;
          svg.append(directorGlyph(x, y, r, n, ds));
          x += r * 2 + 7;
        });
      }
      const room = g.chars(X.target);
      let text;
      if (!dr.enabled) text = "initiative off";
      else {
        const mode = dr.pinned === "auto" ? (dr.last ? `last chose ${dr.last.mode}` : "has not decided yet") : `pinned to ${dr.pinned}`;
        const tasks = dr.tasks.length ? `its tasks: ${dr.tasks.map(t => `${t.id} ${t.mode}`).join(", ")}` : (dr.last?.summary || "");
        text = `${mode}${tasks ? "; " + tasks : ""}`;
      }
      svg.append(sv("text", { x: X.target, y: y + 4.5, class: "m-text" }, cut(text, room)));
      svg.append(sv("line", { x1: 16, x2: g.W - 16, y1: 68, y2: 68, class: "m-lane" }));
    }
    // orders flow in to the router
    svg.append(sv("text", { x: X.inlet, y: g.yMid + 4, class: "m-sub" }, "orders"));
    svg.append(sv("path", { d: `M 70 ${g.yMid} L ${X.router - g.routerR - 4} ${g.yMid}`, class: "m-edge" }));
    if (m.directors) {  // a director takes every order first
      const dx = Math.round((70 + X.router - g.routerR) / 2) - 4;
      svg.append(directorGlyph(dx, g.yMid, 6, { ...m.directors.nodes[0], id: "a director", doing: "takes every order before anything else touches it" }, m.directors));
      svg.append(sv("text", { x: dx, y: g.yMid + 22, "text-anchor": "middle", class: "m-dim" }, "director"));
    }
    m.routes.forEach((r, i) => svg.append(sv("path", { d: routePath(g, i), class: "m-edge" + (r.inactive ? " inactive" : "") })));
    // the router
    const rc = m.router.champion;
    if (rc) {
      svg.append(sv("circle", { cx: X.router, cy: g.yMid, r: g.routerR + 4, class: "m-champ-ring" }));
      svg.append(sv("circle", { cx: X.router, cy: g.yMid, r: g.routerR, class: "m-champ" }, sv("title", {}, nodeTip(rc))));
      svg.append(sv("text", { x: X.router, y: g.yMid + g.routerR + 18, "text-anchor": "middle", class: "m-text" }, "router"));
      svg.append(sv("text", { x: X.router, y: g.yMid + g.routerR + 33, "text-anchor": "middle", class: "m-sub" }, `held-out ${pct(rc.holdout)}`));
      m.router.candidates.forEach((c, j) => svg.append(sv("circle", { cx: X.router - 18 + j * 12, cy: g.yMid - g.routerR - 12, r: 4, class: "m-chall" }, sv("title", {}, "challenger " + nodeTip(c)))));
    } else {
      svg.append(sv("circle", { cx: X.router, cy: g.yMid, r: 9, class: "m-none" }));
      svg.append(sv("text", { x: X.router, y: g.yMid + 28, "text-anchor": "middle", class: "m-sub" }, "router: not built yet"));
      svg.append(sv("text", { x: X.router, y: g.yMid + 43, "text-anchor": "middle", class: "m-dim" }, "teachers route orders"));
    }
    // each route and what serves it
    m.routes.forEach((r, i) => {
      const y = g.y(i);
      svg.append(sv("text", { x: X.label, y: y + 4.5, class: "m-route", "data-route": r.route }, cut(r.route, 26)));
      const tx = X.target;
      if (r.skill) {
        const f = r.skill, c = f.champion;
        let x = tx + 14;
        if (c) {
          const rr = radius(c.params);
          svg.append(sv("circle", { cx: x, cy: y, r: rr, class: "m-champ" }, sv("title", {}, nodeTip(c))));
          x += rr + 9;
        } else {
          svg.append(sv("circle", { cx: x, cy: y, r: 7, class: "m-none" }));
          x += 16;
        }
        f.candidates.forEach(n => { svg.append(sv("circle", { cx: x, cy: y, r: 4.5, class: "m-chall" }, sv("title", {}, "challenger " + nodeTip(n)))); x += 12; });
        f.retired.forEach(n => { svg.append(sv("circle", { cx: x, cy: y, r: 4.5, class: "m-retired" }, sv("title", {}, "retired " + nodeTip(n)))); x += 12; });
        const t = sv("text", { x: x + 6, y: y + 4.5, class: "m-text" });
        const room = g.chars(x + 6);
        if (c) {
          const detail = `${shape(c.arch).split(" ")[0]}, held-out ${pct(c.holdout)}, served ${num(c.serves)}`;
          t.append(sv("tspan", {}, c.id), sv("tspan", { class: "m-sub", dx: 8 }, cut(detail, Math.max(8, room - c.id.length - 2))));
          if (f.candidates.length && room > c.id.length + detail.length + 16) t.append(sv("tspan", { class: "m-dim", dx: 8 }, `${f.candidates.length} challenging`));
        } else {
          t.append(sv("tspan", { class: "m-sub" }, cut(`no node yet (${f.examples} examples); teachers answer`, room)));
        }
        svg.append(t);
      } else if (r.tool) {
        const t = r.tool;
        const cls = { active: "", pending_approval: " pending", approved: " pending", quarantined: " bad", rejected: " bad", retired: " off" }[t.state] ?? " pending";
        svg.append(sv("rect", { x: tx + 6, y: y - 8, width: 16, height: 16, rx: 3, class: "m-tool" + cls }, sv("title", {}, `${t.name} v${t.version} [${t.state}] ${t.note || ""}`)));
        const txt = sv("text", { x: tx + 32, y: y + 4.5, class: "m-text" }, sv("tspan", {}, `${t.name} v${t.version}`));
        txt.append(sv("tspan", { class: "m-sub", dx: 8 }, `${t.state === "active" ? "" : t.state.replace("_", " ") + ", "}${num(t.runs)} runs${t.provides === "knowledge" ? ", research" : ""}`));
        svg.append(txt);
      } else if (r.route === "answer:direct") {
        let x = tx + 12;
        const memNodes = m.memory.slice(-8);
        memNodes.forEach(n => {
          const rr = hexR(n.bytes);
          svg.append(sv("polygon", { points: hexagon(x, y, rr), class: "m-mem" + (n.status === "learning" ? " learning" : "") },
            sv("title", {}, `${n.id}: ${n.facts} facts in ${bytes(n.bytes)}${n.first_bytes > n.bytes ? ` (first ${bytes(n.first_bytes)})` : ""}${n.parts > 1 ? `, merged from ${n.parts} nodes` : ""}\n${n.spec}\ntests ${n.passed}/${n.total}, served ${n.served}`)));
          x += rr * 2 + 4;
        });
        const facts = m.memory.reduce((a, n) => a + n.facts, 0);
        const teachers = r.teachers.length ? `teachers: ${r.teachers.join(", ")}` : "no teacher model (a person is asked)";
        const txt = sv("text", { x: x + 8, y: y + 4.5, class: "m-text" });
        const room = g.chars(x + 8);
        const head = `memory nodes first (${facts} facts)`;
        if (memNodes.length) txt.append(sv("tspan", {}, head), sv("tspan", { class: "m-sub", dx: 8 }, cut(`then ${teachers}`, Math.max(8, room - head.length - 2))));
        else txt.append(sv("tspan", { class: "m-sub" }, cut(teachers, room)));
        svg.append(txt);
      } else {
        svg.append(sv("text", { x: tx, y: y + 4.5, class: "m-dim" }, r.inactive ? "inactive" : ""));
      }
    });
    // every answer passes the emotional engines on its way out
    const ly = g.lane;
    svg.append(sv("line", { x1: 16, x2: g.W - 16, y1: ly - 24, y2: ly - 24, class: "m-lane" }));
    svg.append(sv("text", { x: X.inlet, y: ly + 4, class: "m-sub" }, "answers"));
    {
      let x = 120;
      const long = { affect: "affect reads the tone", disposition: "disposition decides" };
      let stops = m.engines.map(e => ({ label: long[e.capability] || e.capability, short: e.capability, node: e.champion }));
      if (m.engines.length) {
        const guard = m.guard ? `honesty guard: ${m.guard.checked} checked, ${m.guard.changed} changed` : "honesty guard";
        stops.push({ label: guard, short: m.guard ? `guard: ${m.guard.changed} of ${m.guard.checked} changed` : "honesty guard", guard: true });
      }
      if (m.directors) stops.push({ label: m.engines.length ? "a director reviews" : "a director reviews (temperament off)", short: "director", director: true });
      const need = labels => 120 + labels.reduce((a, st) => a + 48 + st.length * 7.3, 0) + 40;
      if (need(stops.map(st => st.label)) > g.W) stops = stops.map(st => ({ ...st, label: st.short }));
      stops.forEach((st, k) => {
        if (k) svg.append(sv("line", { x1: x - 34, x2: x - 12, y1: ly, y2: ly, class: "m-edge" }));
        if (st.director) svg.append(directorGlyph(x, ly, 6, { ...m.directors.nodes[0], id: "a director", doing: "reviews every result before you see it or memory keeps it" }, m.directors));
        else if (st.guard) svg.append(sv("rect", { x: x - 6, y: ly - 6, width: 12, height: 12, rx: 2, class: "m-tool" }));
        else if (st.node) svg.append(sv("circle", { cx: x, cy: ly, r: 6.5, class: "m-champ" }, sv("title", {}, nodeTip(st.node))));
        else svg.append(sv("circle", { cx: x, cy: ly, r: 6, class: "m-none" }, sv("title", {}, "no node yet")));
        const t = sv("text", { x: x + 12, y: ly + 4.5, class: "m-sub" }, st.label);
        svg.append(t);
        x += 48 + st.label.length * 7.3;
      });
      svg.append(sv("line", { x1: x - 34, x2: x - 12, y1: ly, y2: ly, class: "m-edge" }));
      svg.append(sv("text", { x: x - 4, y: ly + 4.5, class: "m-text" }, "you"));
    }
    // dreams: the dream nodes it sleeps and imagines with
    if (m.dreams) {
      const dy = ly + 56, dm = m.dreams;
      svg.append(sv("line", { x1: 16, x2: g.W - 16, y1: dy - 28, y2: dy - 28, class: "m-lane" }));
      svg.append(sv("text", { x: X.inlet, y: dy + 4, class: "m-sub" }, "dreams"));
      let x = X.router - 10;
      dm.nodes.slice(0, 6).forEach(n => {
        const rr = Math.max(5, Math.min(12, 3 + 1.6 * Math.log2(Math.max(1, n.weights / 4000))));
        svg.append(sv("circle", { cx: x, cy: dy, r: rr, class: (n.champion ? "m-dream" : "m-dream-alt") + (dm.asleep ? " asleep" : "") },
          sv("title", {}, `${n.id}: generation ${n.generation}, ${n.shape}, ${num(n.weights)} weights; ${n.fitness === null ? "not checked yet" : pct(n.fitness) + " of its checked dreams right"}${n.champion ? "\nthe best dreamer: imagination uses it" : ""}`)));
        x += rr * 2 + 6;
      });
      const best = dm.nodes.find(n => n.champion);
      const text = dm.asleep ? `asleep, dreaming: ${dm.tonight} dreams tonight (everything else waits)`
        : best ? `awake; ${best.id} dreams best (generation ${best.generation}${best.fitness === null ? "" : `, ${pct(best.fitness)} of checked dreams right`}); "imagine ..." uses it`
        : "no dream nodes yet: it dreams the first time it sleeps";
      svg.append(sv("text", { x: Math.max(x + 8, X.label), y: dy + 4.5, class: dm.asleep ? "m-route hit" : "m-text" }, cut(text, g.chars(Math.max(x + 8, X.label)))));
    }
    return { svg, g };
  }

  function drawLayer(m, g) {
    const svg = sv("svg", { viewBox: `0 0 ${g.W} ${g.H}`, "aria-hidden": "true" });
    m.routes.forEach((r, i) => {
      svg.append(sv("path", { id: `pulse-${i}`, d: `M 70 ${g.yMid} L ${g.X.router} ${g.yMid} ` + routePath(g, i).replace(/^M [\d.]+ [\d.]+ /, ""), fill: "none", stroke: "none" }));
    });
    return svg;
  }

  function pulse(route) {
    const m = S.dash?.mesh;
    if (!m) return;
    const i = m.routes.findIndex(r => r.route === route);
    if (i < 0) return;
    const label = $(`#specimen [data-route="${CSS.escape(route)}"]`);
    if (label) { label.classList.add("hit"); setTimeout(() => label.classList.remove("hit"), 900); }
    if (reduceMotion()) return;
    const layer = $("#specimen .layer svg");
    if (!layer) return;
    const dot = sv("circle", { r: 4.5, class: "m-pulse" });
    const anim = sv("animateMotion", { dur: "0.8s", fill: "freeze", begin: "indefinite", calcMode: "paced" });
    const mp = sv("mpath", {});
    mp.setAttribute("href", `#pulse-${i}`);
    mp.setAttributeNS(XLINK, "xlink:href", `#pulse-${i}`);
    anim.append(mp);
    dot.append(anim);
    layer.append(dot);
    try { anim.beginElement(); } catch { /* no SMIL */ }
    setTimeout(() => dot.remove(), 950);
  }

  function renderSpecimen(m) {
    const host = $("#specimen");
    const width = Math.max(720, Math.floor(host.clientWidth - 18));
    const key = JSON.stringify(m) + width;
    if (key === S.meshKey && host.querySelector("svg")) return;
    S.meshKey = key;
    const { svg, g } = drawMesh(m, width);
    const layerKey = `${m.routes.map(r => r.route).join("|")}:${g.routerR}:${width}`;
    let stage = host.querySelector(".stage");
    if (!stage) { stage = h("div", { class: "stage", style: "position:relative;width:max-content" }); fill(host, stage); }
    const oldLayer = stage.querySelector(".layer");
    const base = stage.querySelector("svg.base");
    svg.setAttribute("class", "base");
    if (base) base.replaceWith(svg); else stage.prepend(svg);
    if (!oldLayer || layerKey !== S.layerKey) {
      S.layerKey = layerKey;
      const layer = h("div", { class: "layer", style: "position:absolute;inset:0;pointer-events:none" }, drawLayer(m, g));
      layer.firstChild.setAttribute("width", g.W);
      layer.firstChild.setAttribute("height", g.H);
      if (oldLayer) oldLayer.replaceWith(layer); else stage.append(layer);
    }
  }

  // ------------------------------------------------------------------ live view
  function renderWork(w) {
    const items = [
      ...w.active.map(a => h("li", {}, h("span", { class: "spin", "aria-hidden": "true" }, "●"),
        h("span", {}, h("b", {}, a.kind), ` ${a.actor}: ${a.text}`), h("span", { class: "t" }, took(a.elapsed)))),
      ...w.recent.slice(0, Math.max(0, 7 - w.active.length)).map(a => h("li", { class: "finished" }, h("span", { class: "done", "aria-hidden": "true" }, "✓"),
        h("span", {}, `${a.kind} ${a.actor}: ${a.text}`), h("span", { class: "t" }, `${took(a.took)}, ${span(a.ago)} ago`))),
    ];
    fill($("#work"), items.length ? items : h("li", { class: "dim" }, h("span", {}), "Nothing running right now.", h("span", {})));
  }

  function logRow(e) {
    return h("li", { class: `lv-${e.level}` }, h("time", { datetime: new Date(e.ts * 1000).toISOString(), title: dateTime(e.ts) }, clock(e.ts)),
      h("span", { class: "tag" }, e.tag), h("span", { class: "msg" }, e.text));
  }

  function renderLiveLog() {
    if (S.rendered.liveLog === S.logVersion) return;
    S.rendered.liveLog = S.logVersion;
    const rows = S.log.slice(-150).reverse().map(logRow);
    fill($("#live-log"), rows.length ? rows : h("li", { class: "dim" }, h("span", {}), h("span", {}), "Nothing logged yet."));
  }

  function replyForm(id, options) {
    const input = h("input", { type: "text", "aria-label": "Your answer", placeholder: "Your answer" });
    const form = h("form", { class: "row", onsubmit: e => {
      e.preventDefault();
      const answer = input.value.trim();
      if (!answer) return;
      act(form.querySelector("button"), async () => { const r = await api("/reply", { inquiry_id: id, answer, by: "web" }); input.value = ""; return r; }, r => r.message || "Sent");
    } }, input, h("button", { type: "submit" }, "Reply"));
    const chips = options?.length ? h("div", { class: "chips" }, options.map(o => h("button", { type: "button", class: "chip", onclick: () => { input.value = o; input.focus(); } }, o))) : null;
    return h("div", {}, chips, form);
  }

  function attentionItems(st, kinds = ["approvals", "inquiries", "questions"]) {
    const out = [];
    for (const a of kinds.includes("approvals") ? st.approvals || [] : []) {
      out.push(h("div", { class: "ask" },
        h("p", {}, h("b", {}, `${a.kind} ${a.subject}`), " needs your approval"),
        h("p", { class: "why" }, a.detail?.permissions ? `Asks for: ${a.detail.permissions.join(", ") || "no permissions"}` : JSON.stringify(a.detail)),
        h("div", { class: "row" },
          h("button", { type: "button", class: "primary", onclick: e => act(e.target, () => api("/approve", { id: a.id, approve: true, by: "web" }), `Approved ${a.subject}`) }, "Approve"),
          h("button", { type: "button", onclick: e => act(e.target, () => api("/approve", { id: a.id, approve: false, by: "web" }), `Denied ${a.subject}`) }, "Deny"))));
    }
    for (const q of kinds.includes("inquiries") ? st.mind?.inquiries || [] : []) {
      out.push(h("div", { class: "ask" }, h("p", {}, q.question), q.why ? h("p", { class: "why" }, `Why it asks: ${q.why}`) : null, replyForm(q.id, q.kind === "clarify" ? [] : q.options)));
    }
    for (const q of kinds.includes("questions") ? st.questions || [] : []) {
      const input = h("input", { type: "text", "aria-label": "Label", placeholder: "label or route" });
      out.push(h("div", { class: "ask" }, h("p", {}, `Teach it: what is “${cut(q.text, 90)}” for ${q.capability}?`), h("p", { class: "why" }, q.reason),
        h("form", { class: "row", onsubmit: e => {
          e.preventDefault();
          if (input.value.trim()) act(e.target.querySelector("button"), () => api("/answer", { question_id: q.id, label: input.value.trim() }), r => r.message || "Thanks");
        } }, input, h("button", { type: "submit" }, "Teach"))));
    }
    return out;
  }

  function renderAttention(st) {
    const host = $("#attention");
    if (busy(host)) return;
    const items = attentionItems(st);
    const more = items.length > 3 ? h("p", { class: "small" }, `${items.length - 3} more in `, h("a", { href: "#status" }, "Status"), " and ", h("a", { href: "#mind" }, "Mind"), ".") : null;
    fill(host, items.length ? h("div", { class: "attention" }, h("h2", {}, "Waiting for you"), items.slice(0, 3), more) : null);
  }

  function renderGoals(st) {
    fill($("#goals"), h("div", { class: "goals", "aria-label": "Directive goals" }, (st.scorecard || []).map(g =>
      h("div", { class: "goal" + (g.met ? " met" : ""), title: g.text }, h("span", { class: "dot", "aria-hidden": "true" }),
        h("span", {}, goalName(g.id)), h("span", { class: "v" }, `${g.value ?? "–"} (${g.direction === "at_least" ? "at least" : "at most"} ${g.target})${g.met ? "" : ", open"}`)))));
  }

  function renderLive() {
    const d = S.dash;
    renderSpecimen(d.mesh);
    renderWork(d.work);
    renderLiveLog();
    renderAttention(d.status);
    renderGoals(d.status);
  }

  // ------------------------------------------------------------------ nodes view
  function spark(values) {
    if (!values || values.length < 2) return null;
    const w = 90, hh = 22, lo = Math.min(...values), hi = Math.max(...values);
    const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${hi === lo ? hh / 2 : hh - 2 - ((v - lo) / (hi - lo)) * (hh - 4)}`);
    return sv("svg", { class: "spark", width: w, height: hh, viewBox: `0 0 ${w} ${hh}`, role: "img", "aria-label": `sizes ${values.join(", ")}` }, sv("path", { d: "M " + pts.join(" L ") }));
  }

  function renderNodes() {
    const host = $("#nodes");
    const st = S.dash.status, evo = st.evolution;
    const rows = Object.entries(st.capabilities).map(([name, c]) => {
      const ch = c.champion;
      const ev = evo?.capabilities?.[name];
      return h("tr", {},
        h("td", {}, h("b", {}, name)), h("td", { class: "n" }, num(c.examples)),
        h("td", {}, ch ? ch.id : h("span", { class: "dim" }, "none yet")),
        h("td", {}, ch ? shape(ch.arch) : ""), h("td", { class: "n" }, ch ? pct(ch.holdout_acc) : ""),
        h("td", { class: "n" }, ch ? num(ch.serves) : ""), h("td", { class: "n" }, ch ? ch.generation : ""),
        h("td", {}, c.candidates.map(x => `${x.id} (${pct(x.holdout_acc)})`).join(", ") || h("span", { class: "dim" }, "none")),
        h("td", {}, ev ? h("span", {}, spark(ev.sizes), ` ${ev.tried} tried, ${ev.promoted} promoted`) : ""),
        h("td", {}, c.retired.length ? h("button", { type: "button", class: "small", title: `Bring back ${c.retired[c.retired.length - 1]}`,
          onclick: e => act(e.target, () => api("/rollback", { capability: name }), r => `${name}: ${r.champion} serves again`) }, "Roll back") : ""));
    });
    if (!busy(host)) {
      fill(host,
        h("div", { class: "section" }, h("h2", {}, "Capabilities"),
          h("p", { class: "lede" }, `${st.nodes.live} nodes live, ${st.nodes.ever} built, ${st.nodes.disposed} disposed. ` +
            (evo?.enabled ? `Evolution grows nodes ${pct(evo.growth_bias)} of the time; the size column shows each champion's size over its generations.` : "Evolution is off.")),
          h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
            h("thead", {}, h("tr", {}, ["Capability", "Examples", "Champion", "Shape", "Held-out", "Served", "Gen", "Challengers", "Evolution", ""].map((t, i) => h("th", { class: [1, 4, 5, 6].includes(i) ? "n" : null }, t)))),
            h("tbody", {}, rows)))),
        h("div", { class: "section" }, h("h2", {}, "Agents"),
          h("div", { class: "scroll-x" }, h("table", { class: "tbl" }, h("tbody", {}, Object.entries(st.agents).map(([a, stats]) =>
            h("tr", {}, h("td", {}, h("b", {}, a)), h("td", {}, Object.entries(stats).map(([k, v]) => `${k} ${num(v)}`).join(", ") || h("span", { class: "dim" }, "idle")))))))));
    }
  }

  // ------------------------------------------------------------------ memory view
  async function loadMemNode(id) {
    S.memNode = id;
    try { S.memDetail = await api(`/memory?node=${encodeURIComponent(id)}`); } catch (e) { S.memDetail = null; toast(e.message); }
    renderMemory(true);
  }

  function renderMemory(force) {
    const host = $("#memory");
    const mem = S.dash.status.memory;
    if (!force && busy(host)) return;
    if (!mem || !mem.enabled) {
      fill(host, h("p", { class: "lede" }, "Memory nodes are off. Set enabled = true under [memory] in directives.toml to have NeuroMesh compress what it learns into tested nodes."));
      return;
    }
    const ask = h("input", { type: "text", placeholder: "Question", "aria-label": "Question" });
    const answer = h("input", { type: "text", placeholder: "Answer", "aria-label": "Answer" });
    const memorize = h("form", { class: "row", onsubmit: e => {
      e.preventDefault();
      if (!ask.value.trim() || !answer.value.trim()) return;
      act(e.target.querySelector("button"), async () => {
        const r = await api("/memorize", { ask: ask.value.trim(), answer: answer.value.trim(), by: "web" });
        ask.value = ""; answer.value = "";
        return r;
      }, r => `${r.status === "new" ? "Learning it" : r.status === "updated" ? "Updating it" : "Already known"} in ${r.node}`);
    } }, ask, answer, h("button", { type: "submit", class: "primary" }, "Memorize"));

    const rows = mem.nodes.map(n => {
      const ratio = n.first_bytes ? n.bytes / n.first_bytes : 1;
      return h("tr", { class: "click" + (n.id === S.memNode ? " sel" : ""), tabindex: 0, onclick: () => loadMemNode(n.id), onkeydown: e => { if (e.key === "Enter") loadMemNode(n.id); } },
        h("td", {}, h("b", {}, n.id)), h("td", {}, n.status, n.parts.length > 1 ? h("div", { class: "dim small" }, `merged from ${n.parts.length} nodes`) : null),
        h("td", { class: "n" }, n.facts),
        h("td", {}, h("div", { class: "size", title: n.first_bytes ? `${(1 / Math.max(ratio, 1e-6)).toFixed(0)}× smaller than when first learned` : "" },
          h("span", {}, bytes(n.bytes), n.first_bytes > n.bytes ? h("span", { class: "dim" }, ` from ${bytes(n.first_bytes)}, ${Math.round(1 / Math.max(ratio, 1e-6)).toLocaleString()}× smaller`) : ""),
          h("div", { class: "bar" }, h("i", { style: `width:${Math.max(1, Math.min(100, ratio * 100)).toFixed(1)}%` })))),
        h("td", { class: "small" }, n.spec),
        h("td", { class: n.total && n.passed === n.total ? "ok" : n.total ? "no" : "wait" }, n.total ? `${n.passed}/${n.total}` : "not checked"),
        h("td", { class: "n" }, num(n.served)));
    });
    const runTests = h("button", { type: "button", onclick: e => act(e.target, async () => {
      S.memTests = await api("/memory/test", {});
      renderMemory(true);
      return S.memTests;
    }, r => (r.passing ? "Every memory node passes its tests" : "Some tests fail - see below")) }, "Run every test");

    const kids = [
      h("div", { class: "section" },
        h("p", { class: "lede" }, `${mem.facts} facts in ${mem.nodes.length} memory node${mem.nodes.length === 1 ? "" : "s"}, ${bytes(mem.bytes)}` +
          (mem.first_bytes > mem.bytes ? ` (first learned in ${bytes(mem.first_bytes)})` : "") +
          `. A node answers at ${pct(mem.threshold)} confidence or more, and is only ever replaced by a smaller one that passes every test in its file` +
          (mem.merge ? `, or merged with another into one node that passes at least ${pct(mem.merge_pass_rate)} of both nodes' tests (what it fails is not served, and relearned)` : "") +
          `. Test files: ${mem.tests_dir}`),
        h("h2", {}, "Teach it a fact"), memorize),
      h("div", { class: "section" }, h("div", { class: "row", style: "justify-content:space-between" }, h("h2", {}, "Memory nodes"), runTests),
        mem.nodes.length
          ? h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
              h("thead", {}, h("tr", {}, ["Node", "State", "Facts", "Size", "Shape", "Tests", "Served"].map((t, i) => h("th", { class: [2, 6].includes(i) ? "n" : null }, t)))),
              h("tbody", {}, rows)))
          : h("p", { class: "dim" }, "Nothing learned yet. It learns from research, teacher answers, challenges and what you teach it.")),
    ];
    if (S.memTests) {
      kids.push(h("div", { class: "section" }, h("h2", {}, "Test run"), h("ul", { class: "rows" }, S.memTests.nodes.map(r =>
        h("li", { class: "ask" }, h("p", {}, h("b", {}, r.node), " ", h("span", { class: r.passed === r.total ? "ok" : "no" }, `${r.passed}/${r.total} pass`)),
          r.results.filter(x => !x.pass).slice(0, 8).map(x => h("p", { class: "why" }, `“${x.ask}”: wanted “${cut(x.want, 60)}”, got “${cut(x.got, 60)}” (${pct(x.confidence)})`)))))));
    }
    const d = S.memDetail;
    if (d && d.id === S.memNode) {
      kids.push(h("div", { class: "section" },
        h("h2", {}, `${d.id}: ${d.facts.length} facts`),
        h("p", { class: "lede" }, `${bytes(d.bytes)} (${d.spec})${d.first_bytes > d.bytes ? `, first learned in ${bytes(d.first_bytes)}` : ""}. Tests ${d.passed}/${d.total}.` +
          (d.merged_from.length ? ` Merged from ${d.merged_from.join(" and ")}; made of ${d.parts.join(", ")}.` : "") + " File: ", h("code", {}, d.file)),
        h("div", { class: "cols" },
          h("div", {}, h("h3", {}, "What it knows"), h("ul", { class: "facts" }, d.facts.map(f => h("li", {},
            h("div", {}, h("div", { class: "q" }, f.ask), h("div", { class: "a" }, f.answer),
              h("div", { class: "s" }, [f.source && `learned from ${f.source}`, f.learned, f.variants.length && `also asked as “${f.variants.join("”, “")}”`].filter(Boolean).join("; "))),
            h("button", { type: "button", class: "quiet small", title: "Forget this fact (its tests go too)", onclick: e => act(e.target, () => api("/command", { line: `/memory forget ${f.ask}` }), "Forgotten") }, "Forget"))))),
          h("div", {}, h("h3", {}, "How it was compressed"), h("ol", { class: "rows small" }, d.history.map(x => h("li", { style: "padding:3px 0" }, x))),
            h("details", {}, h("summary", {}, `${d.unknowns.length} questions it must not answer`),
              h("ul", { class: "rows small" }, d.unknowns.map(u => h("li", { style: "padding:3px 0" }, h("b", {}, u.ask), h("span", { class: "dim" }, ` ${u.reason.replace(/^auto: /, "")}`)))))))));
    }
    fill(host, kids);
  }

  // ------------------------------------------------------------------ mind view
  function meter(name, value, baseline, lo, hi) {
    const p = x => `${(((x - lo) / (hi - lo)) * 100).toFixed(1)}%`;
    return h("div", { class: "meter" }, h("span", {}, name), h("div", { class: "track", title: `baseline ${baseline}` },
      h("b", { style: `width:${p(value)}` }), h("i", { style: `left:${p(baseline)}` })), h("span", { class: "dim" }, value.toFixed(2)));
  }

  function initiative(ag) {
    if (!ag) return null;
    const node = ag.node;
    const select = h("select", { "aria-label": "Director mode", onchange: e => act(e.target, () => api("/director", { mode: e.target.value, by: "web" }), r => r.message) },
      ["auto", ...ag.modes].map(m => h("option", { value: m, selected: m === ag.pinned }, m === "auto" ? "auto (the agency node chooses)" : `${m} (pinned)`)));
    const controls = h("div", { class: "row" },
      h("button", { type: "button", class: ag.enabled ? "" : "primary", onclick: e => act(e.target, () => api("/director", { enabled: !ag.enabled, by: "web" }), r => r.message) },
        ag.enabled ? "Turn initiative off" : "Turn initiative on"),
      h("label", { class: "row small" }, "Mode", select),
      ag.enabled ? h("button", { type: "button", onclick: e => act(e.target, () => api("/director", { decide: true, by: "web" }),
        r => r.decision ? `${r.decision.mode}: ${r.decision.why}${r.decision.summary ? " - " + r.decision.summary : ""}` : "It is already deciding") }, "Take initiative now") : null);
    const who = !ag.enabled ? "Its initiative is off: it only does what people ask."
      : ag.pinned !== "auto" ? `Pinned to ${ag.pinned} by a person.`
      : node && node.trusted ? `The agency node ${node.id} (${shape(node.arch).split(" ")[0]}, held-out ${pct(node.holdout)}) chooses the mode.`
      : `The drive policy chooses while the agency node learns (${ag.examples} situations so far).`;
    const values = Object.entries(ag.values).sort((a, b) => b[1] - a[1]).map(([m, v]) =>
      h("div", { class: "meter" }, h("span", {}, m), h("div", { class: "track" }, h("b", { style: `width:${(v * 100).toFixed(0)}%` })), h("span", { class: "dim" }, `${v.toFixed(2)}, ${ag.counts[m] || 0}×`)));
    const tasks = ag.self_tasks.map(t => h("li", { class: "ask" }, h("div", { class: "row", style: "justify-content:space-between" },
      h("span", {}, h("b", {}, t.id), ` ${t.mode}: ${t.what}`), h("button", { type: "button", class: "quiet small", onclick: e => act(e.target, () => api("/task", { id: t.id, action: "remove", by: "web" }), r => r.message) }, "Remove")),
      h("p", { class: "why" }, `${t.runs} runs, ${span(t.age)} old; ${t.note}`)));
    const decisions = ag.history.slice().reverse().map(d2 => h("li", { class: "ask" },
      h("p", {}, h("time", { class: "dim small", title: dateTime(d2.ts) }, clock(d2.ts)), " ", h("b", {}, d2.mode), `: ${d2.why}`,
        d2.value === null || d2.value === undefined ? null : h("span", { class: d2.value >= 0.5 ? "ok" : d2.value <= 0.15 ? "no" : "dim" }, ` worth ${d2.value.toFixed(2)}`)),
      h("p", { class: "why" }, [d2.how, d2.summary].filter(Boolean).join(" - "))));
    return h("div", { class: "section" },
      h("h2", {}, "Initiative"),
      h("p", { class: "lede" }, `${who} When no one has asked it for anything for a while, it decides what to learn or test next: research, practise, examine, improve, consolidate or rest. ` +
        `${ag.decisions_last_hour} of ${ag.limit} decisions this hour${ag.enabled && ag.next_in !== null ? `; next look in ${span(ag.next_in)}` : ""}.`),
      controls,
      h("div", { class: "cols", style: "margin-top:14px" },
        h("div", {}, h("h3", {}, "Recent decisions"), decisions.length ? h("ul", { class: "rows" }, decisions) : h("p", { class: "dim" }, "None yet.")),
        h("div", {},
          h("h3", {}, "What has paid off"), values.length ? values : h("p", { class: "dim" }, "Nothing scored yet."),
          h("h3", { style: "margin-top:14px" }, "Tasks it set itself"), tasks.length ? h("ul", { class: "rows" }, tasks) : h("p", { class: "dim" }, "None right now."))));
  }

  function directorsPanel(ds) {
    if (!ds) return null;
    const st = ds.stance, c = ds.counts;
    const full = ds.nodes.length >= ds.limit;
    const nodes = ds.nodes.map(n => h("li", { class: "ask" },
      h("div", { class: "row", style: "justify-content:space-between" },
        h("span", {}, h("b", {}, n.id), n.role === "primary" ? " the primary director" : ` a duplicate of ${n.parent}`, ": ",
          h("span", { class: n.doing === "idle" ? "dim" : "hit" }, n.doing)),
        n.role === "duplicate" ? h("button", { type: "button", class: "quiet small", onclick: e => act(e.target, () => api("/directors", { action: "retire", id: n.id, by: "web" }), r => r.message) }, "Retire") : null),
      h("p", { class: "why" }, `${n.handled.intake || 0} orders taken, ${n.handled.review || 0} reviewed, ${n.handled.consult || 0} second opinions, ${n.handled.initiative || 0} initiatives; ` +
        (n.role === "primary" ? `running for ${span(n.age)}; it never retires` : `made ${span(n.age)} ago: ${n.reason}`))));
    const reviews = ds.reviews.slice().reverse().map(r => h("li", { class: "ask" },
      h("p", {}, h("time", { class: "dim small", title: dateTime(r.ts) }, clock(r.ts)), " ", `“${r.text}”`),
      h("p", { class: "why" }, `taken by ${r.intake}, reviewed by ${r.director} (${r.mood}): `,
        h("span", { class: r.second === "disagrees" ? "no" : r.second === "agrees" ? "ok" : "dim" }, r.verdict))));
    const declined = ds.memory.filter(m => !m.kept).reverse().map(m => h("li", { class: "ask" },
      h("p", {}, `“${m.ask}”`), h("p", { class: "why" }, `${m.director}: ${m.why}`)));
    return h("div", { class: "section" }, h("h2", {}, "Directors"),
      h("p", { class: "lede" }, `A director takes every order before anything else touches it, and reviews the result before you see it or anything goes into long-term memory. ` +
        `There is always at least one. It duplicates itself when work waits, when it wants a second opinion, and to take a second mode of initiative; idle duplicates retire. Tools can ask it for facts they need, and when it decides an order was not answered it tries other ways, up to the limit below. ` +
        `${ds.nodes.length} running of at most ${ds.limit}; ${c.review || 0} reviewed, ${c.consult || 0} second opinions, ${c["memory.declined"] || 0} facts kept out of memory.`),
      h("div", { class: "row" },
        h("button", { type: "button", disabled: full, title: full ? `At most ${ds.limit} directors ([directors] max_directors)` : "", onclick: e => act(e.target, () => api("/directors", { action: "spawn", by: "web" }), r => `${r.director} is running`) }, "Add a duplicate"),
        h("label", { class: "row small" }, "Ways to try an unanswered order",
          h("select", { "aria-label": "Ways a director tries an unanswered order before it gives up", onchange: e => act(e.target, () => api("/directors", { attempts: Number(e.target.value), by: "web" }), r => r.message) },
            [1, 2, 3, 4, 5, 6, 8, 10].map(n => h("option", { value: n, selected: n === ds.max_attempts }, n === 1 ? "1 (no retries)" : String(n)))))),
      h("p", { class: "small dim" }, `Unanswered orders tried again ${c.retries || 0} times, ${c.rescued || 0} answered that way, ${c.gave_up || 0} given up; tools asked ${c.queries || 0} questions.`),
      h("div", { class: "cols", style: "margin-top:14px" },
        h("div", {},
          h("h3", {}, `How its mood shapes them: ${st.mood}`),
          h("ul", { class: "rows small" }, st.effects.map(e => h("li", { style: "padding:3px 0" }, e))),
          h("h3", { style: "margin-top:14px" }, "Director nodes"), h("ul", { class: "rows" }, nodes)),
        h("div", {},
          h("h3", {}, "Recent reviews"), reviews.length ? h("ul", { class: "rows" }, reviews) : h("p", { class: "dim" }, "None yet."),
          h("h3", { style: "margin-top:14px" }, "Kept out of long-term memory"), declined.length ? h("ul", { class: "rows" }, declined) : h("p", { class: "dim" }, "Nothing so far."))));
  }

  function dreamsPanel(dr) {
    if (!dr || !dr.enabled) return null;
    const prompt = h("input", { type: "text", placeholder: "a city on Mars, what if Ada Lovelace painted the Mona Lisa, ...", "aria-label": "What to imagine" });
    const imagine = h("form", { class: "row", onsubmit: async e => {
      e.preventDefault();
      if (!prompt.value.trim()) return;
      const b = e.target.querySelector("button");
      b.disabled = true;
      try { showAnswer({ text: `imagine ${prompt.value}`, order: await api("/imagine", { prompt: prompt.value.trim(), source: "web" }) }); prompt.value = ""; window.scrollTo(0, 0); }
      catch (err) { toast(err.message); } finally { b.disabled = false; }
    } }, prompt, h("button", { type: "submit", class: "primary" }, "Imagine"));
    const ln = dr.last_night || {};
    const state = dr.asleep ? `Asleep for ${span(dr.slept_for)}, dreaming: ${dr.dreams_tonight} dreams so far. Everything else waits; an order wakes it.`
      : `Awake for ${span(dr.awake_for)}. When there are no orders for a while it may fall asleep and dream about what it already knows, with no research tools or teachers.`;
    const nodes = dr.nodes.map(n => h("li", { class: "ask" }, h("p", {}, n.champion ? h("b", {}, `${n.id} (dreams best)`) : h("b", {}, n.id),
      ` generation ${n.generation}, ${shape(n.shape)}, ${num(n.weights)} weights: `, h("span", { class: n.fitness === null ? "dim" : "ok" }, n.fitness === null ? "not checked yet" : `${pct(n.fitness)} of checked dreams right`)),
      n.parent ? h("p", { class: "why" }, `from ${n.parent}${n.changes.length ? ": " + n.changes.join("; ") : ""}`) : null));
    const recent = dr.recent.slice().reverse().map(r => h("li", { class: "ask" }, h("p", {}, h("span", { class: "dim small" }, `${r.kind} `), `“${r.question}”`),
      h("p", { class: "why" }, `it answered “${r.answer}” (${pct(r.confidence)}): `, h("span", { class: r.verdict === "correct" ? "ok" : r.verdict === "wrong" ? "no" : "dim" },
        r.verdict === "correct" ? "checked out" : r.verdict === "wrong" ? `wrong${r.truth ? `, it was “${r.truth}”` : ""}` : r.verdict === "nothing" ? "nothing came to mind" : "nothing it knows confirms it"))));
    return h("div", { class: "section" }, h("h2", {}, "Dreams"),
      h("p", { class: "lede" }, state),
      h("div", { class: "row" },
        dr.asleep ? h("button", { type: "button", class: "primary", onclick: e => act(e.target, () => api("/dreams", { action: "wake", by: "web" }), "Awake") }, "Wake up")
          : h("button", { type: "button", onclick: e => act(e.target, () => api("/dreams", { action: "sleep", by: "web" }), r => r.message) }, "Sleep now")),
      h("h3", { style: "margin-top:14px" }, "Imagine"), imagine,
      h("p", { class: "dim small" }, "Its best dream node suggests what comes to mind and blends it into something it remembers. Imagined answers are labelled and never stored as facts."),
      h("div", { class: "cols", style: "margin-top:10px" },
        h("div", {}, h("h3", {}, "Dream nodes"), nodes.length ? h("ul", { class: "rows" }, nodes) : h("p", { class: "dim" }, "None yet: the first one is born the first time it sleeps."),
          ln.dreams ? h("p", { class: "small" }, `Last night: ${ln.dreams} dreams (${ln.rehearsed} rehearsed); ${ln.remembered_correct}/${ln.remembered} remembered answers right, ${ln.imagined_confirmed}/${ln.imagined} imagined ones confirmed by its own memory` +
            (ln.evolved ? `; the genetic algorithm made ${ln.champion} the best dreamer (${pct(ln.fitness)}).` : "; nothing checked out, so no evolution.")) : null),
        h("div", {}, h("h3", {}, "Recent dreams"), recent.length ? h("ul", { class: "rows" }, recent) : h("p", { class: "dim" }, "None yet."))));
  }

  function renderMind() {
    const host = $("#mind");
    if (busy(host)) return;
    const st = S.dash.status, md = st.mind, tm = st.temperament;
    const op = S.extra.opinions, notes = S.extra.notes;
    const goal = h("input", { type: "text", placeholder: "Something to look into, e.g. the Krebs cycle", "aria-label": "Something to look into" });
    const investigate = h("form", { class: "row", onsubmit: e => {
      e.preventDefault();
      if (!goal.value.trim()) return;
      act(e.target.querySelector("button"), async () => { const r = await api("/investigate", { goal: goal.value.trim() }); goal.value = ""; return r; },
        r => cut(r.finding || r.note || JSON.stringify(r), 200));
    } }, goal, h("button", { type: "submit" }, "Look into it"));
    const proposals = (md?.proposals || []).map(p => {
      const labels = h("input", { type: "text", placeholder: "labels (comma separated, optional)", "aria-label": "Labels", value: (p.labels || []).join(", ") });
      labels.dataset.free = "1";
      return h("div", { class: "ask" }, h("p", {}, h("b", {}, `${p.kind} ${p.name}`), p.description ? `: ${p.description}` : ""),
        h("p", { class: "why" }, p.reason),
        h("div", { class: "row" }, p.kind === "skill" ? labels : null,
          h("button", { type: "button", class: "primary", onclick: e => act(e.target, () => api("/adopt", { proposal_id: p.id, labels: p.kind === "skill" ? labels.value.split(",").map(x => x.trim()).filter(Boolean) : undefined, by: "web" }), r => r.message) }, "Adopt"),
          h("button", { type: "button", onclick: e => act(e.target, () => api("/decline", { proposal_id: p.id, by: "web" }), r => r.message) }, "Decline")));
    });
    fill(host,
      directorsPanel(st.directors),
      dreamsPanel(st.dreams),
      initiative(st.agency),
      h("div", { class: "cols" },
        h("div", {},
          h("div", { class: "section" }, h("h2", {}, "Questions for you"),
            md?.inquiries?.length ? md.inquiries.map(q => h("div", { class: "ask" }, h("p", {}, q.question), q.why ? h("p", { class: "why" }, `Why it asks: ${q.why}`) : null, replyForm(q.id, q.kind === "clarify" ? [] : q.options)))
              : h("p", { class: "dim" }, "No questions right now.")),
          h("div", { class: "section" }, h("h2", {}, "Proposals"), proposals.length ? proposals : h("p", { class: "dim" }, "Nothing proposed. It proposes skills, labels and tools when it notices a need; nothing changes until you adopt it.")),
          h("div", { class: "section" }, h("h2", {}, "Investigate"), investigate,
            h("p", { class: "dim small" }, md ? `${md.investigations.last_hour} of ${md.investigations.limit} investigations this hour.` : ""))),
        h("div", {},
          h("div", { class: "section" }, h("h2", {}, `Temperament: ${tm?.mood ?? "–"}`),
            tm && tm.enabled ? [
              meter("Valence", tm.valence, tm.baseline.valence, -1, 1), meter("Arousal", tm.arousal, tm.baseline.arousal, 0, 1),
              meter("Confidence", tm.confidence, tm.baseline.confidence, 0, 1), meter("Trust", tm.trust, tm.baseline.trust, 0, 1),
              meter("Patience", tm.patience, tm.baseline.patience, 0, 1),
              h("p", { class: "dim small" }, `Decisions: ${Object.entries(tm.decisions).map(([k, v]) => `${k} ${v}`).join(", ") || "none yet"}. It may soften, caveat, hold or withhold an answer, never change it.`),
            ] : h("p", { class: "dim" }, "The temperament is off; answers go out unchanged.")),
          h("div", { class: "section" }, h("h2", {}, "Opinions it shares"),
            op ? [op.notes.length ? h("ul", { class: "rows" }, op.notes.map(n => h("li", { class: "ask" }, h("p", {}, h("b", {}, n.subject), ` ${n.stance} (${pct(n.strength)})`), h("p", { class: "why" }, n.text))))
              : h("p", { class: "dim" }, "No shared opinions yet."), op.withheld ? h("p", { class: "dim small" }, `${op.withheld} kept private.`) : null] : h("p", { class: "dim" }, "Loading…")),
          h("div", { class: "section" }, h("h2", {}, "Notes it shares"),
            notes ? [notes.notes.length ? h("ul", { class: "rows" }, notes.notes.slice(0, 20).map(n => h("li", { class: "ask" }, h("p", {}, h("b", {}, n.kind), ` ${n.subject}`), h("p", { class: "why" }, n.text))))
              : h("p", { class: "dim" }, "No shared notes yet."), notes.withheld ? h("p", { class: "dim small" }, `${notes.withheld} private notes (the operator sees them with /mind, which is audited).`) : null] : null))));
  }

  // ------------------------------------------------------------------ logs view
  const LEVELS = ["order", "lesson", "good", "info", "tool", "teacher", "warn", "bad", "work"];
  function renderLogs(force) {
    const host = $("#logs");
    $$(".subnav button", $("#view-logs")).forEach(b => b.setAttribute("aria-selected", String(b.dataset.logs === S.logsTab)));
    if (S.logsTab === "activity") return renderActivity(host, force);
    if (S.logsTab === "audit") {
      const rows = S.extra.audit;
      if (!force && S.rendered.audit === rows) return;
      S.rendered.audit = rows;
      fill(host, rows ? h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
        h("thead", {}, h("tr", {}, ["Time", "Who", "What", "Subject", "Detail"].map(t => h("th", {}, t)))),
        h("tbody", {}, rows.slice().reverse().map(a => h("tr", {}, h("td", { class: "small", title: dateTime(a.ts) }, clock(a.ts)), h("td", {}, a.actor),
          h("td", { class: a.kind.startsWith("deny") || a.kind.includes("rejected") ? "no" : null }, a.kind), h("td", {}, cut(a.subject, 60)),
          h("td", {}, Object.keys(a.detail || {}).length ? h("details", {}, h("summary", { class: "small" }, cut(Object.entries(a.detail).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" "), 70)),
            h("pre", { class: "json" }, JSON.stringify(a.detail, null, 2))) : "")))))) : h("p", { class: "dim" }, "Loading the audit log…"));
      return;
    }
    if (S.logsTab === "events") {
      const evs = S.extra.events;
      if (!force && S.rendered.events === evs) return;
      S.rendered.events = evs;
      fill(host, evs ? h("ol", { class: "log" }, evs.slice().reverse().map(e => h("li", {}, h("time", { title: dateTime(e.ts) }, clock(e.ts)),
        h("span", { class: "tag" }, e.topic), h("details", { class: "msg" }, h("summary", {}, cut(Object.entries(e.data).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" "), 140)),
          h("pre", { class: "json" }, JSON.stringify(e.data, null, 2)))))) : h("p", { class: "dim" }, "Loading events…"));
      return;
    }
    if (S.logsTab === "lineage") {
      const caps = Object.keys(S.dash.status.capabilities);
      const sel = h("select", { "aria-label": "Capability", onchange: e => { S.lineageCap = e.target.value; S.extra.lineage = null; refreshExtra(true); } },
        h("option", { value: "" }, "every capability"), caps.map(c => h("option", { value: c, selected: c === S.lineageCap }, c)));
      if (!force && S.rendered.lineage === S.extra.lineage && host.querySelector("pre")) return;
      S.rendered.lineage = S.extra.lineage;
      fill(host, h("div", { class: "filters" }, h("label", {}, "Family tree of", sel)),
        h("pre", { class: "block" }, S.extra.lineage?.lineage || "Loading…"));
    }
  }

  function renderActivity(host, force) {
    let list = host.querySelector("ol.log");
    if (!list || force) {
      const search = h("input", { type: "search", placeholder: "Search the log", "aria-label": "Search the log", value: S.filter.q,
        oninput: e => { S.filter.q = e.target.value; S.rendered.activity = -1; renderActivity(host); } });
      search.dataset.free = "1";
      const chips = LEVELS.map(l => h("button", { type: "button", class: "chip", "aria-pressed": String(S.filter.levels.has(l)),
        onclick: e => { if (S.filter.levels.has(l)) S.filter.levels.delete(l); else S.filter.levels.add(l); e.target.setAttribute("aria-pressed", String(S.filter.levels.has(l))); S.rendered.activity = -1; renderActivity(host); } }, l));
      const routine = h("input", { type: "checkbox", checked: !S.filter.routine, onchange: e => { S.filter.routine = !e.target.checked; S.rendered.activity = -1; renderActivity(host); } });
      fill(host, h("div", { class: "filters" }, search, h("div", { class: "chips", role: "group", "aria-label": "Show only" }, chips),
        h("label", {}, routine, "Hide routine work"), h("span", { class: "count", id: "log-count" })), h("ol", { class: "log" }));
      list = host.querySelector("ol.log");
      S.rendered.activity = -1;
    }
    if (S.rendered.activity === S.logVersion) return;
    S.rendered.activity = S.logVersion;
    const q = S.filter.q.toLowerCase();
    const shown = S.log.filter(e => (S.filter.routine || !e.routine) && (!S.filter.levels.size || S.filter.levels.has(e.level)) &&
      (!q || e.text.toLowerCase().includes(q) || e.tag.toLowerCase().includes(q)));
    fill(list, shown.slice(-1000).reverse().map(logRow));
    $("#log-count").textContent = `${shown.length.toLocaleString()} of ${S.log.length.toLocaleString()} entries${shown.length > 1000 ? " (newest 1,000 shown)" : ""}`;
  }
  $$(".subnav button", $("#view-logs")).forEach(b => b.addEventListener("click", () => { S.logsTab = b.dataset.logs; renderLogs(true); refreshExtra(true); }));

  // ------------------------------------------------------------------ status view
  function renderStatus() {
    const host = $("#status");
    if (busy(host)) return;
    const st = S.dash.status;
    const spec = h("input", { type: "text", placeholder: "learn jazz every 10m   ·   improve urgency to 0.97   ·   every 5m: what is 2+2", "aria-label": "Background task" });
    const addTask = h("form", { class: "row", onsubmit: e => {
      e.preventDefault();
      if (!spec.value.trim()) return;
      act(e.target.querySelector("button"), async () => { const r = await api("/tasks", { spec: spec.value.trim(), by: "web" }); spec.value = ""; return r; }, r => `Task ${r.id} set`);
    } }, spec, h("button", { type: "submit" }, "Add task"));
    const usage = st.llm_usage || {};
    const k = st.knowledge;
    fill(host,
      h("div", { class: "section" }, h("h2", {}, "Directive goals"),
        h("div", { class: "scroll-x" }, h("table", { class: "tbl" }, h("tbody", {}, (st.scorecard || []).map(g => h("tr", {},
          h("td", { class: g.met ? "ok" : "wait" }, g.met ? "met" : "open"), h("td", {}, h("b", {}, goalName(g.id)), h("div", { class: "dim small" }, g.text)),
          h("td", {}, g.metric), h("td", { class: "n" }, `${g.value ?? "–"}`), h("td", { class: "n" }, `${g.direction === "at_least" ? "≥" : "≤"} ${g.target}`))))))),
      h("div", { class: "section" }, h("h2", {}, "Approvals and lessons waiting for you"),
        (() => { const a = attentionItems(st, ["approvals", "questions"]); return a.length ? a : h("p", { class: "dim" }, "Nothing is waiting for you here; questions the mesh asks you are under Mind."); })()),
      h("div", { class: "cols" },
        h("div", { class: "section" }, h("h2", {}, "Teachers"), h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
          h("thead", {}, h("tr", {}, ["Teacher", "State", "Lessons", "Detail"].map((t, i) => h("th", { class: i === 2 ? "n" : null }, t)))),
          h("tbody", {}, Object.entries(st.teachers).map(([name, t]) => h("tr", {},
            h("td", {}, h("b", {}, name), t.model ? h("div", { class: "dim small" }, `${t.provider}:${t.model}`) : null),
            h("td", { class: t.available ? (t.in_order ? "ok" : "dim") : "dim" }, t.available ? (t.in_order ? "ready" : "ready, not in order") : "off"),
            h("td", { class: "n" }, num(t.lessons)),
            h("td", { class: "small" }, [t.detail, usage[name] && `${usage[name].calls_last_hour} calls this hour`].filter(Boolean).join("; ")))))))),
        h("div", { class: "section" }, h("h2", {}, "Tools"), h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
          h("thead", {}, h("tr", {}, ["Tool", "State", "Runs", "Note"].map((t, i) => h("th", { class: i === 2 ? "n" : null }, t)))),
          h("tbody", {}, Object.entries(st.tools).map(([name, t]) => h("tr", {},
            h("td", {}, h("b", {}, name), h("div", { class: "dim small" }, `v${t.version}${t.permissions.length ? `, needs ${t.permissions.join(", ")}` : ""}`)),
            h("td", { class: t.state === "active" ? "ok" : ["quarantined", "rejected"].includes(t.state) ? "no" : "wait" }, t.state.replace("_", " ")),
            h("td", { class: "n" }, num(t.runs)), h("td", { class: "small" }, t.note)))))))),
      h("div", { class: "section" }, h("h2", {}, "Background tasks"),
        st.tasks.enabled ? [addTask, st.tasks.tasks.length ? h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
          h("thead", {}, h("tr", {}, ["Task", "What", "Every", "State", "Doing", ""].map(t => h("th", {}, t)))),
          h("tbody", {}, st.tasks.tasks.map(t => h("tr", {}, h("td", {}, h("b", {}, t.id)), h("td", {}, `${t.kind} ${cut(t.target, 50)}`), h("td", {}, span(t.interval)),
            h("td", { class: t.status === "active" ? "ok" : "wait" }, t.status), h("td", { class: "small" }, t.note),
            h("td", {}, h("div", { class: "row" },
              h("button", { type: "button", class: "small", onclick: e => act(e.target, () => api("/task", { id: t.id, action: t.status === "active" ? "pause" : "resume", by: "web" }), r => r.message) }, t.status === "active" ? "Pause" : "Resume"),
              h("button", { type: "button", class: "small quiet", onclick: e => act(e.target, () => api("/task", { id: t.id, action: "remove", by: "web" }), r => r.message) }, "Remove")))))))) : h("p", { class: "dim" }, "No background tasks."),
          h("p", { class: "dim small" }, `${st.tasks.usage.last_hour} of ${st.tasks.usage.limit} task runs this hour.`)] : h("p", { class: "dim" }, "Background tasks are off.")),
      h("div", { class: "section" }, h("h2", {}, "Challenges"),
        st.challenges.suites.length ? h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
          h("thead", {}, h("tr", {}, ["Suite", "Kind", "Items", "Last exam", "Best", "Exams", ""].map((t, i) => h("th", { class: [2, 3, 4, 5].includes(i) ? "n" : null }, t)))),
          h("tbody", {}, st.challenges.suites.map(c => h("tr", {}, h("td", {}, h("b", {}, c.suite), h("div", { class: "dim small" }, cut(c.description || c.source, 70))), h("td", {}, c.kind),
            h("td", { class: "n" }, `${c.items} (${c.exam} exam)`), h("td", { class: "n" }, pct(c.last_exam)), h("td", { class: "n" }, pct(c.best_exam)), h("td", { class: "n" }, c.exams),
            h("td", {}, h("div", { class: "row" },
              h("button", { type: "button", class: "small", onclick: e => act(e.target, () => api("/challenge", { suite: c.suite, mode: "practice", by: "web" }), r => r.summary) }, "Practise"),
              h("button", { type: "button", class: "small", onclick: e => act(e.target, () => api("/challenge", { suite: c.suite, mode: "exam", by: "web" }), r => r.summary) }, "Sit exam")))))))) : h("p", { class: "dim" }, "No challenge suites. /challenge fetch gsm8k 200 downloads one.")),
      h("div", { class: "section" }, h("h2", {}, "Knowledge"),
        h("p", { class: "lede" }, `${num(k.documents)} articles and ${num(k.passages)} passages retained; ${num(k.recalled)} recalled and ${num(k.researched)} researched this run; research ${k.research.last_hour} of ${k.research.limit} this hour; ${k.tool ? `researching with ${k.tool}` : "no research tool active"}.`),
        S.extra.knowledge?.documents?.length ? h("ul", { class: "rows" }, S.extra.knowledge.documents.slice(0, 25).map(d => h("li", { style: "padding:4px 0;border-bottom:1px solid var(--rule)" },
          h("a", { href: d.url, target: "_blank", rel: "noopener" }, d.title), h("span", { class: "dim small" }, ` ${d.passages} passages, used ${d.hits}×, ${d.reason}`)))) : null));
  }

  // ------------------------------------------------------------------ views and polling
  const VIEWS = ["live", "nodes", "memory", "mind", "logs", "status"];
  function setView() {
    const v = location.hash.replace("#", "");
    const next = VIEWS.includes(v) ? v : "live";
    if (next !== S.view) window.scrollTo(0, 0);
    S.view = next;
    VIEWS.forEach(name => { $(`#view-${name}`).hidden = name !== S.view; });
    $$("#views a").forEach(a => { if (a.dataset.view === S.view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
    if (S.dash) { render(true); refreshExtra(true); }
  }
  window.addEventListener("hashchange", setView);

  function render(force) {
    const d = S.dash;
    if (!d) return;
    renderHeader(d.status);
    if (S.view === "live") renderLive();
    else if (S.view === "nodes") renderNodes();
    else if (S.view === "memory") renderMemory(force);
    else if (S.view === "mind") renderMind();
    else if (S.view === "logs") renderLogs(force);
    else if (S.view === "status") renderStatus();
  }

  // what some views need beyond /dashboard, fetched while they are open
  async function refreshExtra(force) {
    const now = Date.now();
    const due = (key, ms) => force || !S.extraAt[key] || now - S.extraAt[key] > ms;
    const get = async (key, path) => { S.extraAt[key] = now; try { S.extra[key] = await api(path); } catch { /* shown on the next poll */ } };
    const jobs = [];
    if (S.view === "mind") {
      if (due("opinions", 5000)) jobs.push(get("opinions", "/opinions"));
      if (due("notes", 5000)) jobs.push(get("notes", "/notes"));
    } else if (S.view === "logs") {
      if (S.logsTab === "audit" && due("audit", 4000)) jobs.push(get("audit", "/audit?n=400"));
      if (S.logsTab === "events" && due("events", 3000)) jobs.push(get("events", "/events?n=300"));
      if (S.logsTab === "lineage" && due("lineage", 5000)) jobs.push(get("lineage", `/lineage${S.lineageCap ? `?capability=${encodeURIComponent(S.lineageCap)}` : ""}`));
    } else if (S.view === "status") {
      if (due("knowledge", 10000)) jobs.push(get("knowledge", "/knowledge?n=25"));
    } else if (S.view === "memory" && S.memNode && due("memNode", 3000)) {
      S.extraAt.memNode = now;
      jobs.push(api(`/memory?node=${encodeURIComponent(S.memNode)}`).then(r => { S.memDetail = r; }).catch(() => { S.memDetail = null; S.memNode = null; }));
    }
    if (jobs.length) { await Promise.all(jobs); render(force); }
  }

  function absorb(d) {
    const log = d.log;
    if (log.seq < S.after) { S.log = []; S.after = 0; return false; }  // NeuroMesh restarted: start the log again
    const fresh = log.entries.filter(e => e.seq > S.after);
    if (fresh.length) {
      S.log.push(...fresh);
      if (S.log.length > 5000) S.log.splice(0, S.log.length - 5000);
      S.logVersion++;
    }
    S.after = Math.max(S.after, log.seq);
    const first = !S.dash;
    S.dash = d;
    let k = 0;
    for (const o of d.orders) {
      if (o.ts > S.lastOrderTs) {
        if (!first) { const route = o.route; setTimeout(() => pulse(route), 160 * k++); }  // one after another
        S.lastOrderTs = o.ts;
      }
    }
    return true;
  }

  let timer = 0, polling = false;
  async function tick(now) {
    if (polling && !now) return;
    clearTimeout(timer);
    polling = true;
    try {
      const d = await api(`/dashboard?after=${S.after}`);
      if (!absorb(d)) { await loadHistory(); }
      S.failures = 0;
      setConn(true, `Updated ${clock(d.now)}`);
      render(false);
      refreshExtra(false);
    } catch (e) {
      if (!e.auth) { S.failures++; setConn(false, `Can't reach NeuroMesh at ${location.host}; retrying`); }
    } finally {
      polling = false;
      const wait = S.failures ? Math.min(15000, 1500 * 2 ** Math.min(S.failures, 4)) : 1500;
      timer = setTimeout(tick, document.hidden ? Math.max(wait, 5000) : wait);
    }
  }

  async function loadHistory() {
    try {
      const r = await api("/log?after=0&n=5000");
      S.log = r.entries;
      S.after = r.seq;
      S.logVersion++;
    } catch { /* the dashboard poll fetches what it can */ }
  }

  async function start() {
    setView();
    await loadHistory();
    tick(true);
  }
  document.addEventListener("visibilitychange", () => { if (!document.hidden) tick(true); });
  let resizeTimer = 0;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (S.dash && S.view === "live") renderSpecimen(S.dash.mesh); }, 150); });
  start();
})();
