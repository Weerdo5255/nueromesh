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
    view: "live", logsTab: "activity", mindTab: "questions", lineageCap: "", memNode: null, memDetail: null, memTests: null,
    filter: { q: "", levels: new Set(), routine: true },
    extra: {}, extraAt: {}, open: new Set(), reports: {},
    closer: store.get("closer", "") === "1", sel: { node: null, text: null, order: null }, trace: null, anat: null, checks: new Map(),
    follow: store.get("follow", "1") === "1", fireSeq: 0,
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

  // ------------------------------------------------------------------ its own thinking, after the knowledge
  // A thinking node's thought: the facts it holds put together in its own words, its view and what it wants
  // to find out. Shown beside the answer, never in it (P17); it says which node and how many facts.
  function thoughtAside(t) {
    const view = t.opinion ? (t.opinion.private ? "It has a view of this too, and keeps it to itself." : null) : null;
    return h("aside", { class: "thought", "aria-label": "Its own thinking" },
      h("p", { class: "thought-by" }, "Its own thinking", h("span", {}, ` ${t.node}, from ${t.grounds} fact${t.grounds === 1 ? "" : "s"} it holds; not part of the answer`)),
      h("p", {}, t.text),
      view ? h("p", { class: "dim small" }, view) : null);
  }

  // ------------------------------------------------------------------ was it right?
  // Yes: the nodes that answered learn so. No: it offers the next most likely answers, worked out now,
  // or none of them (and what is right, if you know); it learns from what you pick.
  const UNCHECKED = ["tasks", "objectives", "toolsmith", "mind", "thinking"];  // these hand work on rather than answer (corrections.py)
  function answerCheck(orderId, text) {
    const box = h("div", { class: "check", role: "group", "aria-label": "Was this answer right?" });
    const upper = t => t.charAt(0).toUpperCase() + t.slice(1);
    const again = () => text ? h("button", { type: "button", class: "quiet small", title: "Send the same order again",
      onclick: () => { cmd.value = text; $("#command").requestSubmit(); } }, "Ask again") : null;
    const thanks = r => fill(box, h("p", { class: "learned" }, r.learned?.length ? `Thanks. ${upper(r.learned.join("; "))}.` : "Thanks, noted."), again());
    const ask = () => fill(box, h("span", { class: "q" }, "Was this right?"),
      h("button", { type: "button", class: "small", onclick: e => act(e.target, () => api("/correct", { order_id: orderId, verdict: "right", by: "web" }),
        r => { thanks(r); return "Marked right"; }) }, "Yes"),
      h("button", { type: "button", class: "small", onclick: () => wrong() }, "No"));
    async function wrong() {
      fill(box, h("p", { class: "dim" }, "Working out the next most likely answers…"));
      let alts;
      try { alts = await api(`/alternatives?order=${encodeURIComponent(orderId)}`); }
      catch (e) { fill(box, h("p", { class: "no" }, e.message), h("button", { type: "button", class: "quiet small", onclick: ask }, "Back")); return; }
      const name = `alt-${orderId}`;
      const typed = h("input", { type: "text", placeholder: "The right answer, if you know it", "aria-label": "The right answer, if you know it" });
      const options = alts.alternatives.map((x, i) => h("label", { class: "alt" },
        h("input", { type: "radio", name, value: x.id, checked: i === 0 }),
        h("span", { class: "a" }, x.answer),
        // square-root scale: a 1% long shot still shows, and a near-certain answer fills the bar
        h("span", { class: "like", title: `How likely it found this: ${x.likelihood < 0.005 ? "under 1%" : pct(x.likelihood)}`, "aria-hidden": "true" },
          h("b", { style: `width:${Math.max(2, Math.sqrt(x.likelihood) * 100).toFixed(1)}%` })),
        h("span", { class: "why" }, `${x.source}; ${x.how}`)));
      const none = h("label", { class: "alt none" }, h("input", { type: "radio", name, value: "none", checked: !alts.alternatives.length }),
        h("span", { class: "a" }, "None of these"), typed);
      typed.addEventListener("focus", () => { none.querySelector("input").checked = true; });
      const form = h("form", { onsubmit: e => {
        e.preventDefault();
        const choice = form.querySelector(`input[name="${CSS.escape(name)}"]:checked`)?.value || "none";
        act(form.querySelector("button[type=submit]"), () => api("/correct", { order_id: orderId, verdict: "wrong", choice,
          answer: choice === "none" ? typed.value.trim() : "", by: "web" }), r => { thanks(r); return "Taught"; });
      } },
        h("fieldset", {}, h("legend", {}, alts.alternatives.length ? "Which is right? Its next most likely answers, best first:" : "It has no other answer to offer. Do you know the right one?"),
          options, none),
        h("div", { class: "row" }, h("button", { type: "submit", class: "primary small" }, "Teach it"),
          h("button", { type: "button", class: "quiet small", onclick: ask }, "Cancel")));
      fill(box, form);
      form.querySelector("input[type=radio]:checked")?.focus();
    }
    ask();
    return box;
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
    if (r.thought && r.route !== "thinking") kids.push(thoughtAside(r.thought));
    kids.push(h("div", { class: "meta" }, meta, close));
    if (r.order_id && !UNCHECKED.includes(r.route)) kids.push(answerCheck(r.order_id, a.text));
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

  // a closer look: nodes can be picked, and the order being followed is marked on its path
  function traced() {
    const t = S.closer && S.trace ? (S.traces || []).find(x => x.order === S.trace) : null;
    return t ? new Set(t.hops.map(hp => hp.node).filter(Boolean)) : new Set();
  }
  let TRACED = new Set();
  function mark(id) { return (S.closer && S.sel.node === id ? " sel" : "") + (TRACED.has(id) ? " traced" : ""); }
  function nodeAttrs(id) { return S.closer ? { "data-node": id, tabindex: 0, role: "button", "aria-label": `look inside ${id}` } : { "data-node": id }; }

  function drawMesh(m, width) {
    TRACED = traced();
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
        svg.append(sv("circle", { cx: X.router, cy: y, r: rr, class: (dr.trusted ? "m-champ" : "m-chall") + mark(an.id), ...nodeAttrs(an.id) },
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
    m.routes.forEach((r, i) => svg.append(sv("path", { d: routePath(g, i), class: "m-edge" + (r.inactive ? " inactive" : "")
      + (S.closer && S.trace && (S.traces || []).some(t => t.order === S.trace && t.route === r.route) ? " traced" : "") })));
    // the router
    const rc = m.router.champion;
    if (rc) {
      svg.append(sv("circle", { cx: X.router, cy: g.yMid, r: g.routerR + 4, class: "m-champ-ring" }));
      svg.append(sv("circle", { cx: X.router, cy: g.yMid, r: g.routerR, class: "m-champ" + mark(rc.id), ...nodeAttrs(rc.id) }, sv("title", {}, nodeTip(rc))));
      svg.append(sv("text", { x: X.router, y: g.yMid + g.routerR + 18, "text-anchor": "middle", class: "m-text" }, "router"));
      svg.append(sv("text", { x: X.router, y: g.yMid + g.routerR + 33, "text-anchor": "middle", class: "m-sub" }, `held-out ${pct(rc.holdout)}`));
      m.router.candidates.forEach((c, j) => svg.append(sv("circle", { cx: X.router - 18 + j * 12, cy: g.yMid - g.routerR - 12, r: 4, class: "m-chall" + mark(c.id), ...nodeAttrs(c.id) }, sv("title", {}, "challenger " + nodeTip(c)))));
    } else {
      svg.append(sv("circle", { cx: X.router, cy: g.yMid, r: 9, class: "m-none" }));
      svg.append(sv("text", { x: X.router, y: g.yMid + 28, "text-anchor": "middle", class: "m-sub" }, "router: not built yet"));
      svg.append(sv("text", { x: X.router, y: g.yMid + 43, "text-anchor": "middle", class: "m-dim" }, "teachers route orders"));
    }
    // each route and what serves it
    m.routes.forEach((r, i) => {
      const y = g.y(i);
      const onPath = S.closer && S.trace && (S.traces || []).some(t => t.order === S.trace && t.route === r.route);
      svg.append(sv("text", { x: X.label, y: y + 4.5, class: "m-route" + (onPath ? " hit" : ""), "data-route": r.route }, cut(r.route, 26)));
      const tx = X.target;
      if (r.skill) {
        const f = r.skill, c = f.champion;
        let x = tx + 14;
        if (c) {
          const rr = radius(c.params);
          svg.append(sv("circle", { cx: x, cy: y, r: rr, class: "m-champ" + mark(c.id), ...nodeAttrs(c.id) }, sv("title", {}, nodeTip(c))));
          x += rr + 9;
        } else {
          svg.append(sv("circle", { cx: x, cy: y, r: 7, class: "m-none" }));
          x += 16;
        }
        f.candidates.forEach(n => { svg.append(sv("circle", { cx: x, cy: y, r: 4.5, class: "m-chall" + mark(n.id), ...nodeAttrs(n.id) }, sv("title", {}, "challenger " + nodeTip(n)))); x += 12; });
        f.retired.forEach(n => { svg.append(sv("circle", { cx: x, cy: y, r: 4.5, class: "m-retired" + mark(n.id), ...nodeAttrs(n.id) }, sv("title", {}, "retired " + nodeTip(n)))); x += 12; });
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
          svg.append(sv("polygon", { points: hexagon(x, y, rr), class: "m-mem" + (n.status === "learning" ? " learning" : "") + mark(n.id), ...nodeAttrs(n.id) },
            sv("title", {}, `${n.id}: ${n.facts} facts in ${bytes(n.bytes)}${n.first_bytes > n.bytes ? ` (first ${bytes(n.first_bytes)})` : ""}${n.parts > 1 ? `, merged from ${n.parts} nodes` : ""}\n${n.spec}\ntests ${n.passed}/${n.total}, served ${n.served}`)));
          x += rr * 2 + 4;
        });
        const facts = m.memory.reduce((a, n) => a + n.facts, 0);
        const thinkNodes = (m.thinking || []).slice(-4);
        if (memNodes.length && thinkNodes.length) {  // after the memory nodes: its thinking nodes
          svg.append(sv("line", { x1: x - 2, x2: x + 8, y1: y, y2: y, class: "m-edge" }));
          x += 14;
          thinkNodes.forEach(n => {
            const gl = sv("g", { class: "m-think-g" + mark(n.id), ...nodeAttrs(n.id) }, sv("title", {}, `${n.id}: ${n.thoughts} thoughts in its own words\n${n.spec}\ntests ${n.passed}/${n.total}, brought up ${n.served}`));
            thinkGlyph(gl, x + 6, y, 6.5, n.status === "learning");
            svg.append(gl);
            x += 18;
          });
        }
        const thoughts = thinkNodes.reduce((a, n) => a + n.thoughts, 0);
        const teachers = r.teachers.length ? `teachers: ${r.teachers.join(", ")}` : "no teacher model (a person is asked)";
        const txt = sv("text", { x: x + 8, y: y + 4.5, class: "m-text" });
        const room = g.chars(x + 8);
        const head = `memory nodes first (${facts} facts)` + (thinkNodes.length ? `, then thinking (${thoughts} thoughts)` : "");
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
        else if (st.node) svg.append(sv("circle", { cx: x, cy: ly, r: 6.5, class: "m-champ" + mark(st.node.id), ...nodeAttrs(st.node.id) }, sv("title", {}, nodeTip(st.node))));
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
    const key = JSON.stringify(m) + width + (S.closer ? `|${S.sel.node}|${S.trace}|${(S.traces || []).length && S.traces[0].order}` : "");
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

  // ------------------------------------------------------------------ a closer look
  const STAGE = { director: "director", router: "router", memory: "memory node", thinking: "thinking node", tool: "tool", knowledge: "retained knowledge",
                  teacher: "teacher", affect: "affect", disposition: "disposition" };
  // a thinking node: a thought bubble (a cell with a small one trailing it)
  function thinkGlyph(g, cx, cy, r, learning = false) {
    const cls = "m-think" + (learning ? " learning" : "");
    g.append(sv("circle", { cx: cx + r * 0.15, cy: cy - r * 0.1, r, class: cls }));
    g.append(sv("circle", { cx: cx - r * 0.95, cy: cy + r * 0.95, r: Math.max(1.6, r * 0.28), class: cls }));
    return g;
  }
  function hopGlyph(hp) {
    const g = sv("svg", { width: 18, height: 18, viewBox: "0 0 18 18", "aria-hidden": "true", class: "hop-glyph" });
    if (hp.kind === "thinking") thinkGlyph(g, 10, 8, 5.5);
    else if (hp.kind === "director") g.append(sv("polygon", { points: "9,2 16,9 9,16 2,9", class: "m-director" }));
    else if (hp.kind === "memory") g.append(sv("polygon", { points: hexagon(9, 9, 7), class: "m-mem" }));
    else if (hp.kind === "tool") g.append(sv("rect", { x: 3, y: 3, width: 12, height: 12, rx: 2, class: "m-tool" }));
    else if (hp.kind === "node") g.append(sv("circle", { cx: 9, cy: 9, r: 6.5, class: "m-champ" }));
    else g.append(sv("circle", { cx: 9, cy: 9, r: 6, class: "m-none" }));
    return g;
  }

  function renderTraces(force) {
    const host = $("#mic-traces");
    const ts = S.traces;
    const key = JSON.stringify([ts && ts.map(t => t.order), S.trace, S.sel.node, S.sel.order]);
    if (!force && key === S.rendered.traces) return;
    S.rendered.traces = key;
    if (!ts) return fill(host, h("p", { class: "dim" }, "Loading…"));
    if (!ts.length) return fill(host, h("p", { class: "dim" }, "No orders yet. Send one with the command bar and follow it here."));
    fill(host, h("ol", { class: "traces" }, ts.map(t => {
      const open = t.order === S.trace;
      const row = h("button", { type: "button", class: "trace" + (open ? " open" : ""), "aria-expanded": String(open),
        onclick: () => { S.trace = open ? null : t.order; if (!open) followTrace(t); renderTraces(true); renderSpecimen(S.dash.mesh); } },
        h("span", { class: "t-text" }, cut(t.text, 64)),
        h("span", { class: "t-meta" }, `${clock(t.ts)}, ${t.route || "no route"}${t.ok ? "" : ", not fulfilled"}, ${t.latency_ms ?? "–"} ms`));
      // the same check box is kept across redraws, so a half-made correction is not wiped by new orders arriving
      const checkable = open && t.answer !== undefined && !UNCHECKED.includes(t.route);
      if (checkable && !S.checks.has(t.order)) {
        S.checks.set(t.order, answerCheck(t.order, t.text));
        if (S.checks.size > 40) S.checks.delete(S.checks.keys().next().value);
      }
      const checked = checkable ? h("div", { class: "trace-answer" }, h("p", { class: "small" }, h("b", {}, "Answer: "), cut(t.answer || "(none)", 160)),
        S.checks.get(t.order)) : null;
      const hops = open ? h("ol", { class: "hops" }, t.hops.map(hp => {
        const name = hp.node || hp.tool || STAGE[hp.stage] || hp.stage;
        const what = [hp.stage !== "director" && hp.stage !== hp.node ? STAGE[hp.stage] || hp.stage : null,
          hp.label ? `“${cut(hp.label, 48)}”` : null, hp.confidence != null ? pct(hp.confidence) : null,
          hp.kind === "teacher" ? `(${cut(hp.how || "a teacher", 30)})` : null].filter(Boolean).join(" ");
        const lookable = hp.node && (hp.kind === "node" || hp.kind === "memory" || hp.kind === "thinking");
        const here = lookable && hp.node === S.sel.node && (hp.input || t.text) === S.sel.text;
        return h("li", { class: here ? "here" : "" }, hopGlyph(hp),
          lookable ? h("button", { type: "button", class: "linkish", title: `Look inside ${hp.node} as it fired on this order`,
            onclick: () => selectNode(hp.node, hp.input || t.text, t.order) }, name) : h("span", {}, name),
          h("span", { class: "dim small" }, ` ${what}`));
      }), h("li", {}, h("span", { class: "hop-glyph you", "aria-hidden": "true" }, "→"), h("span", {}, "you"))) : null;
      return h("li", {}, row, checked, hops);
    })));
  }

  function followTrace(t) {
    const hp = t.hops.find(x => x.node && (x.kind === "node" || x.kind === "memory") && x.node === S.sel.node)
      || t.hops.find(x => x.node && (x.kind === "node" || x.kind === "memory"));
    if (hp) selectNode(hp.node, hp.input || t.text, t.order);
  }

  async function selectNode(id, text, order, pinned = false) {
    S.sel = { node: id, text: text || null, order: order || null, pinned };  // pinned: a text you chose, not followed
    S.fireSeq += 1;
    const seq = S.fireSeq;
    if (S.dash) renderSpecimen(S.dash.mesh);
    renderTraces(true);
    try {
      const q = `/anatomy?node=${encodeURIComponent(id)}` + (text ? `&text=${encodeURIComponent(text)}` : "&latest=1");
      const a = await api(q);
      if (seq !== S.fireSeq) return;  // a newer pick won
      S.anat = a;
      if (!S.sel.text && a.fire) S.sel.text = a.fire.text;
      renderAnatomy(true);
    } catch (e) {
      if (seq !== S.fireSeq) return;
      S.anat = { error: e.message, id, capability: e.data?.capability };
      renderAnatomy(true);
    }
  }

  function nodeOptions() {
    const st = S.dash?.status, m = S.dash?.mesh;
    const groups = [];
    if (st) for (const [cap, c] of Object.entries(st.capabilities)) {
      const ids = [c.champion, ...(c.candidates || [])].filter(Boolean).map(n => ({ id: n.id, text: `${n.id} (${n.status})` }))
        .concat((c.retired || []).map(id => ({ id, text: `${id} (retired)` })));
      if (ids.length) groups.push({ label: cap === "dispatch" ? "router" : cap, ids });
    }
    if (m?.memory?.length) groups.push({ label: "memory", ids: m.memory.map(n => ({ id: n.id, text: `${n.id} (${n.facts} facts)` })) });
    if (m?.thinking?.length) groups.push({ label: "thinking", ids: m.thinking.map(n => ({ id: n.id, text: `${n.id} (${n.thoughts} thoughts)` })) });
    return groups;
  }

  function anatomySvg(a) {
    const f = a.fire;
    const W = 980, padL = 200, padR = 250, top = 34;
    if (!f) {  // its construction only: layer by layer, fully connected
      const cols = a.sizes.length, H = 230;
      const x = k => padL + (k * (W - padL - padR)) / Math.max(1, cols - 1);
      const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "anat", role: "img", "aria-label": `${a.id}: ${a.sizes.join(" to ")} neurons, fully connected` });
      const dots = a.sizes.map(n => Math.min(n, 9));
      a.sizes.forEach((n, k) => {
        const ys = Array.from({ length: dots[k] }, (_, i) => top + 20 + (i * (H - top - 50)) / Math.max(1, dots[k] - 1));
        if (k < cols - 1) {
          const nxt = Array.from({ length: dots[k + 1] }, (_, i) => top + 20 + (i * (H - top - 50)) / Math.max(1, dots[k + 1] - 1));
          ys.forEach(y1 => nxt.forEach(y2 => svg.append(sv("line", { x1: x(k), y1, x2: x(k + 1), y2, class: "a-wire" }))));
        }
        ys.forEach(y => svg.append(sv("circle", { cx: x(k), cy: y, r: 5, class: "a-cell" })));
        svg.append(sv("text", { x: x(k), y: top - 12, "text-anchor": "middle", class: "m-sub" },
          k === 0 ? `${num(n)} inputs` : k === cols - 1 ? `${n} outputs` : `${n} hidden (ReLU)`));
      });
      return svg;
    }
    const L = f.layers, cols = L.length;
    const rows = Math.max(...L.map(l => l.shown.length));
    const H = top + 30 + rows * 24;
    const x = k => padL + (k * (W - padL - padR)) / Math.max(1, cols - 1);
    const ys = L.map(l => { const n = l.shown.length, span = (rows - 1) * 24, off = (span - (n - 1) * 24) / 2;
      return new Map(l.shown.map((u, i) => [u.i, top + 26 + off + i * 24])); });
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "anat fire", role: "img",
      "aria-label": `${a.id} fired on “${f.text}”: ${L.map(l => `${l.active} of ${l.size} ${l.kind}`).join(", ")}; it chose ${f.chosen.label}` });
    const step = 0.42;
    L.forEach((l, k) => svg.append(sv("text", { x: x(k), y: top - 12, "text-anchor": "middle", class: "m-sub" },
      l.kind === "input" ? `inputs: ${l.active} of ${num(l.size)} active` : l.kind === "hidden" ? `hidden ${k}: ${l.active} of ${l.size} fired` : `outputs (${l.size})`)));
    // synapses: activation x weight; excitatory or inhibitory; the strongest path drawn last
    const maxSig = {};
    f.synapses.forEach(sy => { maxSig[sy.layer] = Math.max(maxSig[sy.layer] || 1e-9, Math.abs(sy.signal)); });
    const syns = f.synapses.slice().sort((p, q) => (p.path - q.path) || (Math.abs(p.signal) - Math.abs(q.signal)));
    for (const sy of syns) {
      const y1 = ys[sy.layer].get(sy.from), y2 = ys[sy.layer + 1].get(sy.to);
      if (y1 === undefined || y2 === undefined) continue;
      const rel = Math.abs(sy.signal) / maxSig[sy.layer];
      const x1 = x(sy.layer) + 7, x2 = x(sy.layer + 1) - 7, mid = (x1 + x2) / 2;
      svg.append(sv("path", { d: `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`, pathLength: 1,
        class: `a-syn ${sy.signal >= 0 ? "exc" : "inh"}${sy.path ? " path" : ""}`,
        style: `stroke-width:${(sy.path ? 1.6 : 0.5) + 3.4 * rel};opacity:${(sy.path ? 0.95 : 0.18 + 0.6 * rel).toFixed(2)};--d:${(0.15 + sy.layer * step).toFixed(2)}s` },
        sv("title", {}, `synapse ${sy.layer === 0 ? "#" + sy.from : "h" + sy.from} → ${sy.layer + 1 === cols - 1 ? L[cols - 1].shown.find(o => o.i === sy.to)?.label : "h" + sy.to}: weight ${sy.weight >= 0 ? "+" : ""}${sy.weight}, signal ${sy.signal >= 0 ? "+" : ""}${sy.signal} (${sy.signal >= 0 ? "excitatory" : "inhibitory"})${sy.path ? "; on the strongest path" : ""}`)));
    }
    // neurons
    L.forEach((l, k) => {
      const maxV = Math.max(1e-9, ...l.shown.map(u => Math.abs(l.kind === "input" ? u.attr : l.kind === "output" ? u.prob : u.value)));
      l.shown.forEach(u => {
        const y = ys[k].get(u.i), onPath = f.path[k] === u.i;
        const v = l.kind === "input" ? u.attr : l.kind === "output" ? u.prob : u.value;
        const rel = Math.abs(v) / maxV;
        const cls = `a-cell ${l.kind}${l.kind === "input" ? (u.attr >= 0 ? " exc" : " inh") : ""}${onPath ? " path" : ""}${l.kind === "output" && u.i === f.chosen.i ? " chosen" : ""}`;
        const tip = l.kind === "input" ? `input #${u.i} = ${u.value}: ${u.features.join(", ") || "(collisions only)"}\npushes the answer ${u.attr >= 0 ? "towards" : "away from"} ${f.chosen.label} (${u.attr >= 0 ? "+" : ""}${u.attr})`
          : l.kind === "hidden" ? `h${u.i}: fired at ${u.value}` : `${u.label}: ${pct(u.prob)} (logit ${u.logit})`;
        svg.append(sv("circle", { cx: x(k), cy: y, r: 4 + 4 * rel, class: cls,
          style: `--fill:${(0.25 + 0.75 * rel).toFixed(2)};--d:${(k * step).toFixed(2)}s` }, sv("title", {}, tip)));
        if (l.kind === "input") svg.append(sv("text", { x: x(k) - 14, y: y + 4, "text-anchor": "end", class: "a-label" + (onPath ? " path" : ""),
          style: `--d:${(k * step).toFixed(2)}s` }, cut(u.features[0] || `#${u.i}`, 24)));
        else if (l.kind === "output") svg.append(sv("text", { x: x(k) + 14, y: y + 4, class: "a-label" + (u.i === f.chosen.i ? " chosen" : ""),
          style: `--d:${(k * step).toFixed(2)}s` }, `${cut(u.label, 28)} ${pct(u.prob)}`));
        else if (onPath) svg.append(sv("text", { x: x(k), y: y - 11, "text-anchor": "middle", class: "a-tiny path", style: `--d:${(k * step).toFixed(2)}s` }, `h${u.i}`));
      });
    });
    return svg;
  }

  function renderAnatomy(fresh) {
    const host = $("#mic-anatomy");
    if (!fresh && busy(host)) return;
    const a = S.anat;
    const groups = nodeOptions();
    const pick = h("select", { "aria-label": "Node to look inside", onchange: e => selectNode(e.target.value, S.sel.text) },
      h("option", { value: "", disabled: true, selected: !S.sel.node }, "Pick a node…"),
      groups.map(g => h("optgroup", { label: g.label }, g.ids.map(o => h("option", { value: o.id, selected: o.id === S.sel.node }, o.text)))));
    const probe = h("input", { type: "text", placeholder: "Fire it with any text", "aria-label": "Text to fire the node with", value: "" });
    const form = h("form", { class: "row", onsubmit: e => { e.preventDefault(); if (S.sel.node && probe.value.trim()) selectNode(S.sel.node, probe.value.trim(), null, true); } },
      pick, probe, h("button", { type: "submit", disabled: !S.sel.node }, "Fire"));
    const follow = h("label", { class: "follow small" }, h("input", { type: "checkbox", checked: S.follow,
      onchange: e => { S.follow = e.target.checked; store.set("follow", S.follow ? "1" : "0"); if (S.follow) S.sel.pinned = false; } }),
      S.sel.pinned && S.follow ? "Follow new orders through this node (paused while you look at your own text)" : "Follow new orders through this node");
    if (!a) return fill(host, h("h3", {}, "Inside a node"), form, h("p", { class: "dim" }, "Pick a node above, click one in the network, or open an order on the left."));
    if (a.error) return fill(host, form, h("h3", {}, a.id), h("p", {}, a.error),
      a.capability ? h("button", { type: "button", onclick: () => selectNode(a.capability, S.sel.text, S.sel.order) },
        `Look inside the node that serves ${a.capability === "dispatch" ? "the router" : a.capability} now`) : null);
    const f = a.fire;
    const head = h("div", {},
      h("h3", {}, `${a.id} `, h("span", { class: "dim small" }, `${a.capability === "dispatch" ? "the router" : a.capability}, ${a.status}`)),
      h("p", { class: "small" }, `${a.sizes.map(num).join(" → ")} neurons, ${num(a.params)} weights. ${a.encoder}; ${a.activation}.` +
        (a.holdout != null ? ` Held-out ${pct(a.holdout)}.` : "") + (a.tests ? ` Tests ${a.tests}.` : "") +
        (a.generation ? ` Generation ${a.generation}${a.parent ? `, from ${a.parent}` : ""}.` : "")));
    const replay = f ? h("button", { type: "button", class: "quiet small", onclick: () => renderAnatomy(true) }, "Fire again") : null;
    const fired = f ? h("p", {}, "Fired on ", h("b", {}, `“${cut(f.text, 120)}”`), a.text_from ? h("span", { class: "dim" }, ` (${a.text_from})`) : null,
      `: ${f.features} features hashed onto its inputs; it chose `, h("b", {}, f.chosen.label), ` (${pct(f.chosen.prob)}). `, replay) : null;
    const words = f && f.words.length ? h("div", { class: "pushes" }, h("h4", {}, "Words that pushed it"),
      h("ul", { class: "rows" }, f.words.slice(0, 10).map(w => h("li", { class: "push" },
        h("span", { class: "w" }, w.word), h("span", { class: "bar" }, h("b", { class: w.push >= 0 ? "exc" : "inh", style: `width:${Math.min(100, Math.abs(w.push) * 100).toFixed(1)}%` })),
        h("span", { class: "dim small" }, `${w.push >= 0 ? "+" : "−"}${Math.abs(w.push * 100).toFixed(0)}%`))))) : null;
    const pathText = f && f.path.length ? h("p", { class: "small" }, h("b", {}, "Strongest path: "),
      [`input #${f.path[0]} (${f.layers[0].shown.find(u => u.i === f.path[0])?.features[0] || "collisions"})`,
       ...f.path.slice(1, -1).map(i => `h${i}`), f.chosen.label].join(" → "),
      h("span", { class: "dim" }, `; signal ${f.path_signal.map(v => (v >= 0 ? "+" : "") + v.toFixed(3)).join(", ")} (activation × weight at each synapse)`)) : null;
    const layers = h("table", { class: "tbl small" }, h("thead", {}, h("tr", {}, ["Layer", "Synapses", "Mean |w|", "Largest", "Near zero", "Excitatory"].map((t, i) => h("th", { class: i ? "n" : null }, t)))),
      h("tbody", {}, a.layers.map((l, k) => h("tr", {}, h("td", {}, `${num(l.from)} → ${num(l.to)}${k === a.layers.length - 1 ? " (output)" : ""}`), h("td", { class: "n" }, num(l.weights)),
        h("td", { class: "n" }, l.mean_abs.toFixed(4)), h("td", { class: "n" }, l.max_abs.toFixed(3)), h("td", { class: "n" }, pct(l.near_zero)), h("td", { class: "n" }, pct(l.excitatory))))));
    const firings = a.firings.length ? h("ul", { class: "rows small firings" }, a.firings.slice(0, 8).map(x => h("li", {},
      h("button", { type: "button", class: "linkish", onclick: () => selectNode(a.id, x.text, null, true) }, `“${cut(x.text, 56)}”`),
      h("span", { class: "dim" }, ` ${clock(x.ts)} → ${cut(x.label, 30)} ${pct(x.confidence)}${x.used === true ? ", used" : x.used === false ? ", not used" : ""}`)))) : h("p", { class: "dim small" }, "It has not fired on anything since NeuroMesh started.");
    fill(host, form, follow, head, h("div", { class: "anat-wrap" }, anatomySvg(a)), fired,
      f ? h("p", { class: "small dim legend" }, h("span", { class: "key exc" }), "excitatory (pushes the next neuron up)  ", h("span", { class: "key inh" }), "inhibitory  ",
        h("span", { class: "key path" }), "the strongest path. Line width: activation × weight. Showing the most telling of each layer.") : h("p", { class: "dim small" }, "Fire it with some text to see which synapses carry the signal."),
      pathText, words,
      h("details", {}, h("summary", {}, "Its weights, layer by layer"), layers),
      h("h4", {}, "What it fired on lately"), firings);
  }

  function renderMicroscope() {
    const on = S.closer;
    const mic = $("#microscope");
    mic.hidden = !on;
    const btn = $("#closer");
    btn.setAttribute("aria-pressed", String(on));
    btn.textContent = on ? "Closer look: on" : "Closer look";
    $("#specimen").classList.toggle("closer", on);
    $("#specimen-hint").textContent = on ? "Click a node to look inside it; open an order below to follow its path." : "Hover a node for its shape and scores.";
    if (!on) return;
    renderTraces(false);
    if (!S.sel.node && S.dash && S.traces) {  // start with the newest order, at its first node
      const t = S.traces[0];
      if (t) { S.trace = t.order; followTrace(t); }
      else if (S.dash.mesh.router.champion) selectNode(S.dash.mesh.router.champion.id, null);
    }
    if (!$("#mic-anatomy").hasChildNodes()) renderAnatomy(true);
  }

  $("#closer").addEventListener("click", () => {
    S.closer = !S.closer;
    store.set("closer", S.closer ? "1" : "");
    if (S.dash) { renderSpecimen(S.dash.mesh); renderMicroscope(); refreshExtra(true); }
  });
  function pickFromSpecimen(e) {
    if (!S.closer) return;
    const el = e.target.closest("[data-node]");
    if (!el || (e.type === "keydown" && e.key !== "Enter" && e.key !== " ")) return;
    e.preventDefault();
    selectNode(el.dataset.node, null);
    const mic = $("#microscope");
    if (mic.getBoundingClientRect().top > window.innerHeight - 120) mic.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" });
  }
  $("#specimen").addEventListener("click", pickFromSpecimen);
  $("#specimen").addEventListener("keydown", pickFromSpecimen);

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
    const more = items.length > 3 ? h("p", { class: "small" }, `${items.length - 3} more in `, h("a", { href: "#status" }, "Status"), " and ", h("a", { href: "#mind/questions" }, "Mind"), ".") : null;
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
    renderMicroscope();
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

  // ------------------------------------------------------------------ tabs
  // a row of role="tab" buttons: the selected one is marked and is the only one in the tab order
  function syncTabs(list, key, current, panel) {
    $$("[role=tab]", list).forEach(b => {
      const on = b.dataset[key] === current;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
      if (on && panel) panel.setAttribute("aria-labelledby", b.id);
    });
  }
  // click a tab, or move between them with the arrow keys, Home and End
  function tabKeys(list, key, select) {
    const tabs = () => $$("[role=tab]", list);
    list.addEventListener("click", e => { const b = e.target.closest("[role=tab]"); if (b) select(b.dataset[key]); });
    list.addEventListener("keydown", e => {
      const all = tabs(), i = all.indexOf(document.activeElement);
      if (i < 0) return;
      const j = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: all.length - 1 }[e.key];
      if (j === undefined) return;
      e.preventDefault();
      select(all[(j + all.length) % all.length].dataset[key], true);
    });
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
      h("p", { class: "lede" }, `${who} When no one has asked it for anything for a while, it decides what to learn or test next: research, practise, examine, improve, consolidate, build a tool, try one out, or rest. ` +
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

  function toolsmithPanel(ts) {
    if (!ts || !ts.enabled) return null;
    const need = h("input", { type: "text", placeholder: "what the tool should do, e.g. reverse a word, count the vowels", "aria-label": "What the tool should do" });
    const build = h("form", { class: "row", onsubmit: e => {
      e.preventDefault();
      if (!need.value.trim()) return;
      act(e.target.querySelector("button"), async () => { const r = await api("/toolsmith", { build: need.value.trim(), by: "web" }); need.value = ""; return r; }, r => r.message);
    } }, need, h("button", { type: "submit" }, "Write a tool"));
    const node = ts.node;
    const who = node && node.trusted ? `Coder node ${node.id} (${pct(node.holdout)} on held-out requests) picks the recipe.`
      : `The coder node is still learning (${ts.examples} examples), so its code library's cues pick the recipe.`;
    const state = { "awaiting approval": "hit", proven: "ok", active: "ok", installed: "dim", "needs work": "no", declined: "dim", quarantined: "no", retired: "dim" };
    const tools = ts.drafts.slice().reverse().map(d => h("li", { class: "ask" },
      h("div", { class: "row", style: "justify-content:space-between" },
        h("span", {}, h("b", {}, `${d.name} v${d.version}`), " ", h("span", { class: state[d.status] || "dim" }, d.status), `: ${d.title}`),
        d.status === "awaiting approval" ? h("span", { class: "row" },
          h("button", { type: "button", class: "primary small", onclick: e => act(e.target, () => api("/approve", { id: d.approval, approve: true, by: "web" }), `Installed ${d.name}; it is self-tested next`) }, "Approve"),
          h("button", { type: "button", class: "quiet small", onclick: e => act(e.target, () => api("/approve", { id: d.approval, approve: false, by: "web" }), `Declined ${d.name}`) }, "Decline"))
          : ["active", "proven", "needs work"].includes(d.status) ? h("button", { type: "button", class: "quiet small", onclick: e => act(e.target, () => api("/toolsmith", { try: d.name, by: "web" }), r => r.message) }, "Try it out") : null),
      h("p", { class: "why" }, (d.needs.length ? `For “${cut(d.needs[0], 70)}”${d.needs.length > 1 ? ` and ${d.needs.length - 1} more` : ""}` : `You asked for a tool that ${d.asked}`) +
        `; ${d.tests.length} self-tests; written by ${d.by} from the ${d.recipe} recipe` +
        (d.live && d.live !== d.version ? `; v${d.live} stays installed until you decide` : "") +
        (d.score !== null && d.score !== undefined ? `; last experiment ${pct(d.score)} right` : "")),
      h("p", { class: "why dim small" }, d.folder)));
    const needs = ts.needs.map(g => h("li", { style: "padding:3px 0" }, `${g.recipe}: ${g.texts.length} request${g.texts.length === 1 ? "" : "s"}, e.g. “${cut(g.texts[0], 60)}”`));
    const recent = ts.history.slice().reverse().map(x => h("li", { class: "ask" },
      h("p", {}, h("time", { class: "dim small", title: dateTime(x.ts) }, clock(x.ts)), " ", h("b", {}, x.event), ` ${x.line}`)));
    return h("div", { class: "section" }, h("h2", {}, "Tools it writes"),
      h("p", { class: "lede" }, `When requests go unmet, its coder node picks a recipe from its code library (${ts.recipes.length} building blocks), and the toolsmith writes a tool from it with self-tests. ` +
        `Nothing it writes runs until you approve it. Once a tool is approved and passes its self-tests, it tries the tool out on its own and refines what falls short, which asks you again. ${who}`),
      build,
      h("p", { class: "small dim" }, `${ts.builds_left} builds and ${ts.experiments_left} experiments left this hour.`),
      h("div", { class: "cols", style: "margin-top:14px" },
        h("div", {}, h("h3", {}, "Tools it made"), tools.length ? h("ul", { class: "rows" }, tools) : h("p", { class: "dim" }, "None yet."),
          h("h3", { style: "margin-top:14px" }, "Requests it could write a tool for"), needs.length ? h("ul", { class: "rows small" }, needs) : h("p", { class: "dim" }, "None right now.")),
        h("div", {}, h("h3", {}, "Recent work"), recent.length ? h("ul", { class: "rows" }, recent) : h("p", { class: "dim" }, "Nothing yet."))));
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

  function questionsPanel(md) {
    const inquiries = md?.inquiries || [];
    const proposals = (md?.proposals || []).map(p => {
      const labels = h("input", { type: "text", placeholder: "labels (comma separated, optional)", "aria-label": "Labels", value: (p.labels || []).join(", ") });
      labels.dataset.free = "1";
      return h("div", { class: "ask" }, h("p", {}, h("b", {}, `${p.kind} ${p.name}`), p.description ? `: ${p.description}` : ""),
        h("p", { class: "why" }, p.reason),
        h("div", { class: "row" }, p.kind === "skill" ? labels : null,
          h("button", { type: "button", class: "primary", onclick: e => act(e.target, () => api("/adopt", { proposal_id: p.id, labels: p.kind === "skill" ? labels.value.split(",").map(x => x.trim()).filter(Boolean) : undefined, by: "web" }), r => r.message) }, "Adopt"),
          h("button", { type: "button", onclick: e => act(e.target, () => api("/decline", { proposal_id: p.id, by: "web" }), r => r.message) }, "Decline")));
    });
    return h("div", { class: "cols" },
      h("div", { class: "section" }, h("h2", {}, "Questions for you"),
        inquiries.length ? inquiries.map(q => h("div", { class: "ask" }, h("p", {}, q.question), q.why ? h("p", { class: "why" }, `Why it asks: ${q.why}`) : null, replyForm(q.id, q.kind === "clarify" ? [] : q.options)))
          : h("p", { class: "dim" }, "No questions right now. It asks when it could not fulfil an order, when it wants to know your interests, and before a tool it wrote may run.")),
      h("div", { class: "section" }, h("h2", {}, "Proposals"),
        proposals.length ? proposals : h("p", { class: "dim" }, "Nothing proposed. It proposes skills, labels and tools when it notices a need; nothing changes until you adopt it.")));
  }

  // a <details> that stays open across the 1.5 s re-renders
  function fold(key, summary, ...kids) {
    return h("details", { open: S.open.has(key), ontoggle: e => {
      if (e.target.open === S.open.has(key)) return;
      if (e.target.open) S.open.add(key); else S.open.delete(key);
      if (e.target.open && key.startsWith("report:")) refreshExtra(true);
    } }, h("summary", {}, summary), kids);
  }

  function objectivesPanel(ob) {
    if (!ob || !ob.enabled) return offPanel("Objectives", "Objectives are off. Turn them on with [objectives] enabled = true in directives.toml.");
    const aim = h("input", { type: "text", placeholder: "understand the Berlin Wall, find out who wrote Pride and Prejudice and when, compare tea and coffee", "aria-label": "An objective" });
    const set = h("form", { class: "row", onsubmit: e => {
      e.preventDefault();
      if (!aim.value.trim()) return;
      act(e.target.querySelector("button"), async () => { const r = await api("/objectives", { text: aim.value.trim(), by: "web" }); aim.value = ""; return r; }, r => r.message);
    } }, aim, h("button", { type: "submit", class: "primary" }, "Set objective"));
    const tone = { active: "hit", met: "ok", unmet: "no", paused: "dim" };
    const word = { active: "in progress", met: "met", unmet: "ended short", paused: "paused" };
    const stand = q => ["alt", "retry", "settle"].includes(q.role);
    const items = ob.objectives.slice().reverse().map(o => {
      const [done, total] = o.progress;
      const qs = o.questions.filter(q => q.kind !== "research" && !stand(q));
      const reading = o.questions.filter(q => q.kind === "research");
      const write = fold(`report:${o.id}`, "Read the write-up", h("pre", { class: "block report" }, S.reports[o.id] || "Loading the write-up…"),
        h("p", { class: "small dim" }, `Also in ${o.report}`));
      const check = q => q.status !== "answered" ? null
        : q.confirmed === true ? h("span", { class: "ok" }, ` confirmed (${q.checked_by})`)
        : q.confirmed === false ? h("span", { class: "no" }, ` disputed: ${q.checked_by} says “${cut(q.conflict, 80)}”`)
        : q.confirmed === "single" ? h("span", { class: "dim" }, " one source") : h("span", { class: "dim" }, " not checked yet");
      const rows = qs.map(q => h("li", { class: "ask" },
        h("p", {}, h("b", {}, q.ask), q.core ? null : h("span", { class: "dim small" }, q.from === "rumination" ? " follow-up" : " extra")),
        q.status === "answered" ? h("p", { class: "why" }, cut(q.answer, 220), " ", h("span", { class: "dim" }, `(${q.source}, ${pct(q.confidence)})`), check(q))
          : h("p", { class: "why" }, { todo: "Not asked yet.", waiting: `Waiting for you (${q.inquiry}).`, "waiting-task": cut(q.answer, 160), "waiting-tool": cut(q.answer, 160),
            retried: "Not answered yet; trying another way.", unknown: `Not found. Tried: ${q.tried.join("; ") || "nothing it has could answer it"}.`,
            failed: "Not answered; it will think it over.", skipped: "Skipped." }[q.status] || q.status)));
      const thoughts = o.thoughts.slice(-8).reverse().map(t => h("li", { style: "padding:3px 0" }, h("time", { class: "dim small", title: dateTime(t[0]) }, clock(t[0])), " ", h("b", {}, t[1]), ` ${t[2]}`));
      const live = o.status === "active" || o.status === "paused";
      return h("li", { class: "ask objective" },
        h("div", { class: "row", style: "justify-content:space-between;align-items:baseline" },
          h("span", {}, h("b", {}, o.text.charAt(0).toUpperCase() + o.text.slice(1)), " ", h("span", { class: tone[o.status] || "dim" }, word[o.status] || o.status),
            o.self_set ? h("span", { class: "dim small" }, " set by its own initiative") : null),
          live ? h("span", { class: "row" },
            h("button", { type: "button", class: "small", onclick: e => act(e.target, () => api("/objectives", { id: o.id, action: "think", by: "web" }), r => cut(r.message, 200)) }, "Think it over now"),
            h("button", { type: "button", class: "quiet small", onclick: e => act(e.target, () => api("/objectives", { id: o.id, action: o.status === "paused" ? "resume" : "pause", by: "web" }), r => r.message) }, o.status === "paused" ? "Resume" : "Pause"),
            h("button", { type: "button", class: "quiet small", onclick: e => act(e.target, () => api("/objectives", { id: o.id, action: "remove", by: "web" }), r => r.message) }, "Remove")) : null),
        h("div", { class: "progress", role: "img", "aria-label": `${done} of ${total} core questions answered` }, h("b", { style: `width:${total ? (done / total * 100).toFixed(0) : 0}%` })),
        h("p", { class: "why" }, `${o.id}: ${done} of ${total} core questions answered; ${o.steps} steps, ${o.rounds} round${o.rounds === 1 ? "" : "s"} of thought; ${o.strategy} strategy, planned by ${o.planned_by}.` +
          (reading.length ? ` Reading: ${reading.map(q => `${q.ask} (${q.status === "answered" ? "read" : q.note || q.status})`).join(", ")}.` : "")),
        o.note ? h("p", { class: "why" }, o.note) : null,
        o.summary ? h("p", {}, o.summary) : null,
        fold(`questions:${o.id}`, `Questions (${qs.length})`, h("ul", { class: "rows" }, rows)),
        thoughts.length ? fold(`thoughts:${o.id}`, "How it thought it over", h("ul", { class: "rows small" }, thoughts)) : null,
        write);
    });
    const second = ob.second_thoughts.slice().reverse().map(t => h("li", { class: "ask" },
      h("p", {}, h("time", { class: "dim small", title: dateTime(t.ts) }, clock(t.ts)), " ", h("span", { class: t.verdict === "changed its mind" ? "no" : t.verdict === "now answers" ? "ok" : "dim" }, t.verdict)),
      h("p", { class: "why" }, t.line)));
    return h("div", { class: "section" }, h("h2", {}, "Objectives"),
      h("p", { class: "lede" }, "Long-term aims it works on in its spare time. Its planner node picks a strategy, and each question in the plan is asked of the mesh itself: the router, its tools, memory nodes, what it has read, and teachers, with a director at both ends. " +
        "When the plan runs out it thinks things over: it checks each answer against a second source, tries what failed in other words or reads up on it, asks you what nothing it has can answer, and follows up what the answers raise. It writes up everything as it goes."),
      set,
      h("p", { class: "small dim" }, `${ob.steps_left} of ${ob.limit} steps left this hour. Also as an order: “objective: understand photosynthesis”.`),
      h("div", { class: "cols", style: "margin-top:14px" },
        h("div", {}, h("h3", {}, "Its objectives"), items.length ? h("ul", { class: "rows" }, items) : h("p", { class: "dim" }, "None yet. Set one above, or let its initiative set one about something you said you are interested in.")),
        h("div", {},
          h("h3", {}, "Second thoughts"),
          h("p", { class: "small dim" }, `It thinks again about recent answers it was unsure of or could not give. ${ob.doubts ? `${ob.doubts} worth another look now.` : "Nothing waiting for a second look."}`),
          h("button", { type: "button", disabled: !ob.doubts, onclick: e => act(e.target, () => api("/objectives", { ruminate: true, by: "web" }), r => cut(r.message, 240)) }, "Think again now"),
          second.length ? h("ul", { class: "rows", style: "margin-top:10px" }, second) : null)));
  }

  function thinkingPanel(tk, brief) {
    if (brief && !brief.enabled) return offPanel("Thinking", "Thinking nodes are off. Turn them on with [thinking] enabled = true (and [memory] enabled) in directives.toml.");
    if (!tk) return h("div", { class: "section" }, h("h2", {}, "Thinking"), h("p", { class: "dim" }, "Loading its thoughts…"));
    const nodes = new Map(tk.nodes.map(n => [n.id, n]));
    const want = d => !d ? null : h("p", { class: "want" }, h("b", {}, "Wants: "), d.want,
      d.status === "pursuing" ? h("span", { class: "hit" }, ` working on it (${d.objective})`) : d.status === "satisfied" ? h("span", { class: "ok" }, " found out") : null);
    const items = tk.thoughts.map(t => h("li", { class: "ask thought-item" },
      h("div", { class: "row", style: "justify-content:space-between;align-items:baseline" },
        h("span", {}, h("b", {}, t.topic), h("span", { class: "dim small" }, ` ${t.id} in ${t.node || "no node yet"}, from ${t.facts} fact${t.facts === 1 ? "" : "s"}`)),
        nodes.get(t.node)?.status === "ready" ? h("button", { type: "button", class: "linkish small", title: `Open ${t.node} in the closer look`,
          onclick: () => { lookInside(t.node, t.topic); } }, "Look inside") : null),
      h("p", { class: "own" }, t.understanding),
      t.opinion ? h("p", { class: "view" }, h("b", {}, "View: "), t.opinion.private ? h("span", { class: "dim" }, "private (an opinion about a person is never shared, P9)")
        : [t.opinion.stance, h("span", { class: "dim" }, ` because ${t.opinion.why}`)]) : null,
      want(t.desire),
      fold(`grounds:${t.id}`, `What it rests on (${t.facts})`, h("ul", { class: "rows small" }, t.grounds.map(g => h("li", {}, h("b", {}, g.ask), " ", cut(g.answer, 160), h("span", { class: "dim" }, ` (${g.node})`))))),
      h("p", { class: "small dim" }, t.formed_by + (t.rejected ? `. Turned down: ${t.rejected}` : ""))));
    const nodeRows = tk.nodes.map(n => h("li", { class: "ask" },
      h("p", {}, h("svg", { width: 16, height: 16, viewBox: "0 0 18 18", "aria-hidden": "true", class: "inline-glyph" }), h("b", {}, n.id), " ",
        h("span", { class: n.status === "ready" ? "ok" : n.status === "failing" ? "no" : "dim" }, n.status)),
      h("p", { class: "why" }, `${n.thoughts} thought${n.thoughts === 1 ? "" : "s"}; ${n.spec}; tests ${n.passed}/${n.total}; brought up ${n.served} time${n.served === 1 ? "" : "s"}`),
      n.failing.length ? h("p", { class: "why no" }, `Still learning: ${n.failing.map(f => `“${cut(f, 40)}”`).join(", ")}`) : null));
    nodeRows.forEach(li => { const g = li.querySelector("svg"); if (g) thinkGlyph(g, 9, 9, 6); });
    const st = tk.stats;
    return h("div", { class: "section" }, h("h2", {}, "Thinking"),
      h("p", { class: "lede" }, "After its memory nodes come its thinking nodes. For each topic it holds several facts about, it puts them together in its own words, forms a view from evidence it can name, and notes what it wants to find out. " +
        "Every word is checked against the facts underneath, so a thought never adds a fact. When an answer comes from what it knows, the thought appears after it, marked as its own thinking."),
      h("div", { class: "row" },
        h("button", { type: "button", onclick: e => act(e.target, () => api("/thinking", { form: true, by: "web" }), r => `Thought again: ${r.message}`) }, "Think again now"),
        h("span", { class: "small dim" }, `${st.formed} formed, ${st.reformed} re-formed, ${st.expressed} shown after answers, ${st.answered} questions answered, ${st.rejected} drafts turned down. Ask it: “what do you think of …?”, “what do you want to find out?”`)),
      h("div", { class: "cols", style: "margin-top:14px" },
        h("div", {}, h("h3", {}, "Its thoughts"), items.length ? h("ul", { class: "rows" }, items)
          : h("p", { class: "dim" }, "No thoughts yet. They form once its memory nodes hold two or more facts about the same thing: teach it some (/memorize question :: answer) or let it read.")),
        h("div", {},
          h("h3", {}, "What it wants to find out"),
          tk.desires.length ? h("ul", { class: "rows" }, tk.desires.map(d => h("li", { class: "ask" }, h("p", {}, d.want),
            h("p", { class: "why" }, `${d.thought} on ${d.topic} (${d.node}); ${d.status === "pursuing" ? `its director took it up as ${d.objective}` : "open: its director may take it up as an objective"}`))))
            : h("p", { class: "dim" }, "Nothing in particular yet."),
          h("h3", { style: "margin-top:14px" }, "Thinking nodes"),
          nodeRows.length ? h("ul", { class: "rows" }, nodeRows) : h("p", { class: "dim" }, "None yet."))));
  }

  function curiosityPanel(cu, brief) {
    if (brief && !brief.enabled) return offPanel("Curiosity", "Curiosity is off. Turn it on with [curiosity] enabled = true in directives.toml.");
    if (!cu) return h("div", { class: "section" }, h("h2", {}, "Curiosity"), h("p", { class: "dim" }, "Loading…"));
    const bar = (v, cls, label) => h("span", { class: `pair-bar ${cls}`, role: "img", "aria-label": `${label} ${pct(v)}` }, h("b", { style: `width:${(v * 100).toFixed(0)}%` }));
    const areas = cu.areas.map(a => h("li", { class: "ask area" },
      h("div", { class: "area-head" }, h("b", {}, a.name),
        h("span", { class: "area-bars" }, h("span", { class: "small dim" }, "interest"), bar(a.interest, "interest", "interest"),
          h("span", { class: "small dim" }, "lacks"), bar(a.lack, "lack", "what it lacks"))),
      h("p", { class: "why" }, [...a.why_interest, ...a.why_lack].join("; ") || "–"),
      a.gaps.length ? h("ul", { class: "gaps" }, a.gaps.slice(0, 3).map(g => h("li", {}, `“${cut(g.question, 90)}” `, h("span", { class: "dim" }, g.why)))) : null));
    const kind = k => h("span", { class: `kind-chip ${k}` }, k === "gap" ? "lacking" : "interest");
    const aims = cu.aims.map(x => h("li", { class: "ask" }, h("p", {}, kind(x.kind), " ", h("b", {}, x.text)), h("p", { class: "why" }, x.why)));
    const tone = { active: "hit", met: "ok", unmet: "no", paused: "dim", removed: "dim" };
    const mine = cu.set.map(r => h("li", { class: "ask" },
      h("p", {}, h("b", {}, r.text), " ", h("span", { class: tone[r.status] || "dim" }, r.status), h("span", { class: "dim small" }, ` ${r.objective}, ${r.progress[0]} of ${r.progress[1]} answered`)),
      h("p", { class: "why" }, `Why: ${r.why}`)));
    return h("div", { class: "section" }, h("h2", {}, "Curiosity"),
      h("p", { class: "lede" }, "What it finds interesting and where what it knows runs out. Interest grows when people ask about something, say they are interested, or when it reads about it. " +
        "It lacks information where it could not answer, was told it was wrong, an objective left a question open or its sources disagree, or it knows a name only from a mention. When it has time, its director sets itself objectives from here, and says why."),
      h("div", { class: "row" },
        h("button", { type: "button", disabled: !cu.aims.length || !S.dash.status.agency?.enabled,
          title: S.dash.status.agency?.enabled ? "Its director sets itself the first of these now" : "Its initiative is off: turn it on in Initiative",
          onclick: e => act(e.target, () => api("/curiosity", { pursue: true, by: "web" }), r => cut(r.message || "Done", 200)) }, "Set itself one now"),
        h("span", { class: "small dim" }, `${cu.gaps} gap${cu.gaps === 1 ? "" : "s"} in what it knows; people asked ${cu.asked} time${cu.asked === 1 ? "" : "s"} about something it noted.`)),
      h("div", { class: "cols", style: "margin-top:14px" },
        h("div", {}, h("h3", {}, "Areas"), areas.length ? h("ul", { class: "rows" }, areas) : h("p", { class: "dim" }, "Nothing yet: ask it things, or teach it some facts.")),
        h("div", {},
          h("h3", {}, "Objectives it could set itself"),
          aims.length ? h("ul", { class: "rows" }, aims) : h("p", { class: "dim" }, "None right now."),
          h("h3", { style: "margin-top:14px" }, "Objectives it set itself"),
          mine.length ? h("ul", { class: "rows" }, mine) : h("p", { class: "dim" }, "None yet."))));
  }

  // open a node in the Live view's closer look, firing on a text
  function lookInside(id, text) {
    if (!S.closer) { S.closer = true; store.set("closer", "1"); }
    location.hash = "#live";  // setView renders the Live view with the closer look open
    selectNode(id, text, null, true);
    setTimeout(() => $("#microscope")?.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" }), 60);
  }

  function temperamentPanel(tm) {
    if (!tm || !tm.enabled) return h("div", { class: "section" }, h("h2", {}, "Temperament"),
      h("p", { class: "dim" }, "The temperament is off, so answers go out unchanged. Turn it on with [temperament] enabled = true in directives.toml."));
    const decided = Object.entries(tm.decisions).map(([k, v]) => `${k} ${v}`).join(", ") || "none yet";
    const held = (tm.held || []).map(id => h("li", { class: "ask" }, h("div", { class: "row", style: "justify-content:space-between" },
      h("span", {}, h("code", {}, id)),
      h("button", { type: "button", class: "small", onclick: e => act(e.target, () => api("/release", { order_id: id, by: "web" }), r => r.output ? cut(r.output, 240) : r.message) }, "Release"))));
    return h("div", { class: "section" }, h("h2", {}, `Temperament: ${tm.mood}`),
      h("p", { class: "lede" }, "Its mood moves with the tone of what people say and how its work goes, and drifts back to its baseline (the tick on each bar). " +
        "It decides how an answer is delivered (passed, softened, caveated, held or withheld) and how carefully the directors work, never what an answer says."),
      h("div", { class: "cols", style: "margin-top:10px" },
        h("div", {},
          meter("Valence", tm.valence, tm.baseline.valence, -1, 1), meter("Arousal", tm.arousal, tm.baseline.arousal, 0, 1),
          meter("Confidence", tm.confidence, tm.baseline.confidence, 0, 1), meter("Trust", tm.trust, tm.baseline.trust, 0, 1),
          meter("Patience", tm.patience, tm.baseline.patience, 0, 1),
          h("div", { class: "row", style: "margin-top:12px" },
            h("button", { type: "button", onclick: e => act(e.target, () => api("/calm", { by: "web" }), r => `Mood reset: ${r.mood}`) }, "Reset its mood"))),
        h("div", {},
          h("h3", {}, "What it decided"), h("p", { class: "small" }, `${decided}.` + (tm.restored_by_guard ? ` The honesty guard restored ${tm.restored_by_guard} answer${tm.restored_by_guard === 1 ? "" : "s"} it tried to change.` : "")),
          h("h3", { style: "margin-top:14px" }, "Answers it is holding"),
          held.length ? h("ul", { class: "rows" }, held) : h("p", { class: "dim" }, "None. An answer it holds waits here until you release it."))));
  }

  function thoughtsPanel(md, op, notes) {
    const goal = h("input", { type: "text", placeholder: "Something to look into, e.g. the Krebs cycle", "aria-label": "Something to look into" });
    const investigate = h("form", { class: "row", onsubmit: e => {
      e.preventDefault();
      if (!goal.value.trim()) return;
      act(e.target.querySelector("button"), async () => { const r = await api("/investigate", { goal: goal.value.trim() }); goal.value = ""; return r; },
        r => cut(r.finding || r.note || JSON.stringify(r), 200));
    } }, goal, h("button", { type: "submit" }, "Look into it"));
    return [
      h("div", { class: "section" }, h("h2", {}, "Investigate"), investigate,
        h("p", { class: "dim small" }, (md ? `${md.investigations.last_hour} of ${md.investigations.limit} investigations this hour. ` : "") +
          "It looks into it with its own tools and nodes, and what it finds becomes a note.")),
      h("div", { class: "cols" },
        h("div", { class: "section" }, h("h2", {}, "Opinions it shares"),
          op ? [op.notes.length ? h("ul", { class: "rows" }, op.notes.map(n => h("li", { class: "ask" }, h("p", {}, h("b", {}, n.subject), ` ${n.stance} (${pct(n.strength)})`), h("p", { class: "why" }, n.text))))
            : h("p", { class: "dim" }, "No shared opinions yet."), op.withheld ? h("p", { class: "dim small" }, `${op.withheld} kept private.`) : null] : h("p", { class: "dim" }, "Loading…")),
        h("div", { class: "section" }, h("h2", {}, "Notes it shares"),
          notes ? [notes.notes.length ? h("ul", { class: "rows" }, notes.notes.slice(0, 20).map(n => h("li", { class: "ask" }, h("p", {}, h("b", {}, n.kind), ` ${n.subject}`), h("p", { class: "why" }, n.text))))
            : h("p", { class: "dim" }, "No shared notes yet."), notes.withheld ? h("p", { class: "dim small" }, `${notes.withheld} private notes (the operator sees them with /mind, which is audited).`) : null] : h("p", { class: "dim" }, "Loading…"))),
    ];
  }

  function offPanel(title, text) { return h("div", { class: "section" }, h("h2", {}, title), h("p", { class: "dim" }, text)); }

  // the Mind view is divided into tabs; the open one is in the address (#mind/tools) and remembered
  const MIND_TABS = ["questions", "thinking", "curiosity", "objectives", "directors", "initiative", "tools", "dreams", "temperament", "thoughts"];
  if (MIND_TABS.includes(store.get("mindTab", ""))) S.mindTab = store.get("mindTab", "");

  function mindBadges(st) {
    const md = st.mind, ts = st.toolsmith, tm = st.temperament, dr = st.dreams;
    const asks = (md?.inquiries?.length || 0) + (md?.proposals?.length || 0);
    $("#badge-mind-questions").textContent = asks ? String(asks) : "";
    const waiting = (ts?.drafts || []).filter(d => d.status === "awaiting approval").length;
    $("#badge-mind-tools").textContent = waiting ? String(waiting) : "";
    $("#note-mind-dreams").textContent = dr?.enabled && dr.asleep ? "asleep" : "";
    $("#note-mind-temperament").textContent = tm?.enabled ? tm.mood : "";
    const live = (st.objectives?.objectives || []).filter(o => o.status === "active").length;
    $("#note-mind-objectives").textContent = live ? `${live} active` : "";
    $("#note-mind-thinking").textContent = st.thinking?.enabled && st.thinking.thoughts ? `${st.thinking.thoughts}` : "";
    $("#badge-mind-curiosity").textContent = st.curiosity?.enabled && st.curiosity.gaps ? String(st.curiosity.gaps) : "";
  }

  function renderMind(force) {
    const host = $("#mind");
    const st = S.dash.status;
    syncTabs($("#mind-tabs"), "mind", S.mindTab, host);
    mindBadges(st);
    if (!force && busy(host)) return;
    const panel = {
      questions: () => questionsPanel(st.mind),
      thinking: () => thinkingPanel(S.extra.thinking, st.thinking),
      curiosity: () => curiosityPanel(S.extra.curiosity, st.curiosity),
      objectives: () => objectivesPanel(st.objectives),
      directors: () => directorsPanel(st.directors),
      initiative: () => initiative(st.agency) || offPanel("Initiative", "Its initiative is not available in this NeuroMesh."),
      tools: () => toolsmithPanel(st.toolsmith) || offPanel("Tools it writes", "The toolsmith is off. Turn it on with [toolsmith] enabled = true in directives.toml."),
      dreams: () => dreamsPanel(st.dreams) || offPanel("Dreams", "Dreams are off. Turn them on with [dreams] enabled = true in directives.toml."),
      temperament: () => temperamentPanel(st.temperament),
      thoughts: () => thoughtsPanel(st.mind, S.extra.opinions, S.extra.notes),
    }[S.mindTab];
    fill(host, panel());
  }

  function selectMind(tab, focus) {
    if (!MIND_TABS.includes(tab)) return;
    S.mindTab = tab;
    store.set("mindTab", tab);
    if (location.hash !== `#mind/${tab}`) history.replaceState(null, "", `#mind/${tab}`);
    if (focus) $(`#mind-tab-${tab}`).focus();
    if (S.dash) { renderMind(true); refreshExtra(true); }
  }
  tabKeys($("#mind-tabs"), "mind", selectMind);

  // ------------------------------------------------------------------ logs view
  const LEVELS = ["order", "lesson", "good", "info", "tool", "teacher", "warn", "bad", "work"];
  function renderLogs(force) {
    const host = $("#logs");
    syncTabs($("#logs-tabs"), "logs", S.logsTab, host);
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
  tabKeys($("#logs-tabs"), "logs", (tab, focus) => {
    S.logsTab = tab;
    if (focus) $(`#logs-tab-${tab}`).focus();
    renderLogs(true);
    refreshExtra(true);
  });

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
    const [v, sub] = location.hash.replace("#", "").split("/");
    const next = VIEWS.includes(v) ? v : "live";
    if (next === "mind" && MIND_TABS.includes(sub)) { S.mindTab = sub; store.set("mindTab", sub); }
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
    else if (S.view === "mind") renderMind(force);
    else if (S.view === "logs") renderLogs(force);
    else if (S.view === "status") renderStatus();
  }

  // what some views need beyond /dashboard, fetched while they are open
  async function refreshExtra(force) {
    const now = Date.now();
    const due = (key, ms) => force || !S.extraAt[key] || now - S.extraAt[key] > ms;
    const get = async (key, path) => { S.extraAt[key] = now; try { S.extra[key] = await api(path); } catch { /* shown on the next poll */ } };
    const jobs = [];
    if (S.view === "live" && S.closer && due("traces", 2000)) {
      jobs.push((async () => {
        S.extraAt.traces = now;
        try {
          const before = S.traces?.[0]?.order;
          S.traces = (await api("/traces?n=12")).traces;
          const newest = S.traces[0];
          if (S.follow && !S.sel.pinned && newest && newest.order !== before && before !== undefined && S.sel.node) {
            const hp = newest.hops.find(x => x.node === S.sel.node);
            if (hp) { S.trace = newest.order; selectNode(S.sel.node, hp.input || newest.text, newest.order); }
          }
        } catch { /* next poll */ }
      })());
    }
    if (S.view === "mind" && S.mindTab === "objectives") {
      for (const key of S.open) {
        if (!key.startsWith("report:")) continue;
        const id = key.slice(7);
        if (due(key, 4000)) jobs.push((async () => { S.extraAt[key] = now; try { S.reports[id] = (await api(`/objectives?id=${encodeURIComponent(id)}`)).report; } catch { /* next poll */ } })());
      }
    }
    if (S.view === "mind" && S.mindTab === "thinking" && due("thinking", 4000)) jobs.push(get("thinking", "/thinking"));
    if (S.view === "mind" && S.mindTab === "curiosity" && due("curiosity", 4000)) jobs.push(get("curiosity", "/curiosity"));
    if (S.view === "mind" && S.mindTab === "thoughts") {
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
