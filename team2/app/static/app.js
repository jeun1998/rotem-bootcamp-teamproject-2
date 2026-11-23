// ───────── 설정 ─────────

// 위험도는 확률에서 파생되는 표현일 뿐이므로 프론트에서 계산한다 (CONTEXT.md).
// 결항은 기본 발생률이 훨씬 낮으므로 기준을 따로 둔다.
const RISK_THRESHOLDS = {
  delay: { medium: 0.2, high: 0.35 },
  cancel: { medium: 0.02, high: 0.05 },
};
const RECENT_KEY = "recentQueries";
const RECENT_MAX = 5;

const $ = (id) => document.getElementById(id);
const form = $("predict-form");
let OPTIONS = { airlines: [], airports: [] };
const byCode = {};

// ───────── 유틸 ─────────

function localISODate(d) {
  const z = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${z(d.getMonth() + 1)}-${z(d.getDate())}`;
}
function addDays(d, n) { const c = new Date(d); c.setDate(c.getDate() + n); return c; }
function pct(p) { return (p * 100).toFixed(1); }
// 상위 공항과 항공사만 한글 이름이 있다. 나머지는 영문 이름을 쓴다
const displayName = (a) => a.name_ko ?? a.name;
function escapeHTML(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]); }

const storage = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} },
};

function riskLevel(kind, p) {
  const t = RISK_THRESHOLDS[kind];
  if (p >= t.high) return { label: "높음", cls: "high", icon: '<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17v.5"/>' };
  if (p >= t.medium) return { label: "보통", cls: "medium", icon: '<circle cx="12" cy="12" r="9"/><path d="M8 12h8"/>' };
  return { label: "낮음", cls: "low", icon: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>' };
}

// ───────── 테마 ─────────

$("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const isDark = root.dataset.theme
    ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = isDark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch {}
});

// ───────── 아래로 펼치는 패널의 자리 ─────────

// 선택창은 모두 칸 아래로 펼친다. 화면 아래로 넘치면 넘치는 만큼 화면을 올리고, 페이지 끝이 모자라면
// 끝에 그만큼 자리를 덧댄다. 패널이 모두 닫히면 덧댄 자리를 걷고, 결과가 없을 때는 처음 화면으로 돌아간다.
// 스크롤바는 html 의 scrollbar-gutter 로 자리를 늘 비워 두어, 자리가 생기고 걷힐 때 화면 폭이 바뀌지 않는다
const room = {
  open: new Set(),
  extra: 0,
  releasing: 0,
  show(panel) {
    clearTimeout(this.releasing);
    this.open.add(panel);
    const EDGE = 12;
    const bottom = panel.getBoundingClientRect().bottom + EDGE;
    const lift = Math.ceil(bottom - innerHeight);
    if (lift <= 0) return;
    // 펼친 패널도 문서 높이(scrollHeight)를 늘리므로, 패널을 뺀 본문 끝(body.offsetHeight)과 비교한다
    const short = Math.ceil(bottom + scrollY - document.body.offsetHeight);
    if (short > 0) {
      this.extra += short;
      document.body.style.paddingBottom = `${this.extra}px`;
    }
    scrollBy({ top: lift, behavior: "smooth" });
  },
  // 다른 칸으로 바로 옮겨 가면 이 패널이 닫히고 곧바로 다음 패널이 열린다. 잠깐 기다려
  // 다음 패널이 열리면 처음 화면으로 돌아가지 않고 덧댄 자리도 그대로 둔다
  hide(panel) {
    this.open.delete(panel);
    clearTimeout(this.releasing);
    this.releasing = setTimeout(() => {
      if (this.open.size || !this.extra) return;
      const release = () => { if (!this.open.size) { this.extra = 0; document.body.style.paddingBottom = ""; } };
      if (!$("result").hidden) return release();
      scrollTo({ top: 0, behavior: "smooth" });
      addEventListener("scrollend", release, { once: true });
      setTimeout(release, 800);  // scrollend 를 지원하지 않는 브라우저
    }, 100);
  },
};

// ───────── 콤보박스 (목록 드롭다운 + 검색): 항공사, 출발지, 도착지 ─────────

class Combo {
  // allowed(): 지금 고를 수 있는 코드 Set (다른 칸에서 고른 값에 맞는 것). onPick(code): 사용자가 고른 뒤 호출
  constructor(root, items, emptyText, { allowed = () => null, onPick = () => {} } = {}) {
    this.root = root;
    this.items = items;
    this.byCode = new Map(items.map((a) => [a.code, a]));
    this.emptyText = emptyText;
    this.allowed = allowed;
    this.onPick = onPick;
    this.hidden = root.querySelector('input[type="hidden"]');
    this.input = root.querySelector(".combo-input");
    this.list = root.querySelector(".combo-list");
    this.active = -1;
    this.matches = [];

    this.input.addEventListener("focus", () => { this.input.select(); this.open(""); });
    this.input.addEventListener("click", () => { if (this.list.hidden) this.open(""); });
    this.input.addEventListener("input", () => { this.hidden.value = ""; this.open(this.input.value); });
    this.input.addEventListener("keydown", (e) => this.onKey(e));
    this.input.addEventListener("blur", () => setTimeout(() => this.commit(), 0));
    this.root.querySelector(".chev").addEventListener("mousedown", (e) => {
      e.preventDefault();
      if (this.list.hidden) this.input.focus(); else this.close();
    });
    this.list.addEventListener("mousedown", (e) => {
      e.preventDefault(); // blur 방지
      const li = e.target.closest("li[data-code]");
      if (li) { this.pick(li.dataset.code); this.close(); }
    });
  }

  label(code) {
    const a = this.byCode.get(code);
    return a ? `${a.code} ${displayName(a)}` : "";
  }

  get value() { return this.hidden.value; }

  set(code) {
    this.hidden.value = code || "";
    this.input.value = this.label(code);
  }

  pick(code) {
    this.set(code);
    this.onPick(code);
  }

  pool() {
    const ok = this.allowed();
    return ok ? this.items.filter((a) => ok.has(a.code)) : this.items;
  }

  filter(q) {
    q = q.trim().toLowerCase();
    const pool = this.pool();
    if (!q || q === this.label(this.value).toLowerCase()) return pool;
    // 다른 칸과 맞는 것 중에 없으면 전체에서 찾는다. 고르면 맞지 않게 된 칸을 비운다 (onPick)
    const found = search(pool, q);
    return found.length ? found : search(this.items, q);
  }

  open(q) {
    this.matches = this.filter(q);
    this.active = this.matches.findIndex((a) => a.code === this.value);
    if (this.active < 0 && q.trim()) this.active = 0;
    this.render();
    const wasHidden = this.list.hidden;
    this.list.hidden = false;
    this.root.dataset.open = "";
    this.input.setAttribute("aria-expanded", "true");
    this.scrollActive();
    if (wasHidden) room.show(this.list);
  }

  close() {
    if (!this.list.hidden) room.hide(this.list);
    this.list.hidden = true;
    delete this.root.dataset.open;
    this.input.setAttribute("aria-expanded", "false");
    this.input.removeAttribute("aria-activedescendant");
  }

  render() {
    if (!this.matches.length) {
      this.list.innerHTML = `<li class="empty">${this.emptyText}</li>`;
      return;
    }
    const id = this.list.id;
    this.list.innerHTML = this.matches.map((a, i) => `
      <li id="${id}-${a.code}" role="option" data-code="${a.code}"
          aria-selected="${i === this.active}" class="${a.code === this.value ? "current" : ""}">
        <span class="code">${a.code}</span>
        <span class="nm">${escapeHTML(displayName(a))}${a.name_ko ? `<small>${escapeHTML(a.name)}</small>` : ""}</span>
      </li>`).join("");
    if (this.active >= 0) this.input.setAttribute("aria-activedescendant", `${id}-${this.matches[this.active].code}`);
  }

  scrollActive() {
    const el = this.list.querySelector('[aria-selected="true"]');
    if (el) this.list.scrollTop = el.offsetTop - (this.list.clientHeight - el.offsetHeight) / 2;
  }

  onKey(e) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (this.list.hidden) return this.open(this.input.value);
      const n = this.matches.length;
      if (!n) return;
      this.active = (this.active + (e.key === "ArrowDown" ? 1 : -1) + n) % n;
      this.render(); this.scrollActive();
    } else if (e.key === "Enter") {
      if (!this.list.hidden && this.matches[this.active]) {
        e.preventDefault();
        this.pick(this.matches[this.active].code);
        this.close();
      }
    } else if (e.key === "Escape") {
      this.close();
      this.input.value = this.label(this.value);
    }
  }

  // 포커스가 빠질 때: 선택이 없으면 정확히 일치하거나 유일한 후보를 자동 선택
  commit() {
    if (document.activeElement === this.input) return;
    this.close();
    if (this.value) { this.input.value = this.label(this.value); return; }
    const q = this.input.value.trim();
    if (!q) return;
    const exact = this.items.find((a) => a.code.toLowerCase() === q.toLowerCase());
    const matches = this.filter(q);
    const pick = exact || (matches.length === 1 ? matches[0] : null);
    if (pick) this.pick(pick.code);
    else this.hidden.value = q.toUpperCase(); // 서버가 "지원하지 않는 공항"으로 응답
  }
}

function search(items, q) {
  const starts = [], contains = [];
  for (const a of items) {
    const hay = [a.code, a.name, a.name_ko ?? ""].map((s) => s.toLowerCase());
    if (hay[0].startsWith(q)) starts.push(a);
    else if (hay.some((h) => h.includes(q))) contains.push(a);
  }
  return [...starts, ...contains];
}

let airlineCombo, originCombo, destCombo;

// ───────── 출발 예정일: 달력 (Air Datepicker) ─────────

// 폼 값은 숨은 칸에 "YYYY-MM-DD" 로 두고, 보이는 칸에는 "2026. 12. 03 (목)" 으로 보여 준다
const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];
const MONTHS = [...Array(12)].map((_, i) => `${i + 1}월`);
const KO_LOCALE = { days: WEEKDAYS.map((d) => `${d}요일`), daysShort: WEEKDAYS, daysMin: WEEKDAYS, months: MONTHS, monthsShort: MONTHS,
  today: "오늘", clear: "지우기", dateFormat: "yyyy-MM-dd", timeFormat: "HH:mm", firstDay: 0 };
const pad2 = (n) => String(n).padStart(2, "0");
// "YYYY-MM-DD" 를 그대로 Date 에 넣으면 UTC 자정이 되어 미국 시간대에서는 하루 앞 날짜가 된다
const localDate = (iso) => { const [y, m, d] = iso.split("-").map(Number); return new Date(y, m - 1, d); };

class DateField {
  constructor(control, { min, max, note }) {
    this.hidden = control.querySelector('input[type="hidden"]');
    const input = control.querySelector('input[type="text"]');
    this.picker = new AirDatepicker(input, {
      locale: KO_LOCALE, minDate: localDate(min), maxDate: localDate(max), autoClose: true, position: "bottom left",
      dateFormat: (d) => `${d.getFullYear()}. ${pad2(d.getMonth() + 1)}. ${pad2(d.getDate())} (${WEEKDAYS[d.getDay()]})`,
      altField: this.hidden, altFieldDateFormat: "yyyy-MM-dd",
      navTitles: { days: (dp) => `${dp.viewDate.getFullYear()}년 ${dp.viewDate.getMonth() + 1}월` },
      // 펼침 애니메이션 중에는 제자리보다 아래에 있어서, 다 펼친 뒤에 자리를 잰다
      onShow: (done) => { if (done) room.show(this.picker.$datepicker); else this.addNote(note); },
      onHide: (done) => { if (!done) room.hide(this.picker.$datepicker); },
    });
    control.querySelector(".chev").addEventListener("mousedown", (e) => { e.preventDefault(); input.focus(); this.picker.show(); });
  }

  // 고를 수 없는 날이 왜 막혔는지 달력 안에서 알려 준다. 달력 내용은 처음 펼칠 때 만들어진다
  addNote(note) {
    const cal = this.picker.$datepicker;
    if (note && !cal.querySelector(".adp-note")) cal.insertAdjacentHTML("beforeend", `<p class="adp-note">${escapeHTML(note)}</p>`);
  }

  get value() { return this.hidden.value; }
  // 달력이 숨은 칸을 늦게 채우므로, 바로 이어서 예측을 보내도 맞는 날짜가 가도록 직접 넣는다
  set value(iso) {
    this.hidden.value = iso;
    this.picker.selectDate(localDate(iso), { silent: true });
    this.picker.setViewDate(localDate(iso));
  }
}

// ───────── 출발·도착 시각: 오전/오후 · 시 · 분 세 칸 드롭다운 ─────────

// 폼 값은 숨은 칸에 24시간 "HH:MM" 으로 두고, 보이는 칸에는 "오전 09:30" 으로 보여 준다.
// 세 칸에서 하나씩 고르거나, 보이는 칸에 "14:30", "오후 2:30", "1430" 처럼 직접 입력한다
const MINUTE_STEP = 5;
const PERIODS = ["오전", "오후"];
const to12 = (hhmm) => { const [h, m] = hhmm.split(":").map(Number); return { period: PERIODS[+(h >= 12)], hour: h % 12 || 12, minute: m }; };
const to24 = ({ period, hour, minute }) => `${pad2((hour % 12) + (period === "오후" ? 12 : 0))}:${pad2(minute)}`;
const showTime = (hhmm) => { const t = to12(hhmm); return `${t.period} ${pad2(t.hour)}:${pad2(t.minute)}`; };

function parseTime(text) {
  const s = String(text).trim();
  const period = /오후|pm/i.test(s) ? "오후" : /오전|am/i.test(s) ? "오전" : null;
  const m = s.replace(/오전|오후|am|pm/gi, "").trim().match(/^(\d{1,2})(?::?(\d{2}))?$/);
  if (!m) return null;
  const hour = Number(m[1]), minute = Number(m[2] ?? 0);
  if (minute > 59 || hour > 23 || (period && (hour < 1 || hour > 12))) return null;
  return period ? to24({ period, hour, minute }) : `${pad2(hour)}:${pad2(minute)}`;
}

class TimeField {
  constructor(control) {
    this.hidden = control.querySelector('input[type="hidden"]');
    this.input = control.querySelector('input[type="text"]');
    this.panel = control.querySelector(".time-cols");
    const column = (part, options) => `<ul data-part="${part}" role="listbox">${
      options.map(([v, label]) => `<li role="option" data-v="${v}">${label}</li>`).join("")}</ul>`;
    this.panel.innerHTML = column("period", PERIODS.map((p) => [p, p]))
      + column("hour", [...Array(12)].map((_, i) => [i + 1, pad2(i + 1)]))
      + column("minute", [...Array(60 / MINUTE_STEP)].map((_, i) => [i * MINUTE_STEP, pad2(i * MINUTE_STEP)]));

    this.input.addEventListener("focus", () => { this.input.select(); this.open(); });
    this.input.addEventListener("click", () => this.open());
    // 입력하는 동안은 세 칸에 미리 보여 주기만 하고, 값은 칸을 떠날 때 정한다
    this.input.addEventListener("input", () => { const t = parseTime(this.input.value); if (t) this.mark(true, t); });
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === "Escape") { e.preventDefault(); this.input.blur(); }
    });
    this.input.addEventListener("blur", () => this.close());
    control.querySelector(".chev").addEventListener("mousedown", (e) => {
      e.preventDefault();
      if (this.panel.hidden) this.input.focus(); else this.input.blur();
    });
    this.panel.addEventListener("mousedown", (e) => {
      e.preventDefault();  // 칸의 포커스를 지켜 패널이 닫히지 않게 한다
      const li = e.target.closest("li");
      if (!li) return;
      const t = to12(parseTime(this.input.value) ?? (this.value || "09:00"));
      const part = li.parentElement.dataset.part;
      t[part] = part === "period" ? li.dataset.v : Number(li.dataset.v);
      this.value = to24(t);
    });
  }

  get value() { return this.hidden.value; }
  set value(hhmm) {
    this.hidden.value = hhmm;
    this.input.value = hhmm ? showTime(hhmm) : "";
    this.mark(false);
  }

  // 값을 각 칸에 표시한다. center 면 고른 줄이 칸 가운데 오도록 스크롤한다
  mark(center, hhmm = this.value) {
    const t = hhmm ? to12(hhmm) : null;
    for (const ul of this.panel.querySelectorAll("ul")) {
      const v = t ? String(t[ul.dataset.part]) : null;
      for (const li of ul.children) li.setAttribute("aria-selected", String(li.dataset.v === v));
      const cur = ul.querySelector('[aria-selected="true"]');
      if (center && cur) ul.scrollTop = cur.offsetTop - (ul.clientHeight - cur.offsetHeight) / 2;
    }
  }

  open() {
    if (!this.panel.hidden) return;
    this.panel.hidden = false;
    this.input.setAttribute("aria-expanded", "true");
    this.mark(true);
    room.show(this.panel);
  }

  close() {
    if (this.panel.hidden) return;
    this.panel.hidden = true;
    this.input.setAttribute("aria-expanded", "false");
    // 잘못 입력했으면 마지막으로 맞았던 값으로 되돌린다
    this.value = parseTime(this.input.value) ?? this.value;
    room.hide(this.panel);
  }
}

let dateField, depTime, arrTime;

// ───────── 실제 운항 노선으로 서로 거르기 ─────────

let ROUTES = {}; // 항공사 → 출발 → [도착] (서버 /api/options, ml/build_options.py 가 만든 실제 운항 노선)
const FIELD_LABEL = { airline: "항공사", origin: "출발지", dest: "도착지" };
const combos = () => ({ airline: airlineCombo, origin: originCombo, dest: destCombo });
const known = (combo) => (combo.byCode.has(combo.value) ? combo.value : "");

// field 칸에서 고를 수 있는 코드: 나머지 두 칸에서 고른 값과 함께 실제로 운항하는 노선이 있는 것
function allowedFor(field) {
  const picked = { airline: known(airlineCombo), origin: known(originCombo), dest: known(destCombo) };
  const out = new Set();
  for (const [airline, byOrigin] of Object.entries(ROUTES)) {
    if (field !== "airline" && picked.airline && airline !== picked.airline) continue;
    for (const [origin, dests] of Object.entries(byOrigin)) {
      if (field !== "origin" && picked.origin && origin !== picked.origin) continue;
      for (const dest of dests) {
        if (field !== "dest" && picked.dest && dest !== picked.dest) continue;
        out.add({ airline, origin, dest }[field]);
      }
    }
  }
  return out;
}

function showNote(field, text) {
  form.querySelector(`.error[data-for="${field}"]`).textContent = text;
}

// changed 칸을 고른 뒤, 그 값과 맞지 않게 된 다른 칸은 비우고 이유를 알려 준다
function dropUnreachable(changed) {
  showNote(changed, "");
  for (const [field, combo] of Object.entries(combos())) {
    if (field === changed || !known(combo) || allowedFor(field).has(combo.value)) continue;
    combo.set("");
    showNote(field, `고른 ${FIELD_LABEL[changed]}로는 운항하지 않는 곳이라 ${FIELD_LABEL[field]}를 비웠어요.`);
  }
}

// ───────── 폼 ─────────

function showErrors(errors) {
  form.querySelectorAll(".error").forEach((el) => { el.textContent = errors[el.dataset.for] || ""; });
}

function fillForm(q) {
  airlineCombo.set(q.airline);
  originCombo.set(q.origin);
  destCombo.set(q.dest);
  if (q.date) dateField.value = q.date;
  depTime.value = q.dep_time;
  arrTime.value = q.arr_time;
}

const toMinutes = (hhmm) => { const [h, m] = hhmm.split(":").map(Number); return h * 60 + m; };
const toHHMM = (min) => { const m = ((min % 1440) + 1440) % 1440; return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`; };

// 출발 시각만 바꾸고 예정 소요시간은 유지한다 (서버의 시간대별 비교와 같은 규칙)
function withDeparture(q, hhmm) {
  return { ...q, dep_time: hhmm, arr_time: toHHMM(toMinutes(q.arr_time) + toMinutes(hhmm) - toMinutes(q.dep_time)) };
}

$("swap").addEventListener("click", () => {
  const o = originCombo.value, d = destCombo.value;
  originCombo.set(d); destCombo.set(o);
  if (known(destCombo) && !allowedFor("dest").has(destCombo.value)) {
    destCombo.set("");
    showNote("dest", "반대 방향은 고른 항공사가 운항하지 않아 도착지를 비웠어요.");
  }
});

async function postJSON(url, payload) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  return { ok: res.ok, data: await res.json().catch(() => ({})) };
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  showErrors({});
  const payload = Object.fromEntries(new FormData(form));
  const button = form.querySelector(".primary");
  button.disabled = true;
  try {
    const [single, hourly] = await Promise.all([
      postJSON("/api/predict", payload),
      postJSON("/api/predict/hourly", payload),
    ]);
    if (!single.ok) { showErrors(single.data.errors || { _: "요청에 실패했습니다." }); return; }
    renderResult(payload, single.data, hourly.ok ? hourly.data : null);
    remember(payload);
  } catch {
    showErrors({ _: "서버에 연결할 수 없습니다." });
  } finally {
    button.disabled = false;
  }
});

// ───────── 최근 조회 ─────────

function chip(q) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  const d = q.date ? `<small>${q.date.slice(5).replace("-", "/")} ${q.dep_time}</small>` : "";
  b.innerHTML = `${escapeHTML(q.airline)} ${escapeHTML(q.origin)} → ${escapeHTML(q.dest)}${d}`;
  b.addEventListener("click", () => { fillForm(q); form.requestSubmit(); });
  return b;
}

function renderRecent() {
  const today = localISODate(new Date());
  const items = storage.get(RECENT_KEY, []).filter((q) => q.date >= today && q.arr_time);
  $("recent-row").hidden = !items.length;
  $("recent-list").replaceChildren(...items.map((q) => chip(q)));
}

function remember(q) {
  const key = (x) => [x.airline, x.origin, x.dest, x.date, x.dep_time, x.arr_time].join("|");
  const items = storage.get(RECENT_KEY, []).filter((x) => key(x) !== key(q));
  storage.set(RECENT_KEY, [q, ...items].slice(0, RECENT_MAX));
  renderRecent();
}

// ───────── 결과 ─────────

function renderMetric(kind, p) {
  $(`${kind}-prob`).textContent = pct(p);
  requestAnimationFrame(() => { $(`${kind}-meter`).style.width = `${Math.max(p * 100, 1)}%`; });
  const r = riskLevel(kind, p);
  const el = $(`${kind}-risk`);
  el.className = `risk ${r.cls}`;
  el.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${r.icon}</svg>위험도 ${r.label}`;
}

function renderResult(q, single, hourly) {
  const al = OPTIONS.airlines.find((a) => a.code === q.airline);
  $("r-origin").textContent = q.origin;
  $("r-origin-name").textContent = byCode[q.origin] ? displayName(byCode[q.origin]) : "";
  $("r-dest").textContent = q.dest;
  $("r-dest-name").textContent = byCode[q.dest] ? displayName(byCode[q.dest]) : "";
  $("r-airline").textContent = al ? `${al.name_ko} (${al.code})` : q.airline;
  const [y, m, d] = q.date.split("-").map(Number);
  $("r-date").textContent = new Intl.DateTimeFormat("ko-KR", { dateStyle: "full" }).format(new Date(y, m - 1, d));
  $("r-time").textContent = `출발 ${q.dep_time} → 도착 ${q.arr_time}`;

  renderMetric("delay", single.delay_probability);
  renderMetric("cancel", single.cancel_probability);

  // 차트 라벨 크기를 재려면 그리기 전에 화면에 보여야 한다
  const section = $("result");
  const wasHidden = section.hidden;
  section.hidden = false;
  document.querySelector(".hourly").hidden = !hourly;

  const selectedHour = Number(q.dep_time.split(":")[0]);
  if (hourly) {
    renderHourlyChart($("chart-delay"), "delay", hourly.hours, selectedHour, q);
    renderHourlyChart($("chart-cancel"), "cancel", hourly.hours, selectedHour, q);
    $("hourly-table").innerHTML = hourly.hours.map((h) => `
      <tr class="${h.hour === selectedHour ? "current" : ""}">
        <td>${String(h.hour).padStart(2, "0")}:00</td><td>${pct(h.delay_probability)}%</td><td>${pct(h.cancel_probability)}%</td>
      </tr>`).join("");
  }
  if (wasHidden) section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ───────── 시간대별 막대 차트 ─────────

function niceMax(v) {
  const steps = [0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1];
  for (const s of steps) if (v <= s * 4) return s * 4 > 1 ? 1 : s * 4;
  return 1;
}

function renderHourlyChart(el, kind, hours, selectedHour, q) {
  const key = `${kind}_probability`;
  const W = 420, H = 180, padL = 34, padR = 6, padT = 22, padB = 22;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const max = niceMax(Math.max(...hours.map((h) => h[key])));
  const band = innerW / hours.length;
  const bw = Math.min(16, band - 4);
  const y = (v) => padT + innerH - (v / max) * innerH;

  const ticks = [0, max / 2, max];
  const fmtTick = (v) => `${+(v * 100).toFixed(1)}%`;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="시간대별 ${kind === "delay" ? "지연" : "결항"} 확률 막대 차트">`;
  for (const t of ticks) {
    svg += `<line class="gridline" x1="${padL}" x2="${W - padR}" y1="${y(t)}" y2="${y(t)}"/>`;
    svg += `<text class="tick" x="${padL - 6}" y="${y(t) + 4}" text-anchor="end">${fmtTick(t)}</text>`;
  }

  const minIdx = hours.reduce((best, h, i) => (h[key] < hours[best][key] ? i : best), 0);
  const selIdx = hours.findIndex((h) => h.hour === selectedHour);
  const hasSel = selIdx >= 0;
  const bars = [];

  hours.forEach((h, i) => {
    const cx = padL + band * i + band / 2;
    const v = h[key];
    const top = y(v), base = y(0);
    const hgt = Math.max(base - top, 1);
    const r = Math.min(4, hgt, bw / 2);
    const x0 = cx - bw / 2, x1 = cx + bw / 2;
    bars.push({ x0, x1, top });
    const path = `M${x0},${base} V${top + r} Q${x0},${top} ${x0 + r},${top} H${x1 - r} Q${x1},${top} ${x1},${top + r} V${base} Z`;
    const dim = hasSel && i !== selIdx ? " dim" : "";
    svg += `<rect class="hit" x="${padL + band * i}" y="${padT}" width="${band}" height="${innerH}" tabindex="0" role="button"
              data-i="${i}" aria-label="${h.hour}시 출발: ${pct(v)}%. 이 시각으로 다시 예측"/>`;
    svg += `<path class="bar${dim}" d="${path}"/>`;
    if (h.hour % 2 === 0) svg += `<text class="tick" x="${cx}" y="${H - 6}" text-anchor="middle">${h.hour}시</text>`;
  });

  // 선택한 시각과 최저 시각에만 값 라벨 (겹치면 선택 시각만)
  // 양 끝 막대의 라벨은 축 눈금과 겹치지 않도록 막대 쪽으로 정렬
  const label = (i, text, cls) => {
    const cx = padL + band * i + band / 2;
    const [x, anchor] = i <= 1 ? [cx - bw / 2, "start"] : i >= hours.length - 2 ? [cx + bw / 2, "end"] : [cx, "middle"];
    svg += `<text class="lbl ${cls}" x="${x}" y="${y(hours[i][key]) - 6}" text-anchor="${anchor}">${text}</text>`;
  };
  if (hasSel) label(selIdx, `${pct(hours[selIdx][key])}%`, "");
  if (!hasSel || Math.abs(minIdx - selIdx) > 1) label(minIdx, `${String(hours[minIdx].hour).padStart(2, "0")}시 · 최저 ${pct(hours[minIdx][key])}%`, "sub");
  svg += "</svg>";
  el.innerHTML = svg;
  const minLabel = el.querySelector(".lbl.sub");
  if (minLabel) wrapInBadge(minLabel, bars[minIdx].top, [...el.querySelectorAll(".lbl:not(.sub)")]);

  el.querySelectorAll(".hit").forEach((hit) => {
    const h = hours[Number(hit.dataset.i)];
    const show = () => showTooltip(hit, h, kind);
    hit.addEventListener("mouseenter", show);
    hit.addEventListener("focus", show);
    hit.addEventListener("mouseleave", hideTooltip);
    hit.addEventListener("blur", hideTooltip);
    // 그 시각의 확률은 이미 받은 시간대별 결과에 있고, 출발 시각만 바뀐 요청의 시간대별 결과도 같다.
    // 서버에 다시 묻지 않고 그 값으로 결과를 바꾼다. 폼은 차트를 만든 조회 기준으로 맞춘다 (그 사이 고친 칸이 있어도)
    const pick = () => {
      hideTooltip();
      const picked = withDeparture(q, `${String(h.hour).padStart(2, "0")}:00`);
      showErrors({});
      fillForm(picked);
      renderResult(picked, h, { hours });
      remember(picked);
    };
    hit.addEventListener("click", pick);
    hit.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
}

// 배지는 불투명해서 옆 막대를 덮어도 되지만, 값 라벨과 겹치면 둘 다 못 읽으므로 그때만 올린다
function wrapInBadge(text, barTop, others) {
  const padX = 6, padY = 3, gap = 4;
  const box = text.getBBox();
  const left = box.x - padX, right = box.x + box.width + padX;
  const height = box.height + padY * 2;
  let rectY = barTop - gap - height;
  for (const o of others.map((t) => t.getBBox())) {
    const crosses = o.x < right && o.x + o.width > left && o.y < rectY + height && o.y + o.height > rectY;
    if (crosses) rectY = o.y - gap - height;
  }
  rectY = Math.max(0, rectY);
  text.setAttribute("y", Number(text.getAttribute("y")) + (rectY + padY - box.y));
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  Object.entries({ class: "badge", x: left, y: rectY, width: right - left, height, rx: height / 2 }).forEach(([k, v]) => rect.setAttribute(k, v));
  text.before(rect);
}

const tooltip = $("tooltip");
function showTooltip(target, h, kind) {
  const color = kind === "delay" ? "var(--series-delay)" : "var(--series-cancel)";
  const name = kind === "delay" ? "지연" : "결항";
  tooltip.innerHTML = `<div>${String(h.hour).padStart(2, "0")}:00 출발</div>
    <div><span class="sw" style="background:${color}"></span>${name} 확률 <b>${pct(h[`${kind}_probability`])}%</b></div>`;
  tooltip.hidden = false;
  // 막대 바로 위에 띄운다 (막대는 hit 영역 다음 형제)
  const bar = target.nextElementSibling.getBoundingClientRect();
  const r = target.getBoundingClientRect();
  const t = tooltip.getBoundingClientRect();
  let left = r.left + r.width / 2 - t.width / 2;
  left = Math.max(8, Math.min(left, innerWidth - t.width - 8));
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${Math.max(8, bar.top - t.height - 26)}px`;
}
function hideTooltip() { tooltip.hidden = true; }

// ───────── 초기화 ─────────

async function init() {
  OPTIONS = await (await fetch("/api/options")).json();
  OPTIONS.airports.forEach((a) => { byCode[a.code] = a; });

  ROUTES = OPTIONS.routes;
  const combo = (name, items, emptyText) => new Combo(form.querySelector(`[data-name="${name}"]`), items, emptyText,
    { allowed: () => allowedFor(name), onPick: () => dropUnreachable(name) });
  airlineCombo = combo("airline", OPTIONS.airlines, "조건에 맞는 항공사가 없어요");
  originCombo = combo("origin", OPTIONS.airports, "조건에 맞는 공항이 없어요");
  destCombo = combo("dest", OPTIONS.airports, "조건에 맞는 공항이 없어요");

  const { min, max, limit_note } = OPTIONS.dates;
  dateField = new DateField($("date").closest(".control"), { min, max, note: limit_note });
  const twoWeeks = localISODate(addDays(new Date(), 14));
  dateField.value = twoWeeks <= max ? twoWeeks : max;
  depTime = new TimeField($("dep_time").closest(".control"));
  arrTime = new TimeField($("arr_time").closest(".control"));
  depTime.value = "09:00";

  renderRecent();
}

if (!matchMedia("(prefers-reduced-motion: reduce)").matches) $("sky-video").play().catch(() => {});

// 스크롤한 정도(첫 화면 높이의 80%까지를 0~1)를 CSS 에 넘겨 배경 덮개(.sky-veil)를 짙게 한다
let scrollFrame = 0;
addEventListener("scroll", () => {
  cancelAnimationFrame(scrollFrame);
  scrollFrame = requestAnimationFrame(() => {
    document.documentElement.style.setProperty("--scrolled", Math.min(scrollY / (innerHeight * 0.8), 1).toFixed(3));
  });
}, { passive: true });

init();
