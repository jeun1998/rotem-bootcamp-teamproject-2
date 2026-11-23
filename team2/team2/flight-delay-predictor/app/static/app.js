// 위험도는 확률에서 파생되는 표현일 뿐이므로 프론트에서 계산한다 (CONTEXT.md 참고).
// 결항은 기본 발생률이 훨씬 낮으므로 기준을 따로 둔다.
const RISK_THRESHOLDS = {
  delay: { medium: 0.2, high: 0.35 },
  cancel: { medium: 0.02, high: 0.05 },
};

function riskLevel(kind, p) {
  const t = RISK_THRESHOLDS[kind];
  if (p >= t.high) return { label: "높음", cls: "high" };
  if (p >= t.medium) return { label: "보통", cls: "medium" };
  return { label: "낮음", cls: "low" };
}

const form = document.getElementById("predict-form");

function fillSelect(name, items) {
  const select = form.elements[name];
  select.innerHTML = '<option value="">선택하세요</option>' +
    items.map((i) => `<option value="${i.code}">${i.code} · ${i.name}</option>`).join("");
}

function showErrors(errors) {
  form.querySelectorAll(".error").forEach((el) => {
    el.textContent = errors[el.dataset.for] || "";
  });
}

function renderOutcome(kind, p) {
  const risk = riskLevel(kind, p);
  document.getElementById(`${kind}-prob`).textContent = `${(p * 100).toFixed(1)}%`;
  const badge = document.getElementById(`${kind}-risk`);
  badge.textContent = `위험도 ${risk.label}`;
  badge.className = `badge ${risk.cls}`;
}

async function init() {
  const res = await fetch("/api/options");
  const options = await res.json();
  fillSelect("airline", options.airlines);
  fillSelect("origin", options.airports);
  fillSelect("dest", options.airports);
  form.elements.date.valueAsDate = new Date();
  form.elements.dep_time.value = "09:00";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  showErrors({});
  const payload = Object.fromEntries(new FormData(form));
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      showErrors(data.errors || { _: "요청에 실패했습니다." });
      return;
    }
    renderOutcome("delay", data.delay_probability);
    renderOutcome("cancel", data.cancel_probability);
    document.getElementById("model-version").textContent = `모델: ${data.model_version}`;
    document.getElementById("result").hidden = false;
  } catch {
    showErrors({ _: "서버에 연결할 수 없습니다." });
  } finally {
    button.disabled = false;
  }
});

init();
