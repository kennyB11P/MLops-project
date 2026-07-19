from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/ui", response_class=HTMLResponse)
def ui_page() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Reviews Analytics MVP</title>
  <style>
    :root {
      --bg: #f4f6fb;
      --card: #ffffff;
      --border: #d9dee8;
      --text: #111827;
      --muted: #667085;
      --blue: #2563eb;
      --blue-dark: #1d4ed8;
      --danger: #b42318;
      --ok: #027a48;
      --code: #0f172a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      padding: 22px;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    h1 {
      margin: 0 0 4px;
      font-size: 28px;
      line-height: 1.2;
    }

    .subtitle {
      color: var(--muted);
      margin-bottom: 28px;
      font-size: 14px;
    }

    .layout {
      display: grid;
      grid-template-columns: 420px minmax(520px, 1fr);
      gap: 20px;
      align-items: start;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
      margin-bottom: 14px;
    }

    h2 {
      margin: 0 0 14px;
      font-size: 24px;
    }

    label {
      display: block;
      font-weight: 650;
      font-size: 13px;
      margin: 12px 0 7px;
    }

    input,
    select,
    textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      background: white;
      outline: none;
    }

    textarea {
      min-height: 86px;
      resize: vertical;
    }

    input:focus,
    select:focus,
    textarea:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .help {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .btn {
      width: 100%;
      border: 0;
      border-radius: 10px;
      padding: 12px 14px;
      font-weight: 700;
      font-size: 15px;
      cursor: pointer;
      color: white;
      background: var(--blue);
      margin-top: 14px;
    }

    .btn:hover {
      background: var(--blue-dark);
    }

    .btn.secondary {
      background: #111827;
    }

    .checkbox-line {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 12px;
      font-size: 14px;
      font-weight: 650;
    }

    .checkbox-line input {
      width: auto;
    }

    .labels-box {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      background: white;
      max-height: 245px;
      overflow: auto;
    }

    .label-option {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      margin: 0;
      padding: 7px 4px;
      font-weight: 500;
      font-size: 13px;
      line-height: 1.25;
      cursor: pointer;
    }

    .label-option input {
      width: auto;
      margin-top: 2px;
    }

    .small-actions {
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }

    .mini-btn {
      border: 1px solid var(--border);
      background: white;
      border-radius: 8px;
      padding: 7px 10px;
      cursor: pointer;
      color: var(--text);
      font-weight: 600;
    }

    .mini-btn:hover {
      background: #f8fafc;
    }

    .status {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      color: var(--muted);
      background: #f8fafc;
      margin-bottom: 14px;
    }

    .status.ok { color: var(--ok); }
    .status.err { color: var(--danger); }

    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }

    .metric {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      background: #f8fafc;
      min-width: 155px;
    }

    .metric .name {
      color: var(--muted);
      font-size: 12px;
    }

    .metric .value {
      font-size: 22px;
      font-weight: 750;
    }

    .table-wrap {
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-top: 12px;
      max-height: 520px;
    }

    table {
      border-collapse: collapse;
      width: 100%;
      background: white;
      font-size: 13px;
    }

    th,
    td {
      border-bottom: 1px solid var(--border);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      max-width: 520px;
    }

    th {
      background: #f8fafc;
      position: sticky;
      top: 0;
      z-index: 1;
    }

    pre {
      background: var(--code);
      color: #dbeafe;
      padding: 12px;
      border-radius: 12px;
      overflow: auto;
      max-height: 360px;
      font-size: 12px;
      line-height: 1.45;
    }

    details {
      margin-top: 14px;
    }

    summary {
      cursor: pointer;
      color: var(--muted);
      font-weight: 650;
    }

    @media (max-width: 980px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <h1>Reviews Analytics MVP</h1>
  <div class="subtitle">PostgreSQL-аналитика отзывов: шаблоны, чат, таблицы и примеры.</div>

  <div class="layout">
    <div>
      <div class="card">
        <h2>Шаблон</h2>

        <label for="templateSelect">Сценарий</label>
        <select id="templateSelect"></select>

        <label>Labels</label>
        <div id="labelsBox" class="labels-box"></div>
        <div class="small-actions">
          <button class="mini-btn" type="button" onclick="clearLabels()">Очистить</button>
          <button class="mini-btn" type="button" onclick="selectProblemLabels()">Все проблемы</button>
        </div>
        <div class="help">Можно выбрать несколько labels. Можно оставить пустым.</div>

        <div class="row">
          <div>
            <label for="dateFrom">Дата от</label>
            <input id="dateFrom" type="date" value="2025-09-01" />
          </div>
          <div>
            <label for="dateTo">Дата до</label>
            <input id="dateTo" type="date" value="2025-11-30" />
          </div>
        </div>

        <div class="row">
          <div>
            <label for="groupBy">Группировка</label>
            <select id="groupBy">
              <option value="">по умолчанию</option>
              <option value="day">day</option>
              <option value="week">week</option>
              <option value="month">month</option>
              <option value="label">label</option>
            </select>
          </div>
          <div>
            <label for="limit">Limit</label>
            <input id="limit" type="number" min="1" max="200" value="10" />
          </div>
        </div>

        <label for="keyword">Keyword</label>
        <input id="keyword" type="text" placeholder="например: синтетика" />
        <div class="help">Используется в keyword_search и review_samples.</div>

        <div class="checkbox-line">
          <input id="llmSummary" type="checkbox" />
          <span>Добавить аналитический вывод через LLM</span>
        </div>

        <button class="btn" type="button" onclick="runTemplate()">Выполнить шаблон</button>
      </div>

      <div class="card">
        <h2>Чат-запрос</h2>
        <textarea id="chatMessage">сколько жалоб на упаковку в сентябре?</textarea>
        <button class="btn secondary" type="button" onclick="runChat()">Отправить chat/ask</button>
      </div>

      <div class="card">
        <h2>База</h2>
        <button class="mini-btn" type="button" onclick="loadDbStats()">Обновить статистику</button>
        <pre id="dbStats">Нажми “Обновить статистику”.</pre>
      </div>
    </div>

    <div class="card">
      <h2>Результат</h2>
      <div id="status" class="status">Выполни шаблон или chat-запрос.</div>
      <div id="answer"></div>
      <div id="metrics" class="metrics"></div>
      <div id="table"></div>
      <details>
        <summary>Raw JSON</summary>
        <pre id="rawJson">{}</pre>
      </details>
    </div>
  </div>

<script>
const TEMPLATES = [
  ["count_by_problem", "count_by_problem — сколько отзывов по проблеме"],
  ["top_problems", "top_problems — топ проблем"],
  ["problem_dynamics", "problem_dynamics — динамика проблем"],
  ["review_samples", "review_samples — примеры отзывов"],
  ["top_products_by_problem", "top_products_by_problem — топ товаров по проблеме"],
  ["period_comparison", "period_comparison — сравнение периодов"],
  ["problem_share", "problem_share — доли проблем"],
  ["problem_growth", "problem_growth — рост проблем"],
  ["label_cooccurrence", "label_cooccurrence — совместные проблемы"],
  ["keyword_search", "keyword_search — поиск по слову"],
  ["positive_vs_problem", "positive_vs_problem — положительные vs проблемные"]
];

const LABELS = [
  "Положительный / нейтральный отзыв",
  "Проблема с размером / посадкой",
  "Проблема с качеством товара",
  "Проблема с комплектацией / упаковкой",
  "Несоответствие карточке товара",
  "Цена / ценность",
  "Проблема с возвратом",
  "Проблема доставки / получения",
  "Другая проблема"
];

const POSITIVE_LABEL = "Положительный / нейтральный отзыв";

function init() {
  const templateSelect = document.getElementById("templateSelect");
  templateSelect.innerHTML = TEMPLATES
    .map(([id, title]) => `<option value="${escapeHtml(id)}">${escapeHtml(title)}</option>`)
    .join("");

  const labelsBox = document.getElementById("labelsBox");
  labelsBox.innerHTML = LABELS.map((label, i) => `
    <label class="label-option">
      <input type="checkbox" name="labelCheckbox" value="${escapeHtml(label)}" />
      <span>${escapeHtml(label)}</span>
    </label>
  `).join("");

  document.getElementById("templateSelect").value = "top_problems";
  loadDbStats();
}

function getSelectedLabels() {
  return Array.from(document.querySelectorAll('input[name="labelCheckbox"]:checked'))
    .map(x => x.value);
}

function clearLabels() {
  document.querySelectorAll('input[name="labelCheckbox"]').forEach(x => x.checked = false);
}

function selectProblemLabels() {
  document.querySelectorAll('input[name="labelCheckbox"]').forEach(x => {
    x.checked = x.value !== POSITIVE_LABEL;
  });
}

function getPayload() {
  const dateFrom = document.getElementById("dateFrom").value || null;
  const dateTo = document.getElementById("dateTo").value || null;
  const keyword = document.getElementById("keyword").value.trim() || null;
  const groupBy = document.getElementById("groupBy").value || null;
  const limit = Number(document.getElementById("limit").value || 10);
  const addLLM = document.getElementById("llmSummary").checked;

  return {
    filters: {
      date_from: dateFrom,
      date_to: dateTo,
      labels: getSelectedLabels(),
      keyword: keyword
    },
    group_by: groupBy,
    add_analytical_summary: addLLM,
    limit: limit
  };
}

async function runTemplate() {
  const templateId = document.getElementById("templateSelect").value;
  setStatus(`Выполняю ${templateId}...`);

  try {
    const res = await fetch(`/api/v1/templates/${encodeURIComponent(templateId)}/execute`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(getPayload())
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(JSON.stringify(data, null, 2));
    }

    renderResult(data);
    setStatus("Готово.", "ok");
  } catch (err) {
    setStatus(String(err), "err");
  }
}

async function runChat() {
  const message = document.getElementById("chatMessage").value.trim();
  if (!message) {
    setStatus("Введите chat-запрос.", "err");
    return;
  }

  setStatus("Выполняю chat/ask...");

  try {
    const res = await fetch("/api/v1/chat/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ message })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(JSON.stringify(data, null, 2));
    }

    renderResult(data);
    setStatus("Готово.", "ok");
  } catch (err) {
    setStatus(String(err), "err");
  }
}

async function loadDbStats() {
  try {
    const res = await fetch("/api/v1/debug/db-stats");
    const data = await res.json();
    document.getElementById("dbStats").textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    document.getElementById("dbStats").textContent = String(err);
  }
}

function renderResult(data) {
  document.getElementById("rawJson").textContent = JSON.stringify(data, null, 2);

  const answerText = data.answer_text || data?.result?.answer_text || "";
  document.getElementById("answer").innerHTML = answerText
    ? `<div class="status ok">${escapeHtml(answerText)}</div>`
    : "";

  const metrics = data?.result?.metrics || data?.metrics || [];
  renderMetrics(metrics);

  const rows = extractRows(data);
  renderTable(rows);
}

function extractRows(data) {
  const directRows = data?.result?.rows || data?.rows || [];
  if (!Array.isArray(directRows)) return [];

  return directRows.map(row => row.data ? row.data : row);
}

function renderMetrics(metrics) {
  const box = document.getElementById("metrics");
  if (!metrics || metrics.length === 0) {
    box.innerHTML = "";
    return;
  }

  box.innerHTML = metrics.map(m => `
    <div class="metric">
      <div class="name">${escapeHtml(m.name || "")}</div>
      <div class="value">${escapeHtml(String(m.value ?? ""))}</div>
      <div class="name">${escapeHtml(m.unit || "")}</div>
    </div>
  `).join("");
}

function renderTable(rows) {
  const target = document.getElementById("table");

  if (!rows || rows.length === 0) {
    target.innerHTML = `<div class="status">Строк нет.</div>`;
    return;
  }

  const columns = Array.from(new Set(rows.flatMap(row => Object.keys(row))));
  const html = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>${columns.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows.map(row => `
            <tr>
              ${columns.map(c => `<td>${formatCell(row[c])}</td>`).join("")}
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;

  target.innerHTML = html;
}

function formatCell(value) {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return escapeHtml(value.join(", "));
  if (typeof value === "object") return `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
  return escapeHtml(String(value));
}

function setStatus(text, kind = "") {
  const el = document.getElementById("status");
  el.className = `status ${kind}`;
  el.textContent = text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>
        """
    )
