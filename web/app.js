const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const byId = (id) => document.getElementById(id);
const palette = {
  text: "#ecf3ff",
  muted: "#91a4bf",
  line: "#203755",
  accent: "#54d6bb",
  blue: "#75a7ff",
  yellow: "#f7c96a",
  red: "#f17b88",
  calm: "#54d6bb",
  volatile: "#f7c96a",
  dislocated: "#f17b88",
};

function shortNode(node) {
  return node.replace("BINANCE:", "BN ").replace("COINBASE:", "CB ").replace("KRAKEN:", "KR ");
}

function setStatus(message) {
  byId("status").textContent = message;
}

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function fitCanvas(canvas) {
  const scale = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * scale);
  canvas.height = Math.round(rect.height * scale);
  const context = canvas.getContext("2d");
  context.setTransform(scale, 0, 0, scale, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function arrow(context, from, to, width, alpha) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  const radius = 37;
  const end = { x: to.x - Math.cos(angle) * radius, y: to.y - Math.sin(angle) * radius };
  context.beginPath();
  context.moveTo(from.x + Math.cos(angle) * radius, from.y + Math.sin(angle) * radius);
  context.lineTo(end.x, end.y);
  context.strokeStyle = `rgba(84, 214, 187, ${alpha})`;
  context.lineWidth = width;
  context.stroke();
  context.beginPath();
  context.moveTo(end.x, end.y);
  context.lineTo(end.x - 8 * Math.cos(angle - Math.PI / 6), end.y - 8 * Math.sin(angle - Math.PI / 6));
  context.lineTo(end.x - 8 * Math.cos(angle + Math.PI / 6), end.y - 8 * Math.sin(angle + Math.PI / 6));
  context.closePath();
  context.fillStyle = `rgba(84, 214, 187, ${alpha})`;
  context.fill();
}

function drawNetwork(nodes, edges) {
  const canvas = byId("network-chart");
  const { context, width, height } = fitCanvas(canvas);
  context.clearRect(0, 0, width, height);
  const center = { x: width / 2, y: height / 2 };
  const radius = Math.min(width, height) * 0.32;
  const positions = new Map();
  nodes.forEach((node, index) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * index) / nodes.length;
    positions.set(node.id, { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius });
  });
  const max = Math.max(...edges.map((edge) => edge.excitation), 0.01);
  edges.forEach((edge) => {
    const from = positions.get(edge.source);
    const to = positions.get(edge.target);
    if (from && to) arrow(context, from, to, 1.2 + 4.5 * edge.excitation / max, 0.3 + 0.65 * edge.excitation / max);
  });
  nodes.forEach((node) => {
    const point = positions.get(node.id);
    context.beginPath();
    context.arc(point.x, point.y, 33, 0, Math.PI * 2);
    context.fillStyle = "#132b46";
    context.fill();
    context.strokeStyle = palette.blue;
    context.lineWidth = 1.2;
    context.stroke();
    context.textAlign = "center";
    context.fillStyle = palette.text;
    context.font = "700 11px system-ui";
    context.fillText(shortNode(node.id), point.x, point.y - 3);
    context.fillStyle = palette.muted;
    context.font = "10px system-ui";
    context.fillText(`${node.events} events`, point.x, point.y + 13);
  });
  if (!edges.length) {
    context.fillStyle = palette.muted;
    context.font = "14px system-ui";
    context.textAlign = "center";
    context.fillText("No edges exceeded the selected sparse-fit threshold.", center.x, center.y);
  }
}

function drawRegimes(data) {
  const canvas = byId("regime-chart");
  const { context, width, height } = fitCanvas(canvas);
  context.clearRect(0, 0, width, height);
  const pad = { left: 38, right: 13, top: 19, bottom: 30 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const maximum = Math.max(...data.map((row) => row.stress_score), 1);
  context.strokeStyle = "rgba(145, 164, 191, 0.18)";
  context.lineWidth = 1;
  [0, 0.5, 1].forEach((step) => {
    const y = pad.top + chartH * step;
    context.beginPath(); context.moveTo(pad.left, y); context.lineTo(width - pad.right, y); context.stroke();
  });
  const barWidth = Math.max(1, chartW / data.length);
  data.forEach((row, index) => {
    const x = pad.left + index * barWidth;
    context.fillStyle = `${palette[row.regime] || palette.calm}20`;
    context.fillRect(x, pad.top, barWidth + 0.3, chartH);
  });
  context.beginPath();
  data.forEach((row, index) => {
    const x = pad.left + (index / Math.max(data.length - 1, 1)) * chartW;
    const y = pad.top + chartH - (row.stress_score / maximum) * chartH;
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.strokeStyle = palette.blue;
  context.lineWidth = 2;
  context.stroke();
  context.fillStyle = palette.muted;
  context.font = "10px system-ui";
  context.textAlign = "right";
  context.fillText(`${maximum.toFixed(1)}×`, pad.left - 6, pad.top + 4);
  context.fillText("0×", pad.left - 6, pad.top + chartH + 3);
  context.textAlign = "left";
  context.fillText("replay start", pad.left, height - 9);
  context.textAlign = "right";
  context.fillText("replay end", width - pad.right, height - 9);
}

function drawEquity(data) {
  const canvas = byId("equity-chart");
  const { context, width, height } = fitCanvas(canvas);
  context.clearRect(0, 0, width, height);
  const pad = { left: 54, right: 14, top: 18, bottom: 25 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const values = data.map((row) => row.equity);
  const low = Math.min(...values, 100000);
  const high = Math.max(...values, 100000);
  const spread = Math.max(1, high - low);
  context.strokeStyle = "rgba(145, 164, 191, 0.18)";
  [0, 0.5, 1].forEach((step) => {
    const y = pad.top + chartH * step;
    context.beginPath(); context.moveTo(pad.left, y); context.lineTo(width - pad.right, y); context.stroke();
  });
  context.beginPath();
  data.forEach((row, index) => {
    const x = pad.left + (index / Math.max(data.length - 1, 1)) * chartW;
    const y = pad.top + chartH - ((row.equity - low) / spread) * chartH;
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.strokeStyle = values.at(-1) >= 100000 ? palette.accent : palette.red;
  context.lineWidth = 2.3;
  context.stroke();
  context.fillStyle = palette.muted;
  context.font = "10px system-ui";
  context.textAlign = "right";
  context.fillText(currency.format(high), pad.left - 6, pad.top + 4);
  context.fillText(currency.format(low), pad.left - 6, pad.top + chartH + 3);
}

function renderEdges(edges) {
  const container = byId("edge-list");
  if (!edges.length) {
    container.innerHTML = '<p class="caption">No relationships reached the sparse threshold.</p>';
    return;
  }
  const maximum = Math.max(...edges.map((edge) => edge.excitation));
  container.innerHTML = edges.slice(0, 6).map((edge) => `
    <div class="edge-row">
      <header><span>${shortNode(edge.source)}</span><b>→</b><span>${shortNode(edge.target)}</span></header>
      <small>Excitation ${edge.excitation.toFixed(3)} · confidence ${(edge.confidence * 100).toFixed(0)}%</small>
      <div class="edge-bar"><i style="width:${Math.max(8, edge.excitation / maximum * 100)}%"></i></div>
    </div>
  `).join("");
}

function renderTrades(trades) {
  const table = byId("trade-table");
  if (!trades.length) {
    table.innerHTML = '<tr><td colspan="8">No qualifying out-of-sample signals for this replay.</td></tr>';
    return;
  }
  table.innerHTML = trades.slice(-10).reverse().map((trade) => `
    <tr>
      <td>${(trade.timestamp_ms / 1000).toFixed(0)}s</td>
      <td>${shortNode(trade.source)}</td>
      <td>${shortNode(trade.target)}</td>
      <td class="side">${trade.side}</td>
      <td><span class="regime-pill">${trade.regime}</span></td>
      <td>${trade.gross_bps.toFixed(2)}</td>
      <td>${trade.cost_bps.toFixed(2)}</td>
      <td class="${trade.net_pnl_usd >= 0 ? "profit" : "loss"}">${currency.format(trade.net_pnl_usd)}</td>
    </tr>
  `).join("");
}

function updateDashboard(metadata, network, regimes, backtest) {
  const summary = backtest.summary;
  const baseline = backtest.baseline_summary;
  byId("event-count").textContent = metadata.events.toLocaleString();
  byId("source-label").textContent = `${metadata.nodes.length} venues · ${metadata.duration_seconds / 60} min replay`;
  byId("edge-count").textContent = network.edges.length.toString();
  byId("pnl").textContent = currency.format(summary.net_pnl_usd);
  byId("trade-count").textContent = `${summary.trades} test-set trades · ${summary.win_rate_pct}% win rate`;
  byId("sharpe").textContent = summary.sharpe_like.toFixed(2);
  byId("drawdown").textContent = `${summary.max_drawdown_pct}% max drawdown`;
  byId("model-label").textContent = metadata.model;
  byId("baseline-label").textContent = `Static baseline ${currency.format(baseline.net_pnl_usd)}`;
  byId("comparison").innerHTML = `
    <span><b>Model:</b> ${currency.format(summary.net_pnl_usd)} net after ${summary.average_cost_bps.toFixed(2)} bps average cost</span>
    <span><b>Baseline:</b> ${currency.format(baseline.net_pnl_usd)} without regime avoidance</span>
    <span><b>Test split:</b> ${metadata.test_bins} unseen 1-second bins</span>
  `;
  drawNetwork(network.nodes, network.edges);
  drawRegimes(regimes);
  drawEquity(backtest.equity_curve);
  renderEdges(network.edges);
  renderTrades(backtest.trades);
}

async function loadDashboard() {
  setStatus("Loading model output…");
  const [metadata, network, regimes, backtest] = await Promise.all([
    request("/api/metadata"), request("/api/network"), request("/api/regimes"), request("/api/backtest"),
  ]);
  updateDashboard(metadata, network, regimes, backtest);
  setStatus(`Run ready · seed ${metadata.seed} · ${metadata.events.toLocaleString()} normalized events`);
}

async function runAnalysis() {
  const button = byId("run-button");
  button.disabled = true;
  button.textContent = "Running…";
  setStatus("Generating event replay and fitting sparse diffusion network…");
  try {
    await request("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed: Number(byId("seed").value), duration_seconds: Number(byId("duration").value), bin_ms: 1000 }),
    });
    await loadDashboard();
  } catch (error) {
    setStatus(`Analysis failed: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Run analysis";
  }
}

byId("run-button").addEventListener("click", runAnalysis);
window.addEventListener("resize", () => loadDashboard().catch(() => {}));
loadDashboard().catch((error) => setStatus(`Unable to load dashboard: ${error.message}`));

