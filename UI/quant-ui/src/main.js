import { createChart, CandlestickSeries, HistogramSeries, LineSeries, createSeriesMarkers } from 'lightweight-charts';
import './style.css';

const API_BASE = "http://100.90.15.23:8000/run_backtest";

let chart;
let candleSeries;
let markersApi;
let totalBars = 0;
let isUpdatingSlider = false;
let volumeSeries;
let predSeries, labelSeries;
let isMeasuring = false;
let startPoint = null;
let rulerRect, rulerLabel;
let equitySeries;

window.changeTimeframe = async (tf) => {
    document.querySelectorAll('.tf-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    await loadData(tf);
};

async function loadData(timeframe = "") {
    const loadingEl = document.getElementById("loading");
    loadingEl.style.display = "block";
    loadingEl.innerText = "Loading Data...";

    try {
        const url = timeframe ? `${API_BASE}?tf=${timeframe}` : API_BASE;

        const res = await fetch(url);
        const data = await res.json();

        if (data.error) {
            alert("Error in backtest: " + data.error);
            loadingEl.style.display = "none";
            return;
        }

        renderStats(data.statistics);
        updateChart(data.candles, data.markers, data.statistics);

        loadingEl.style.display = "none";

    } catch (err) {
        console.error(err);
        loadingEl.innerText = "Connection to server failed. Please check if the backend is running.";
    }
}

// eg: getByPath(data, "params.strategy.commission")
function getByPath(obj, path, defaultValue = null) {
    if (!obj || !path) return defaultValue;

    return path.split(".").reduce((cur, key) => {
        if (cur !== null && cur !== undefined && cur[key] !== undefined && cur[key] !== null) {
            return cur[key];
        }
        return defaultValue;
    }, obj);
}

function formatStatValue(rawValue, item) {
    if (rawValue === null || rawValue === undefined) return null;

        if (item.type === "precision_base") {
        const cls = String(item.classId);

        const report = getByPath(item.rootData, `model_metrics.classification_report.${cls}`);
        const trueDist = getByPath(item.rootData, "model_metrics.label_distribution_true");

        if (!report || !trueDist) return null;

        const precision = report.precision;

        const total = Object.values(trueDist).reduce((sum, v) => sum + Number(v), 0);
        const baseCount = Number(trueDist[cls] ?? 0);
        const baseRate = total > 0 ? baseCount / total : null;

        if (precision === null || precision === undefined || baseRate === null) return null;

        return `${(precision * 100).toFixed(item.decimals ?? 2)}% : ${(baseRate * 100).toFixed(item.decimals ?? 2)}%`;
    }

    if (item.isPeriod) {
        const start = rawValue?.start;
        const end = rawValue?.end;

        if (!start || !end) return null;

        const startText = new Date(start).toLocaleDateString();
        const endText = new Date(end).toLocaleDateString();

        return `${startText} ~ ${endText}`;
    }

    if (item.isDateTime) {
        const date = new Date(rawValue);
        if (!Number.isNaN(date.getTime())) {
            return date.toLocaleString();
        }
        return String(rawValue);
    }

    let val = rawValue;

    if (typeof val === "string" && val.trim() !== "") {
        const parsed = Number(val);
        if (!Number.isNaN(parsed)) {
            val = parsed;
        }
    }

    if (typeof val !== "number") {
        return String(val);
    }

    if (item.scale100) {
        val = val * 100;
    }

    if (item.isCurrency) {
        return "$" + val.toLocaleString(undefined, {
            minimumFractionDigits: item.decimals ?? 2,
            maximumFractionDigits: item.decimals ?? 2
        });
    }

    const decimals = item.decimals ?? (val % 1 === 0 ? 0 : 4);
    let text = val.toFixed(decimals);

    if (item.showPercent) {
        text += "%";
    }

    return text;
}

const clearRuler = () => {
    isMeasuring = false;
    startPoint = null;

    rulerRect.style.display = 'none';
    rulerLabel.style.display = 'none';

    rulerRect.style.width = '0px';
    rulerRect.style.height = '0px';
    rulerLabel.innerHTML = '';

    chart.applyOptions({ handleScroll: { pressedMouseMove: true } });
};

function renderStats(statsArray) {
    const panel = document.getElementById("stats-panel");
    panel.innerHTML = "";

    if (!statsArray || !statsArray[1]) return;

    const data = statsArray[1];

    const config = [
        {
            group: "Backtest Period",
            fields: [
                {
                    key: "time",
                    title: "Period",
                    isPeriod: true,
                    wide: true
                }
            ]
        },
        {
            group: "Performance",
            fields: [
                {
                    key: "performance.gross_return",
                    title: "Total Return",
                    scale100: true,
                    showPercent: true,
                    decimals: 2
                },
                {
                    key: "performance.cagr",
                    title: "CAGR",
                    scale100: true,
                    showPercent: true,
                    decimals: 2
                },
                {
                    key: "performance.sharpe",
                    title: "Sharpe",
                    decimals: 2
                },
                {
                    key: "performance.calmar",
                    title: "Calmar",
                    decimals: 2
                },
                {
                    key: "performance.end_value",
                    title: "End Value",
                    isCurrency: true,
                    decimals: 2
                }
            ]
        },
        {
            group: "Risk & Drawdown",
            fields: [
                {
                    key: "drawdown.max_dd_pct",
                    title: "Max Drawdown %",
                    scale100: false,
                    showPercent: true,
                    decimals: 2
                },
                {
                    key: "drawdown.max_daily_dd",
                    title: "Max Daily DD",
                    scale100: true,
                    showPercent: true,
                    decimals: 2
                },
                {
                    key: "drawdown.dd_5_pct_days",
                    title: "Days > 5% DD",
                    decimals: 0
                },
                {
                    key: "drawdown.max_hwm_duration_days",
                    title: "Max Recovery Days",
                    decimals: 0
                }
            ]
        },
        {
            group: "Trades & Execution",
            fields: [
                {
                    key: "trades.total",
                    title: "Total Trades",
                    decimals: 0
                },
                {
                    key: "trades.win_rate",
                    title: "Win Rate",
                    scale100: true,
                    showPercent: true,
                    decimals: 2
                },
                {
                    key: "params.strategy.commission",
                    title: "Commission",
                    scale100: false,
                    showPercent: true,
                    decimals: 2
                },
                {
                    key: "trades.long_pnl",
                    title: "Long Return",
                    scale100: false,
                    suffix: "",
                    decimals: 4
                },
                {
                    key: "trades.short_pnl",
                    title: "Short Return",
                    scale100: false,
                    suffix: "",
                    decimals: 4
                },
                {
                    key: "trades.daily_freq",
                    title: "Daily Freq",
                    decimals: 4
                }
            ]
        },
        {
            group: "Exposure & Model",
            fields: [
                {
                    key: "exposure.avg_pos",
                    title: "Avg Exposure",
                    scale100: true,
                    showPercent: true,
                    decimals: 4
                },
                {
                    key: "model_metrics.accuracy",
                    title: "Model Acc",
                    scale100: true,
                    showPercent: true,
                    decimals: 2
                },
                {
                    key: "model_metrics.f1_macro",
                    title: "Macro F1",
                    scale100: false,
                    suffix: "",
                    decimals: 4
                },
                {
                    key: "model_metrics.f1_weighted",
                    title: "F1 Weighted",
                    scale100: false,
                    suffix: "",
                    decimals: 4
                },
                {
                    key: "model_metrics.classification_report.0",
                    title: "Short Presicion : Actual Rate",
                    type: "precision_base",
                    classId: 0,
                    decimals: 2,
                    wide: true
                },
                {
                    key: "model_metrics.classification_report.2",
                    title: "Long Presicion : Actual Rate",
                    type: "precision_base",
                    classId: 2,
                    decimals: 2,
                    wide: true
                }
            ]
        }
    ];

    config.forEach(section => {
        const sectionHeader = document.createElement("h2");
        sectionHeader.className = "section-title";
        sectionHeader.innerText = section.group;
        panel.appendChild(sectionHeader);

        const grid = document.createElement("div");
        grid.className = "stats-grid";

        section.fields.forEach(item => {
            const rawVal = getByPath(data, item.key);

            if (rawVal === null || rawVal === undefined) return;

            const displayValue = formatStatValue(rawVal, { ...item, rootData: data });

            let colorClass = "";

            const numericVal = Number(rawVal);

            if (!Number.isNaN(numericVal)) {
                if (numericVal < 0) {
                    colorClass = "loss";
                }

                if (
                    numericVal > 0 &&
                    (
                        item.key === "performance.gross_return" ||
                        item.key === "performance.cagr" ||
                        item.key === "trades.win_rate"
                    )
                ) {
                    colorClass = "win";
                }
            }

            const div = document.createElement("div");
            div.className = "card";
            if (item.wide) {
                div.classList.add("wide-card");
            }
            div.innerHTML = `<h3>${item.title}</h3><p class="${colorClass}">${displayValue}</p>`;
            grid.appendChild(div);
        });

        panel.appendChild(grid);
    });
}

function initChart() {
    const container = document.getElementById("chart-container");
    container.innerHTML = "";

    chart = createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            background: { color: "#ffffff" },
            textColor: "#333"
        },
        grid: {
            vertLines: { color: "#f0f0f0" },
            horzLines: { color: "#f0f0f0" }
        },
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
            borderColor: '#E1E1E1',
            barSpacing: 6,
            minBarSpacing: 0.005
        },
    });

    candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: "#26a69a",
        downColor: "#ef5350",
        borderVisible: false,
        wickUpColor: "#26a69a",
        wickDownColor: "#ef5350",
    });

    markersApi = createSeriesMarkers(candleSeries);

    equitySeries = chart.addSeries(LineSeries, {
        color: '#2962FF',
        lineWidth: 2,
        priceScaleId: 'equity-scale',
    });

    chart.priceScale('equity-scale').applyOptions({
        scaleMargins: {
            top: 0.75,
            bottom: 0.05,
        },
    });

    volumeSeries = chart.addSeries(HistogramSeries, {
        color: '#26a69a',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume-scale',
    });

    chart.priceScale('volume-scale').applyOptions({
        scaleMargins: {
            top: 0.8,
            bottom: 0,
        },
    });

    labelSeries = chart.addSeries(HistogramSeries, {
        title: 'Label',
        priceScaleId: 'label-scale',
    });

    predSeries = chart.addSeries(HistogramSeries, {
        title: 'Prediction',
        priceScaleId: 'pred-scale',
    });

    const fixedRange = () => ({
        priceRange: {
            minValue: 0,
            maxValue: 1
        },
    });

    chart.priceScale('label-scale').applyOptions({
        scaleMargins: {
            top: 0.02,
            bottom: 0.93
        },
    });

    labelSeries.applyOptions({
        autoscaleInfoProvider: fixedRange
    });

    chart.priceScale('pred-scale').applyOptions({
        scaleMargins: {
            top: 0.08,
            bottom: 0.87
        },
    });

    predSeries.applyOptions({
        autoscaleInfoProvider: fixedRange
    });

    window.chart = chart;
    window.volumeSeries = volumeSeries;

    window.addEventListener("resize", () => {
        chart.applyOptions({
            width: container.clientWidth,
            height: container.clientHeight
        });
    });

    chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
        if (isUpdatingSlider) return;

        requestAnimationFrame(() => {
            const slider = document.getElementById("time-scrollbar");
            if (!slider || totalBars === 0) return;

            const position = chart.timeScale().scrollPosition();
            const newVal = totalBars - position;

            if (Math.abs(slider.value - newVal) > 0.5) {
                slider.value = newVal;
            }
        });
    });

    const slider = document.getElementById("time-scrollbar");

    slider.oninput = function () {
        isUpdatingSlider = true;

        const distFromRight = totalBars - parseFloat(this.value);
        chart.timeScale().scrollToPosition(-distFromRight, false);

        isUpdatingSlider = false;
    };

    const tooltip = document.createElement('div');
    tooltip.className = 'floating-tooltip';
    container.appendChild(tooltip);

    chart.subscribeCrosshairMove(param => {
        if (
            param.point === undefined ||
            !param.time ||
            param.point.x < 0 ||
            param.point.x > container.clientWidth ||
            param.point.y < 0 ||
            param.point.y > container.clientHeight
        ) {
            tooltip.style.display = 'none';
            return;
        }

        const data = param.seriesData.get(candleSeries);

        if (!data) {
            tooltip.style.display = 'none';
            return;
        }

        tooltip.style.display = 'block';

        const change = data.close - data.open;
        const changePercentage = (change / data.open * 100).toFixed(2);
        const colorClass = change >= 0 ? 'win' : 'loss';
        const sign = change >= 0 ? '+' : '';

        tooltip.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 4px;">${new Date(data.time * 1000).toLocaleString()}</div>
            <div class="tooltip-row">Open: ${data.open.toFixed(2)}</div>
            <div class="tooltip-row">High: ${data.high.toFixed(2)}</div>
            <div class="tooltip-row">Low: ${data.low.toFixed(2)}</div>
            <div class="tooltip-row">Colse: ${data.close.toFixed(2)}</div>
            <div class="tooltip-row ${colorClass}">
                Range: ${sign}${changePercentage}% (${sign}${change.toFixed(2)})
            </div>
        `;

        const tooltipWidth = 180;
        const tooltipHeight = 150;
        const x = param.point.x;
        const y = param.point.y;

        let left = x + 20;
        if (left > container.clientWidth - tooltipWidth) {
            left = x - tooltipWidth - 20;
        }

        let top = y + 20;
        if (top > container.clientHeight - tooltipHeight) {
            top = y - tooltipHeight - 20;
        }

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    });

    rulerRect = document.createElement('div');
    rulerRect.className = 'ruler-rect';
    container.appendChild(rulerRect);

    rulerLabel = document.createElement('div');
    rulerLabel.className = 'ruler-label';
    container.appendChild(rulerLabel);

    container.addEventListener('mousedown', (e) => {
        if (!e.shiftKey) {
            clearRuler();
            return;
        }

        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const logicalIndex = chart.timeScale().coordinateToLogical(x);
        const price = candleSeries.coordinateToPrice(y);

        if (logicalIndex === null || price === null) return;

        isMeasuring = true;
        startPoint = {
            logicalIndex,
            price,
            x,
            y
        };

        chart.applyOptions({
            handleScroll: {
                pressedMouseMove: false
            }
        });

        rulerRect.style.display = 'block';
        rulerRect.style.width = '0px';
        rulerRect.style.height = '0px';
        rulerLabel.style.display = 'block';
    });

    container.addEventListener('mousemove', (e) => {
        if (!isMeasuring || !startPoint) return;

        const rect = container.getBoundingClientRect();
        const curX = e.clientX - rect.left;
        const curY = e.clientY - rect.top;

        const curLogicalIndex = chart.timeScale().coordinateToLogical(curX);
        const curPrice = candleSeries.coordinateToPrice(curY);

        if (curLogicalIndex === null || curPrice === null) return;

        const startXAtCurrentScale = chart.timeScale().logicalToCoordinate(startPoint.logicalIndex);

        const rectLeft = Math.min(startXAtCurrentScale, curX);
        const rectTop = Math.min(startPoint.y, curY);
        const rectWidth = Math.abs(curX - startXAtCurrentScale);
        const rectHeight = Math.abs(curY - startPoint.y);

        rulerRect.style.left = `${rectLeft}px`;
        rulerRect.style.top = `${rectTop}px`;
        rulerRect.style.width = `${rectWidth}px`;
        rulerRect.style.height = `${rectHeight}px`;

        const labelX = rectLeft + rectWidth + 5;
        const labelY = rectTop - 45;

        rulerLabel.style.left = `${labelX}px`;
        rulerLabel.style.top = `${labelY}px`;

        const barCount = Math.abs(Math.floor(curLogicalIndex) - Math.floor(startPoint.logicalIndex)) + 1;
        const priceDiff = curPrice - startPoint.price;
        const percentChange = ((priceDiff / startPoint.price) * 100).toFixed(2);

        const isUp = priceDiff >= 0;

        rulerRect.style.backgroundColor = isUp ? 'rgba(38, 166, 154, 0.2)' : 'rgba(239, 83, 80, 0.2)';
        rulerRect.style.borderColor = isUp ? '#26a69a' : '#ef5350';

        rulerLabel.innerHTML = `
            <b style="color: ${isUp ? '#26a69a' : '#ef5350'}">
                ${isUp ? '▲' : '▼'} ${percentChange}%
            </b><br>
            Vol: ${priceDiff.toFixed(2)}<br>
            Bars: ${barCount}
        `;
    });

    window.addEventListener('mouseup', () => {
        if (isMeasuring) {
            isMeasuring = false;

            chart.applyOptions({
                handleScroll: {
                    pressedMouseMove: true
                }
            });
        }
    });

    chart.subscribeClick(() => {
        clearRuler();
    });

    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            clearRuler();
        }
    });
}

function updateChart(candles, markers, statistics) {
    if (!chart) initChart();

    candleSeries.setData(candles);

    const volumeData = candles.map(c => ({
        time: c.time,
        value: c.volume,
        color: c.close >= c.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
    }));

    volumeSeries.setData(volumeData);

    const colors = {
        POSITIVE: '#26a69a',
        NEGATIVE: '#ef5350',
        NEUTRAL: '#b2b2b2',
        WARNING: '#ffff00'
    };

    const labelData = [];
    const predData = [];

    for (let i = 0; i < candles.length; i++) {
        const c = candles[i];
        const l = Math.round(c.label);
        const p = Math.round(c.pred);

        const getSignalColor = (val) => {
            if (val === 2) return colors.POSITIVE;
            if (val === 0) return colors.NEGATIVE;
            return colors.NEUTRAL;
        };

        labelData.push({
            time: c.time,
            value: 1,
            color: getSignalColor(l)
        });

        predData.push({
            time: c.time,
            value: 1,
            color: getSignalColor(p)
        });
    }

    labelSeries.setData(labelData);
    predSeries.setData(predData);

    if (markers && markers.length > 0) {
        markersApi.setMarkers(markers);
    } else {
        markersApi.setMarkers([]);
    }

    totalBars = candles.length;

    const slider = document.getElementById("time-scrollbar");
    slider.max = totalBars;
    slider.min = 0;
    slider.value = totalBars;

    chart.timeScale().scrollToPosition(0, false);

    // === Balance ===
    // try {
    //     const dailyList = statistics?.[1]?.drawdown?.daily_loss_list;
    //
    //     if (dailyList && dailyList.length > 0) {
    //         const equityData = dailyList.map(d => ({
    //             time: Math.floor(new Date(d.date).getTime() / 1000),
    //             value: d.equity
    //         }));
    //
    //         equitySeries.setData(equityData);
    //     }
    // } catch (e) {
    //     console.error("Equity parse error:", e);
    // }
}

// --- Performance switch ---
window.toggleStats = () => {
    const panel = document.getElementById("stats-panel");
    const btn = document.getElementById("performance-btn");

    panel.classList.toggle('hidden');
    btn.classList.toggle('active-toggle');

    if (window.chart) {
        const container = document.getElementById("chart-container");

        setTimeout(() => {
            window.chart.applyOptions({
                width: container.clientWidth,
                height: container.clientHeight
            });
        }, 50);
    }
};

loadData();