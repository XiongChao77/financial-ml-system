import { createChart, CandlestickSeries, HistogramSeries, LineSeries, createSeriesMarkers } from 'lightweight-charts'; 
import './style.css';

const API_BASE = "http://100.90.15.23:8000/run_backtest";

// --- 全局变量 ---
let chart;         // 图表实例
let candleSeries;  // K线序列实例
let markersApi;    // 【新增】用于保存 Marker 控制器实例
let totalBars = 0; // 当前数据的总条数
let isUpdatingSlider = false; // 防抖锁
let volumeSeries; // 【新增】全局变量
let predSeries, labelSeries;
let isMeasuring = false;  // 提升到全局
let startPoint = null;    // 提升到全局
let rulerRect, rulerLabel; // 提升到全局
let equitySeries;
// --- 暴露给 HTML 调用的切换函数 ---
window.changeTimeframe = async (tf) => {
    // 1. UI 更新: 切换按钮激活状态
    document.querySelectorAll('.tf-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // 2. 重新加载数据
    await loadData(tf);
};

// --- 数据加载逻辑 ---
async function loadData(timeframe = "") {
    const loadingEl = document.getElementById("loading");
    loadingEl.style.display = "block";
    loadingEl.innerText = "正在加载数据...";

    try {
        // 构建带参数的 URL
        const url = timeframe ? `${API_BASE}?tf=${timeframe}` : API_BASE;
        
        const res = await fetch(url);
        const data = await res.json();

        if (data.error) {
            alert("回测出错: " + data.error);
            loadingEl.style.display = "none";
            return;
        }

        renderStats(data.statistics);
        updateChart(data.candles, data.markers, data.statistics); // 更新图表

        loadingEl.style.display = "none";

    } catch (err) {
        console.error(err);
        loadingEl.innerText = "连接服务器失败，请检查后端是否启动。";
    }
}

// 在 initChart 内部定义这个重置函数
const clearRuler = () => {
    isMeasuring = false;
    startPoint = null;
    
    // 1. 隐藏元素
    rulerRect.style.display = 'none';
    rulerLabel.style.display = 'none';
    
    // 2. 【关键】清空物理尺寸，防止下次显示时“闪现”旧形状
    rulerRect.style.width = '0px';
    rulerRect.style.height = '0px';
    rulerLabel.innerHTML = '';
    // 恢复图表滚动功能
    chart.applyOptions({ handleScroll: { pressedMouseMove: true } });
};

// --- 统计面板渲染 ---
function renderStats(stats) {
    const panel = document.getElementById("stats-panel");
    panel.innerHTML = "";
    if (!stats) return;

    const fields = [
        { key: "gross_return", title: "总收益率", color: stats.gross_return.includes('-') ? 'loss' : 'win' },
        { key: "win_rate", title: "胜率", color: "" },
        { key: "sharpe", title: "夏普比率", color: "" },
        { key: "max_drawdown", title: "最大回撤", color: "loss" },
        { key: "total_trades", title: "交易次数", color: "" },
        { key: "cagr", title: "年化收益", color: "" },
        { key: "start_value", title: "初始资金", color: "" },
        { key: "end_value", title: "最终资金", color: "win" },
    ];

    fields.forEach(item => {
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `<h3>${item.title}</h3><p class="${item.color}">${stats[item.key]}</p>`;
        panel.appendChild(div);
    });
}

// --- 图表初始化 (只执行一次) ---
function initChart() {
    const container = document.getElementById("chart-container");
    container.innerHTML = ""; 

    chart = createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: { background: { color: "#ffffff" }, textColor: "#333" },
        grid: { vertLines: { color: "#f0f0f0" }, horzLines: { color: "#f0f0f0" } },
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
            borderColor: '#E1E1E1',
            barSpacing: 6,
            // 【关键】允许无限缩小，查看数年数据
            minBarSpacing: 0.005, 
        },
    });

    // 添加 K 线系列
    candleSeries = chart.addSeries(CandlestickSeries, { 
        upColor: "#26a69a", downColor: "#ef5350",
        borderVisible: false, 
        wickUpColor: "#26a69a", wickDownColor: "#ef5350",
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

    // 3. 【修正】正确添加成交量系列
    volumeSeries = chart.addSeries(HistogramSeries, {
        color: '#26a69a',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume-scale', // 创建一个独立的坐标轴
    });

    // 4. 设置成交量显示在底部 (占 20% 高度)
    chart.priceScale('volume-scale').applyOptions({
        scaleMargins: {
            top: 0.8,
            bottom: 0,
        },
    });

    //label  {
    // 1. 创建实际标签行 (Top Row)
    labelSeries = chart.addSeries(HistogramSeries, {
        title: 'Label',
        priceScaleId: 'label-scale',
    });

    // 2. 创建预测结果行 (Bottom Row)
    predSeries = chart.addSeries(HistogramSeries, {
        title: 'Prediction',
        priceScaleId: 'pred-scale',
    });

    // --- 【关键】修复报错：改用 autoscaleInfoProvider 锁定范围 ---
    const fixedRange = () => ({
        priceRange: { minValue: 0, maxValue: 1 }, // 锁定高度为 1
    });

    // 配置第一行位置 (顶部 2% - 7%)
    chart.priceScale('label-scale').applyOptions({
        scaleMargins: { top: 0.02, bottom: 0.93 },
    });
    labelSeries.applyOptions({ autoscaleInfoProvider: fixedRange });

    // 配置第二行位置 (顶部 8% - 13%)
    chart.priceScale('pred-scale').applyOptions({
        scaleMargins: { top: 0.08, bottom: 0.87 },
    });
    predSeries.applyOptions({ autoscaleInfoProvider: fixedRange });
    //label  }

    window.volumeSeries = volumeSeries;
    // 窗口大小自适应
    window.addEventListener("resize", () => {
        chart.applyOptions({ 
            width: container.clientWidth,
            height: container.clientHeight
        });
    });

    // --- 绑定图表拖动事件 -> 更新滑动条 ---
    chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
        if (isUpdatingSlider) return;
        
        // 简单的防抖，避免频繁操作 DOM
        requestAnimationFrame(() => {
            const slider = document.getElementById("time-scrollbar");
            if (!slider || totalBars === 0) return;

            // scrollPosition() 返回距离最右侧的偏移量
            // 0 = 最右边 (最新数据)
            // positive = 向左偏移
            const position = chart.timeScale().scrollPosition();
            
            // 滑动条逻辑：最右边是 max (totalBars)
            const newVal = totalBars - position;
            
            // 只有差异较大时才更新，防止死循环
            if (Math.abs(slider.value - newVal) > 0.5) {
                slider.value = newVal;
            }
        });
    });

    // --- 绑定滑动条拖动事件 -> 更新图表 ---
    const slider = document.getElementById("time-scrollbar");
    slider.oninput = function() {
        isUpdatingSlider = true;
        // 计算距离右侧的距离
        const distFromRight = totalBars - parseFloat(this.value);
        // 跳转，关闭动画以保证跟手
        chart.timeScale().scrollToPosition(-distFromRight, false);
        isUpdatingSlider = false;
    };

    // 在 initChart 函数内部添加
    const tooltip = document.createElement('div');
    tooltip.className = 'floating-tooltip';
    container.appendChild(tooltip);

    chart.subscribeCrosshairMove(param => {
        // 如果鼠标移出图表或在空白区域
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

        // 获取当前鼠标对应 K 线的数据
        const data = param.seriesData.get(candleSeries);
        if (!data) {
            tooltip.style.display = 'none';
            return;
        }

        tooltip.style.display = 'block';

        // 计算涨跌幅
        const change = data.close - data.open;
        const changePercentage = (change / data.open * 100).toFixed(2);
        const colorClass = change >= 0 ? 'win' : 'loss';
        const sign = change >= 0 ? '+' : '';

        // 填充内容
        tooltip.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 4px;">${new Date(data.time * 1000).toLocaleString()}</div>
            <div class="tooltip-row">开: ${data.open.toFixed(2)}</div>
            <div class="tooltip-row">高: ${data.high.toFixed(2)}</div>
            <div class="tooltip-row">低: ${data.low.toFixed(2)}</div>
            <div class="tooltip-row">收: ${data.close.toFixed(2)}</div>
            <div class="tooltip-row ${colorClass}">
                幅度: ${sign}${changePercentage}% (${sign}${change.toFixed(2)})
            </div>
        `;

        // 动态调整位置 (防止超出右边界)
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

    // --- 【新增】测量工具 DOM 元素 ---
    rulerRect = document.createElement('div');
    rulerRect.className = 'ruler-rect';
    container.appendChild(rulerRect);

    rulerLabel = document.createElement('div');
    rulerLabel.className = 'ruler-label';
    container.appendChild(rulerLabel);


    // --- 2. 鼠标按下事件 ---
    container.addEventListener('mousedown', (e) => {
        // 如果没有按住 Shift，直接清除并退出
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

        // 只有按住 Shift 时才执行以下逻辑
        isMeasuring = true;
        startPoint = { logicalIndex, price, x, y };

        // 禁用图表滚动，避免测量时图形乱跑
        chart.applyOptions({ handleScroll: { pressedMouseMove: false } });

        rulerRect.style.display = 'block';
        rulerRect.style.width = '0px';
        rulerRect.style.height = '0px';
        rulerLabel.style.display = 'block';
    });

    // --- 3. 鼠标移动事件 ---
    container.addEventListener('mousemove', (e) => {
        if (!isMeasuring || !startPoint) return;

        const rect = container.getBoundingClientRect();
        const curX = e.clientX - rect.left;
        const curY = e.clientY - rect.top;

        const curLogicalIndex = chart.timeScale().coordinateToLogical(curX);
        const curPrice = candleSeries.coordinateToPrice(curY);

        if (curLogicalIndex === null || curPrice === null) return;

        // 1. 计算矩形位置
        const startXAtCurrentScale = chart.timeScale().logicalToCoordinate(startPoint.logicalIndex);
        
        const rectLeft = Math.min(startXAtCurrentScale, curX);
        const rectTop = Math.min(startPoint.y, curY);
        const rectWidth = Math.abs(curX - startXAtCurrentScale);
        const rectHeight = Math.abs(curY - startPoint.y);

        rulerRect.style.left = `${rectLeft}px`;
        rulerRect.style.top = `${rectTop}px`;
        rulerRect.style.width = `${rectWidth}px`;
        rulerRect.style.height = `${rectHeight}px`;

        // 2. 【关键修改】计算标签位置：固定在矩形的右上方
        // X 坐标：矩形左边缘 + 宽度 + 5px 偏移
        // Y 坐标：矩形顶边缘 - 标签高度（约 40px）
        const labelX = rectLeft + rectWidth + 5;
        const labelY = rectTop - 45; // 减去 45px 让它悬浮在矩形上方

        rulerLabel.style.left = `${labelX}px`;
        rulerLabel.style.top = `${labelY}px`;

        // 3. 计算数据内容
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
            价差: ${priceDiff.toFixed(2)}<br>
            周期: ${barCount} 根 K 线
        `;
    });

    // --- 4. 鼠标松开事件 ---
    window.addEventListener('mouseup', () => {
        if (isMeasuring) {
            isMeasuring = false;
            // 恢复图表滚动
            chart.applyOptions({ handleScroll: { pressedMouseMove: true } });
        }
    });

    // --- 5. 图表点击清除 ---
    chart.subscribeClick(() => {
        // 普通点击（不带 Shift）直接清除
        clearRuler();
    });

    // 监听键盘按下（可选）：如果用户按下 Esc 键也清除测量
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            clearRuler();
        }
    });
}

// --- 数据更新与渲染 ---
function updateChart(candles, markers, statistics) {
    if (!chart) initChart();

    // 1. 设置数据
    candleSeries.setData(candles);

    // 2. 【新增】转换并设置成交量数据
    const volumeData = candles.map(c => ({
        time: c.time,
        value: c.volume,
        // 涨绿跌红逻辑
        color: c.close >= c.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
    }));
    volumeSeries.setData(volumeData);

    //label {
    const colors = {
        POSITIVE : '#26a69a',    // 绿色 (2)
        NEGATIVE: '#ef5350',   // 红色 (0)
        NEUTRAL: '#b2b2b2', // 灰色 (1)
        WARNING: '#ffff00'  // 【强调】黄色：预测与实际冲突
    };

    const labelData = [];
    const predData = [];

    for (let i = 0; i < candles.length; i++) {
        const c = candles[i];
        const l = Math.round(c.label); // 实际
        const p = Math.round(c.pred);  // 预测
        const isCorrect = l === p;

        // 颜色映射函数
        const getSignalColor = (val) => {
            if (val === 2) return colors.POSITIVE ;
            if (val === 0) return colors.NEGATIVE;
            return colors.NEUTRAL;
        };

        // 上行：显示实际状态
        labelData.push({
            time: c.time,
            value: 1, // 固定高度
            color: getSignalColor(l)
        });

        // 下行：显示预测状态，如果错误则变黄强调
        predData.push({
            time: c.time,
            value: 1, // 固定高度
            color: getSignalColor(p) 
        });
    }

    labelSeries.setData(labelData);
    predSeries.setData(predData);
    //label }

    // 2. 设置标记 (必须排序)
    if (markers && markers.length > 0) {
        markersApi.setMarkers(markers);
    } else {
        markersApi.setMarkers([]);
    }

    // 3. 更新滑动条范围
    totalBars = candles.length;
    const slider = document.getElementById("time-scrollbar");
    slider.max = totalBars;
    slider.min = 0;
    
    // 默认跳转到最新数据
    slider.value = totalBars; 
    
    // 自动适配视野
    // chart.timeScale().fitContent();
    
    // 强制重置滚动位置到最右
    chart.timeScale().scrollToPosition(0, false);

    // === 资金曲线 ===
    try {
        const dailyList = statistics?.[1]?.drawdown?.daily_loss_list;

        if (dailyList && dailyList.length > 0) {
            const equityData = dailyList.map(d => ({
                time: Math.floor(new Date(d.date).getTime() / 1000),
                value: d.equity
            }));

            equitySeries.setData(equityData);
        }
    } catch (e) {
        console.error("Equity parse error:", e);
    }
}

// 启动
loadData();