import { createChart, CandlestickSeries, createSeriesMarkers } from 'lightweight-charts'; 
import './style.css';

const API_BASE = "http://100.90.15.23/run_backtest";

// --- 全局变量 ---
let chart;         // 图表实例
let candleSeries;  // K线序列实例
let markersApi;    // 【新增】用于保存 Marker 控制器实例
let totalBars = 0; // 当前数据的总条数
let isUpdatingSlider = false; // 防抖锁

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
        updateChart(data.candles, data.markers); // 更新图表

        loadingEl.style.display = "none";

    } catch (err) {
        console.error(err);
        loadingEl.innerText = "连接服务器失败，请检查后端是否启动。";
    }
}

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
        chart.timeScale().scrollToPosition(distFromRight, false);
        isUpdatingSlider = false;
    };
}

// --- 数据更新与渲染 ---
function updateChart(candles, markers) {
    if (!chart) initChart();

    // 1. 设置数据
    candleSeries.setData(candles);
  
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
}

// 启动
loadData();