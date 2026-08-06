import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PROFILER_EVENT = "NNProfiler";
let timerRunning = false;
let timerFrame = null;
const slow_time = 10;

const config = {
  activeColor: "#4fc3f7",
  badgeColor: "#1e7832",
  badgeColorSlow: "#962828",
  slowTime: 10,
  precision: 2,
};

function updateConfig(cfg) {
  Object.assign(config, cfg);
}

function startTimerLoop() {
  if (timerRunning) {
    return;
  }

  timerRunning = true;

  function tick() {
    let active = false;

    for (const node of app.graph.nodes) {
      if (node.nnProfiler && node.nnProfiler.running) {
        active = true;
        node.setDirtyCanvas(true, true);
      }
    }

    if (active) {
      timerFrame = requestAnimationFrame(tick);
    } else {
      timerRunning = false;
      timerFrame = null;
    }
  }

  timerFrame = requestAnimationFrame(tick);
}

function findNode(path, graph = app.graph, index = 0) {
  const parts = Array.isArray(path) ? path : String(path).split(":");

  const node = graph.getNodeById(Number(parts[index]));

  if (!node) {
    return null;
  }

  if (index === parts.length - 1) {
    return node;
  }

  const nextGraph = node.subgraph;

  if (!nextGraph) {
    return null;
  }

  return findNode(parts, nextGraph, index + 1);
}

function formatTime(seconds) {
  if (seconds === undefined || seconds === null) {
    return "";
  }

  return `${seconds.toFixed(config.precision)}s`;
}

function getLiveTime(data) {
  if (!data.running || !data.started) {
    return data.time ?? 0;
  }

  return (performance.now() - data.started) / 1000;
}

function resetNodeBadge(node) {
  if (node.nnProfiler) {
    node.nnProfiler.reset = true;
  }
  node.setDirtyCanvas(true, true);
}

function setNodeBadge(id, data) {
  const node = findNode(id);

  if (!node) {
    return;
  }

  node.nnProfiler = data;
  node.setDirtyCanvas(true, true);
}

function hexToRgb(hex) {
  hex = hex.replace("#", "");

  return [
    parseInt(hex.substring(0, 2), 16),
    parseInt(hex.substring(2, 4), 16),
    parseInt(hex.substring(4, 6), 16),
  ];
}

function runtimeColor(seconds) {
  const t = Math.min(seconds / config.slowTime, 1);

  const start = hexToRgb(config.badgeColor);
  const end = hexToRgb(config.badgeColorSlow);

  const r = Math.round(start[0] + (end[0] - start[0]) * t);
  const g = Math.round(start[1] + (end[1] - start[1]) * t);
  const b = Math.round(start[2] + (end[2] - start[2]) * t);

  return `rgb(${r}, ${g}, ${b})`;
}

function drawBadge(node, orig, args) {
  const ctx = args[0];

  const result = orig?.apply(node, args);

  if (!node.nnProfiler) {
    return result;
  }

  if (
    // node.flags?.collapsed ||
    node.constructor.title_mode === LiteGraph.NO_TITLE
  ) {
    return result;
  }

  let text = "";
  let runtime = 0;

  if (node.nnProfiler.type === "time") {
    runtime = getLiveTime(node.nnProfiler);
    text = formatTime(runtime);
  } else if (node.nnProfiler.type === "subgraph") {
    if (
      node.nnProfiler.active !== null &&
      node.nnProfiler.active !== undefined
    ) {
      runtime = node.nnProfiler.total + getLiveTime(node.nnProfiler);
      const child = findNode(node.nnProfiler.child);
      text =
        `${formatTime(runtime)} - ` +
        `${child?.title || node.nnProfiler.class_type}: ` +
        `${formatTime(getLiveTime(node.nnProfiler))}`;
    } else {
      runtime = node.nnProfiler.total;
      text = formatTime(runtime);
    }
  }

  if (!text) {
    return result;
  }

  ctx.save();
  ctx.font = "12px sans-serif";

  const padding = 6;
  const height = 20;
  const y = -LiteGraph.NODE_TITLE_HEIGHT - height;
  const width = ctx.measureText(text).width + padding * 2;
  const isReset = node.nnProfiler.reset;

  ctx.fillStyle = isReset
    ? LiteGraph.NODE_DEFAULT_BGCOLOR
    : runtimeColor(runtime);

  ctx.beginPath();
  ctx.roundRect(0, y, width, height, 5);
  ctx.fill();

  if (node.nnProfiler.running) {
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 2;
    ctx.strokeStyle = config.activeColor;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  ctx.fillStyle = isReset ? LiteGraph.NODE_TEXT_COLOR : "#ececec";

  ctx.fillText(text, padding, y + 14);

  ctx.restore();
}

function installBadge(node) {
  if (node.nnProfilerInstalled) {
    return;
  }

  const original = node.onDrawForeground;

  node.onDrawForeground = function (ctx) {
    return drawBadge(this, original, arguments);
  };

  node.nnProfilerInstalled = true;
}

app.registerExtension({
  name: "NNCustomNodes.Profiler",

  async setup() {
    api.addEventListener(PROFILER_EVENT, ({ detail }) => {
      const data = detail;

      switch (data.type) {
        case "config":
          updateConfig(data);
          break;

        case "node_reset":
          for (const id of data.nodes) {
            const node = findNode(id);

            if (node) {
              resetNodeBadge(node);
            }
          }

          break;

        case "node_start":
          setNodeBadge(data.node, {
            type: "time",
            time: 0,
            running: true,
            started: performance.now(),
            reset: false,
          });

          startTimerLoop();
          break;

        case "node_time":
          setNodeBadge(data.node, {
            type: "time",
            time: data.time,
            running: false,
            started: null,
          });
          break;

        case "subgraph_start":
          setNodeBadge(data.node, {
            type: "subgraph",
            class_type: data.class_type,
            child: data.child,
            total: data.total,
            active: 0,
            running: true,
            started: performance.now(),
            reset: false,
          });

          startTimerLoop();
          break;

        case "subgraph_time":
          setNodeBadge(data.node, {
            type: "subgraph",
            total: data.total,
            active: null,
            time: data.time,
            running: false,
            started: null,
          });

          break;
      }
    });
  },

  async nodeCreated(node) {
    installBadge(node);
  },

  async loadedGraphNode(node) {
    installBadge(node);
  },
});
