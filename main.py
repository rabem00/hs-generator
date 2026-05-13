from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import socket
import webbrowser
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image


TERRAIN_PRESETS = {
    "Island": {
        "octaves": 6,
        "scale": 3.2,
        "persistence": 0.52,
        "lacunarity": 2.0,
        "ridge": 0.2,
        "terrace": 0.0,
        "island": 0.95,
    },
    "Mountainous": {
        "octaves": 7,
        "scale": 4.5,
        "persistence": 0.58,
        "lacunarity": 2.12,
        "ridge": 0.75,
        "terrace": 0.08,
        "island": 0.15,
    },
    "Volcanic": {
        "octaves": 6,
        "scale": 3.8,
        "persistence": 0.54,
        "lacunarity": 2.05,
        "ridge": 0.55,
        "terrace": 0.2,
        "island": 0.55,
    },
    "Normal": {
        "octaves": 6,
        "scale": 3.0,
        "persistence": 0.5,
        "lacunarity": 2.0,
        "ridge": 0.25,
        "terrace": 0.02,
        "island": 0.0,
    },
    "Realistic": {
        "octaves": 8,
        "scale": 3.6,
        "persistence": 0.55,
        "lacunarity": 2.08,
        "ridge": 0.48,
        "terrace": 0.05,
        "island": 0.35,
    },
    "Sea": {
        "octaves": 5,
        "scale": 2.5,
        "persistence": 0.44,
        "lacunarity": 2.0,
        "ridge": 0.08,
        "terrace": 0.0,
        "island": 0.8,
    },
}


SPLAT_COLORS = [
    ("Water", "#24d6d6"),
    ("Rock", "#c72727"),
    ("Sand", "#e5dc23"),
    ("Grass", "#18db28"),
]


@dataclass
class TerrainSettings:
    shape: str
    size: int
    max_height: float
    seed: int
    smoothing_passes: int
    noise_layers: int
    thresholds: list[float]
    colors: list[str]
    warp_strength: float
    warp_frequency: float
    warp_twist: float
    river_frequency: float
    river_depth: float
    river_width: float
    river_spacing: float


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def hashed_random_array(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    value = (
        ix.astype(np.uint64) * np.uint64(374761393)
        + iy.astype(np.uint64) * np.uint64(668265263)
        + np.uint64(seed & 0xFFFFFFFF) * np.uint64(2147483647)
        + (ix.astype(np.uint64) * iy.astype(np.uint64) * np.uint64(1274126177))
    ) & np.uint64(0xFFFFFFFF)
    value = ((value ^ (value >> np.uint64(13))) * np.uint64(1274126177)) & np.uint64(0xFFFFFFFF)
    value = (value ^ (value >> np.uint64(16))) & np.uint64(0xFFFFFFFF)
    return value.astype(np.float32) / np.float32(0xFFFFFFFF)


def value_noise_array(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = (x - x0).astype(np.float32)
    ty = (y - y0).astype(np.float32)
    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)

    a = hashed_random_array(x0, y0, seed)
    b = hashed_random_array(x0 + 1, y0, seed)
    c = hashed_random_array(x0, y0 + 1, seed)
    d = hashed_random_array(x0 + 1, y0 + 1, seed)
    ab = a + (b - a) * tx
    cd = c + (d - c) * tx
    return ab + (cd - ab) * ty


def fbm_array(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    octaves: int,
    persistence: float,
    lacunarity: float,
) -> np.ndarray:
    amplitude = 1.0
    frequency = 1.0
    total = np.zeros_like(x, dtype=np.float32)
    norm = 0.0
    for octave in range(octaves):
        total += value_noise_array(x * frequency, y * frequency, seed + octave * 1013) * amplitude
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / max(norm, 0.000001)


def normalize_map(heightmap: np.ndarray, max_height: float) -> np.ndarray:
    low = float(np.min(heightmap))
    high = float(np.max(heightmap))
    spread = max(0.000001, high - low)
    return ((heightmap - low) / spread * max_height).astype(np.float32)


def smooth_map(heightmap: np.ndarray, passes: int) -> np.ndarray:
    current = heightmap
    for _ in range(passes):
        padded = np.pad(current, 1, mode="edge")
        current = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1] * 2.0
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 10.0
    return current.astype(np.float32)


def apply_terraces_array(values: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return values
    steps = 8.0 + (1.0 - amount) * 20.0
    terraced = np.floor(values * steps) / steps
    return (values + (terraced - values) * amount).astype(np.float32)


def carve_rivers(heightmap: np.ndarray, settings: TerrainSettings) -> np.ndarray:
    if settings.river_depth <= 0.0 or settings.river_frequency <= 0.0:
        return heightmap

    size = heightmap.shape[0]
    spacing = max(0.2, settings.river_spacing)
    width = max(0.001, settings.river_width)
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    nx, ny = np.meshgrid(axis, axis)
    turbulence = fbm_array(
        nx * settings.river_frequency * 7.0,
        ny * settings.river_frequency * 7.0,
        settings.seed + 9001,
        4,
        0.55,
        2.0,
    )
    river_line = np.abs(
        np.sin(
            (nx * 2.2 + turbulence * 0.9 + ny * 0.35)
            * math.pi
            * spacing
            * settings.river_frequency
        )
    )
    channel = np.clip(1.0 - river_line / width, 0.0, 1.0)
    channel = channel * channel * (3.0 - 2.0 * channel)
    return np.maximum(0.0, heightmap - channel * settings.river_depth).astype(np.float32)


def generate_heightmap(settings: TerrainSettings) -> np.ndarray:
    preset = TERRAIN_PRESETS[settings.shape]
    size = settings.size
    center = 0.5
    layers = max(1, settings.noise_layers)
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    nx, ny = np.meshgrid(axis, axis)
    dx = nx - center
    dy = ny - center
    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx) + settings.warp_twist * radius * 2.3
    warped_radius = radius * (1.0 + settings.warp_strength * 0.2)
    wx = center + np.cos(angle) * warped_radius
    wy = center + np.sin(angle) * warped_radius

    if settings.warp_strength > 0.0:
        warp_x = fbm_array(nx * settings.warp_frequency, ny * settings.warp_frequency, settings.seed + 313, 3, 0.5, 2.0)
        warp_y = fbm_array(nx * settings.warp_frequency, ny * settings.warp_frequency, settings.seed + 811, 3, 0.5, 2.0)
        wx += (warp_x - 0.5) * settings.warp_strength * 0.45
        wy += (warp_y - 0.5) * settings.warp_strength * 0.45

    heightmap = np.zeros((size, size), dtype=np.float32)
    for layer in range(layers):
        offset = layer * 0.173
        heightmap += fbm_array(
            (wx + offset) * preset["scale"],
            (wy - offset) * preset["scale"],
            settings.seed + layer * 1777,
            int(preset["octaves"]),
            preset["persistence"],
            preset["lacunarity"],
        )
    heightmap /= layers

    if preset["ridge"] > 0.0:
        ridge = 1.0 - np.abs(heightmap * 2.0 - 1.0)
        heightmap = heightmap + (ridge - heightmap) * preset["ridge"]

    if preset["island"] > 0.0:
        falloff = np.clip(1.0 - radius * 1.55, 0.0, 1.0)
        falloff = falloff * falloff * (3.0 - 2.0 * falloff)
        heightmap = heightmap + (heightmap * falloff - heightmap) * preset["island"]

    if settings.shape == "Volcanic":
        crater = np.clip(1.0 - np.abs(radius - 0.18) / 0.09, 0.0, 1.0)
        heightmap += crater * 0.28
        heightmap -= np.clip(1.0 - radius / 0.13, 0.0, 1.0) * 0.22

    if settings.shape == "Sea":
        heightmap *= 0.42

    heightmap = apply_terraces_array(heightmap, preset["terrace"])
    heightmap = normalize_map(heightmap, settings.max_height)
    heightmap = carve_rivers(heightmap, settings)
    if settings.smoothing_passes:
        heightmap = smooth_map(heightmap, settings.smoothing_passes)
    return normalize_map(heightmap, settings.max_height)


def make_splatmap(heightmap: np.ndarray, settings: TerrainSettings) -> np.ndarray:
    max_height = max(0.000001, settings.max_height)
    normalized = np.clip(heightmap / max_height, 0.0, 1.0)
    thresholds = np.array(settings.thresholds, dtype=np.float32)
    order = np.argsort(thresholds)
    sorted_thresholds = thresholds[order]
    sorted_colors = np.array([hex_to_rgb(settings.colors[index]) for index in order], dtype=np.uint8)
    indexes = np.searchsorted(sorted_thresholds, normalized, side="left")
    indexes = np.clip(indexes, 0, len(sorted_colors) - 1)
    return sorted_colors[indexes]


def png_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def data_url_png(array: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(array)).decode("ascii")


def settings_from_payload(payload: dict[str, Any]) -> TerrainSettings:
    colors = payload.get("colors") or [color for _, color in SPLAT_COLORS]
    thresholds = payload.get("thresholds") or [0.22, 0.42, 0.68, 1.0]
    return TerrainSettings(
        shape=str(payload.get("shape", "Realistic")),
        size=max(64, min(2048, int(payload.get("size", 512)))),
        max_height=max(0.01, float(payload.get("maxHeight", 1.0))),
        seed=int(payload.get("seed", 1337)),
        smoothing_passes=max(0, min(12, int(payload.get("smoothingPasses", 1)))),
        noise_layers=max(1, min(24, int(payload.get("noiseLayers", 5)))),
        thresholds=[max(0.0, min(1.0, float(value))) for value in thresholds[:4]],
        colors=[str(value) for value in colors[:4]],
        warp_strength=max(0.0, min(1.5, float(payload.get("warpStrength", 0.28)))),
        warp_frequency=max(0.1, min(12.0, float(payload.get("warpFrequency", 3.0)))),
        warp_twist=max(0.0, min(2.0, float(payload.get("warpTwist", 0.18)))),
        river_frequency=max(0.0, min(8.0, float(payload.get("riverFrequency", 1.8)))),
        river_depth=max(0.0, min(1.5, float(payload.get("riverDepth", 0.12)))),
        river_width=max(0.001, min(0.8, float(payload.get("riverWidth", 0.13)))),
        river_spacing=max(0.1, min(8.0, float(payload.get("riverSpacing", 2.4)))),
    )


def generate_payload(settings: TerrainSettings) -> dict[str, Any]:
    heightmap = generate_heightmap(settings)
    splatmap = make_splatmap(heightmap, settings)
    height_pixels = np.clip(heightmap / max(0.000001, settings.max_height) * 255, 0, 255).astype(np.uint8)
    return {
        "size": settings.size,
        "maxHeight": settings.max_height,
        "heightmap": data_url_png(height_pixels),
        "splatmap": data_url_png(splatmap.astype(np.uint8)),
    }


def export_zip_bytes(settings: TerrainSettings) -> bytes:
    heightmap = generate_heightmap(settings)
    splatmap = make_splatmap(heightmap, settings)
    height_pixels = np.clip(heightmap / max(0.000001, settings.max_height) * 255, 0, 255).astype(np.uint8)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("terrain_heightmap.png", png_bytes(height_pixels))
        zip_file.writestr("terrain_splatmap.png", png_bytes(splatmap.astype(np.uint8)))
    return archive.getvalue()


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Heightmap / Splatmap Generator</title>
  <style>
    :root {
      color-scheme: dark;
      --panel: #23262b;
      --panel-2: #181a1f;
      --line: #3a3f46;
      --text: #eff2f6;
      --muted: #aab1bc;
      --blue: #3d7bea;
      --green: #32b642;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      overflow: hidden;
      font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111318;
      color: var(--text);
    }
    #app {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 430px;
      height: 100vh;
    }
    #viewport {
      position: relative;
      min-width: 0;
      background: #89aeb8;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
    #status {
      position: absolute;
      left: 14px;
      bottom: 12px;
      padding: 6px 9px;
      color: #f5f7fa;
      background: rgb(15 18 24 / 72%);
      border: 1px solid rgb(255 255 255 / 14%);
    }
    aside {
      overflow: auto;
      background: var(--panel);
      border-left: 1px solid #111318;
      padding: 10px;
    }
    label, .label {
      display: block;
      color: var(--muted);
      margin: 7px 0 3px;
    }
    .shape-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 4px;
      margin-bottom: 8px;
    }
    .shape-grid button,
    .actions button,
    select,
    input[type="text"] {
      min-height: 28px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 4px;
      padding: 4px 7px;
    }
    .shape-grid button.active {
      border-color: var(--blue);
      background: var(--blue);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 52px;
      align-items: center;
      gap: 8px;
      margin-bottom: 5px;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--blue);
    }
    .swatch-row {
      display: grid;
      grid-template-columns: 58px 1fr 42px 44px;
      align-items: center;
      gap: 8px;
      margin: 4px 0;
    }
    input[type="color"] {
      width: 100%;
      height: 26px;
      padding: 0;
      border: 0;
      background: transparent;
    }
    #gradient {
      position: relative;
      height: 24px;
      border: 1px solid var(--line);
      margin: 3px 0 7px;
    }
    #gradient i {
      position: absolute;
      top: 0;
      width: 2px;
      height: 100%;
      background: #fff;
      transform: translateX(-1px);
      pointer-events: none;
    }
    .section-title {
      margin-top: 10px;
      color: #dce2ea;
    }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0 9px;
    }
    .actions button {
      border-color: transparent;
      cursor: pointer;
    }
    #generate { background: var(--blue); }
    #download { background: var(--green); }
    .previews {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .preview {
      background: #050505;
      aspect-ratio: 1;
      overflow: hidden;
    }
    .preview img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: pixelated;
    }
    @media (max-width: 900px) {
      #app { grid-template-columns: 1fr; grid-template-rows: 58vh 42vh; }
      aside { border-left: 0; border-top: 1px solid #111318; }
    }
  </style>
</head>
<body>
  <div id="app">
    <main id="viewport">
      <canvas id="gl"></canvas>
      <div id="status">Starting WebGL shader preview...</div>
    </main>
    <aside>
      <div class="label">Terrain Shape</div>
      <div class="shape-grid" id="shapeGrid"></div>

      <label for="size">Terrain Dimensions</label>
      <select id="size">
        <option value="256">256 x 256</option>
        <option value="512" selected>512 x 512</option>
        <option value="1024">1024 x 1024</option>
        <option value="2048">2048 x 2048</option>
      </select>

      <div id="controls"></div>

      <div class="section-title">Splatmap Colors/Threshold</div>
      <div id="gradient"></div>
      <div id="splatRows"></div>

      <div class="section-title">Domain Warping</div>
      <div id="warpControls"></div>

      <div class="section-title">River Carving</div>
      <div id="riverControls"></div>

      <div class="actions">
        <button id="generate">Generate</button>
        <button id="download">Export PNGs</button>
      </div>

      <div class="previews">
        <div class="preview"><img id="heightPreview" alt="Heightmap preview" /></div>
        <div class="preview"><img id="splatPreview" alt="Splatmap preview" /></div>
      </div>
    </aside>
  </div>

  <script type="module">
    const shapes = ["Island", "Mountainous", "Volcanic", "Normal", "Realistic", "Sea"];
    const splats = [
      ["Water", "#24d6d6"],
      ["Rock", "#c72727"],
      ["Sand", "#e5dc23"],
      ["Grass", "#18db28"],
    ];
    const state = {
      shape: "Realistic",
      size: 512,
      maxHeight: 1,
      seed: Math.floor(Math.random() * 2147483647),
      smoothingPasses: 1,
      noiseLayers: 5,
      thresholds: [0.22, 0.42, 0.68, 1],
      colors: splats.map((s) => s[1]),
      warpStrength: 0.28,
      warpFrequency: 3,
      warpTwist: 0.18,
      riverFrequency: 1.8,
      riverDepth: 0.12,
      riverWidth: 0.13,
      riverSpacing: 2.4,
    };

    const $ = (id) => document.getElementById(id);
    const status = $("status");
    const shapeGrid = $("shapeGrid");
    const controls = $("controls");
    const warpControls = $("warpControls");
    const riverControls = $("riverControls");
    const splatRows = $("splatRows");
    const gradient = $("gradient");
    const heightPreview = $("heightPreview");
    const splatPreview = $("splatPreview");

    let latestHeightUrl = "";
    let latestSplatUrl = "";
    let busy = false;
    let debounce = null;

    function control(container, key, label, min, max, step) {
      const wrap = document.createElement("div");
      wrap.className = "row";
      const title = document.createElement("label");
      title.textContent = label;
      title.htmlFor = key;
      title.style.gridColumn = "1 / 3";
      const input = document.createElement("input");
      input.type = key === "seed" ? "text" : "range";
      input.id = key;
      input.value = state[key];
      const value = document.createElement("span");
      value.textContent = String(state[key]);
      if (input.type === "range") {
        input.min = min;
        input.max = max;
        input.step = step;
      }
      input.addEventListener("input", () => {
        state[key] = input.type === "range" ? Number(input.value) : input.value;
        if (key === "seed") state[key] = Number.parseInt(input.value || "0", 10) || 0;
        value.textContent = input.type === "range" && step < 1 ? Number(state[key]).toFixed(2) : String(state[key]);
        queueGenerate();
      });
      wrap.append(title, input, value);
      container.appendChild(wrap);
    }

    shapes.forEach((shape) => {
      const button = document.createElement("button");
      button.textContent = shape;
      button.addEventListener("click", () => {
        state.shape = shape;
        document.querySelectorAll(".shape-grid button").forEach((item) => item.classList.toggle("active", item.textContent === shape));
        queueGenerate();
      });
      if (shape === state.shape) button.classList.add("active");
      shapeGrid.appendChild(button);
    });

    $("size").addEventListener("change", (event) => {
      state.size = Number(event.target.value);
      if (state.size > 512) {
        status.textContent = "Large map selected. Press Generate when ready.";
        return;
      }
      queueGenerate();
    });

    control(controls, "maxHeight", "Max Height (relative)", 0.1, 2.5, 0.01);
    control(controls, "seed", "Terrain Seed", 0, 0, 1);
    control(controls, "smoothingPasses", "Smoothing Passes", 0, 8, 1);
    control(controls, "noiseLayers", "Noise Layer Stacks", 1, 16, 1);
    control(warpControls, "warpStrength", "Warp Strength", 0, 1, 0.01);
    control(warpControls, "warpFrequency", "Warp Frequency", 0.5, 8, 0.1);
    control(warpControls, "warpTwist", "Twist / Bend", 0, 1.2, 0.01);
    control(riverControls, "riverFrequency", "River Frequency", 0, 5, 0.1);
    control(riverControls, "riverDepth", "River Depth", 0, 0.8, 0.01);
    control(riverControls, "riverWidth", "River Width", 0.02, 0.45, 0.01);
    control(riverControls, "riverSpacing", "River Spacing", 0.5, 6, 0.1);

    function redrawGradient() {
      const stops = state.thresholds.map((value, index) => `${state.colors[index]} ${Math.round(value * 100)}%`);
      gradient.style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
      gradient.innerHTML = "";
      state.thresholds.forEach((value) => {
        const marker = document.createElement("i");
        marker.style.left = `${value * 100}%`;
        gradient.appendChild(marker);
      });
    }

    splats.forEach(([name], index) => {
      const row = document.createElement("div");
      row.className = "swatch-row";
      const color = document.createElement("input");
      color.type = "color";
      color.value = state.colors[index];
      const threshold = document.createElement("input");
      threshold.type = "range";
      threshold.min = 0;
      threshold.max = 1;
      threshold.step = 0.01;
      threshold.value = state.thresholds[index];
      const value = document.createElement("span");
      value.textContent = state.thresholds[index].toFixed(2);
      color.addEventListener("input", () => {
        state.colors[index] = color.value;
        redrawGradient();
        renderer.setMaterialColors(state.colors);
        queueGenerate();
      });
      threshold.addEventListener("input", () => {
        state.thresholds[index] = Number(threshold.value);
        value.textContent = state.thresholds[index].toFixed(2);
        redrawGradient();
        queueGenerate();
      });
      row.append(name, threshold, value, color);
      splatRows.appendChild(row);
    });
    redrawGradient();

    function queueGenerate() {
      if (state.size > 512) {
        status.textContent = "Large map selected. Press Generate when ready.";
        return;
      }
      clearTimeout(debounce);
      debounce = setTimeout(generate, 280);
    }

    async function generate() {
      if (busy) return;
      busy = true;
      status.textContent = `Generating ${state.size} x ${state.size}...`;
      try {
        const response = await fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(state),
        });
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        latestHeightUrl = payload.heightmap;
        latestSplatUrl = payload.splatmap;
        heightPreview.src = latestHeightUrl;
        splatPreview.src = latestSplatUrl;
        await renderer.setTextures(latestHeightUrl, latestSplatUrl, payload.maxHeight);
        status.textContent = `Generated ${payload.size} x ${payload.size}. Preview is WebGL shader rendered.`;
      } catch (error) {
        status.textContent = `Generation failed: ${error.message}`;
      } finally {
        busy = false;
      }
    }

    $("generate").addEventListener("click", generate);
    $("download").addEventListener("click", async () => {
      status.textContent = "Preparing heightmap and splatmap export...";
      try {
        const response = await fetch("/export", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(state),
        });
        if (!response.ok) throw new Error(await response.text());
        const blob = await response.blob();
        download(URL.createObjectURL(blob), "terrain_maps.zip", true);
        status.textContent = "Exported terrain_maps.zip with heightmap and splatmap PNGs.";
      } catch (error) {
        status.textContent = `Export failed: ${error.message}`;
      }
    });

    function download(url, name, revoke = false) {
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      if (revoke) setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    class Renderer {
      constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext("webgl2", { antialias: true });
        if (!this.gl) throw new Error("WebGL2 is not available in this browser.");
        this.rotation = 0;
        this.heightScale = 0.45;
        this.program = this.createProgram(vertexShader, fragmentShader);
        this.attribs = {
          position: this.gl.getAttribLocation(this.program, "aPosition"),
          uv: this.gl.getAttribLocation(this.program, "aUv"),
        };
        this.uniforms = {
          projection: this.gl.getUniformLocation(this.program, "uProjection"),
          view: this.gl.getUniformLocation(this.program, "uView"),
          model: this.gl.getUniformLocation(this.program, "uModel"),
          heightmap: this.gl.getUniformLocation(this.program, "uHeightmap"),
          splatmap: this.gl.getUniformLocation(this.program, "uSplatmap"),
          heightScale: this.gl.getUniformLocation(this.program, "uHeightScale"),
          texelSize: this.gl.getUniformLocation(this.program, "uTexelSize"),
          materialColors: this.gl.getUniformLocation(this.program, "uMaterialColors[0]"),
        };
        this.buildMesh(220);
        this.heightTexture = this.makeTexture();
        this.splatTexture = this.makeTexture();
        this.setMaterialColors(state.colors);
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.disable(this.gl.CULL_FACE);
        requestAnimationFrame((time) => this.frame(time));
      }

      buildMesh(resolution) {
        const vertices = [];
        const indices = [];
        for (let y = 0; y <= resolution; y++) {
          for (let x = 0; x <= resolution; x++) {
            const u = x / resolution;
            const v = y / resolution;
            vertices.push((u - 0.5) * 2, (v - 0.5) * 2, u, v);
          }
        }
        for (let y = 0; y < resolution; y++) {
          for (let x = 0; x < resolution; x++) {
            const i = y * (resolution + 1) + x;
            indices.push(i, i + 1, i + resolution + 1, i + 1, i + resolution + 2, i + resolution + 1);
          }
        }
        const gl = this.gl;
        this.indexCount = indices.length;
        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);
        const vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(this.attribs.position);
        gl.vertexAttribPointer(this.attribs.position, 2, gl.FLOAT, false, 16, 0);
        gl.enableVertexAttribArray(this.attribs.uv);
        gl.vertexAttribPointer(this.attribs.uv, 2, gl.FLOAT, false, 16, 8);
        const indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(indices), gl.STATIC_DRAW);
        gl.bindVertexArray(null);
      }

      createProgram(vs, fs) {
        const gl = this.gl;
        const vertex = this.compile(gl.VERTEX_SHADER, vs);
        const fragment = this.compile(gl.FRAGMENT_SHADER, fs);
        const program = gl.createProgram();
        gl.attachShader(program, vertex);
        gl.attachShader(program, fragment);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
          throw new Error(gl.getProgramInfoLog(program));
        }
        return program;
      }

      compile(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
          throw new Error(gl.getShaderInfoLog(shader));
        }
        return shader;
      }

      makeTexture() {
        const gl = this.gl;
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
        return texture;
      }

      imageFromUrl(url) {
        return new Promise((resolve, reject) => {
          const image = new Image();
          image.onload = () => resolve(image);
          image.onerror = reject;
          image.src = url;
        });
      }

      async setTextures(heightUrl, splatUrl, maxHeight) {
        const [heightImage, splatImage] = await Promise.all([this.imageFromUrl(heightUrl), this.imageFromUrl(splatUrl)]);
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.heightTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, heightImage);
        gl.bindTexture(gl.TEXTURE_2D, this.splatTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, splatImage);
        this.texelSize = [1 / heightImage.width, 1 / heightImage.height];
        this.heightScale = 0.28 + maxHeight * 0.28;
      }

      setMaterialColors(colors) {
        const values = colors.flatMap((color) => {
          const n = Number.parseInt(color.slice(1), 16);
          return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
        });
        this.materialColors = new Float32Array(values);
      }

      resize() {
        const width = Math.max(1, Math.floor(this.canvas.clientWidth * devicePixelRatio));
        const height = Math.max(1, Math.floor(this.canvas.clientHeight * devicePixelRatio));
        if (this.canvas.width !== width || this.canvas.height !== height) {
          this.canvas.width = width;
          this.canvas.height = height;
          this.gl.viewport(0, 0, width, height);
        }
      }

      frame() {
        this.resize();
        this.rotation += 0.0045;
        const gl = this.gl;
        gl.clearColor(0.50, 0.66, 0.70, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(this.program);
        const aspect = this.canvas.width / this.canvas.height;
        gl.uniformMatrix4fv(this.uniforms.projection, false, perspective(42 * Math.PI / 180, aspect, 0.1, 20));
        gl.uniformMatrix4fv(this.uniforms.view, false, lookAt([0, 1.65, 3.15], [0, 0.05, 0], [0, 1, 0]));
        gl.uniformMatrix4fv(this.uniforms.model, false, multiply(rotateY(this.rotation), rotateX(-0.55)));
        gl.uniform1f(this.uniforms.heightScale, this.heightScale);
        gl.uniform2fv(this.uniforms.texelSize, this.texelSize || [1 / 512, 1 / 512]);
        gl.uniform3fv(this.uniforms.materialColors, this.materialColors);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.heightTexture);
        gl.uniform1i(this.uniforms.heightmap, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.splatTexture);
        gl.uniform1i(this.uniforms.splatmap, 1);
        gl.bindVertexArray(this.vao);
        gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);
        requestAnimationFrame((time) => this.frame(time));
      }
    }

    const vertexShader = `#version 300 es
      precision highp float;
      in vec2 aPosition;
      in vec2 aUv;
      uniform sampler2D uHeightmap;
      uniform mat4 uProjection;
      uniform mat4 uView;
      uniform mat4 uModel;
      uniform float uHeightScale;
      uniform vec2 uTexelSize;
      out vec2 vUv;
      out vec3 vNormal;
      out float vHeight;
      void main() {
        float h = texture(uHeightmap, aUv).r;
        float hx = texture(uHeightmap, aUv + vec2(uTexelSize.x, 0.0)).r - texture(uHeightmap, aUv - vec2(uTexelSize.x, 0.0)).r;
        float hy = texture(uHeightmap, aUv + vec2(0.0, uTexelSize.y)).r - texture(uHeightmap, aUv - vec2(0.0, uTexelSize.y)).r;
        vec3 normal = normalize(vec3(-hx * uHeightScale * 8.0, 0.16, -hy * uHeightScale * 8.0));
        vec3 position = vec3(aPosition.x, h * uHeightScale, aPosition.y);
        vUv = aUv;
        vHeight = h;
        vNormal = mat3(uModel) * normal;
        gl_Position = uProjection * uView * uModel * vec4(position, 1.0);
      }
    `;

    const fragmentShader = `#version 300 es
      precision highp float;
      uniform sampler2D uSplatmap;
      uniform vec3 uMaterialColors[4];
      in vec2 vUv;
      in vec3 vNormal;
      in float vHeight;
      out vec4 outColor;
      float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }
      float colorMask(vec3 splat, vec3 target) {
        return 1.0 - smoothstep(0.08, 0.45, distance(splat, target));
      }
      void main() {
        vec3 splat = texture(uSplatmap, vUv).rgb;
        float water = colorMask(splat, uMaterialColors[0]);
        float rock = colorMask(splat, uMaterialColors[1]);
        float sand = colorMask(splat, uMaterialColors[2]);
        float grass = colorMask(splat, uMaterialColors[3]);
        float total = max(0.0001, water + rock + sand + grass);
        vec2 tiled = vUv * 52.0;
        float grain = hash(floor(tiled));
        vec3 waterColor = mix(vec3(0.02, 0.38, 0.48), vec3(0.06, 0.78, 0.82), 0.45 + grain * 0.2);
        vec3 rockColor = mix(vec3(0.30, 0.29, 0.27), vec3(0.58, 0.56, 0.50), grain);
        vec3 sandColor = mix(vec3(0.68, 0.60, 0.34), vec3(0.88, 0.82, 0.50), grain);
        vec3 grassColor = mix(vec3(0.15, 0.36, 0.13), vec3(0.45, 0.61, 0.22), grain);
        vec3 base = (waterColor * water + rockColor * rock + sandColor * sand + grassColor * grass) / total;
        vec3 normal = normalize(vNormal);
        vec3 lightDir = normalize(vec3(0.35, 0.85, 0.42));
        float diffuse = max(dot(normal, lightDir), 0.0);
        float slope = 1.0 - clamp(normal.y, 0.0, 1.0);
        base = mix(base, rockColor, smoothstep(0.42, 0.82, slope) * 0.55);
        base += vec3(0.08, 0.09, 0.10) * smoothstep(0.72, 1.0, vHeight);
        outColor = vec4(base * (0.38 + diffuse * 0.78), 1.0);
      }
    `;

    function perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2);
      const nf = 1 / (near - far);
      return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
      ]);
    }

    function lookAt(eye, center, up) {
      const z = normalize([eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]]);
      const x = normalize(cross(up, z));
      const y = cross(z, x);
      return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
      ]);
    }

    function rotateX(angle) {
      const c = Math.cos(angle), s = Math.sin(angle);
      return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
    }

    function rotateY(angle) {
      const c = Math.cos(angle), s = Math.sin(angle);
      return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
    }

    function multiply(a, b) {
      const out = new Float32Array(16);
      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
          out[col * 4 + row] =
            a[0 * 4 + row] * b[col * 4 + 0] +
            a[1 * 4 + row] * b[col * 4 + 1] +
            a[2 * 4 + row] * b[col * 4 + 2] +
            a[3 * 4 + row] * b[col * 4 + 3];
        }
      }
      return out;
    }

    function normalize(v) {
      const length = Math.hypot(...v) || 1;
      return v.map((value) => value / length);
    }

    function cross(a, b) {
      return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    }

    function dot(a, b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    const renderer = new Renderer($("gl"));
    generate();
  </script>
</body>
</html>
"""


class TerrainRequestHandler(BaseHTTPRequestHandler):
    server_version = "HSTerrain/0.2"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path != "/":
            self.send_error(404)
            return
        self.send_bytes(APP_HTML.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/generate", "/export"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            settings = settings_from_payload(payload)
            if path == "/export":
                archive = export_zip_bytes(settings)
                self.send_bytes(archive, "application/zip", "attachment; filename=terrain_maps.zip")
                return
            response = generate_payload(settings)
        except Exception as exc:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(str(exc).encode("utf-8"))
            return
        self.send_bytes(json.dumps(response).encode("utf-8"), "application/json")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_bytes(self, body: bytes, content_type: str, disposition: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()
        self.wfile.write(body)


def available_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Heightmap / Splatmap generator with WebGL shader preview.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser.")
    args = parser.parse_args()

    port = available_port(args.port)
    server = ThreadingHTTPServer((args.host, port), TerrainRequestHandler)
    url = f"http://{args.host}:{port}/"
    print(f"Heightmap / Splatmap Generator running at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
