from __future__ import annotations

import math
import random
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk


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


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def lerp(left: float, right: float, t: float) -> float:
    return left + (right - left) * t


def hashed_random(ix: int, iy: int, seed: int) -> float:
    value = (
        ix * 374761393
        + iy * 668265263
        + seed * 2147483647
        + (ix * iy * 1274126177)
    ) & 0xFFFFFFFF
    value = (value ^ (value >> 13)) * 1274126177 & 0xFFFFFFFF
    value = (value ^ (value >> 16)) & 0xFFFFFFFF
    return value / 0xFFFFFFFF


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


def value_noise(x: float, y: float, seed: int) -> float:
    x0 = math.floor(x)
    y0 = math.floor(y)
    tx = smoothstep(x - x0)
    ty = smoothstep(y - y0)

    a = hashed_random(x0, y0, seed)
    b = hashed_random(x0 + 1, y0, seed)
    c = hashed_random(x0, y0 + 1, seed)
    d = hashed_random(x0 + 1, y0 + 1, seed)
    return lerp(lerp(a, b, tx), lerp(c, d, tx), ty)


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
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty


def fbm(
    x: float,
    y: float,
    seed: int,
    octaves: int,
    persistence: float,
    lacunarity: float,
) -> float:
    amplitude = 1.0
    frequency = 1.0
    total = 0.0
    norm = 0.0
    for octave in range(octaves):
        total += value_noise(x * frequency, y * frequency, seed + octave * 1013) * amplitude
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / norm if norm else 0.0


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
        next_map = (
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
        current = next_map
    return current.astype(np.float32)


def apply_terraces(value: float, amount: float) -> float:
    if amount <= 0.0:
        return value
    steps = 8.0 + (1.0 - amount) * 20.0
    terraced = math.floor(value * steps) / steps
    return lerp(value, terraced, amount)


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
        warp_x = fbm_array(
            nx * settings.warp_frequency,
            ny * settings.warp_frequency,
            settings.seed + 313,
            3,
            0.5,
            2.0,
        )
        warp_y = fbm_array(
            nx * settings.warp_frequency,
            ny * settings.warp_frequency,
            settings.seed + 811,
            3,
            0.5,
            2.0,
        )
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


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def color_for_height(value: float, thresholds: list[float], colors: list[str]) -> tuple[int, int, int]:
    sorted_stops = sorted(zip(thresholds, colors), key=lambda item: item[0])
    chosen = sorted_stops[-1][1]
    for threshold, color in sorted_stops:
        if value <= threshold:
            chosen = color
            break
    return hex_to_rgb(chosen)


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


class TerrainGeneratorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Heightmap / Splatmap Generator")
        self.geometry("1380x860")
        self.minsize(1120, 740)

        self.heightmap: np.ndarray | None = None
        self.splatmap: np.ndarray | None = None
        self.preview_points: list[tuple[float, float, float, tuple[int, int, int]]] = []
        self.height_preview_image: ImageTk.PhotoImage | None = None
        self.splat_preview_image: ImageTk.PhotoImage | None = None
        self.rotation = 0.0
        self.last_generated = 0.0

        self.shape_var = tk.StringVar(value="Realistic")
        self.size_var = tk.IntVar(value=512)
        self.seed_var = tk.StringVar(value=str(random.randint(1, 2_147_483_647)))
        self.max_height_var = tk.DoubleVar(value=1.0)
        self.smoothing_var = tk.IntVar(value=1)
        self.layers_var = tk.IntVar(value=5)
        self.warp_strength_var = tk.DoubleVar(value=0.28)
        self.warp_frequency_var = tk.DoubleVar(value=3.0)
        self.warp_twist_var = tk.DoubleVar(value=0.18)
        self.river_frequency_var = tk.DoubleVar(value=1.8)
        self.river_depth_var = tk.DoubleVar(value=0.12)
        self.river_width_var = tk.DoubleVar(value=0.13)
        self.river_spacing_var = tk.DoubleVar(value=2.4)
        self.threshold_vars = [
            tk.DoubleVar(value=0.22),
            tk.DoubleVar(value=0.42),
            tk.DoubleVar(value=0.68),
            tk.DoubleVar(value=1.0),
        ]
        self.color_vars = [tk.StringVar(value=color) for _, color in SPLAT_COLORS]

        self._build_ui()
        self.after(150, self.generate)
        self.after(33, self.animate)

    def _build_ui(self) -> None:
        self.configure(bg="#17191d")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg="#89aeb8", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(self, padding=10)
        side.grid(row=0, column=1, sticky="ns")
        side.columnconfigure(0, weight=1)
        self._configure_style()

        ttk.Label(side, text="Terrain Shape").grid(row=0, column=0, sticky="w")
        shape_frame = ttk.Frame(side)
        shape_frame.grid(row=1, column=0, sticky="ew", pady=(2, 10))
        for index, shape in enumerate(TERRAIN_PRESETS):
            ttk.Radiobutton(
                shape_frame,
                text=shape,
                variable=self.shape_var,
                value=shape,
                command=self.queue_generate,
            ).grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 8))

        row = 2
        row = self._add_dropdown(side, row, "Terrain Dimensions", self.size_var, [256, 512, 1024, 2048])
        row = self._add_slider(side, row, "Max Height (relative)", self.max_height_var, 0.1, 2.5, 0.01)
        row = self._add_seed(side, row)
        row = self._add_slider(side, row, "Smoothing Passes", self.smoothing_var, 0, 8, 1)
        row = self._add_slider(side, row, "Noise Layer Stacks", self.layers_var, 1, 16, 1)

        ttk.Label(side, text="Splatmap Colors/Threshold").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        self.gradient_canvas = tk.Canvas(side, height=24, width=430, highlightthickness=1, highlightbackground="#30343a")
        self.gradient_canvas.grid(row=row, column=0, sticky="ew", pady=(2, 6))
        self.gradient_canvas.bind("<Button-1>", self._set_nearest_threshold)
        self.gradient_canvas.bind("<B1-Motion>", self._set_nearest_threshold)
        row += 1
        for index, (name, _) in enumerate(SPLAT_COLORS):
            item = ttk.Frame(side)
            item.grid(row=row, column=0, sticky="ew", pady=2)
            item.columnconfigure(1, weight=1)
            ttk.Button(item, text=name, command=lambda i=index: self.choose_color(i)).grid(row=0, column=0, sticky="w")
            ttk.Scale(
                item,
                from_=0.0,
                to=1.0,
                variable=self.threshold_vars[index],
                command=lambda _value: self._threshold_changed(),
            ).grid(row=0, column=1, sticky="ew", padx=8)
            ttk.Label(item, textvariable=self.threshold_vars[index], width=5).grid(row=0, column=2)
            row += 1

        row = self._add_separator(side, row, "Domain Warping")
        row = self._add_slider(side, row, "Warp Strength", self.warp_strength_var, 0.0, 1.0, 0.01)
        row = self._add_slider(side, row, "Warp Frequency", self.warp_frequency_var, 0.5, 8.0, 0.1)
        row = self._add_slider(side, row, "Twist / Bend", self.warp_twist_var, 0.0, 1.2, 0.01)

        row = self._add_separator(side, row, "River Carving")
        row = self._add_slider(side, row, "River Frequency", self.river_frequency_var, 0.0, 5.0, 0.1)
        row = self._add_slider(side, row, "River Depth", self.river_depth_var, 0.0, 0.8, 0.01)
        row = self._add_slider(side, row, "River Width", self.river_width_var, 0.02, 0.45, 0.01)
        row = self._add_slider(side, row, "River Spacing", self.river_spacing_var, 0.5, 6.0, 0.1)

        buttons = ttk.Frame(side)
        buttons.grid(row=row, column=0, sticky="ew", pady=(10, 8))
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(buttons, text="Generate", command=self.generate).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Export PNGs", command=self.export_pngs).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        row += 1

        preview_frame = ttk.Frame(side)
        preview_frame.grid(row=row, column=0, sticky="ew")
        preview_frame.columnconfigure((0, 1), weight=1)
        self.height_canvas = tk.Canvas(preview_frame, width=210, height=210, bg="#050505", highlightthickness=0)
        self.splat_canvas = tk.Canvas(preview_frame, width=210, height=210, bg="#050505", highlightthickness=0)
        self.height_canvas.grid(row=0, column=0, padx=(0, 5))
        self.splat_canvas.grid(row=0, column=1, padx=(5, 0))

        row += 1
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(side, textvariable=self.status_var).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        self._redraw_gradient()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#25282d", foreground="#e8eaee", fieldbackground="#15171b")
        style.configure("TFrame", background="#25282d")
        style.configure("TLabel", background="#25282d", foreground="#e8eaee")
        style.configure("TButton", background="#3b73d9", foreground="#ffffff", padding=5)
        style.configure("TRadiobutton", background="#25282d", foreground="#e8eaee")
        style.configure("TCombobox", fieldbackground="#15171b", foreground="#e8eaee")
        style.configure("Horizontal.TScale", background="#25282d")

    def _add_separator(self, parent: ttk.Frame, row: int, label: str) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(10, 0))
        return row + 1

    def _add_dropdown(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.IntVar,
        values: list[int],
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        row += 1
        dropdown = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        dropdown.grid(row=row, column=0, sticky="ew", pady=(2, 8))
        dropdown.bind("<<ComboboxSelected>>", lambda _event: self.queue_generate())
        return row + 1

    def _add_seed(self, parent: ttk.Frame, row: int) -> int:
        ttk.Label(parent, text="Terrain Seed").grid(row=row, column=0, sticky="w")
        row += 1
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=(2, 8))
        frame.columnconfigure(0, weight=1)
        ttk.Entry(frame, textvariable=self.seed_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(frame, text="Random", command=self.randomize_seed).grid(row=0, column=1, padx=(6, 0))
        return row + 1

    def _add_slider(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.Variable,
        minimum: float,
        maximum: float,
        resolution: float,
    ) -> int:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        value_label = ttk.Label(frame, width=5)
        value_label.grid(row=0, column=1, sticky="e")

        def update_label(*_args: object) -> None:
            value = variable.get()
            if resolution >= 1:
                value_label.configure(text=str(int(float(value))))
            else:
                value_label.configure(text=f"{float(value):.2f}")

        variable.trace_add("write", update_label)
        update_label()
        slider = ttk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            variable=variable,
            command=lambda value: self._slider_changed(variable, value, resolution),
        )
        slider.grid(row=1, column=0, columnspan=2, sticky="ew")
        return row + 1

    def _slider_changed(self, variable: tk.Variable, value: str, resolution: float) -> None:
        number = float(value)
        if resolution >= 1:
            variable.set(int(round(number)))
        else:
            variable.set(round(number / resolution) * resolution)
        self.queue_generate()

    def _threshold_changed(self) -> None:
        self._redraw_gradient()
        self.queue_generate()

    def _set_nearest_threshold(self, event: tk.Event) -> None:
        width = max(1, self.gradient_canvas.winfo_width())
        target = max(0.0, min(1.0, event.x / width))
        nearest = min(
            range(len(self.threshold_vars)),
            key=lambda index: abs(self.threshold_vars[index].get() - target),
        )
        self.threshold_vars[nearest].set(round(target, 2))
        self._threshold_changed()

    def choose_color(self, index: int) -> None:
        color = colorchooser.askcolor(self.color_vars[index].get(), parent=self)[1]
        if color:
            self.color_vars[index].set(color)
            self._redraw_gradient()
            self.queue_generate()

    def randomize_seed(self) -> None:
        self.seed_var.set(str(random.randint(1, 2_147_483_647)))
        self.queue_generate()

    def queue_generate(self) -> None:
        if int(self.size_var.get()) > 512:
            self.status_var.set("Large map selected. Press Generate when ready.")
            self._redraw_gradient()
            return
        if time.time() - self.last_generated > 0.35:
            self.after(80, self.generate)

    def settings(self) -> TerrainSettings:
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            seed = abs(hash(self.seed_var.get())) % 2_147_483_647
        return TerrainSettings(
            shape=self.shape_var.get(),
            size=int(self.size_var.get()),
            max_height=float(self.max_height_var.get()),
            seed=seed,
            smoothing_passes=int(self.smoothing_var.get()),
            noise_layers=int(self.layers_var.get()),
            thresholds=[float(var.get()) for var in self.threshold_vars],
            colors=[var.get() for var in self.color_vars],
            warp_strength=float(self.warp_strength_var.get()),
            warp_frequency=float(self.warp_frequency_var.get()),
            warp_twist=float(self.warp_twist_var.get()),
            river_frequency=float(self.river_frequency_var.get()),
            river_depth=float(self.river_depth_var.get()),
            river_width=float(self.river_width_var.get()),
            river_spacing=float(self.river_spacing_var.get()),
        )

    def generate(self) -> None:
        self.last_generated = time.time()
        settings = self.settings()
        self.status_var.set(f"Generating {settings.size}x{settings.size}...")
        self.update_idletasks()
        self.heightmap = generate_heightmap(settings)
        self.splatmap = make_splatmap(self.heightmap, settings)
        self.preview_points = self._build_preview_points(self.heightmap, self.splatmap, settings.max_height)
        self.draw_previews()
        self.status_var.set(f"Generated {settings.size}x{settings.size} terrain")

    def _build_preview_points(
        self,
        heightmap: np.ndarray,
        splatmap: np.ndarray,
        max_height: float,
    ) -> list[tuple[float, float, float, tuple[int, int, int]]]:
        size = heightmap.shape[0]
        stride = max(1, size // 72)
        points = []
        for y in range(0, size, stride):
            for x in range(0, size, stride):
                px = (x / max(1, size - 1) - 0.5) * 2.0
                py = (y / max(1, size - 1) - 0.5) * 2.0
                pz = float(heightmap[y, x] / max(0.000001, max_height))
                points.append((px, py, pz, tuple(int(channel) for channel in splatmap[y, x])))
        return points

    def draw_previews(self) -> None:
        if self.heightmap is None or self.splatmap is None:
            return
        settings = self.settings()
        grayscale = np.clip(self.heightmap / max(0.000001, settings.max_height) * 255, 0, 255).astype(np.uint8)
        height_rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
        self.height_preview_image = self._canvas_image(self.height_canvas, height_rgb)
        self.splat_preview_image = self._canvas_image(self.splat_canvas, self.splatmap.astype(np.uint8))
        self._redraw_gradient()

    def _canvas_image(self, canvas: tk.Canvas, data: np.ndarray) -> ImageTk.PhotoImage:
        canvas.delete("all")
        width = int(canvas["width"])
        height = int(canvas["height"])
        image = Image.fromarray(data).resize((width, height), Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        return photo

    def _redraw_gradient(self) -> None:
        width = max(1, self.gradient_canvas.winfo_width())
        height = max(1, self.gradient_canvas.winfo_height())
        self.gradient_canvas.delete("all")
        thresholds = [var.get() for var in self.threshold_vars]
        colors = [var.get() for var in self.color_vars]
        for x in range(width):
            value = x / max(1, width - 1)
            color = color_for_height(value, thresholds, colors)
            fill = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            self.gradient_canvas.create_line(x, 0, x, height, fill=fill)
        for threshold in thresholds:
            px = int(threshold * width)
            self.gradient_canvas.create_line(px, 0, px, height, fill="#ffffff", width=2)

    def animate(self) -> None:
        self.rotation += 0.012
        self.draw_terrain()
        self.after(33, self.animate)

    def draw_terrain(self) -> None:
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if not self.preview_points:
            self.canvas.create_text(width / 2, height / 2, text="Generating terrain...", fill="#ffffff", font=("Arial", 22))
            return

        self._draw_grid(width, height)
        cos_a = math.cos(self.rotation)
        sin_a = math.sin(self.rotation)
        scale = min(width, height) * 0.32
        projected = []
        for x, y, z, color in self.preview_points:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            sx = width / 2 + (rx - ry) * scale * 0.75
            sy = height / 2 + (rx + ry) * scale * 0.25 - z * scale * 0.45
            shade = 0.72 + min(0.28, z * 0.28)
            shaded = tuple(max(0, min(255, int(channel * shade))) for channel in color)
            projected.append((sy, sx, sy, shaded, z))

        projected.sort(key=lambda item: item[0])
        point_size = max(2, int(scale / 70))
        for _depth, sx, sy, color, z in projected:
            fill = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            radius = point_size + int(z * 2)
            self.canvas.create_oval(sx - radius, sy - radius, sx + radius, sy + radius, fill=fill, outline=fill)

    def _draw_grid(self, width: int, height: int) -> None:
        center_x = width / 2
        base_y = height * 0.68
        color = "#9fc0c8"
        for index in range(-12, 13):
            offset = index * 32
            self.canvas.create_line(center_x - 520 + offset, base_y - 170, center_x + 520 + offset, base_y + 170, fill=color)
            self.canvas.create_line(center_x - 520 + offset, base_y + 170, center_x + 520 + offset, base_y - 170, fill=color)

    def export_pngs(self) -> None:
        if self.heightmap is None or self.splatmap is None:
            self.generate()
        if self.heightmap is None or self.splatmap is None:
            return

        base_path = filedialog.asksaveasfilename(
            parent=self,
            title="Export heightmap and splatmap",
            defaultextension=".png",
            filetypes=[("PNG images", "*.png")],
            initialfile="terrain.png",
        )
        if not base_path:
            return
        if base_path.lower().endswith(".png"):
            base_path = base_path[:-4]

        max_height = max(0.000001, self.settings().max_height)
        height_pixels = np.clip(self.heightmap / max_height * 255, 0, 255).astype(np.uint8)
        height_path = f"{base_path}_heightmap.png"
        splat_path = f"{base_path}_splatmap.png"
        Image.fromarray(height_pixels).save(height_path)
        Image.fromarray(self.splatmap.astype(np.uint8)).save(splat_path)
        messagebox.showinfo("Export complete", f"Saved:\n{height_path}\n{splat_path}", parent=self)


def main() -> None:
    app = TerrainGeneratorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
