# Heightmap / Splatmap Generator

A Python terrain generator with a browser-based WebGL preview, adjustable heightmap controls, splatmap thresholds, domain warping, and river carving.

The live preview uses GLSL shaders:

- Vertex shader displacement from the generated heightmap texture
- Fragment shader texture splatting from the generated splatmap texture
- GPU lighting, slope-aware rock blending, and procedural surface variation

Run it with:

```bash
uv run python main.py
```

Then open the printed local URL, usually:

```text
http://127.0.0.1:8765/
```

You can also launch the browser automatically:

```bash
uv run python main.py --open
```

Use **Generate** to rebuild the terrain and **Export PNGs** to download `terrain_maps.zip`, which contains:

- `<name>_heightmap.png`
- `<name>_splatmap.png`

Large 1024 and 2048 maps are generated only when you press **Generate**, so slider changes stay responsive.
