# Heightmap / Splatmap Generator

A Python terrain generator with a Tkinter interface, rotating terrain preview, adjustable heightmap controls, splatmap thresholds, domain warping, and river carving.

Run it with:

```bash
uv run python main.py
```

Use **Generate** to rebuild the terrain and **Export PNGs** to save:

- `<name>_heightmap.png`
- `<name>_splatmap.png`

Large 1024 and 2048 maps are generated only when you press **Generate**, so slider changes stay responsive.
