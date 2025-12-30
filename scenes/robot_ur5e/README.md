# UR5E Multi-Robot Scenes

This directory contains multi-UR5E robot arm scenes for MuJoCo benchmarking.

## Directory Structure

```
robot_ur5e/
├── universal_robots_ur5e/   # Base model files (copied from mujoco_menagerie)
│   ├── ur5e.xml            # UR5E model definition
│   ├── scene.xml           # Original single-robot scene
│   └── assets/             # Mesh files (.obj)
├── square/                  # Basic multi-robot scenes in square grid
│   ├── ur5e.xml            # Base model for attach mechanism
│   ├── gen_ur5e_square.py  # Basic scene generator
│   └── *_ur5e_square.xml   # Generated scenes
└── complex/                 # Complex scenes for benchmarking
    ├── ur5e.xml            # Base model for attach
    ├── gen_ur5e_complex.py # Advanced scene generator
    └── *_ur5e_complex_*.xml # Complex scenes with tables & objects
```

## Complex Scenes (Recommended for Benchmarking)

The `complex/` folder contains scenes optimized for benchmarking with:
- **Random initial joint positions** - each robot starts in a unique pose
- **Random control inputs** - motors actively working from step 0
- **Work tables** - robots mounted on tables (z=0.8m)
- **Interactive objects** - boxes and spheres that fall and collide
- **Denser spacing** (1.0m vs 1.2m default)

### Available Complex Scenes

| File | Robots | Objects | Description |
|------|--------|---------|-------------|
| `4_ur5e_complex_tables_objects.xml` | 4 | 8 | 2×2 grid |
| `9_ur5e_complex_tables_objects.xml` | 9 | 18 | 3×3 grid |
| `16_ur5e_complex_tables_objects.xml` | 16 | 32 | 4×4 grid |
| `25_ur5e_complex_tables_objects.xml` | 25 | 50 | 5×5 grid |
| `36_ur5e_complex_tables_objects.xml` | 36 | 72 | 6×6 grid |
| `49_ur5e_complex_tables_objects.xml` | 49 | 98 | 7×7 grid |
| `64_ur5e_complex_tables_objects.xml` | 64 | 128 | 8×8 grid |
| `100_ur5e_complex_tables_objects.xml` | 100 | 200 | 10×10 grid |

## Generating Custom Scenes

### Basic scenes (square/)
```bash
cd square
python gen_ur5e_square.py <count> [--spacing <meters>]
```

### Complex scenes (complex/)
```bash
cd complex
python gen_ur5e_complex.py <count> [options]

Options:
  --spacing <m>         Robot spacing (default: 1.0m)
  --with-tables         Add work tables under robots
  --with-objects        Add boxes/spheres for interaction
  --objects-per-robot   Objects per robot (default: 2)
  --seed <int>          Random seed for reproducibility
```

Example:
```bash
python gen_ur5e_complex.py 144 --with-tables --with-objects --spacing 0.9
```
