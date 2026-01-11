#!/usr/bin/env python3
"""
Generate fruitfly square grid scenes for benchmarking.
Fruitflies are arranged in a square grid pattern.
"""

import os
import math
import shutil

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "square")
MENAGERIE_FRUITFLY = os.path.join(os.path.dirname(__file__), "..", "..", "mujoco_menagerie", "flybody", "fruitfly.xml")

# Fruitfly body is small but has legs and wings that extend outward
# Wingspan and legs span about 0.3-0.4m, so 0.5m spacing ensures no overlap
SPACING = 1  # meters between fruitflies in grid

# Counts to generate (perfect squares for clean grids)
COUNTS = [4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 196, 256]


def generate_square_grid_scene(count: int, spacing: float) -> str:
    """
    Generate XML content for a square grid of fruitflies.
    
    Args:
        count: Number of fruitflies (should be a perfect square)
        spacing: Distance between fruitflies
    
    Returns:
        XML content as string
    """
    side = int(math.sqrt(count))
    assert side * side == count, f"Count {count} must be a perfect square"
    
    # Calculate grid offset to center at origin
    offset = (side - 1) * spacing / 2
    
    # Floor size should accommodate all fruitflies with some margin
    floor_size = max(side * spacing * 1.5, 1.0)
    
    # Generate individual frame positions
    frames = []
    for row in range(side):
        for col in range(side):
            x = col * spacing - offset
            y = row * spacing - offset
            # Fruitfly initial height - they are small, so 0.1 is good
            z = 0.3
            prefix = f"_r{row}_c{col}_"
            frames.append(f'''      <frame pos="{x:.6f} {y:.6f} {z}">
        <attach model="fruitfly" body="thorax" prefix="{prefix}"/>
      </frame>''')
    
    frames_content = "\n".join(frames)
    
    xml_content = f'''<mujoco model="{count} Fruitflies Square">
  <option timestep="0.002" solver="CG" integrator="implicitfast"/>

  <asset>
    <model name="fruitfly" file="fruitfly.xml"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom name="floor" size="{floor_size:.1f} {floor_size:.1f} .05" type="plane" material="grid" contype="1" conaffinity="1"/>

    <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5" dir="0 0 -1" castshadow="false"/>
    <light name="spotlight" mode="targetbodycom" target="world" diffuse="1 1 1" specular="0.3 0.3 0.3" pos="-6 -6 4" cutoff="60"/>
    
    <!-- {count} fruitflies in {side}x{side} grid, spacing={spacing}m -->
{frames_content}
  </worldbody>
</mujoco>
'''
    return xml_content


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Copy original fruitfly.xml to output directory
    dest_fruitfly = os.path.join(OUTPUT_DIR, "fruitfly.xml")
    if not os.path.exists(dest_fruitfly):
        shutil.copy(MENAGERIE_FRUITFLY, dest_fruitfly)
        print(f"Copied fruitfly.xml to {OUTPUT_DIR}")
    
    # Copy assets directory
    assets_src = os.path.join(os.path.dirname(MENAGERIE_FRUITFLY), "assets")
    assets_dst = os.path.join(OUTPUT_DIR, "assets")
    if os.path.exists(assets_src) and not os.path.exists(assets_dst):
        shutil.copytree(assets_src, assets_dst)
        print(f"Copied assets directory to {OUTPUT_DIR}")
    
    # Generate scenes
    for count in COUNTS:
        xml_content = generate_square_grid_scene(count, SPACING)
        output_file = os.path.join(OUTPUT_DIR, f"{count}_fruitflies.xml")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        print(f"Generated {output_file}")
    
    print(f"\nGenerated {len(COUNTS)} scene files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
