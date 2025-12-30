#!/usr/bin/env python3
"""
Generate complex multi-UR5E robot scenes with:
- Random initial joint positions
- Random control inputs
- Interactive objects (boxes, spheres)
- Work tables
- Denser layouts for more collisions
"""

import argparse
import math
import random
from pathlib import Path

# UR5E joint limits (in radians)
JOINT_LIMITS = {
    'shoulder_pan': (-6.28, 6.28),
    'shoulder_lift': (-6.28, 6.28),
    'elbow': (-3.14, 3.14),
    'wrist_1': (-6.28, 6.28),
    'wrist_2': (-6.28, 6.28),
    'wrist_3': (-6.28, 6.28),
}

# Safe operational ranges (more conservative for realistic poses)
SAFE_RANGES = {
    'shoulder_pan': (-2.0, 2.0),
    'shoulder_lift': (-2.5, -0.5),
    'elbow': (0.5, 2.5),
    'wrist_1': (-2.5, -0.5),
    'wrist_2': (-2.0, 2.0),
    'wrist_3': (-2.0, 2.0),
}

DEFAULT_SPACING = 1.0  # Closer spacing for more interactions

XML_HEADER = '''<mujoco model="{n} UR5E Robots Complex Scene">
  
  <option timestep="0.005" solver="CG" integrator="implicit"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <texture name="box_tex" type="2d" builtin="flat" width="64" height="64" rgb1=".8 .4 .1"/>
    <texture name="table_tex" type="2d" builtin="flat" width="128" height="128" rgb1=".3 .25 .2"/>
    <material name="floor" texture="floor" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <material name="box_mat" texture="box_tex" specular="0.3" shininess="0.1"/>
    <material name="sphere_mat" rgba=".2 .6 .8 1" specular="0.5" shininess="0.3"/>
    <material name="table_mat" texture="table_tex" specular="0.2" shininess="0.05"/>
    <model name="ur5e" file="ur5e.xml"/>
  </asset>

  <visual>
    <map force="0.1" zfar="100"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>

  <worldbody>
    <geom name="floor" size="{floor_size} {floor_size} .05" type="plane" material="floor" condim="3"/>
    <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5" dir="0 0 -1" castshadow="false"/>
    <light name="spotlight" mode="targetbodycom" target="world" diffuse="1 1 1" specular="0.3 0.3 0.3" pos="0 0 10" cutoff="60"/>
'''

XML_FOOTER = '''  </worldbody>
</mujoco>
'''

FRAME_TEMPLATE = '''    <frame pos="{x:.4f} {y:.4f} {z:.4f}">
      <attach model="ur5e" body="base" prefix="u{idx}_"/>
    </frame>
'''

TABLE_TEMPLATE = '''    <!-- Table for robot {idx} -->
    <body name="table_{idx}" pos="{x:.4f} {y:.4f} 0">
      <geom type="box" size="0.4 0.4 0.4" pos="0 0 0.4" material="table_mat" mass="50"/>
    </body>
'''

BOX_TEMPLATE = '''    <!-- Box {idx} -->
    <body name="box_{idx}" pos="{x:.4f} {y:.4f} {z:.4f}">
      <freejoint name="box_{idx}_joint"/>
      <geom type="box" size="{size:.3f} {size:.3f} {size:.3f}" material="box_mat" mass="{mass:.2f}"/>
    </body>
'''

SPHERE_TEMPLATE = '''    <!-- Sphere {idx} -->
    <body name="sphere_{idx}" pos="{x:.4f} {y:.4f} {z:.4f}">
      <freejoint name="sphere_{idx}_joint"/>
      <geom type="sphere" size="{size:.3f}" material="sphere_mat" mass="{mass:.2f}"/>
    </body>
'''


def generate_random_qpos():
    """Generate random joint positions within safe ranges."""
    qpos = []
    for joint in ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']:
        low, high = SAFE_RANGES[joint]
        qpos.append(random.uniform(low, high))
    return qpos


def generate_random_ctrl():
    """Generate random control inputs (normalized to [-1, 1] range)."""
    return [random.uniform(-0.5, 0.5) for _ in range(6)]


def generate_square_positions(count: int, spacing: float):
    """Generate positions for a square grid layout."""
    side = math.ceil(math.sqrt(count))
    positions = []
    
    # Center the grid
    offset = (side - 1) * spacing / 2
    
    for i in range(count):
        row = i // side
        col = i % side
        x = col * spacing - offset
        y = row * spacing - offset
        positions.append((x, y))
    
    return positions, side


def generate_keyframe(count: int, object_positions: list):
    """Generate keyframe with random initial states.
    
    Args:
        count: Number of robots
        object_positions: List of (x, y, z) tuples for each object's initial position
    """
    lines = ['  <keyframe>']
    
    # Generate qpos for all robots
    all_qpos = []
    all_ctrl = []
    
    for i in range(count):
        qpos = generate_random_qpos()
        ctrl = generate_random_ctrl()
        all_qpos.extend(qpos)
        all_ctrl.extend(ctrl)
    
    # Add object positions if present (freejoint: pos3 + quat4)
    for (ox, oy, oz) in object_positions:
        # Use actual position from XML, identity quaternion (w, x, y, z) = (1, 0, 0, 0)
        all_qpos.extend([ox, oy, oz, 1, 0, 0, 0])
    
    qpos_str = ' '.join(f'{v:.4f}' for v in all_qpos)
    ctrl_str = ' '.join(f'{v:.4f}' for v in all_ctrl)
    
    lines.append(f'    <key name="random_init" qpos="{qpos_str}" ctrl="{ctrl_str}"/>')
    lines.append('  </keyframe>')
    
    return '\n'.join(lines)


def generate_scene(count: int, spacing: float, with_tables: bool, 
                   with_objects: bool, objects_per_robot: int, seed: int = None) -> str:
    """Generate a complex multi-robot scene XML."""
    
    if seed is not None:
        random.seed(seed)
    
    positions, side = generate_square_positions(count, spacing)
    
    # Floor size should accommodate all robots with margin
    floor_size = side * spacing * 0.75 + 1
    
    xml_parts = [XML_HEADER.format(n=count, floor_size=floor_size)]
    
    # Table height (robots mounted on tables)
    table_height = 0.8 if with_tables else 0
    
    # Add tables first
    if with_tables:
        for idx, (x, y) in enumerate(positions):
            xml_parts.append(TABLE_TEMPLATE.format(idx=idx, x=x, y=y))
    
    # Add robots
    for idx, (x, y) in enumerate(positions):
        xml_parts.append(FRAME_TEMPLATE.format(x=x, y=y, z=table_height, idx=idx))
    
    # Add objects
    if with_objects:
        obj_idx = 0
        for idx, (x, y) in enumerate(positions):
            for _ in range(objects_per_robot):
                # Random offset from robot base
                ox = x + random.uniform(-0.3, 0.3)
                oy = y + random.uniform(-0.3, 0.3)
                oz = table_height + 0.5 + random.uniform(0, 0.3)
                
                # Randomly choose box or sphere
                if random.random() < 0.5:
                    size = random.uniform(0.03, 0.06)
                    mass = random.uniform(0.1, 0.5)
                    xml_parts.append(BOX_TEMPLATE.format(idx=obj_idx, x=ox, y=oy, z=oz, 
                                                          size=size, mass=mass))
                else:
                    size = random.uniform(0.03, 0.05)
                    mass = random.uniform(0.1, 0.3)
                    xml_parts.append(SPHERE_TEMPLATE.format(idx=obj_idx, x=ox, y=oy, z=oz,
                                                             size=size, mass=mass))
                obj_idx += 1
    
    xml_parts.append(XML_FOOTER)
    
    return ''.join(xml_parts)


def main():
    parser = argparse.ArgumentParser(description='Generate complex multi-UR5E robot scene')
    parser.add_argument('count', type=int, help='Number of robots')
    parser.add_argument('--spacing', type=float, default=DEFAULT_SPACING,
                        help=f'Spacing between robots (default: {DEFAULT_SPACING}m)')
    parser.add_argument('--with-tables', action='store_true',
                        help='Add work tables under robots')
    parser.add_argument('--with-objects', action='store_true',
                        help='Add interactive objects (boxes, spheres)')
    parser.add_argument('--objects-per-robot', type=int, default=2,
                        help='Number of objects per robot (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path')
    
    args = parser.parse_args()
    
    if args.output is None:
        suffix = '_complex'
        if args.with_tables:
            suffix += '_tables'
        if args.with_objects:
            suffix += '_objects'
        output_path = Path(__file__).parent / f'{args.count}_ur5e{suffix}.xml'
    else:
        output_path = Path(args.output)
    
    xml_content = generate_scene(
        count=args.count,
        spacing=args.spacing,
        with_tables=args.with_tables,
        with_objects=args.with_objects,
        objects_per_robot=args.objects_per_robot,
        seed=args.seed
    )
    
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    print(f'Generated {output_path}')
    print(f'  - Robots: {args.count}')
    print(f'  - Spacing: {args.spacing}m')
    print(f'  - Tables: {args.with_tables}')
    print(f'  - Objects: {args.with_objects} ({args.objects_per_robot} per robot)')


if __name__ == '__main__':
    main()
