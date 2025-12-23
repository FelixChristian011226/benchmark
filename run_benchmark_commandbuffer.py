import subprocess
import re
import csv
import os
import sys
from typing import Dict, Any, List

# --- 1. 配置区域 ---

DEFAULT_STEPS = 1000
DEFAULT_WORLDS = 1

OUTPUT_FILE = "benchmark_results.csv"
TIMEOUT_SECONDS = 600

# CUDA launch queue scaling settings
# 支持：0.25x, 0.5x, 1x, 2x, 4x
CUDA_LAUNCH_QUEUE_SCALES = ["1x", "4x"]

# .exe 测试配置列表
EXE_TEST_CONFIGS = [
    {
        "engine_name": "mujoco",
        "model_type": "dense",
        "exe_path": r"D:\Learn\CLion\original\mujoco\cmake-build-release\bin\testspeed.exe",
        "model_base_path": r"D:\Learn\CLion\original\humanoid"
    },
    {
        "engine_name": "mujoco",
        "model_type": "sparse",
        "exe_path": r"D:\Learn\CLion\original\mujoco\cmake-build-release\bin\testspeed.exe",
        "model_base_path": r"D:\Learn\CLion\original\humanoid\sparse"
    },
    {
        "engine_name": "cuda_mujoco",
        "model_type": "default",
        "exe_path": r"D:\Learn\CLion\cuda_mujoco\cmake-build-release\bin\testspeed.exe",
        "model_base_path": r"D:\Learn\CLion\original\humanoid"
    }
]

# MuJoCo Warp 配置
WARP_MODEL_BASE_PATH = r"benchmark/humanoid"

WARP_ENGINES = {
    "mujoco_warp": {
        "root_path": r"D:\Learn\CLion\original\mujoco_warp",
        "env_script": r".\env\Scripts\Activate.ps1",
        "command": "mjwarp-testspeed"
    }
}

# ModelScaling 模型列表
MODEL_SCALING_LIST = [
    (1, "humanoid.xml"),
    (8, "8_humanoids.xml"),
    (22, "22_humanoids.xml"),
    (50, "50_humanoids.xml"),
    (100, "100_humanoids.xml"),
    (200, "200_humanoids.xml")
]

# WorldScaling 测试
NWORLD_SCALING_LIST = [1, 4, 16, 64, 256]

NWORLD_WARP_BASE_MODEL = "humanoid.xml"
CUDA_NWORLD_MODEL_BASE_PATH = r"D:\Learn\CLion\original\humanoid\n_humanoid"

# --- 2. 解析函数 ---

def parse_exe_output(output_text: str) -> Dict[str, Any]:
    data = {}

    simple_metrics = [
        ("SimulationTime_s", r"Simulation time\s*:\s*([\d.]+)\s*s"),
        ("StepsPerSecond", r"Steps per second\s*:\s*([\d.]+)"),
        ("RealtimeFactor", r"Realtime factor\s*:\s*([\d.]+)\s*x"),
        ("TimePerStep_us", r"Time per step\s*:\s*([\d.]+)\s*(\?s|μs|us)"),
        ("CG_iters_per_step", r"CG iters / step\s*:\s*([\d.]+)"),
        ("Contacts_per_step", r"Contacts / step\s*:\s*([\d.]+)"),
        ("Constraints_per_step", r"Constraints / step\s*:\s*([\d.]+)"),
        ("DOF", r"Degrees of freedom\s*:\s*([\d.]+)")
    ]

    for name, pattern in simple_metrics:
        match = re.search(pattern, output_text)
        if match:
            data[name] = float(match.group(1))

    profiler_keys = [
        "step", "forward", "position", "velocity", "actuation",
        "constraint", "advance", "other", "position total",
        "kinematics", "inertia", "collision", "broadphase",
        "narrowphase", "make", "project"
    ]

    for key in profiler_keys:
        pattern = rf"^\s*{re.escape(key)}\s*:\s*([\d.]+)\s*\("
        match = re.search(pattern, output_text, re.MULTILINE)
        if match:
            csv_key_name = f"Profiler_{key.replace(' ', '_')}_us"
            data[csv_key_name] = float(match.group(1))

    return data


def parse_warp_output(output_text: str) -> Dict[str, Any]:
    data = {}

    simple_metrics = [
        ("SimulationTime_s", r"Total simulation time\s*:\s*([\d.]+)\s*s"),
        ("StepsPerSecond", r"Total steps per second\s*:\s*([\d.,]+)"),
        ("RealtimeFactor", r"Total realtime factor\s*:\s*([\d.,]+)\s*x"),
        ("TimePerStep_ns", r"Total time per step\s*:\s*([\d.]+)\s*ns"),
    ]

    for name, pattern in simple_metrics:
        match = re.search(pattern, output_text)
        if match:
            value_str = match.group(1).replace(",", "")
            value = float(value_str)
            if name == "TimePerStep_ns":
                data["TimePerStep_us"] = value / 1000.0
            else:
                data[name] = value

    profiler_map = {
        "step": "Profiler_step_us",
        "forward": "Profiler_forward_us",
        "fwd_position": "Profiler_position_us",
        "fwd_velocity": "Profiler_velocity_us",
        "fwd_actuation": "Profiler_actuation_us",
        "solve": "Profiler_constraint_us",
        "euler": "Profiler_advance_us",
        "collision": "Profiler_collision_us",
        "nxn_broadphase": "Profiler_broadphase_us",
        "make_constraint": "Profiler_make_us",
        "primitive_narrowphase": "Profiler_narrowphase_us"
    }

    trace_match = re.search(r"Event trace:", output_text)
    if trace_match:
        profiler_text = output_text[trace_match.end():]

        for warp_key, csv_key in profiler_map.items():
            pattern = rf"^\s*{re.escape(warp_key)}\s*:\s*([\d.]+)"
            match = re.search(pattern, profiler_text, re.MULTILINE)
            if match:
                data[csv_key] = float(match.group(1)) / 1000.0

    return data


# --- 3. 通用执行函数 ---

def run_command_and_parse(command: List[str], engine_name: str, model_id: str,
                          is_powershell: bool = False, parser_type: str = "exe") -> Dict[str, Any]:

    print(f"\n正在运行: [引擎: {engine_name}] - [测试: {model_id}]")
    if is_powershell:
        print(f"命令: {command[0]} {command[1]} '{command[2]}'")
    else:
        print(f"命令: {' '.join(command)}")

    run_kwargs = {
        "capture_output": True,
        "text": True,
        "encoding": 'latin-1',
        "timeout": TIMEOUT_SECONDS
    }

    if is_powershell:
        if sys.platform == "win32":
            run_kwargs["shell"] = True
        else:
            command = ['pwsh'] + command[1:]

    try:
        result = subprocess.run(command, **run_kwargs)
        if result.returncode != 0:
            print("!! 错误: 执行失败")
            return {"error": "ReturnCode_NonZero", "stderr": result.stderr}

        stdout = result.stdout
        parsed_data = parse_warp_output(stdout) if parser_type == "warp" else parse_exe_output(stdout)

        if not parsed_data:
            print("!! 错误: 无法解析输出")
            return {"error": "Parsing_Failed", "stdout": stdout}

        print(f"成功。 每步时间: {parsed_data.get('TimePerStep_us', 'N/A')} µs")
        return parsed_data

    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}

    except Exception as e:
        return {"error": str(e)}


# --- 4. 主程序 ---

def main():

    results = []
    all_fieldnames = set(["Engine", "ModelType", "Humanoids", "ModelFile",
                          "Error", "TestCategory", "Worlds", "CudaLaunchQueueScale"])

    print("开始执行基准测试...")

    # =====================================================================
    # 1. .EXE Engines - ModelScaling
    # =====================================================================
    print("\n--- 正在测试 .EXE 引擎 [ModelScaling] ---")

    for config in EXE_TEST_CONFIGS:
        engine_name = config["engine_name"]
        model_type = config["model_type"]
        exe_path = config["exe_path"]
        model_base_path = config["model_base_path"]

        if not os.path.exists(exe_path):
            print(f"跳过：找不到 {exe_path}")
            continue

        # ========================================================
        # ① CUDA 引擎：循环所有 scale
        # ========================================================
        if engine_name == "cuda_mujoco":

            for scale in CUDA_LAUNCH_QUEUE_SCALES:
                print(f"\n>>> 设置 CUDA_SCALE_LAUNCH_QUEUES = {scale}")
                os.environ["CUDA_SCALE_LAUNCH_QUEUES"] = scale

                for humanoid_count, model_file in MODEL_SCALING_LIST:
                    model_path = os.path.join(model_base_path, model_file)
                    if not os.path.exists(model_path):
                        continue

                    command = [exe_path, model_path, str(DEFAULT_STEPS)]
                    parsed = run_command_and_parse(
                        command,
                        f"{engine_name} ({model_type})",
                        f"{model_file} ({humanoid_count}a, scale={scale})",
                        parser_type="exe"
                    )

                    parsed["TestCategory"] = "ModelScaling"
                    parsed["Engine"] = engine_name
                    parsed["ModelType"] = model_type
                    parsed["Humanoids"] = humanoid_count
                    parsed["Worlds"] = 1
                    parsed["ModelFile"] = model_file
                    parsed["CudaLaunchQueueScale"] = scale

                    results.append(parsed)
                    all_fieldnames.update(parsed.keys())

            continue

        # ========================================================
        # ② 非 CUDA 引擎：正常执行一次
        # ========================================================
        for humanoid_count, model_file in MODEL_SCALING_LIST:
            model_path = os.path.join(model_base_path, model_file)
            if not os.path.exists(model_path):
                continue

            command = [exe_path, model_path, str(DEFAULT_STEPS)]

            parsed = run_command_and_parse(
                command,
                f"{engine_name} ({model_type})",
                f"{model_file} ({humanoid_count}a)",
                parser_type="exe"
            )

            parsed["TestCategory"] = "ModelScaling"
            parsed["Engine"] = engine_name
            parsed["ModelType"] = model_type
            parsed["Humanoids"] = humanoid_count
            parsed["Worlds"] = 1
            parsed["ModelFile"] = model_file
            parsed["CudaLaunchQueueScale"] = "N/A"

            results.append(parsed)
            all_fieldnames.update(parsed.keys())


    # =====================================================================
    # 2. Warp Engines - ModelScaling
    # =====================================================================
    print("\n--- 正在测试 Warp 引擎 [ModelScaling] ---")

    for engine_name, config in WARP_ENGINES.items():

        root_path = config["root_path"]
        env_script = config["env_script"]
        run_command = config["command"]

        env_script_full_path = os.path.join(root_path, env_script)

        if not os.path.exists(root_path) or not os.path.exists(env_script_full_path):
            print(f"跳过 Warp（路径错误）")
            continue

        for humanoid_count, model_file in MODEL_SCALING_LIST:
            model_path_for_warp = f"{WARP_MODEL_BASE_PATH}/{model_file}".replace("\\", "/")

            ps_cmd = (
                f"& {{ Set-Location '{root_path}'; "
                f"& '{env_script}'; "
                f"{run_command} {model_path_for_warp} --event_trace=True "
                f"--nstep={DEFAULT_STEPS} --nworld={DEFAULT_WORLDS} }}"
            )

            command = ['powershell', '-Command', ps_cmd]

            parsed = run_command_and_parse(
                command,
                engine_name,
                f"{model_file} ({humanoid_count}a, {DEFAULT_WORLDS}w)",
                is_powershell=True,
                parser_type="warp"
            )

            parsed["TestCategory"] = "ModelScaling"
            parsed["Engine"] = engine_name
            parsed["ModelType"] = "default"
            parsed["Humanoids"] = humanoid_count
            parsed["Worlds"] = DEFAULT_WORLDS
            parsed["ModelFile"] = model_file
            parsed["CudaLaunchQueueScale"] = "N/A"

            results.append(parsed)
            all_fieldnames.update(parsed.keys())


    # =====================================================================
    # 3. Warp Engines - WorldScaling
    # =====================================================================
    print("\n--- 正在测试 Warp 引擎 [WorldScaling] ---")

    if "mujoco_warp" in WARP_ENGINES:

        config = WARP_ENGINES["mujoco_warp"]
        root_path = config["root_path"]
        env_script = config["env_script"]
        run_command = config["command"]

        env_script_full_path = os.path.join(root_path, env_script)

        if os.path.exists(root_path) and os.path.exists(env_script_full_path):

            model_file = NWORLD_WARP_BASE_MODEL
            model_path_for_warp = f"{WARP_MODEL_BASE_PATH}/{model_file}".replace("\\", "/")

            for nworld in NWORLD_SCALING_LIST:

                ps_cmd = (
                    f"& {{ Set-Location '{root_path}'; "
                    f"& '{env_script}'; "
                    f"{run_command} {model_path_for_warp} --event_trace=True "
                    f"--nstep={DEFAULT_STEPS} --nworld={nworld} }}"
                )

                command = ['powershell', '-Command', ps_cmd]

                parsed = run_command_and_parse(
                    command,
                    "mujoco_warp",
                    f"{model_file} (1a, {nworld}w)",
                    is_powershell=True,
                    parser_type="warp"
                )

                parsed["TestCategory"] = "WorldScaling"
                parsed["Engine"] = "mujoco_warp"
                parsed["ModelType"] = "nworld_scaling"
                parsed["Humanoids"] = 1
                parsed["Worlds"] = nworld
                parsed["ModelFile"] = model_file
                parsed["CudaLaunchQueueScale"] = "N/A"

                results.append(parsed)
                all_fieldnames.update(parsed.keys())


    # =====================================================================
    # 4. CUDA_MUJOCO - WorldScaling（含 scale）
    # =====================================================================
    print("\n--- 正在测试 cuda_mujoco 引擎 [WorldScaling] ---")

    cuda_cfg = None
    for cfg in EXE_TEST_CONFIGS:
        if cfg["engine_name"] == "cuda_mujoco":
            cuda_cfg = cfg
            break

    if cuda_cfg:

        exe_path = cuda_cfg["exe_path"]
        model_base_path = CUDA_NWORLD_MODEL_BASE_PATH

        if os.path.exists(exe_path) and os.path.exists(model_base_path):

            for scale in CUDA_LAUNCH_QUEUE_SCALES:

                print(f"\n>>> 设置 CUDA_SCALE_LAUNCH_QUEUES = {scale}")
                os.environ["CUDA_SCALE_LAUNCH_QUEUES"] = scale

                for count in NWORLD_SCALING_LIST:

                    model_file = "humanoid.xml" if count == 1 else f"{count}_humanoid.xml"
                    model_path = os.path.join(model_base_path, model_file)

                    if not os.path.exists(model_path):
                        continue

                    command = [exe_path, model_path, str(DEFAULT_STEPS)]

                    parsed = run_command_and_parse(
                        command,
                        "cuda_mujoco",
                        f"{model_file} ({count}a, scale={scale})",
                        parser_type="exe"
                    )

                    parsed["TestCategory"] = "WorldScaling"
                    parsed["Engine"] = "cuda_mujoco"
                    parsed["ModelType"] = "nworld_equivalent"
                    parsed["Humanoids"] = count
                    parsed["Worlds"] = count
                    parsed["ModelFile"] = model_file
                    parsed["CudaLaunchQueueScale"] = scale

                    results.append(parsed)
                    all_fieldnames.update(parsed.keys())


    # =====================================================================
    # 5. 写入 CSV
    # =====================================================================

    if not results:
        print("没有结果。")
        return

    print(f"\n写入 CSV 文件: {OUTPUT_FILE}")

    ordered_fields = [
        "TestCategory", "Engine", "ModelType",
        "Humanoids", "Worlds", "ModelFile",
        "CudaLaunchQueueScale",
        "SimulationTime_s", "StepsPerSecond", "TimePerStep_us", "DOF",
        "Profiler_step_us", "Profiler_constraint_us", "Profiler_collision_us",
        "Error", "stderr"
    ]

    final_fields = ordered_fields + sorted([f for f in all_fieldnames if f not in ordered_fields])

    with open(OUTPUT_FILE, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_fields, restval="N/A")
        writer.writeheader()
        writer.writerows(results)

    print("完成！")


if __name__ == "__main__":
    main()
