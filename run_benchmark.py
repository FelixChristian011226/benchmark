import subprocess
import re
import csv
import os
import sys
from typing import Dict, Any, List

# --- 1. 配置区域 ---

DEFAULT_STEPS = 1000
DEFAULT_WORLDS = 1 # 原始 ModelScaling 测试的默认 world 数

OUTPUT_FILE = "benchmark_results.csv"

TIMEOUT_SECONDS = 3600

# .exe 测试配置列表
EXE_TEST_CONFIGS = [
    {
        "engine_name": "mujoco",
        "model_type": "dense",
        "exe_path": r"D:\Learn\CLion\original\mujoco\cmake-build-release\bin\testspeed.exe",
        "model_base_path": r"D:\Learn\CLion\original\humanoid"
        # "steps": 1000  # 可选: 覆盖默认步数
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


# MuJoCo Warp 测试配置
WARP_MODEL_BASE_PATH = r"benchmark/humanoid"

WARP_ENGINES = {
    "mujoco_warp": {
        "root_path": r"D:\Learn\CLion\original\mujoco_warp",
        "env_script": r".\env\Scripts\Activate.ps1",
        "command": "mjwarp-testspeed"
        # "steps" 
    }
}

# 模型列表 (用于 "ModelScaling" 测试)
MODEL_SCALING_LIST = [
    (1, "humanoid.xml"),
    (8, "8_humanoids.xml"),
    (22, "22_humanoids.xml"),
    (50, "50_humanoids.xml"),
    (100, "100_humanoids.xml"),
    (200, "200_humanoids.xml")
]

# --- 1b. (新增) WorldScaling 测试配置 ---

# 用于 mujoco_warp (--nworld N) 和 cuda_mujoco (N_humanoid.xml) 的测试
NWORLD_SCALING_LIST = [1, 4, 16, 64, 256]

# (Warp 用) Warp nworld 测试使用的基础模型
NWORLD_WARP_BASE_MODEL = "humanoid.xml" 

# (Cuda 用) Cuda nworld 等效测试的模型基础路径
CUDA_NWORLD_MODEL_BASE_PATH = r"D:\Learn\CLion\original\humanoid\n_humanoid"


# --- 2. 解析函数 ---

def parse_exe_output(output_text: str) -> Dict[str, Any]:
    """
    使用正则表达式解析 testspeed.exe (mujoco, cuda_mujoco) 的输出。
    """
    data = {}
    
    # 定义要查找的简单指标 (名称, 正则表达式)
    simple_metrics = [
        ("SimulationTime_s", r"Simulation time\s*:\s*([\d.]+)\s*s"),
        ("StepsPerSecond", r"Steps per second\s*:\s*([\d.]+)"),
        ("RealtimeFactor", r"Realtime factor\s*:\s*([\d.]+)\s*x"),
        # 兼容 ?s (编码错误), μs (正确), 和 us
        ("TimePerStep_us", r"Time per step\s*:\s*([\d.]+)\s*(\?s|μs|us)"),
        ("CG_iters_per_step", r"CG iters / step\s*:\s*([\d.]+)"),
        ("Contacts_per_step", r"Contacts / step\s*:\s*([\d.]+)"),
        ("Constraints_per_step", r"Constraints / step\s*:\s*([\d.]+)"),
        ("DOF", r"Degrees of freedom\s*:\s*([\d.]+)")
    ]

    for name, pattern in simple_metrics:
        match = re.search(pattern, output_text)
        if match:
            # TimePerStep_us 有两个捕获组，我们只需要第一个（数值）
            data[name] = float(match.group(1))

    # 定义要查找的内部 profiler 指标
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
    """
    (新增)
    使用正则表达式解析 mjwarp-testspeed 的输出。
    它会将 ns (纳秒) 转换为 μs (微秒) 以保持一致性。
    """
    data = {}
    
    # 定义要查找的简单指标 (名称, 正则表达式)
    simple_metrics = [
        ("SimulationTime_s", r"Total simulation time\s*:\s*([\d.]+)\s*s"),
        ("StepsPerSecond", r"Total steps per second\s*:\s*([\d.,]+)"),
        ("RealtimeFactor", r"Total realtime factor\s*:\s*([\d.,]+)\s*x"),
        # Warp 使用 ns (纳秒)
        ("TimePerStep_ns", r"Total time per step\s*:\s*([\d.]+)\s*ns"),
    ]

    for name, pattern in simple_metrics:
        match = re.search(pattern, output_text)
        if match:
            value_str = match.group(1).replace(",", "") # 移除千位分隔符
            value = float(value_str)
            
            if name == "TimePerStep_ns":
                # 将 ns 转换为 μs (微秒)
                data["TimePerStep_us"] = value / 1000.0
            else:
                data[name] = value

    # 定义要查找的 Event trace (profiler) 指标
    # (Warp 键名, 标准CSV列名)
    profiler_map = {
        "step": "Profiler_step_us",
        "forward": "Profiler_forward_us",
        "fwd_position": "Profiler_position_us",
        "fwd_velocity": "Profiler_velocity_us",
        "fwd_actuation": "Profiler_actuation_us",
        "solve": "Profiler_constraint_us",     # 映射 'solve' 到 'constraint'
        "euler": "Profiler_advance_us",        # 映射 'euler' 到 'advance'
        "collision": "Profiler_collision_us",
        "nxn_broadphase": "Profiler_broadphase_us", # 映射 'nxn_broadphase'
        "make_constraint": "Profiler_make_us",  # 映射 'make_constraint'
        "primitive_narrowphase": "Profiler_narrowphase_us" # 映射 'primitive_narrowphase'
    }

    # 查找 "Event trace:" 标记
    trace_match = re.search(r"Event trace:", output_text)
    if trace_match:
        # 从该点开始查找所有 profiler 键
        profiler_text = output_text[trace_match.end():]
        
        for warp_key, csv_key in profiler_map.items():
            pattern = rf"^\s*{re.escape(warp_key)}\s*:\s*([\d.]+)"
            match = re.search(pattern, profiler_text, re.MULTILINE)
            if match:
                # Warp 的 profiler 单位也是 ns (纳秒)。
                # 所以我们也需要 / 1000.0
                data[csv_key] = float(match.group(1)) / 1000.0

    return data


# --- 3. 主执行函数 ---

def run_command_and_parse(
    command: List[str], 
    engine_name: str, 
    model_id: str, # (已修改) 使用更通用的 ID (可能是文件名或 NWorld)
    is_powershell: bool = False,
    parser_type: str = "exe"
) -> Dict[str, Any]:
    """
    一个通用的函数, 用于执行命令、捕获输出并调用解析器。
    """
    # (已修改) 更新日志
    print(f"\n正在运行: [引擎: {engine_name}] - [测试: {model_id}]")
    
    # 为PowerShell命令打印更清晰的日志
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
    
    # PowerShell 命令需要在 shell=True (Windows) 或特定可执行路径下运行
    if is_powershell:
        if sys.platform == "win32":
            # 在Windows上, 'powershell' 应该在 PATH 中
            run_kwargs["shell"] = True
        else:
            # 在 Linux/Mac 上, 可能需要指定 'pwsh'
            command = ['pwsh'] + command[1:] # 替换 'powershell' 为 'pwsh'
            # (注意: 您的 .ps1 脚本可能在 Linux 上有不同行为)

    try:
        result = subprocess.run(command, **run_kwargs)

        if result.returncode != 0:
            print(f"!! 错误: 命令执行失败。返回码: {result.returncode}")
            print(f"STDOUT: {result.stdout[:500]}...")
            print(f"STDERR: {result.stderr[:500]}...") # 打印前500个字符的错误信息
            return {"error": "ReturnCode_NonZero", "stderr": result.stderr}

        stdout = result.stdout
        
        # 根据 parser_type 调用不同的解析器
        if parser_type == "warp":
            parsed_data = parse_warp_output(stdout)
        else:
            parsed_data = parse_exe_output(stdout)

        if not parsed_data:
            print(f"!! 错误: 无法从输出中解析到任何数据。")
            print(f"STDOUT: {stdout[:500]}...")
            return {"error": "Parsing_Failed", "stdout": stdout}
        
        print(f"成功。 每步时间: {parsed_data.get('TimePerStep_us', 'N/A')} µs")
        return parsed_data

    except subprocess.TimeoutExpired:
        print(f"!! 错误: 命令执行超时 (超过 {TIMEOUT_SECONDS} 秒)。")
        return {"error": "Timeout"}
    except Exception as e:
        print(f"!! 发生意外错误: {e}")
        return {"error": str(e)}


def main():
    results: List[Dict[str, Any]] = []  # 存储所有结果
    # 'ModelType' 已添加
    # (新增) 添加了 'TestCategory' 和 'Worlds'
    all_fieldnames = set(["Engine", "ModelType", "Humanoids", "ModelFile", "Error", "TestCategory", "Worlds"]) # 动态收集所有列名

    print("开始执行基准测试...")
    
    # ---
    # 循环 1: 运行 EXE 引擎 (ModelScaling)
    # ---
    print("\n--- 正在测试 .EXE 引擎 [ModelScaling] ---")
    for config in EXE_TEST_CONFIGS:
        engine_name = config["engine_name"]
        model_type = config["model_type"]
        exe_path = config["exe_path"]
        model_base_path = config["model_base_path"]
        steps_to_run = DEFAULT_STEPS 

        if not os.path.exists(exe_path):
            print(f"\n--- 警告: 未找到 '{engine_name}' ({model_type}) 的 .exe, 跳过。检查路径: {exe_path} ---")
            continue

        # (已修改) 使用 MODEL_SCALING_LIST
        for humanoid_count, model_file in MODEL_SCALING_LIST:
            model_path = os.path.join(model_base_path, model_file)
            
            if not os.path.exists(model_path):
                print(f"\n--- 警告: 未找到模型 '{model_file}' (用于 {engine_name} - {model_type}), 跳过。")
                print(f"--- 检查路径: {model_path} ---")
                continue

            command = [exe_path, model_path, str(steps_to_run)] 
            
            # (已修改) 更新日志
            log_engine_name = f"{engine_name} ({model_type})"
            log_model_id = f"{model_file} ({humanoid_count}a)"
            
            parsed_data = run_command_and_parse(
                command, 
                log_engine_name,
                log_model_id,
                is_powershell=False,
                parser_type="exe"
            )
            
            # 添加上下文信息
            parsed_data["TestCategory"] = "ModelScaling" # (新增)
            parsed_data["Engine"] = engine_name 
            parsed_data["ModelType"] = model_type 
            parsed_data["Humanoids"] = humanoid_count
            parsed_data["Worlds"] = 1 # (新增)
            parsed_data["ModelFile"] = model_file
            
            results.append(parsed_data)
            all_fieldnames.update(parsed_data.keys())

    # ---
    # 循环 2: 运行 Warp 引擎 (ModelScaling)
    # ---
    print("\n--- 正在测试 Warp (PowerShell) 引擎 [ModelScaling] ---")
    for engine_name, config in WARP_ENGINES.items():
        root_path = config["root_path"]
        env_script = config["env_script"]
        run_command = config["command"]
        
        if not os.path.exists(root_path) or not os.path.isdir(root_path):
            print(f"\n--- 警告: 未找到 '{engine_name}' 的根目录, 跳过。检查路径: {root_path} ---")
            continue
            
        env_script_full_path = os.path.join(root_path, env_script)
        if not os.path.exists(env_script_full_path):
            print(f"\n--- 警告: 未找到 '{engine_name}' 的激活脚本, 跳过。检查路径: {env_script_full_path} ---")
            continue
            
        # (已修改) 使用 MODEL_SCALING_LIST
        for humanoid_count, model_file in MODEL_SCALING_LIST:
            model_path_for_warp = f"{WARP_MODEL_BASE_PATH}/{model_file}".replace("\\", "/")

            ps_command_string = (
                f"& {{ "
                f"Set-Location '{root_path}'; "
                f"& '{env_script}'; "
                f"{run_command} {model_path_for_warp} --event_trace=True --nstep={DEFAULT_STEPS}  --nworld={DEFAULT_WORLDS}"
                f" }}"
            )

            command = ['powershell', '-Command', ps_command_string]
            
            log_model_id = f"{model_file} ({humanoid_count}a, {DEFAULT_WORLDS}w)"
            
            parsed_data = run_command_and_parse(
                command, 
                engine_name,
                log_model_id, 
                is_powershell=True,
                parser_type="warp"
            )

            # 添加上下文信息
            parsed_data["TestCategory"] = "ModelScaling" # (新增)
            parsed_data["Engine"] = engine_name
            parsed_data["ModelType"] = "default" 
            parsed_data["Humanoids"] = humanoid_count
            parsed_data["Worlds"] = DEFAULT_WORLDS # (新增)
            parsed_data["ModelFile"] = model_file
            
            results.append(parsed_data)
            all_fieldnames.update(parsed_data.keys())

    # ---
    # (新增) 循环 3: 运行 Warp 引擎 (WorldScaling)
    # ---
    print("\n--- 正在测试 Warp (PowerShell) 引擎 [WorldScaling] ---")
    if "mujoco_warp" in WARP_ENGINES:
        engine_name = "mujoco_warp"
        config = WARP_ENGINES[engine_name]
        
        root_path = config["root_path"]
        env_script = config["env_script"]
        run_command = config["command"]
        
        # 检查 Warp 路径 (复用之前的检查)
        env_script_full_path = os.path.join(root_path, env_script)
        if os.path.exists(root_path) and os.path.isdir(root_path) and os.path.exists(env_script_full_path):
            
            model_file = NWORLD_WARP_BASE_MODEL
            model_path_for_warp = f"{WARP_MODEL_BASE_PATH}/{model_file}".replace("\\", "/")

            for nworld_count in NWORLD_SCALING_LIST:
                
                ps_command_string = (
                    f"& {{ "
                    f"Set-Location '{root_path}'; "
                    f"& '{env_script}'; "
                    f"{run_command} {model_path_for_warp} --event_trace=True --nstep={DEFAULT_STEPS} --nworld={nworld_count}" # <-- 核心变化
                    f" }}"
                )

                command = ['powershell', '-Command', ps_command_string]
                
                log_model_id = f"{model_file} (1a, {nworld_count}w)"
                
                parsed_data = run_command_and_parse(
                    command, 
                    engine_name,
                    log_model_id, 
                    is_powershell=True,
                    parser_type="warp"
                )

                # 添加上下文信息
                parsed_data["TestCategory"] = "WorldScaling"
                parsed_data["Engine"] = engine_name
                parsed_data["ModelType"] = "nworld_scaling" # 新类型
                parsed_data["Humanoids"] = 1 # 固定为1个
                parsed_data["Worlds"] = nworld_count # 变化的值
                parsed_data["ModelFile"] = model_file
                
                results.append(parsed_data)
                all_fieldnames.update(parsed_data.keys())
        else:
            print(f"\n--- 警告: 未找到 '{engine_name}' 的路径或脚本, 跳过 [WorldScaling] 测试。 ---")
    else:
        print("\n--- 警告: 未在 WARP_ENGINES 中配置 'mujoco_warp', 跳过 [WorldScaling] 测试。 ---")

    # ---
    # (新增) 循环 4: 运行 cuda_mujoco 引擎 (WorldScaling)
    # ---
    print("\n--- 正在测试 cuda_mujoco 引擎 [WorldScaling] ---")
    
    cuda_config = None
    for cfg in EXE_TEST_CONFIGS:
        if cfg["engine_name"] == "cuda_mujoco":
            cuda_config = cfg
            break
            
    if cuda_config:
        engine_name = cuda_config["engine_name"]
        exe_path = cuda_config["exe_path"]
        model_base_path = CUDA_NWORLD_MODEL_BASE_PATH # <-- 使用新路径
        
        if not os.path.exists(exe_path):
            print(f"\n--- 警告: 未找到 '{engine_name}' 的 .exe, 跳过 [WorldScaling] 测试。检查路径: {exe_path} ---")
        elif not os.path.exists(model_base_path) or not os.path.isdir(model_base_path):
            print(f"\n--- 警告: (!!) 未找到 'cuda_mujoco' [WorldScaling] 的模型基础路径, 跳过。")
            print(f"--- (!!) 请检查配置变量 'CUDA_NWORLD_MODEL_BASE_PATH': {model_base_path} ---")
        else:
            
            for count in NWORLD_SCALING_LIST:
                # 根据 N 生成模型文件名
                if count == 1:
                    model_file = "humanoid.xml"
                else:
                    model_file = f"{count}_humanoid.xml" # 假设命名: 4_humanoid.xml
                
                model_path = os.path.join(model_base_path, model_file)

                if not os.path.exists(model_path):
                    print(f"\n--- 警告: 未找到模型 '{model_file}' (用于 {engine_name} - WorldScaling), 跳过。")
                    print(f"--- 检查路径: {model_path} ---")
                    continue

                command = [exe_path, model_path, str(DEFAULT_STEPS)]
                
                log_engine_name = f"{engine_name} (nworld_equivalent)"
                log_model_id = f"{model_file} ({count}a)"
                
                parsed_data = run_command_and_parse(
                    command, 
                    log_engine_name,
                    log_model_id,
                    is_powershell=False,
                    parser_type="exe"
                )
                
                # 添加上下文信息
                parsed_data["TestCategory"] = "WorldScaling"
                parsed_data["Engine"] = engine_name
                parsed_data["ModelType"] = "nworld_equivalent" # 新类型
                parsed_data["Humanoids"] = count # 变化的值
                parsed_data["Worlds"] = count # 标记为等效
                parsed_data["ModelFile"] = model_file
                
                results.append(parsed_data)
                all_fieldnames.update(parsed_data.keys())
    else:
        print("\n--- 警告: 未在 EXE_TEST_CONFIGS 中找到 'cuda_mujoco' 配置, 跳过 [WorldScaling] 测试。 ---")


    # --- 4. 写入CSV文件 ---
    if not results:
        print("\n没有收集到任何结果，程序退出。")
        return

    print(f"\n...测试完成。正在将结果写入 {OUTPUT_FILE}...")

    # (已修改) 确保新列的顺序一致且易于阅读
    ordered_fields = [
        "TestCategory", "Engine", "ModelType", 
        "Humanoids", "Worlds", "ModelFile", 
        "SimulationTime_s", "StepsPerSecond", "TimePerStep_us", "DOF",
        "Profiler_step_us", "Profiler_constraint_us", "Profiler_collision_us",
        "Error", "stderr"
    ]
    final_fieldnames = ordered_fields + sorted([f for f in all_fieldnames if f not in ordered_fields])

    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames, restval="N/A")
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n成功！所有结果已保存到 {OUTPUT_FILE}。")
        print("您可以直接使用 Excel 打开此文件进行分析。")

    except Exception as e:
        print(f"!! 错误: 写入CSV文件失败: {e}")


if __name__ == "__main__":
    main()