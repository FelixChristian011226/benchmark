import subprocess
import re
import pandas as pd
import os
import time
from scene_converter import prepare_scenes_for_all_engines, DEFAULT_ENGINE_OPTIONS

# ================= 配置区域 =================

CONFIG = {
    "global_steps": 1000,  # 全局测试步数
    "ctrlnoise": 0.4,      # 控制噪声幅度 (0.0-1.0)，用于增加场景复杂度
    
    # [新增] 统一的源场景目录（所有引擎共用）
    "source_scene_dir": "scenes/humanoid/generated_dense_rings/",
    
    # [新增] 临时目录，存放各引擎的场景副本
    "temp_dir": "temp",
    
    # [移除] 这里的 scenes 列表不再需要，脚本会自动去下面的文件夹里找
    # "scenes": [...], 

    # 引擎配置
    # scene_prefix 将由脚本自动设置为 temp/{engine_name}/
    # ctrlnoise 支持情况：
    #   - mujoco: ✅ 支持 (命令行参数)
    #   - cuda_mujoco: ✅ 支持 (命令行参数)
    #   - mjx: ❌ 不支持 
    #   - mujoco_warp: ❌ 不支持 
    "engines": {
        "mujoco": {
            "enabled": True,
            # testspeed 参数顺序: modelfile [nstep nthread ctrlnoise]
            "cmd_template": "mujoco/build/bin/testspeed {full_path} {steps} 1 {ctrlnoise}",
            "shell": False
        },
        "mjx": {
            "enabled": True,
            "cmd_template": "mjx-testspeed --mjcf {full_path} --base_path . --batch_size 1 --nstep {steps}",
            "shell": False
        },
        "mujoco_warp": {
            "enabled": True,
            "cmd_template": "source env/bin/activate && mjwarp-testspeed {full_path} --event_trace=True --nworld=1 --nstep={steps}",
            "cwd": "mujoco_warp",  # 切换工作目录
            "shell": True 
        },
        "cuda_mujoco": {
            "enabled": True,
            "cmd_template": "cuda_mujoco/build/bin/testspeed_cuda {full_path} {steps} 1 {ctrlnoise}",
            "shell": False
        }
    }
}

# ================= 解析逻辑 =================

def parse_output(engine_name, stdout_text):
    """
    根据不同引擎的输出格式解析关键数据。
    """
    data = {
        "Simulation Time (s)": None,
        "SPS": None,
        "RTF": None,
        "Time per Step (µs)": None
    }
    
    patterns = {}
    if engine_name == "mujoco":
        patterns = {
            "Simulation Time (s)": r"Simulation time\s+:\s+([\d\.]+)\s+s",
            "SPS": r"Steps per second\s+:\s+([\d\.]+)",
            "RTF": r"Realtime factor\s+:\s+([\d\.]+)\s+x",
            "Time per Step (µs)": r"Time per step\s+:\s+([\d\.]+)\s+µs"
        }
    elif engine_name == "mjx":
        patterns = {
            "Simulation Time (s)": r"Total simulation time:\s+([\d\.]+)\s+s",
            "SPS": r"Total steps per second:\s+([\d\.]+)",
            "RTF": r"Total realtime factor:\s+([\d\.]+)\s+x",
            "Time per Step (µs)": r"Total time per step:\s+([\d\.]+)\s+µs"
        }
    elif engine_name == "mujoco_warp":
        patterns = {
            "Simulation Time (s)": r"Total simulation time:\s+([\d\.]+)\s+s",
            "SPS": r"Total steps per second:\s+([\d\.]+)",
            "RTF": r"Total realtime factor:\s+([\d\.]+)\s+x",
            "Time per Step (ns)": r"Total time per step:\s+([\d\.]+)\s+ns" 
        }
    elif engine_name == "cuda_mujoco":
        patterns = {
            "Simulation Time (s)": r"Total wall time\s+:\s+([\d\.]+)\s+s",
            "SPS": r"Steps per second\s+:\s+([\d\.]+)",
            "RTF": r"Realtime factor\s+:\s+([\d\.]+)\s+x",
            "Time per Step (µs)": r"Time per step\s+:\s+([\d\.]+)\s+µs"
        }

    for key, pattern in patterns.items():
        match = re.search(pattern, stdout_text)
        if match:
            val = float(match.group(1))
            if key == "Time per Step (ns)":
                data["Time per Step (µs)"] = val / 1000.0
            else:
                data[key] = val

    return data

# ================= 主执行逻辑 (已修改) =================

def run_benchmarks():
    summary_results = []
    detailed_logs = []
    
    print(f"🚀 开始执行测试...")
    
    # 0. 获取启用的引擎列表
    enabled_engines = [
        name for name, cfg in CONFIG['engines'].items() 
        if cfg.get("enabled", True)
    ]
    
    if not enabled_engines:
        print("⚠️ 没有启用任何引擎")
        return summary_results, detailed_logs
    
    # 1. 准备场景文件（复制并修改）
    engine_scene_dirs = prepare_scenes_for_all_engines(
        source_dir=CONFIG['source_scene_dir'],
        temp_dir=CONFIG['temp_dir'],
        enabled_engines=enabled_engines
    )
    
    # 2. 遍历引擎进行测试
    for engine_name, engine_cfg in CONFIG['engines'].items():
        if not engine_cfg.get("enabled", True):
            continue
        
        print(f"\n[Engine] {engine_name}")

        # 3. 确定场景目录（从临时目录获取）
        base_cwd = engine_cfg.get("cwd", ".") 
        
        # 获取该引擎的临时场景目录
        if engine_name not in engine_scene_dirs:
            print(f"  ❌ Error: 引擎 {engine_name} 的场景目录未准备，跳过")
            continue
        
        temp_scene_dir = engine_scene_dirs[engine_name]
        
        # 对于有 cwd 的引擎，scene_prefix 需要是相对于 cwd 的路径
        if base_cwd != ".":
            scene_prefix = os.path.relpath(temp_scene_dir, base_cwd)
        else:
            scene_prefix = temp_scene_dir
        
        scan_dir = temp_scene_dir
        
        # 检查目录是否存在
        if not os.path.exists(scan_dir):
            print(f"  ❌ Error: 目录不存在，跳过: {scan_dir}")
            continue
            
        # 4. 扫描该目录下的所有 XML 文件
        try:
            files = [f for f in os.listdir(scan_dir) if f.endswith('.xml')]
            
            # === 修复的排序逻辑 ===
            # 逻辑：如果没有数字，视作 -1（排在最前）；如果有数字，按数字大小排。
            # 返回元组 (数字, 文件名) 确保类型一致且能处理同名冲突。
            def get_sort_key(filename):
                match = re.search(r'\d+', filename)
                if match:
                    return (int(match.group()), filename)
                return (-1, filename) # 也就是 humanoid.xml 会被视为 -1，排在 8_humanoids.xml 之前

            files.sort(key=get_sort_key)
            # ===================
            
            if not files:
                print(f"  ⚠️ Warning: 目录 {scan_dir} 下没有找到 .xml 文件")
                continue
                
            print(f"  -> 在 {scan_dir} 扫描到 {len(files)} 个场景文件")
            
        except Exception as e:
            print(f"  ❌ Error scanning directory: {e}")
            # 打印详细堆栈以便调试（可选）
            # import traceback
            # traceback.print_exc()
            continue

        # 5. 遍历找到的文件进行测试
        for filename in files:
            scene_name_no_ext = os.path.splitext(filename)[0]
            print(f"    -> Testing Scene: {scene_name_no_ext}")
            
            full_path_for_cmd = os.path.join(scene_prefix, filename)
            cwd = engine_cfg.get("cwd", os.getcwd())
            use_shell = engine_cfg.get("shell", False)
            
            cmd = engine_cfg["cmd_template"].format(
                full_path=full_path_for_cmd, 
                steps=CONFIG["global_steps"],
                xml_path=full_path_for_cmd,
                ctrlnoise=CONFIG.get("ctrlnoise", 0.01)
            )
            
            try:
                if use_shell:
                    process = subprocess.run(
                        cmd, 
                        shell=True, 
                        executable='/bin/bash',
                        cwd=cwd,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT, 
                        text=True
                    )
                else:
                    cmd_parts = cmd.split()
                    process = subprocess.run(
                        cmd_parts, 
                        cwd=cwd,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT, 
                        text=True
                    )
                
                output = process.stdout
                metrics = parse_output(engine_name, output)
                
                result_row = {
                    "Scene": scene_name_no_ext,
                    "Engine": engine_name,
                    "Steps": CONFIG["global_steps"],
                    **metrics
                }
                summary_results.append(result_row)
                
                detailed_logs.append({
                    "Scene": scene_name_no_ext,
                    "Engine": engine_name,
                    "Raw Output": output
                })
                
                print(f"       Done. SPS: {metrics.get('SPS', 'N/A')}")

            except Exception as e:
                print(f"       ERROR: {e}")
                summary_results.append({
                    "Scene": scene_name_no_ext,
                    "Engine": engine_name,
                    "Error": str(e)
                })

    return summary_results, detailed_logs

# ================= 保存逻辑 (保持不变) =================

def save_to_excel(summary, logs, filename="benchmark_results.xlsx"):
    if not summary:
        print("\n⚠️ 没有数据可保存")
        return

    df_summary = pd.DataFrame(summary)
    df_logs = pd.DataFrame(logs)
    
    # 调整列顺序
    cols = ["Scene", "Engine", "Steps", "Simulation Time (s)", "SPS", "RTF", "Time per Step (µs)"]
    existing_cols = [c for c in cols if c in df_summary.columns]
    # 把剩余的列（比如 Error）也加上
    remaining_cols = [c for c in df_summary.columns if c not in cols]
    df_summary = df_summary[existing_cols + remaining_cols]

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_logs.to_excel(writer, sheet_name='Detailed_Logs', index=False)
    
    print(f"\n✅ 测试完成！结果已保存至: {filename}")

if __name__ == "__main__":
    summary_data, log_data = run_benchmarks()
    save_to_excel(summary_data, log_data)