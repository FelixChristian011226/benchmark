"""
场景转换模块：为不同仿真引擎生成定制化的场景文件副本。

根据各引擎对 <option> 标签的不同要求，自动修改 XML 配置。
"""

import os
import shutil
import re
from typing import Dict, Optional, List


# 各引擎的默认 option 配置
DEFAULT_ENGINE_OPTIONS = {
    "mujoco": {
        "integrator": "implicit",
        "jacobian": None,  # None 表示移除该属性
    },
    "cuda_mujoco": {
        "integrator": "implicit",
        "jacobian": "dense",
    },
    "mjx": {
        "integrator": "implicitfast",
        "jacobian": "dense",
    },
    "mujoco_warp": {
        "integrator": "implicitfast",
        "jacobian": "dense",
    }
}


def modify_option_tag(xml_content: str, engine_options: Dict[str, Optional[str]]) -> str:
    """
    修改 XML 内容中的 <option> 标签属性。
    
    Args:
        xml_content: XML 文件内容
        engine_options: 引擎的 option 配置，如 {"integrator": "implicit", "jacobian": None}
                       值为 None 时移除该属性
    
    Returns:
        修改后的 XML 内容
    """
    # 匹配 <option ... /> 标签
    option_pattern = r'(<option\s+[^>]*)/>'
    
    def replace_option(match):
        option_tag = match.group(1)
        
        for attr_name, attr_value in engine_options.items():
            # 匹配已有的属性
            attr_pattern = rf'{attr_name}="[^"]*"'
            
            if attr_value is None:
                # 移除属性
                option_tag = re.sub(rf'\s*{attr_pattern}', '', option_tag)
            elif re.search(attr_pattern, option_tag):
                # 替换已有属性
                option_tag = re.sub(attr_pattern, f'{attr_name}="{attr_value}"', option_tag)
            else:
                # 添加新属性（在 <option 后面添加）
                option_tag = option_tag.replace('<option ', f'<option {attr_name}="{attr_value}" ', 1)
        
        return option_tag + '/>'
    
    return re.sub(option_pattern, replace_option, xml_content)


def prepare_scenes_for_all_engines(
    source_dir: str,
    temp_dir: str,
    enabled_engines: List[str],
    engine_options: Dict[str, Dict] = None
) -> Dict[str, str]:
    """
    为所有启用的引擎准备场景文件。
    
    Args:
        source_dir: 源场景目录路径
        temp_dir: 临时目录路径
        enabled_engines: 启用的引擎名称列表
        engine_options: 自定义引擎 option 配置，默认使用 DEFAULT_ENGINE_OPTIONS
    
    Returns:
        字典，key 为引擎名，value 为该引擎的场景目录路径
    """
    if engine_options is None:
        engine_options = DEFAULT_ENGINE_OPTIONS
    
    # 1. 清空临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"📁 准备场景文件...")
    print(f"   源目录: {source_dir}")
    print(f"   临时目录: {temp_dir}")
    
    engine_scene_dirs = {}
    
    # 2. 为每个启用的引擎复制并修改场景
    for engine_name in enabled_engines:
        if engine_name not in engine_options:
            print(f"   ⚠️ 跳过未知引擎: {engine_name}")
            continue
        
        engine_temp_dir = os.path.join(temp_dir, engine_name)
        
        # 完整复制源目录
        shutil.copytree(source_dir, engine_temp_dir)
        
        # 扫描并修改所有 XML 文件
        xml_count = 0
        for root, dirs, files in os.walk(engine_temp_dir):
            for filename in files:
                if filename.endswith('.xml'):
                    filepath = os.path.join(root, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    modified_content = modify_option_tag(content, engine_options[engine_name])
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    xml_count += 1
        
        print(f"   ✓ {engine_name}: 已处理 {xml_count} 个 XML 文件")
        engine_scene_dirs[engine_name] = engine_temp_dir
    
    print()
    return engine_scene_dirs


if __name__ == "__main__":
    # 测试用例
    test_xml = '''<mujoco model="test">
  <option timestep="0.005" solver="CG" integrator="implicitfast" jacobian="dense"/>
  <worldbody>
    <geom name="floor" size="1 1 .05" type="plane"/>
  </worldbody>
</mujoco>'''
    
    print("原始 XML:")
    print(test_xml)
    print()
    
    # 测试 mujoco 配置（移除 jacobian）
    mujoco_result = modify_option_tag(test_xml, DEFAULT_ENGINE_OPTIONS["mujoco"])
    print("mujoco 配置后:")
    print(mujoco_result)
    print()
    
    # 测试 cuda_mujoco 配置
    cuda_result = modify_option_tag(test_xml, DEFAULT_ENGINE_OPTIONS["cuda_mujoco"])
    print("cuda_mujoco 配置后:")
    print(cuda_result)
