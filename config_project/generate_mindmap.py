import yaml
import os

# 读取YAML文件
yaml_path = os.path.join(os.path.dirname(__file__), 'signal_dictionary.yaml')
with open(yaml_path, 'r', encoding='utf-8') as f:
    signal_dict = yaml.safe_load(f)

# 构建层级结构：L0 -> L1 -> L2 -> L3
structure = {}

for key, value in signal_dict.items():
    if isinstance(value, dict):
        l1 = value.get('L1_factor')
        l2 = value.get('L2_factor')
        l3 = value.get('L3_factor', key)
        
        if l1 and l2 and l3:
            if l1 not in structure:
                structure[l1] = {}
            if l2 not in structure[l1]:
                structure[l1][l2] = []
            if l3 not in structure[l1][l2]:
                structure[l1][l2].append(l3)

# 生成文本格式的思维导图
def generate_text_mindmap(structure):
    """生成文本格式的思维导图"""
    lines = []
    lines.append("=" * 80)
    lines.append("信号因子层级结构思维导图")
    lines.append("=" * 80)
    lines.append("")
    lines.append("L0signal")
    lines.append("│")
    
    l1_list = sorted(structure.keys())
    for i, l1 in enumerate(l1_list):
        is_last_l1 = (i == len(l1_list) - 1)
        prefix = "└── " if is_last_l1 else "├── "
        lines.append(prefix + f"L1: {l1}")
        
        l2_list = sorted(structure[l1].keys())
        for j, l2 in enumerate(l2_list):
            is_last_l2 = (j == len(l2_list) - 1)
            l1_prefix = "    " if is_last_l1 else "│   "
            l2_prefix = "└── " if is_last_l2 else "├── "
            lines.append(l1_prefix + l2_prefix + f"L2: {l2}")
            
            l3_list = sorted(structure[l1][l2])
            for k, l3 in enumerate(l3_list):
                is_last_l3 = (k == len(l3_list) - 1)
                l2_prefix_space = "    " if is_last_l2 else "│   "
                l1_prefix_space = "    " if is_last_l1 else "│   "
                l3_prefix = "└── " if is_last_l3 else "├── "
                lines.append(l1_prefix_space + l2_prefix_space + l3_prefix + f"L3: {l3}")
    
    return "\n".join(lines)

# 生成Mermaid格式的思维导图（可以在Markdown中渲染）
def generate_mermaid_mindmap(structure):
    """生成Mermaid格式的思维导图"""
    lines = []
    lines.append("```mermaid")
    lines.append("mindmap")
    lines.append("  root((L0signal))")
    
    for l1 in sorted(structure.keys()):
        lines.append(f"    {l1}")
        for l2 in sorted(structure[l1].keys()):
            lines.append(f"      {l2}")
            for l3 in sorted(structure[l1][l2]):
                lines.append(f"        {l3}")
    
    lines.append("```")
    return "\n".join(lines)

# 生成更详细的文本格式思维导图（带统计信息）
def generate_detailed_text_mindmap(structure):
    """生成详细的文本格式思维导图，包含统计信息"""
    lines = []
    lines.append("=" * 80)
    lines.append("信号因子层级结构思维导图")
    lines.append("=" * 80)
    lines.append("")
    
    # 统计信息
    total_l1 = len(structure)
    total_l2 = sum(len(l2_dict) for l2_dict in structure.values())
    total_l3 = sum(len(l3_list) for l2_dict in structure.values() for l3_list in l2_dict.values())
    
    lines.append(f"统计信息：")
    lines.append(f"  L1因子数量: {total_l1}")
    lines.append(f"  L2因子数量: {total_l2}")
    lines.append(f"  L3因子数量: {total_l3}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    lines.append("L0signal")
    lines.append("│")
    
    l1_list = sorted(structure.keys())
    for i, l1 in enumerate(l1_list):
        is_last_l1 = (i == len(l1_list) - 1)
        prefix = "└── " if is_last_l1 else "├── "
        l2_count = len(structure[l1])
        lines.append(prefix + f"L1: {l1} ({l2_count}个L2因子)")
        
        l2_list = sorted(structure[l1].keys())
        for j, l2 in enumerate(l2_list):
            is_last_l2 = (j == len(l2_list) - 1)
            l1_prefix = "    " if is_last_l1 else "│   "
            l2_prefix = "└── " if is_last_l2 else "├── "
            l3_count = len(structure[l1][l2])
            lines.append(l1_prefix + l2_prefix + f"L2: {l2} ({l3_count}个L3因子)")
            
            l3_list = sorted(structure[l1][l2])
            for k, l3 in enumerate(l3_list):
                is_last_l3 = (k == len(l3_list) - 1)
                l2_prefix_space = "    " if is_last_l2 else "│   "
                l1_prefix_space = "    " if is_last_l1 else "│   "
                l3_prefix = "└── " if is_last_l3 else "├── "
                lines.append(l1_prefix_space + l2_prefix_space + l3_prefix + f"L3: {l3}")
    
    return "\n".join(lines)

# 生成并保存
text_mindmap = generate_text_mindmap(structure)
detailed_text_mindmap = generate_detailed_text_mindmap(structure)
mermaid_mindmap = generate_mermaid_mindmap(structure)

# 保存到文件
output_path = os.path.join(os.path.dirname(__file__), 'signal_mindmap.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(detailed_text_mindmap)

output_mermaid_path = os.path.join(os.path.dirname(__file__), 'signal_mindmap.md')
with open(output_mermaid_path, 'w', encoding='utf-8') as f:
    f.write("# 信号因子层级结构思维导图\n\n")
    f.write(mermaid_mindmap)
    f.write("\n\n## 文本格式（详细版）\n\n")
    f.write("```\n")
    f.write(detailed_text_mindmap)
    f.write("\n```\n")

print("思维导图已生成！")
print("\n文本格式已保存到: signal_mindmap.txt")
print("Mermaid格式已保存到: signal_mindmap.md")
print("\n" + "=" * 80)
print(detailed_text_mindmap)

