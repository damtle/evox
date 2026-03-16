import os

# ================= 配置区 =================
CONFIG = {
    # 1. 入口文件列表 (如: ["main.py"])。为空则不单独指定入口。
    "MAIN_FILES": ["test_relc.py"],

    # 2. 需要扫描的源码目录 (如: ["src", "utils"])。为空且 MAIN_FILES 也为空时，合并当前目录下所有 .py。
    "TARGET_DIRS": ["rlec"],

    # 3. 输出文件名
    "OUTPUT_FILE": "merged_code.txt"
}


# ==========================================

def generate_tree(path, level=0, exclude_files=None):
    """递归生成目录树"""
    if exclude_files is None: exclude_files = []
    if not os.path.exists(path): return ""

    indent = '    ' * level
    if os.path.isfile(path):
        return f"{indent}{os.path.basename(path)}\n"

    tree_str = f"{indent}{os.path.basename(path) or './'}/\n"
    try:
        items = sorted(os.listdir(path))
        for item in items:
            if item.startswith('.') or item == "__pycache__" or item in exclude_files:
                continue
            tree_str += generate_tree(os.path.join(path, item), level + 1, exclude_files)
    except Exception:
        pass
    return tree_str


def merge_files():
    output_file = CONFIG["OUTPUT_FILE"]
    main_files = CONFIG["MAIN_FILES"]
    target_dirs = CONFIG["TARGET_DIRS"]
    script_name = os.path.basename(__file__)

    # 自动判断：如果两个列表都为空，则扫描当前整个目录
    is_full_scan = not main_files and not target_dirs
    if is_full_scan:
        target_dirs = ["."]

    processed_paths = set()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # --- 第一步：生成目录结构 ---
        outfile.write("### 1. 项目结构概览\n" + "-" * 40 + "\n")
        if is_full_scan:
            outfile.write(generate_tree(".", exclude_files=[script_name, output_file]))
        else:
            for f in main_files: outfile.write(f"[Main] {f}\n")
            for d in target_dirs: outfile.write(generate_tree(d))

        outfile.write("\n\n### 2. 代码详细内容\n" + "=" * 50 + "\n\n")

        # --- 第二步：写入 Main 文件 ---
        for file_path in main_files:
            if os.path.exists(file_path):
                abs_path = os.path.abspath(file_path)
                write_content(file_path, outfile, "MAIN 入口")
                processed_paths.add(abs_path)

        # --- 第三步：写入目录下的文件 ---
        for directory in target_dirs:
            for root, _, files in os.walk(directory):
                for file in sorted(files):
                    # 基础过滤：必须是.py，不能是脚本自己，不能是输出文件
                    if file.endswith('.py') and file != script_name and file != output_file:
                        full_path = os.path.join(root, file)
                        abs_path = os.path.abspath(full_path)

                        if abs_path not in processed_paths:
                            # 计算相对路径，让显示更清晰
                            rel_path = os.path.relpath(full_path, os.getcwd())
                            write_content(full_path, outfile, "MODULE", rel_path)
                            processed_paths.add(abs_path)

    print(f"✅ 处理完成！模式: {'全量合并' if is_full_scan else '指定合并'}")
    print(f"📄 结果保存至: {output_file}")


def write_content(path, outfile, tag, display_path=None):
    display_path = display_path or path
    outfile.write(f"// [{tag}] File: {display_path}\n")
    outfile.write("-" * 40 + "\n")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            outfile.write(f.read())
    except Exception as e:
        outfile.write(f"Error reading: {e}\n")
    outfile.write("\n\n" + "=" * 50 + "\n\n")


if __name__ == "__main__":
    merge_files()