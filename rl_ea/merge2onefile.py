import os


def generate_directory_tree(startpath):
    """生成目录树结构的字符串"""
    tree_str = "### 1. 目录框架 (Directory Structure)\n"
    tree_str += "--------------------------------------\n"
    for root, dirs, files in os.walk(startpath):
        # 排除脚本自身和隐藏文件夹
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree_str += f"{indent}{os.path.basename(root)}/\n"
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if f.endswith('.py') and f != 'merge_py.py':
                tree_str += f"{sub_indent}{f}\n"
    return tree_str


def merge_python_files(output_file="merged_code2.txt"):
    current_dir = os.getcwd()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 第一步：写入目录树
        outfile.write(generate_directory_tree(current_dir))
        outfile.write("\n\n### 2. 代码详情 (Code Details)\n")
        outfile.write("======================================\n\n")

        # 第二步：递归遍历并写入代码
        for root, _, files in os.walk(current_dir):
            for file in files:
                if file.endswith('.py') and file != 'merge_py.py':
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, current_dir)

                    outfile.write(f"// File: {relative_path}\n")
                    outfile.write("-" * 40 + "\n")

                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}\n")

                    outfile.write("\n\n" + "=" * 50 + "\n\n")

    print(f"成功！所有 .py 文件已合并至: {output_file}")


if __name__ == "__main__":
    merge_python_files()