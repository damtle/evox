import os
from pathlib import Path


def create_rlec_structure(base_path="rlec"):
    """
    一键创建 RLEC (Reinforcement Learning for Evolutionary Control) 框架的目录与空白文件
    """
    # 定义目录树结构
    structure = {
        "": ["__init__.py"],
        "algorithms": ["__init__.py", "rlec_wrapper.py", "code_rlec.py"],
        "control": ["__init__.py", "intent_vector.py", "interpreter.py"],
        "features": ["__init__.py", "macro_state.py", "stage_reward.py"],
        "rl": ["__init__.py", "ppo.py", "networks.py", "rollout_buffer.py"],
        "utils": ["__init__.py", "population_metrics.py"]
    }

    base_dir = Path(base_path)

    print(f"{'=' * 50}")
    print(f"🚀 开始构建 RLEC 空间站骨架...")
    print(f"{'=' * 50}\n")

    for folder, files in structure.items():
        folder_path = base_dir / folder
        # 创建文件夹
        folder_path.mkdir(parents=True, exist_ok=True)

        if folder != "":
            print(f"📁 创建目录: {folder_path}/")
        else:
            print(f"📁 创建根目录: {folder_path}/")

        # 创建空白的 .py 文件
        for file in files:
            file_path = folder_path / file
            file_path.touch(exist_ok=True)
            print(f"   📄 创建文件: {file}")

    print(f"\n{'=' * 50}")
    print("✅ RLEC 顶级目录与文件骨架已全部初始化完毕！")
    print("🚩 我们现在可以正式开始 Sprint 1 的代码编写了！")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    create_rlec_structure()