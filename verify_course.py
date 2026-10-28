from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = [
    "README.md",
    "tutorial/00_环境精度与性能测量.md",
    "tutorial/01_PyTorch张量与形状.md",
    "tutorial/02_自动求导与反向传播.md",
    "tutorial/03_线性层与激活函数.md",
    "tutorial/04_Transformer_FFN与Qwen3_SwiGLU.md",
    "tutorial/05_nano-vLLM真实计算链.md",
    "tutorial/06_算子融合原理与实践.md",
    "tutorial/07_Profile_Triton与CUDA进阶.md",
    "labs/common.py",
    "labs/triton_silu_mul.py",
    "labs/triton_linear_relu.py",
    "labs/09_linear_relu_fusion.py",
]


def main() -> int:
    errors: list[str] = []
    for relative in EXPECTED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing file: {relative}")

    python_paths = [
        *sorted(ROOT.glob("*.py")),
        *sorted((ROOT / "labs").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]
    for path in python_paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"syntax error: {exc}")

    link_pattern = re.compile(r"\]\(([^)]+)\)")
    for path in [ROOT / "README.md", *sorted((ROOT / "tutorial").glob("*.md"))]:
        for target in link_pattern.findall(path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            local = (path.parent / target.split("#", 1)[0]).resolve()
            if not local.exists():
                errors.append(f"broken local link: {path.relative_to(ROOT)} -> {target}")

    source_repo = ROOT / "reference" / "nano-vllm"
    if source_repo.exists():
        commit = subprocess.check_output(
            ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        print(f"nano-vLLM reference commit: {commit}")

    if errors:
        print("Course verification FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Course verification PASS: {len(EXPECTED)} required files, Python AST and Markdown links")
    return 0


if __name__ == "__main__":
    sys.exit(main())

