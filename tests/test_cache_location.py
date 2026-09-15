"""Cache của HuggingFace phải nằm trong project, không rơi ra ~/.cache.

Đây là test chống hồi quy cho một lỗi tốn đĩa rất khó thấy: chỉ cần import
sai thứ tự (huggingface_hub được import trước qwen_lightning) là HF_HOME quay
về mặc định và hàng chục GB weight rơi vào ~/.cache/huggingface.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"


def _run(snippet: str, extra_env: dict[str, str] | None = None) -> str:
    """Chạy snippet trong tiến trình con sạch env HF_*, trả về stdout.

    src/ được nhét vào sys.path ngay trong snippet thay vì qua PYTHONPATH:
    PYTHONPATH tách bằng dấu ":" nên hỏng nếu project nằm dưới đường dẫn có
    chứa ":" (ví dụ một mount gvfs/sftp).
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_")}
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(SRC)!r})\n{snippet}",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


def test_hf_home_is_inside_project():
    out = _run(
        "import qwen_lightning, os; print(os.environ['HF_HOME'])"
    )
    assert Path(out).is_relative_to(PROJECT_ROOT), out


def test_hf_home_not_in_user_cache():
    out = _run(
        "import qwen_lightning, os; print(os.environ['HF_HOME'])"
    )
    assert ".cache/huggingface" not in out, out


def test_hub_cache_follows_hf_home():
    out = _run(
        "import qwen_lightning, os; "
        "print(os.environ['HF_HUB_CACHE'])"
    )
    assert Path(out).is_relative_to(PROJECT_ROOT), out


def test_explicit_hf_home_is_respected():
    """Docker set HF_HOME riêng (kèm HF_HUB_OFFLINE=1) — không được ghi đè."""
    out = _run(
        "import qwen_lightning, os; print(os.environ['HF_HOME'])",
        {"HF_HOME": "/opt/custom-hf"},
    )
    assert out == "/opt/custom-hf"


def _download_ast() -> ast.Module:
    source = (SRC / "qwen_lightning" / "download.py").read_text("utf-8")
    return ast.parse(source)


def test_download_never_calls_login():
    """login() ghi token ra $HF_HOME/token — tạo state trên đĩa ngoài ý muốn.

    Token phải được truyền thẳng vào từng lời gọi download thay vì login().
    Kiểm bằng AST chứ không so khớp chuỗi, để chữ "login()" trong comment
    không làm test đỏ oan.
    """
    tree = _download_ast()

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "login" not in imported

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "login" not in called


def test_download_uses_local_dir_and_no_removed_kwarg():
    """Mọi lời gọi download phải ghim local_dir và bỏ kwarg đã bị xoá.

    ``local_dir_use_symlinks`` bị gỡ khỏi huggingface_hub 1.x; cả
    snapshot_download lẫn hf_hub_download đều là keyword-only và không nhận
    **kwargs, nên truyền vào là TypeError ngay lúc chạy.
    """
    tree = _download_ast()
    targets = {"snapshot_download", "hf_hub_download"}
    seen = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in targets:
            continue
        seen.add(node.func.id)
        kwargs = {kw.arg for kw in node.keywords}
        assert "local_dir" in kwargs, node.func.id
        assert "local_dir_use_symlinks" not in kwargs, node.func.id

    assert seen == targets, f"thiếu lời gọi: {targets - seen}"
