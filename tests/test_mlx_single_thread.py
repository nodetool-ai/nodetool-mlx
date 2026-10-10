"""Guard: all MLX work must run on the single dedicated MLX thread.

MLX >= 0.32 only lets the thread that created lazily loaded arrays/streams evaluate
them, so offloading to asyncio's multi-thread default pool breaks model reuse.
"""

import ast
import pathlib

import nodetool.nodes.mlx as mlx_nodes
from nodetool.nodes.mlx._mlx_thread import MLX_EXECUTOR

NODES_DIR = pathlib.Path(list(mlx_nodes.__path__)[0])


def test_mlx_executor_is_single_threaded():
    assert MLX_EXECUTOR._max_workers == 1


def test_all_executor_calls_use_mlx_thread():
    offenders = []
    for path in sorted(NODES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            if node.func.attr == "run_in_executor":
                first = node.args[0] if node.args else None
                if not (isinstance(first, ast.Name) and first.id == "MLX_EXECUTOR"):
                    offenders.append(f"{path.name}:{node.lineno} run_in_executor")
            elif node.func.attr == "to_thread":
                offenders.append(f"{path.name}:{node.lineno} asyncio.to_thread")
    assert not offenders, "MLX work must use MLX_EXECUTOR:\n" + "\n".join(offenders)
