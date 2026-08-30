"""The README's examples are executable, and stay that way.

Every ```python block in the README asserts its own expected output, so running
them is enough to catch documentation that has drifted from the code.
"""

import pathlib
import re

import pytest

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def python_blocks():
    source = README.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", source, re.DOTALL)


def test_readme_has_examples():
    assert len(python_blocks()) >= 6


@pytest.mark.parametrize("index", range(len(python_blocks())))
def test_readme_block_runs(index):
    block = python_blocks()[index]
    exec(compile(block, f"{README.name} block {index + 1}", "exec"), {})
