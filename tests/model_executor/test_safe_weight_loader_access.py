# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests that model load_weights methods use safe weight_loader access.

After process_weights_after_loading (e.g., AWQ quantization), specialized
Parameter subclasses (with weight_loader property) may be replaced with
plain nn.Parameter (without weight_loader). The load_weights methods
should use getattr(param, "weight_loader", default_weight_loader) instead
of directly accessing param.weight_loader to avoid AttributeError.
"""

import ast
import os

import pytest


def _get_model_files():
    """Get all Python files in the models directory."""
    models_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "vllm",
        "model_executor",
        "models",
    )
    models_dir = os.path.normpath(models_dir)
    model_files = []
    for fname in sorted(os.listdir(models_dir)):
        if fname.endswith(".py") and not fname.startswith("_"):
            model_files.append(os.path.join(models_dir, fname))
    return model_files


def _find_unsafe_weight_loader_access(filepath: str) -> list[int]:
    """
    Find lines that access param.weight_loader directly without getattr.

    Searches for the pattern:
        weight_loader = param.weight_loader
    which should instead be:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)

    Returns list of line numbers with unsafe access.
    """
    unsafe_lines = []
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=filepath)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        # Look for: weight_loader = <something>.weight_loader
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "weight_loader"
        ):
            # Check target is a simple name 'weight_loader'
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "weight_loader":
                    # The value is ast.Attribute (direct access), not
                    # ast.Call (getattr), so this is unsafe
                    unsafe_lines.append(node.lineno)
    return unsafe_lines


class TestSafeWeightLoaderAccess:
    """Ensure all model files use safe weight_loader access patterns."""

    def test_no_unsafe_param_weight_loader_access(self):
        """
        Verify that no model file directly accesses param.weight_loader.

        Models should use:
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
        instead of:
            weight_loader = param.weight_loader

        The direct access pattern fails after process_weights_after_loading
        replaces specialized Parameter subclasses (e.g., PackedvLLMParameter
        with weight_loader property) with plain nn.Parameter objects.
        """
        violations = {}
        for filepath in _get_model_files():
            unsafe_lines = _find_unsafe_weight_loader_access(filepath)
            if unsafe_lines:
                fname = os.path.basename(filepath)
                violations[fname] = unsafe_lines

        if violations:
            msg_parts = [
                "Found unsafe param.weight_loader access "
                "(should use getattr with default_weight_loader fallback):"
            ]
            for fname, lines in sorted(violations.items()):
                lines_str = ", ".join(str(ln) for ln in lines)
                msg_parts.append(f"  {fname}: line(s) {lines_str}")
            pytest.fail("\n".join(msg_parts))
