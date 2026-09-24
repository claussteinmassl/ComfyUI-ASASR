"""Real key inventory of the ASASR checkpoints (recorded 2026-08-05).

Both files share the same 13 module patterns; ranks differ (SR=128, DPO=16).
"""

DOUBLE_PATTERNS = [
    "transformer_blocks.{}.attn.to_q",
    "transformer_blocks.{}.attn.to_k",
    "transformer_blocks.{}.attn.to_v",
    "transformer_blocks.{}.attn.to_out.0",
    "transformer_blocks.{}.ff.net.2",
    "transformer_blocks.{}.norm1.linear",
]
SINGLE_PATTERNS = [
    "single_transformer_blocks.{}.attn.to_q",
    "single_transformer_blocks.{}.attn.to_k",
    "single_transformer_blocks.{}.attn.to_v",
    "single_transformer_blocks.{}.norm.linear",
    "single_transformer_blocks.{}.proj_mlp",
    "single_transformer_blocks.{}.proj_out",
]
TOP_LEVEL = ["x_embedder"]
N_DOUBLE = 19
N_SINGLE = 38


def all_module_paths(n_double=N_DOUBLE, n_single=N_SINGLE):
    """Yields every diffusers module path present in the real checkpoints."""
    for i in range(n_double):
        for p in DOUBLE_PATTERNS:
            yield p.format(i)
    for i in range(n_single):
        for p in SINGLE_PATTERNS:
            yield p.format(i)
    yield from TOP_LEVEL
