from vllm_rollout import checkpoint_name


def test_text_model_params_use_conditional_checkpoint_layout():
    assert checkpoint_name("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert checkpoint_name("model.layers.0.linear_attn.in_proj_qkv.weight") == (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    )
    assert checkpoint_name("model.layers.3.self_attn.q_proj.weight") == (
        "model.language_model.layers.3.self_attn.q_proj.weight"
    )


def test_lm_head_is_left_for_vllm_mapper():
    # The conditional model's mapper rewrites lm_head.* itself; it is not nested under model.
    assert checkpoint_name("lm_head.weight") == "lm_head.weight"


def test_unknown_prefix_passes_through():
    assert checkpoint_name("foo.bar.weight") == "foo.bar.weight"
