from vllm_rollout import match_weight_names


def test_matches_by_suffix_across_multimodal_prefixes():
    vllm_names = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.lm_head.weight",
        "visual.blocks.0.norm1.weight",
    ]
    hf_names = ["model.embed_tokens.weight", "model.layers.0.linear_attn.A_log"]

    assert match_weight_names(vllm_names, hf_names) == {
        "model.embed_tokens.weight": "language_model.model.embed_tokens.weight",
        "model.layers.0.linear_attn.A_log": "language_model.model.layers.0.linear_attn.A_log",
    }


def test_exact_names_pass_through_and_unmatched_are_skipped():
    vllm_names = ["model.norm.weight", "visual.blocks.0.norm1.weight"]
    hf_names = ["model.norm.weight", "model.not_in_vllm.weight", "visual.blocks.0.norm1.weight"]

    mapping = match_weight_names(vllm_names, hf_names)

    assert mapping["model.norm.weight"] == "model.norm.weight"
    assert mapping["visual.blocks.0.norm1.weight"] == "visual.blocks.0.norm1.weight"
    assert "model.not_in_vllm.weight" not in mapping


def test_ambiguous_suffix_prefers_shortest_match():
    vllm_names = ["visual.model.norm.weight", "model.norm.weight"]
    assert match_weight_names(vllm_names, ["model.norm.weight"]) == {"model.norm.weight": "model.norm.weight"}
