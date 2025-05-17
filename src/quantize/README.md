
        # ☑️ Add prefix -> this format:
        # [
        #   ScaleEntry(name="model.model.layers.0.input_layernorm", subnames=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], value=tensor([...])),
        #   ScaleEntry(name="model.model.layers.0.post_attention_layernorm", subnames=["mlp.gate_proj", "mlp.up_proj"], value=tensor([...])),
        #   ScaleEntry(name="model.model.layers.0.mlp.up_proj", subnames=["mlp.down_proj"], value=tensor([...]))
        # ]

        # ☑️ flatten nested before saving -> expected format (in tuples already):
        # [
        #   ("model.model.layers.0.input_layernorm.self_attn.q_proj", tensor([...]))
        #   ("model.model.layers.0.input_layernorm.self_attn.k_proj", tensor([...]))
        #   ("model.model.layers.0.input_layernorm.self_attn.v_proj", tensor([...]))
        # ]