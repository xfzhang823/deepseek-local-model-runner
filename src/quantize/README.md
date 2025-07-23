### ✅ Scale Format Examples

#### 1. **Grouped Scale Entries** (before flattening)
Each entry contains:
- A common layer `name`
- A list of submodule `subnames`
- A shared tensor `value`

```python
[
    ScaleEntry(
        name="model.model.layers.0.input_layernorm",
        subnames=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        value=tensor([...])
    ),
    ScaleEntry(
        name="model.model.layers.0.post_attention_layernorm",
        subnames=["mlp.gate_proj", "mlp.up_proj"],
        value=tensor([...])
    ),
    ScaleEntry(
        name="model.model.layers.0.mlp.up_proj",
        subnames=["mlp.down_proj"],
        value=tensor([...])
    )
]
```

#### 2. **Flattened Scale Entries** (after expanding nested structure)
Each tuple combines the full parameter path and its individual scale tensor:

```python
[
    ("model.model.layers.0.input_layernorm.self_attn.q_proj", tensor([...])),
    ("model.model.layers.0.input_layernorm.self_attn.k_proj", tensor([...])),
    ("model.model.layers.0.input_layernorm.self_attn.v_proj", tensor([...])),
    # ...
]
```


### Input Activations X (batch of size B)

**INPUT DIMENSIONS →**

```
 ┌────────────────────────────────────────────────────────────────────┐
 │ group_0 │ group_1 │ ... │ group_11                                 │
 │ [0–127] │ [128–255]       [1408–1535]                              │
 ├────────────────────────────────────────────────────────────────────┤
 │        x[0, :]                →  1st token hidden state            │
 │        x[1, :]                →  2nd token hidden state            │
 │        ...                       B rows                            │
 └────────────────────────────────────────────────────────────────────┘
X ∈ [B, 1536]
```

---

### Weight Matrix W (used as Wᵀ in `X @ Wᵀ`)

**INPUT DIMENSIONS →**

```
 ┌────────────────────────────────────────────────────────────────────┐
 │ group_0 │ group_1 │ ... │ group_11                                 │
 │ [0–127] │ [128–255]       [1408–1535]                              │
 ├────────────────────────────────────────────────────────────────────┤
 │        w[0, :]                →  1st output neuron                 │
 │        w[1, :]                →  2nd output neuron                 │
 │        ...                       1536 rows                         │
 └────────────────────────────────────────────────────────────────────┘
W ∈ [1536, 1536]
```

## Attention Layers
```text
           [input]
              |
      ┌───────┴──────┐
      |       |      |
   q_proj   k_proj v_proj
      |       |      |
      |       |      |
      └─────Attention─────┐
                          |
                    [attn output]
                          |
                       o_proj
                          |
                     [output to next layer]
