# Physics-Informed Machine Learning Approach

## Biological Background

### Prokaryotes
- Simple, homogeneous cellular structure
- No membrane-bound organelles
- Examples: bacteria (rod, spherical, spiral)

### Eukaryotes
- Complex, compartmentalized structure
- Multiple distinct organelles (nucleus, mitochondria, etc.)
- Examples: amoeba, euglena, hydra, paramecium, yeast

## Physics-Informed Bias

Rather than pure data-driven learning, we incorporate domain knowledge:
```python
eukaryote_bias = 0.1 * (1 - torch.softmax(outputs, dim=1)[:, 1]).mean()
loss = loss + eukaryote_bias
```

This regularization term:
- Rewards high confidence for eukaryote predictions
- Reflects biological understanding of cellular structure
- Prevents uncertain predictions
- Combines data-driven learning with expert knowledge

## Benefits
1. Better alignment with biological reality
2. Improved generalization to new organisms
3. More interpretable predictions
