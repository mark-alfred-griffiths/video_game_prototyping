# README for `pc_jpc_tensorflow_npc_demo.py`

A single-file demonstration of a narrative NPC architecture built from:

- Predictive Coding (PC)
- JAX + JPC
- TensorFlow/Keras
- Bayesian belief distributions
- Free-energy-style inference dynamics

The project demonstrates how a predictive-coding latent representation can be used as the internal cognitive state of an NPC, while TensorFlow heads learn:

1. belief updates
2. dialogue policy selection

---

# Overview

This script implements a minimal cognitive game agent.

The NPC ("Alice") maintains uncertain beliefs about hidden narrative states called **engrams**.

The player makes dialogue choices.

Those choices generate observations with varying:

- strength
- reliability
- ambiguity

The NPC:

1. converts observations into numerical features,
2. encodes them through a predictive-coding network,
3. updates internal probabilistic beliefs,
4. estimates uncertainty,
5. computes free energy,
6. and selects narrative actions.

---

# High-Level Architecture

```text
player dialogue choice
        ↓
observation generation
        ↓
raw psychological feature vector
        ↓
JPC predictive-coding encoder
        ↓
latent representation
        ↓
TensorFlow belief-update head
        ↓
updated Beta belief distribution
        ↓
TensorFlow policy head
        ↓
NPC action selection
        ↓
dialogue response
```

---

# Core Concepts

## 1. Engrams

An engram represents a hidden psychological belief or narrative state.

Example:

```python
Engram(
    id="x_mattered",
    description="The protagonist cared about X more than they admit.",
    prior=0.40,
)
```

The NPC attempts to infer whether this hidden proposition is true.

---

# 2. Beta Belief Distribution

The NPC does not store beliefs as simple scalar values.

Instead, beliefs are represented using a Beta distribution:

```text
Beta(alpha, beta)
```

The belief mean is:

```text
belief = alpha / (alpha + beta)
```

The variance becomes a measure of uncertainty.

High concentration:
- confident belief

Low concentration:
- uncertain belief

---

# 3. Predictive Coding Encoder (JPC)

The predictive-coding network is implemented using:

- JAX
- Equinox
- Optax
- JPC

The encoder learns latent representations of:

- prediction error
- uncertainty
- social state
- belief state
- free energy

The model is created with:

```python
jpc.make_mlp(...)
```

and updated using:

```python
jpc.make_pc_step(...)
```

This is a genuine predictive-coding parameter update rather than standard backpropagation through a TensorFlow model.

---

# 4. Free Energy

The script manually computes a simplified variational free energy term:

```python
0.5 * evidence_precision * sensory_error²
+
0.5 * prior_precision * prior_error²
```

Where:

```text
sensory_error = observation - belief
prior_error   = belief - prior
```

Lower free energy corresponds to more internally coherent beliefs.

---

# 5. TensorFlow Heads

The TensorFlow model contains two heads:

## Belief Head

Predicts:

```text
delta-alpha
delta-beta
```

These update the Beta distribution.

## Policy Head

Predicts logits over:

```text
DISMISS
PROBE
REVEAL
CONFRONT
```

The highest logit becomes the NPC action.

---

# Runtime Flow

1. Player selects dialogue.
2. Observation is generated.
3. Raw features are created.
4. JPC encodes the features.
5. TensorFlow predicts belief updates.
6. Beta distribution updates.
7. TensorFlow predicts policy logits.
8. NPC selects an action.
9. Dialogue response is generated.

---

# Dependencies

Install requirements:

```bash
pip install tensorflow equinox optax "jax[cpu]"
pip install git+https://github.com/thebuckleylab/jpc.git
```

---

# Running

Run scripted demo:

```bash
python pc_jpc_tensorflow_npc_demo.py
```

Run interactive mode:

```bash
python pc_jpc_tensorflow_npc_demo.py --interactive
```

---

# Educational Purpose

This repository is intentionally compact and interpretable.

It demonstrates:

- predictive coding
- probabilistic beliefs
- latent representation learning
- free-energy-style reasoning
- NPC cognitive architectures

rather than production-scale game AI.

---

# Future Extensions

Possible future directions include:

- recurrent predictive coding
- active inference planning
- Kalman/Kalman-Bucy filtering
- differentiable memory systems
- multi-agent inference
- generative dialogue models
