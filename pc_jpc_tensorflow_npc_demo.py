"""
pc_jpc_tensorflow_npc_demo.py

Single-file demo of an NPC whose internal model uses:

1. Actual JPC for the predictive-coding network
   - JPC/JAX model: `jpc.make_mlp(...)`
   - PC training step: `jpc.make_pc_step(...)`
   - The JPC network learns a predictive-coding latent representation of the
     current NPC belief state + observation.

2. TensorFlow/Keras for the NPC heads
   - Belief-update MLP head: predicts raw delta-alpha and raw delta-beta.
   - Policy head: predicts the NPC action.

Conceptual loop:

player choice
-> observation
-> raw PC input features
-> JPC predictive-coding encoder
-> TensorFlow MLP belief head
-> updated Beta belief distribution
-> TensorFlow policy head
-> NPC action/dialogue

Install dependencies:

    pip install tensorflow equinox optax "jax[cpu]"
    pip install git+https://github.com/thebuckleylab/jpc.git

Run:

    python pc_jpc_tensorflow_npc_demo.py
    python pc_jpc_tensorflow_npc_demo.py --interactive

Note:
This is a concept demo, not a production game architecture. It deliberately fails
with a clear dependency message if JPC or TensorFlow is unavailable, because the
point is to demonstrate actual JPC + actual TensorFlow rather than a fake fallback.
"""

from __future__ import annotations

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# -----------------------------------------------------------------------------
# Required ML dependencies
# -----------------------------------------------------------------------------


IMPORT_ERRORS: list[str] = []

try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    import jax.random as jr
except Exception as exc:  # pragma: no cover - dependency guard
    IMPORT_ERRORS.append(f"JAX import failed: {exc}")

try:
    import equinox as eqx
except Exception as exc:  # pragma: no cover - dependency guard
    IMPORT_ERRORS.append(f"Equinox import failed: {exc}")

try:
    import optax
except Exception as exc:  # pragma: no cover - dependency guard
    IMPORT_ERRORS.append(f"Optax import failed: {exc}")

try:
    import jpc
except Exception as exc:  # pragma: no cover - dependency guard
    IMPORT_ERRORS.append(f"JPC import failed: {exc}")

try:
    import numpy as np
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - dependency guard
    IMPORT_ERRORS.append(f"TensorFlow/NumPy import failed: {exc}")

if IMPORT_ERRORS:
    details = "\n".join(f"  - {msg}" for msg in IMPORT_ERRORS)
    raise SystemExit(
        "This demo requires actual JPC/JAX and TensorFlow.\n"
        "Install with:\n"
        "  pip install tensorflow equinox optax 'jax[cpu]'\n"
        "  pip install git+https://github.com/thebuckleylab/jpc.git\n\n"
        f"Import errors:\n{details}"
    )


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def softplus(x: float) -> float:
    if x > 30.0:
        return x
    return math.log1p(math.exp(x))


def inverse_softplus(y: float) -> float:
    y = max(1e-6, y)
    if y > 30.0:
        return y
    return math.log(math.exp(y) - 1.0)


# -----------------------------------------------------------------------------
# Story/model types
# -----------------------------------------------------------------------------


class NPCAction(str, Enum):
    DISMISS = "dismiss"
    PROBE = "probe"
    REVEAL = "reveal"
    CONFRONT = "confront"


@dataclass(slots=True)
class Engram:
    id: str
    description: str
    prior: float = 0.40


@dataclass(slots=True)
class Observation:
    engram_id: str
    strength: float
    reliability: float
    source: str

    @property
    def weighted_strength(self) -> float:
        return self.strength * self.reliability


@dataclass(slots=True)
class NPCState:
    name: str = "Alice"
    belief_alpha: dict[str, float] = field(default_factory=dict)
    belief_beta: dict[str, float] = field(default_factory=dict)
    memory: set[str] = field(default_factory=set)

    default_belief: float = 0.40
    default_concentration: float = 5.0

    evidence_precision: float = 2.5
    prior_precision: float = 2.0

    trust: float = 0.20
    suspicion: float = 0.50
    instability: float = 0.00

    def default_alpha_beta(self) -> tuple[float, float]:
        mean = clamp01(self.default_belief)
        c = max(1e-3, self.default_concentration)
        return max(1e-3, mean * c), max(1e-3, (1.0 - mean) * c)

    def get_params(self, engram_id: str) -> tuple[float, float]:
        if engram_id not in self.belief_alpha or engram_id not in self.belief_beta:
            alpha, beta = self.default_alpha_beta()
            self.belief_alpha[engram_id] = alpha
            self.belief_beta[engram_id] = beta
        return self.belief_alpha[engram_id], self.belief_beta[engram_id]

    def set_params(self, engram_id: str, alpha: float, beta: float) -> None:
        self.belief_alpha[engram_id] = max(1e-3, float(alpha))
        self.belief_beta[engram_id] = max(1e-3, float(beta))
        self.memory.add(engram_id)

    def belief(self, engram_id: str) -> float:
        alpha, beta = self.get_params(engram_id)
        return alpha / (alpha + beta)

    def variance(self, engram_id: str) -> float:
        alpha, beta = self.get_params(engram_id)
        total = alpha + beta
        return (alpha * beta) / ((total * total) * (total + 1.0))

    def uncertainty(self, engram_id: str) -> float:
        return clamp01(self.variance(engram_id) / 0.25)

    def confidence(self, engram_id: str) -> float:
        return 1.0 - self.uncertainty(engram_id)


# -----------------------------------------------------------------------------
# Actual JPC predictive-coding encoder
# -----------------------------------------------------------------------------


class JPCPredictiveCodingEncoder:
    """
    JPC-backed predictive-coding representation model.

    Input: raw NPC/observation feature vector.
    Output: learned PC latent vector.

    During demo training, `jpc.make_pc_step(...)` updates this encoder using PC.
    The TensorFlow heads then consume this latent vector.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        width: int,
        depth: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.key = jr.PRNGKey(seed)

        self.model = jpc.make_mlp(
            self.key,
            input_dim=input_dim,
            width=width,
            depth=depth,
            output_dim=latent_dim,
            act_fn="relu",
        )
        self.optim = optax.adam(learning_rate)
        self.opt_state = self.optim.init((eqx.filter(self.model, eqx.is_array), None))

    def train_pc_step(self, x: list[float], target_latent: list[float]) -> None:
        x_batch = jnp.asarray([x], dtype=jnp.float32)  # shape: (1, input_dim)
        target_batch = jnp.asarray([target_latent], dtype=jnp.float32)  # shape: (1, latent_dim)

        result = jpc.make_pc_step(
            model=self.model,
            optim=self.optim,
            opt_state=self.opt_state,
            output=target_batch,
            input=x_batch,
        )

        self.model = result["model"]
        self.opt_state = result["opt_state"]  #

    def encode(self, x: list[float]) -> "np.ndarray":
        y = jnp.asarray(x, dtype=jnp.float32)

        for block in self.model:
            y = block(y)

        return np.asarray(y, dtype=np.float32)


# -----------------------------------------------------------------------------
# TensorFlow heads
# -----------------------------------------------------------------------------


class TensorFlowNPCHeads:
    """
    TensorFlow/Keras model with two heads:
    - belief_head: raw delta-alpha, raw delta-beta
    - policy_head: logits over NPCAction
    """

    def __init__(self, *, latent_dim: int, hidden_dim: int, action_count: int, learning_rate: float) -> None:
        inputs = tf.keras.Input(shape=(latent_dim,), name="jpc_pc_latent")
        hidden = tf.keras.layers.Dense(hidden_dim, activation="relu", name="npc_hidden_1")(inputs)
        hidden = tf.keras.layers.Dense(hidden_dim, activation="relu", name="npc_hidden_2")(hidden)

        belief_raw_delta = tf.keras.layers.Dense(2, name="belief_raw_delta_alpha_beta")(hidden)
        policy_logits = tf.keras.layers.Dense(action_count, name="policy_logits")(hidden)

        self.model = tf.keras.Model(
            inputs=inputs,
            outputs={
                "belief_raw_delta": belief_raw_delta,
                "policy_logits": policy_logits,
            },
            name="tensorflow_npc_heads",
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @tf.function
    def _train_step(
        self,
        latent_batch: "tf.Tensor",
        belief_target_batch: "tf.Tensor",
        action_target_batch: "tf.Tensor",
    ) -> tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"]:
        with tf.GradientTape() as tape:
            outputs = self.model(latent_batch, training=True)
            belief_loss = self.mse(belief_target_batch, outputs["belief_raw_delta"])
            policy_loss = self.ce(action_target_batch, outputs["policy_logits"])
            total_loss = belief_loss + policy_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss, belief_loss, policy_loss

    def train_step(
        self,
        latent: "np.ndarray",
        belief_target: list[float],
        action_label: int,
        action_count: int,
    ) -> tuple[float, float, float]:
        latent_batch = tf.convert_to_tensor(latent.reshape(1, -1), dtype=tf.float32)
        belief_target_batch = tf.convert_to_tensor([belief_target], dtype=tf.float32)
        action_target = np.zeros((1, action_count), dtype=np.float32)
        action_target[0, action_label] = 1.0
        action_target_batch = tf.convert_to_tensor(action_target, dtype=tf.float32)

        total, belief, policy = self._train_step(latent_batch, belief_target_batch, action_target_batch)
        return float(total), float(belief), float(policy)

    def predict(self, latent: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        outputs = self.model(latent.reshape(1, -1), training=False)
        raw_delta = outputs["belief_raw_delta"].numpy()[0]
        logits = outputs["policy_logits"].numpy()[0]
        return raw_delta, logits


# -----------------------------------------------------------------------------
# Combined JPC + TensorFlow NPC controller
# -----------------------------------------------------------------------------


class JPCTensorFlowNPCController:
    action_labels = [NPCAction.DISMISS, NPCAction.PROBE, NPCAction.REVEAL, NPCAction.CONFRONT]

    raw_feature_names = [
        "belief_mean",
        "alpha_scaled",
        "beta_scaled",
        "uncertainty",
        "confidence",
        "prior",
        "evidence_strength",
        "reliability",
        "weighted_strength",
        "trust",
        "suspicion",
        "instability",
    ]

    def __init__(self, *, seed: int = 7) -> None:
        tf.keras.utils.set_random_seed(seed)
        self.pc_encoder = JPCPredictiveCodingEncoder(
            input_dim=len(self.raw_feature_names),
            latent_dim=8,
            width=32,
            depth=3,
            learning_rate=1e-3,
            seed=seed,
        )
        self.tf_heads = TensorFlowNPCHeads(
            latent_dim=8,
            hidden_dim=32,
            action_count=len(self.action_labels),
            learning_rate=1e-3,
        )
        self.last_trace: dict[str, object] = {}

    def raw_features(self, npc: NPCState, engram: Engram, obs: Observation) -> list[float]:
        alpha, beta = npc.get_params(engram.id)
        return [
            npc.belief(engram.id),
            alpha / 10.0,
            beta / 10.0,
            npc.uncertainty(engram.id),
            npc.confidence(engram.id),
            engram.prior,
            obs.strength,
            obs.reliability,
            obs.weighted_strength,
            npc.trust,
            npc.suspicion,
            npc.instability,
        ]

    def free_energy(self, npc: NPCState, engram: Engram, obs: Observation) -> float:
        belief = npc.belief(engram.id)
        sensory_error = obs.weighted_strength - belief
        prior_error = belief - engram.prior
        return (
            0.5 * npc.evidence_precision * sensory_error * sensory_error
            + 0.5 * npc.prior_precision * prior_error * prior_error
        )

    def pc_latent_teacher(self, npc: NPCState, engram: Engram, obs: Observation) -> list[float]:
        """
        Teacher latent used only for this demo's synthetic training.

        In a larger system, this target would come from episode logs or a richer
        generative PC objective. Here it makes the PC encoder learn a compact
        representation of prediction errors, belief, uncertainty, and social state.
        """
        belief = npc.belief(engram.id)
        sensory_error = obs.weighted_strength - belief
        prior_error = belief - engram.prior
        fe = self.free_energy(npc, engram, obs)
        return [
            belief,
            obs.weighted_strength,
            sensory_error,
            prior_error,
            fe,
            npc.uncertainty(engram.id),
            npc.trust - npc.suspicion,
            npc.instability,
        ]

    def belief_delta_teacher(self, npc: NPCState, engram: Engram, obs: Observation) -> list[float]:
        """
        Converts PC prediction error into a target Beta update.

        TensorFlow learns the raw values; runtime applies softplus so alpha/beta
        increments remain positive.
        """
        belief = npc.belief(engram.id)
        error = obs.weighted_strength - belief
        gain = 0.10 + 1.80 * obs.reliability

        if error >= 0:
            delta_alpha = gain * abs(error)
            delta_beta = 0.05 * gain
        else:
            delta_alpha = 0.05 * gain
            delta_beta = gain * abs(error)

        return [inverse_softplus(delta_alpha), inverse_softplus(delta_beta)]

    def policy_teacher(self, npc: NPCState, engram: Engram) -> int:
        b = npc.belief(engram.id)
        u = npc.uncertainty(engram.id)

        if b < 0.35:
            return self.action_labels.index(NPCAction.DISMISS)
        if u > 0.55:
            return self.action_labels.index(NPCAction.PROBE)
        if b > 0.68 and npc.suspicion > npc.trust:
            return self.action_labels.index(NPCAction.CONFRONT)
        if b > 0.55:
            return self.action_labels.index(NPCAction.REVEAL)
        return self.action_labels.index(NPCAction.PROBE)

    def train_step(self, npc: NPCState, engram: Engram, obs: Observation) -> tuple[float, float, float]:
        x = self.raw_features(npc, engram, obs)
        target_latent = self.pc_latent_teacher(npc, engram, obs)

        # Actual JPC predictive-coding parameter update.
        self.pc_encoder.train_pc_step(x, target_latent)

        # TensorFlow heads train from the JPC latent.
        latent = self.pc_encoder.encode(x)
        belief_target = self.belief_delta_teacher(npc, engram, obs)
        action_label = self.policy_teacher(npc, engram)

        return self.tf_heads.train_step(
            latent,
            belief_target,
            action_label,
            action_count=len(self.action_labels),
        )

    def update_belief_and_act(self, npc: NPCState, engram: Engram, obs: Observation) -> NPCAction:
        before_fe = self.free_energy(npc, engram, obs)
        old_belief = npc.belief(engram.id)
        old_uncertainty = npc.uncertainty(engram.id)

        x = self.raw_features(npc, engram, obs)
        latent = self.pc_encoder.encode(x)
        raw_delta, logits = self.tf_heads.predict(latent)

        delta_alpha = softplus(float(raw_delta[0]))
        delta_beta = softplus(float(raw_delta[1]))

        alpha, beta = npc.get_params(engram.id)
        npc.set_params(engram.id, alpha + delta_alpha, beta + delta_beta)

        action_index = int(np.argmax(logits))
        action = self.action_labels[action_index]
        after_fe = self.free_energy(npc, engram, obs)

        self.last_trace = {
            "backend": "actual_jpc_encoder_plus_tensorflow_heads",
            "raw_features": dict(zip(self.raw_feature_names, x)),
            "jpc_latent": [round(float(v), 4) for v in latent.tolist()],
            "old_belief": old_belief,
            "new_belief": npc.belief(engram.id),
            "old_uncertainty": old_uncertainty,
            "new_uncertainty": npc.uncertainty(engram.id),
            "free_energy_before": before_fe,
            "free_energy_after": after_fe,
            "raw_delta_alpha": float(raw_delta[0]),
            "raw_delta_beta": float(raw_delta[1]),
            "delta_alpha": delta_alpha,
            "delta_beta": delta_beta,
            "policy_logits": {
                self.action_labels[i].value: round(float(logits[i]), 4)
                for i in range(len(self.action_labels))
            },
            "chosen_action": action.value,
        }
        return action

