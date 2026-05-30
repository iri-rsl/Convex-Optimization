"""Feedforward neural network for binary classification.

This module intentionally avoids heavyweight deep learning dependencies so it can
run in a minimal Python environment while still matching the project brief:
ReLU hidden layers, sigmoid output, binary cross-entropy loss, and SGD/Adam
optimization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]


def load_processed_dataset(path: str | Path = "data/processed/clean_student_data.csv") -> pd.DataFrame:
    """Load the processed dataset expected by the training notebook."""
    return pd.read_csv(path)


def prepare_binary_classification_data(
    df: pd.DataFrame,
    target_column: str = "Depression",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Split a processed dataframe into feature and target arrays.

    The target is converted from {-1, 1} to {0, 1} for sigmoid + cross-entropy
    training, while keeping the original label semantics in the caller.
    """

    if target_column not in df.columns:
        raise KeyError(f"Missing target column: {target_column}")

    features = df.drop(columns=[target_column])
    target = df[target_column].astype(int).map({-1: 0, 1: 1})

    if target.isna().any():
        raise ValueError("Target column must contain only -1 and 1 values.")

    return features.to_numpy(dtype=np.float64), target.to_numpy(dtype=np.float64), list(features.columns)


def train_validation_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train, validation, and test partitions."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    train_end = int(len(X) * train_ratio)
    validation_end = train_end + int(len(X) * validation_ratio)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_validation = X[train_end:validation_end]
    y_validation = y[train_end:validation_end]
    X_test = X[validation_end:]
    y_test = y[validation_end:]

    return X_train, y_train, X_validation, y_validation, X_test, y_test


@dataclass
class TrainingHistory:
    train_loss: List[float]
    validation_loss: List[float]
    train_accuracy: List[float]
    validation_accuracy: List[float]


class BinaryMLPClassifier:
    """Simple feedforward neural network for binary classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int] = (64, 32),
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        optimizer: str = "adam",
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        l2_penalty: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer.lower()
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.l2_penalty = l2_penalty
        self.rng = np.random.default_rng(seed)

        layer_sizes = [input_dim, *self.hidden_layers, 1]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / fan_in) if fan_out != 1 else np.sqrt(1.0 / fan_in)
            self.weights.append(self.rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))

        self.history = TrainingHistory([], [], [], [])
        self._velocity_w = [np.zeros_like(weight) for weight in self.weights]
        self._velocity_b = [np.zeros_like(bias) for bias in self.biases]
        self._adam_m_w = [np.zeros_like(weight) for weight in self.weights]
        self._adam_v_w = [np.zeros_like(weight) for weight in self.weights]
        self._adam_m_b = [np.zeros_like(bias) for bias in self.biases]
        self._adam_v_b = [np.zeros_like(bias) for bias in self.biases]
        self._adam_step = 0

    def _get_config(self) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_layers": list(self.hidden_layers),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "l2_penalty": self.l2_penalty,
        }

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(np.float64)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        y_prob = np.clip(y_prob, 1e-8, 1.0 - 1e-8)
        return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))

    @staticmethod
    def _accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        predictions = (y_prob >= 0.5).astype(np.float64)
        return float(np.mean(predictions == y_true))

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        pre_activations = []

        current = X
        for index in range(len(self.weights) - 1):
            z = current @ self.weights[index] + self.biases[index]
            current = self._relu(z)
            pre_activations.append(z)
            activations.append(current)

        z_output = current @ self.weights[-1] + self.biases[-1]
        output = self._sigmoid(z_output)
        pre_activations.append(z_output)
        activations.append(output)

        return activations, pre_activations

    def _backward(
        self,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
        y_true: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        m = y_true.shape[0]
        gradients_w = [np.zeros_like(weight) for weight in self.weights]
        gradients_b = [np.zeros_like(bias) for bias in self.biases]

        delta = activations[-1] - y_true.reshape(-1, 1)
        gradients_w[-1] = (activations[-2].T @ delta) / m + self.l2_penalty * self.weights[-1]
        gradients_b[-1] = np.mean(delta, axis=0, keepdims=True)

        next_delta = delta
        for layer_index in range(len(self.weights) - 2, -1, -1):
            next_delta = (next_delta @ self.weights[layer_index + 1].T) * self._relu_derivative(pre_activations[layer_index])
            gradients_w[layer_index] = (activations[layer_index].T @ next_delta) / m + self.l2_penalty * self.weights[layer_index]
            gradients_b[layer_index] = np.mean(next_delta, axis=0, keepdims=True)

        return gradients_w, gradients_b

    def _update_parameters(self, gradients_w: List[np.ndarray], gradients_b: List[np.ndarray]) -> None:
        if self.optimizer == "sgd":
            for index, (grad_w, grad_b) in enumerate(zip(gradients_w, gradients_b)):
                self._velocity_w[index] = self.momentum * self._velocity_w[index] - self.learning_rate * grad_w
                self._velocity_b[index] = self.momentum * self._velocity_b[index] - self.learning_rate * grad_b
                self.weights[index] += self._velocity_w[index]
                self.biases[index] += self._velocity_b[index]
            return

        if self.optimizer != "adam":
            raise ValueError("optimizer must be either 'sgd' or 'adam'")

        self._adam_step += 1
        for index, (grad_w, grad_b) in enumerate(zip(gradients_w, gradients_b)):
            self._adam_m_w[index] = self.beta1 * self._adam_m_w[index] + (1.0 - self.beta1) * grad_w
            self._adam_v_w[index] = self.beta2 * self._adam_v_w[index] + (1.0 - self.beta2) * (grad_w ** 2)
            self._adam_m_b[index] = self.beta1 * self._adam_m_b[index] + (1.0 - self.beta1) * grad_b
            self._adam_v_b[index] = self.beta2 * self._adam_v_b[index] + (1.0 - self.beta2) * (grad_b ** 2)

            corrected_m_w = self._adam_m_w[index] / (1.0 - self.beta1 ** self._adam_step)
            corrected_v_w = self._adam_v_w[index] / (1.0 - self.beta2 ** self._adam_step)
            corrected_m_b = self._adam_m_b[index] / (1.0 - self.beta1 ** self._adam_step)
            corrected_v_b = self._adam_v_b[index] / (1.0 - self.beta2 ** self._adam_step)

            self.weights[index] -= self.learning_rate * corrected_m_w / (np.sqrt(corrected_v_w) + self.epsilon)
            self.biases[index] -= self.learning_rate * corrected_m_b / (np.sqrt(corrected_v_b) + self.epsilon)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: Optional[np.ndarray] = None,
        y_validation: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Train the network and return the recorded history."""

        for epoch in range(1, self.epochs + 1):
            indices = self.rng.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start_index in range(0, len(X_shuffled), self.batch_size):
                stop_index = start_index + self.batch_size
                batch_X = X_shuffled[start_index:stop_index]
                batch_y = y_shuffled[start_index:stop_index]

                activations, pre_activations = self._forward(batch_X)
                gradients_w, gradients_b = self._backward(activations, pre_activations, batch_y)
                self._update_parameters(gradients_w, gradients_b)

            train_probabilities = self.predict_proba(X_train)
            train_loss = self._binary_cross_entropy(y_train, train_probabilities)
            train_accuracy = self._accuracy(y_train, train_probabilities)

            self.history.train_loss.append(train_loss)
            self.history.train_accuracy.append(train_accuracy)

            if X_validation is not None and y_validation is not None and len(X_validation) > 0:
                validation_probabilities = self.predict_proba(X_validation)
                validation_loss = self._binary_cross_entropy(y_validation, validation_probabilities)
                validation_accuracy = self._accuracy(y_validation, validation_probabilities)
            else:
                validation_loss = float("nan")
                validation_accuracy = float("nan")

            self.history.validation_loss.append(validation_loss)
            self.history.validation_accuracy.append(validation_accuracy)

            if verbose:
                print(
                    f"Epoch {epoch:03d}/{self.epochs} | "
                    f"loss={train_loss:.4f} | acc={train_accuracy:.4f} | "
                    f"val_loss={validation_loss:.4f} | val_acc={validation_accuracy:.4f}"
                )

        return self.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities."""

        activations, _ = self._forward(X)
        return activations[-1].reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions in the original {-1, 1} label space."""

        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, -1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute binary cross-entropy and accuracy."""

        probabilities = self.predict_proba(X)
        return {
            "loss": self._binary_cross_entropy(y, probabilities),
            "accuracy": self._accuracy(y, probabilities),
        }

    def save(self, path: str | Path) -> Path:
        """Save model configuration and learned parameters to a compressed file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, np.ndarray] = {
            "config": np.array(json.dumps(self._get_config())),
            "weights_count": np.array(len(self.weights), dtype=np.int64),
            "biases_count": np.array(len(self.biases), dtype=np.int64),
        }

        for index, weight in enumerate(self.weights):
            payload[f"weight_{index}"] = weight
        for index, bias in enumerate(self.biases):
            payload[f"bias_{index}"] = bias

        np.savez_compressed(output_path, **payload)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "BinaryMLPClassifier":
        """Load a saved model from disk."""

        input_path = Path(path)
        with np.load(input_path, allow_pickle=False) as data:
            config = json.loads(str(data["config"]))
            model = cls(**config)

            weights_count = int(data["weights_count"])
            biases_count = int(data["biases_count"])

            model.weights = [data[f"weight_{index}"] for index in range(weights_count)]
            model.biases = [data[f"bias_{index}"] for index in range(biases_count)]

            model._velocity_w = [np.zeros_like(weight) for weight in model.weights]
            model._velocity_b = [np.zeros_like(bias) for bias in model.biases]
            model._adam_m_w = [np.zeros_like(weight) for weight in model.weights]
            model._adam_v_w = [np.zeros_like(weight) for weight in model.weights]
            model._adam_m_b = [np.zeros_like(bias) for bias in model.biases]
            model._adam_v_b = [np.zeros_like(bias) for bias in model.biases]
            model._adam_step = 0

        return model
