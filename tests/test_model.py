"""
test_model.py — Unit tests for model.py.

Tests model initialization and forward pass on a tiny synthetic example.

Uses microsoft/deberta-v3-small by default. This requires the model to
be downloadable. Tests are marked with @pytest.mark.integration if they
require network access / model download.
"""

import pytest
import torch

from src.member3.model import (
    DEFAULT_MODEL_NAME,
    get_device,
    load_model,
    load_tokenizer,
)


# ---------------------------------------------------------------------------
# Basic tests (no model download required)
# ---------------------------------------------------------------------------

class TestDeviceDetection:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# Integration tests (require model download)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(DEFAULT_MODEL_NAME)


@pytest.fixture(scope="module")
def model():
    return load_model(DEFAULT_MODEL_NAME)


@pytest.mark.integration
class TestModelInitialization:
    """Tests that require downloading the DeBERTa model."""

    def test_tokenizer_loads(self, tokenizer):
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")

    def test_model_loads(self, model):
        assert model is not None
        assert hasattr(model, "forward")

    def test_model_has_2_labels(self, model):
        assert model.config.num_labels == 2

    def test_forward_pass(self, model, tokenizer):
        """Run a forward pass on a tiny synthetic example."""
        text_a = "Name: Acme Corp | Address: 123 Main St | Country: US"
        text_b = "Name: Acme Corporation | Address: 123 Main Street | Country: US"

        encoding = tokenizer(
            text_a, text_b,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )

        device = get_device()
        model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )

        assert outputs.logits.shape == (1, 2)
        probs = torch.softmax(outputs.logits, dim=-1)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_forward_pass_with_labels(self, model, tokenizer):
        """Forward pass with labels should return a loss."""
        text_a = "Name: Test | Address: Addr | Country: US"
        text_b = "Name: Test | Address: Addr | Country: US"

        encoding = tokenizer(
            text_a, text_b,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )

        device = get_device()
        model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
                labels=torch.tensor([1]).to(device),
            )

        assert outputs.loss is not None
        assert outputs.loss.item() > 0
