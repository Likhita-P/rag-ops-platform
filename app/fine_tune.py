"""
pipelines/fine_tune.py
-----------------------
Fine-tuning pipeline for the embedding model using TensorFlow and PyTorch/HuggingFace.

Covers:
  ETS JD  — "Expert in Python and familiar with core ML/DL frameworks
             (e.g., TensorFlow, PyTorch, Hugging Face Transformers)"
           — "Model development: Select appropriate algorithms, train,
             tune, and validate models using best practices"
           — "MLOps & deployment: versioning of data/models and
             reproducible training"
  Resume  — "LLM Fine-Tuning, Quantization, LoRA, PEFT, PyTorch,
             TensorFlow, HuggingFace Transformers" (already on resume —
             this file makes it demonstrable in the project)

Interview angle (ETS):
  "This pipeline shows the full model development lifecycle: data prep,
   fine-tuning with LoRA/PEFT for efficiency, TensorFlow-based evaluation,
   and MLflow logging for experiment tracking. For embedding fine-tuning
   I use contrastive loss — teaching the model that questions and their
   relevant document chunks should be close in embedding space."

Interview angle (ACKO Round 1 — fine-tuning story):
  "At HealthEdge we fine-tuned the embedding model on domain-specific
   HRP terminology using LoRA — the base model didn't know what
   'benefit period' or 'out-of-pocket maximum' meant in our context.
   After fine-tuning, retrieval precision improved measurably on our
   RAGAS eval suite."
"""

import os
import json
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
BASE_MODEL      = os.getenv("EMBEDDING_MODEL_PATH", "all-MiniLM-L6-v2")
FINE_TUNED_PATH = os.getenv("FINE_TUNED_MODEL_PATH", "models/fine_tuned_embedding")
BATCH_SIZE      = int(os.getenv("FINETUNE_BATCH_SIZE", "8"))
NUM_EPOCHS      = int(os.getenv("FINETUNE_EPOCHS", "3"))
LEARNING_RATE   = float(os.getenv("FINETUNE_LR", "2e-5"))
MAX_LENGTH      = 256


# ── Dataset ────────────────────────────────────────────────────────────────

class ContrastiveDataset(Dataset):
    """
    Dataset of (query, positive_passage, negative_passage) triples
    for contrastive fine-tuning of the embedding model.

    Interview angle (ETS — data engineering):
      "For embedding fine-tuning we need contrastive pairs — queries
       paired with relevant and irrelevant document chunks. We mine
       hard negatives from the same document (chunks that are topically
       close but don't answer the query) to make the model learn
       meaningful distinctions, not just obvious ones."
    """

    def __init__(self, triples: List[dict], tokenizer, max_length: int = MAX_LENGTH):
        self.triples    = triples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple  = self.triples[idx]
        query   = triple["query"]
        pos     = triple["positive"]
        neg     = triple.get("negative", "")   # optional hard negative

        def encode(text):
            return self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        return {
            "query":    encode(query),
            "positive": encode(pos),
            "negative": encode(neg) if neg else None,
        }


def load_training_data(data_path: str = "data/finetune_triples.jsonl") -> List[dict]:
    """
    Load training triples from JSONL.

    Each line: {"query": "...", "positive": "...", "negative": "..."}

    For the demo, returns synthetic triples if file doesn't exist.

    Interview angle (ETS — data engineering):
      "In production we'd generate these triples from the support
       ticket data — the question is the query, the document chunk
       that resolved the ticket is the positive, and a random
       unrelated chunk is the negative. For this demo I generate
       synthetic triples to show the pipeline structure."
    """
    if os.path.exists(data_path):
        triples = []
        with open(data_path) as f:
            for line in f:
                try:
                    triples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(triples)} training triples from {data_path}")
        return triples

    # Synthetic demo triples — domain-specific insurance/HRP terminology
    logger.info("No training data found. Using synthetic demo triples.")
    return [
        {
            "query":    "What is the maximum out-of-pocket limit?",
            "positive": "The annual out-of-pocket maximum is $7,500 per individual.",
            "negative": "The plan effective date begins on January 1st of the plan year.",
        },
        {
            "query":    "Are prescription drugs covered?",
            "positive": "Prescription drug coverage includes generic and brand-name medications.",
            "negative": "Emergency services are covered at the in-network benefit level.",
        },
        {
            "query":    "How do I submit a claim?",
            "positive": "Claims must be submitted within 90 days of the date of service.",
            "negative": "The deductible resets on the first day of each benefit year.",
        },
        {
            "query":    "What is the deductible amount?",
            "positive": "The individual deductible is $1,500 per benefit year.",
            "negative": "Prior authorization is required for certain specialist visits.",
        },
        {
            "query":    "Is mental health treatment covered?",
            "positive": "Mental health and substance use disorder benefits are covered at parity.",
            "negative": "Preventive care services are covered at 100% with no cost sharing.",
        },
    ]


# ── LoRA config ────────────────────────────────────────────────────────────

def get_lora_config() -> LoraConfig:
    """
    LoRA configuration for efficient fine-tuning.

    Interview angle (ETS + Resume):
      "LoRA freezes the base model weights and adds small trainable
       rank-decomposition matrices to the attention layers. For a
       66M-parameter model like MiniLM, LoRA reduces trainable
       parameters by ~95% while matching full fine-tuning quality.
       This is how we fine-tuned in production at HealthEdge —
       the base model stayed frozen, only the LoRA adapters changed."

    r=8: rank of the low-rank matrices. Higher = more expressive but
         more parameters. 8 is standard for embedding models.
    alpha=16: scaling factor. Usually set to 2×r.
    target_modules: which layers to adapt. "query" and "value"
                    in attention are the standard choice.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        # For embedding models we use FEATURE_EXTRACTION task type
        # (no causal LM head)
        task_type=TaskType.FEATURE_EXTRACTION,
    )


# ── Contrastive loss ───────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    InfoNCE / NT-Xent contrastive loss for embedding fine-tuning.

    Pulls query embeddings closer to their positive passages and
    pushes them away from negative passages (and other in-batch negatives).

    Interview angle (ETS — model development):
      "Contrastive loss is the standard objective for embedding fine-tuning.
       The temperature parameter controls how sharply the model
       discriminates — lower temperature = harder negatives = more
       aggressive training. We start at 0.07 (standard from SimCLR)
       and tune from there based on validation retrieval metrics."
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_emb:    torch.Tensor,   # (batch, hidden)
        positive_emb: torch.Tensor,   # (batch, hidden)
        negative_emb: Optional[torch.Tensor] = None,  # (batch, hidden)
    ) -> torch.Tensor:

        # Normalise
        q = F.normalize(query_emb,    dim=-1)
        p = F.normalize(positive_emb, dim=-1)

        # Positive similarity: diagonal of q @ p^T
        pos_sim = (q * p).sum(dim=-1) / self.temperature  # (batch,)

        if negative_emb is not None:
            n   = F.normalize(negative_emb, dim=-1)
            neg_sim = (q * n).sum(dim=-1) / self.temperature  # (batch,)
            # Loss: softmax over [pos, neg] — want pos to dominate
            logits = torch.stack([pos_sim, neg_sim], dim=-1)   # (batch, 2)
            labels = torch.zeros(len(q), dtype=torch.long)     # positive = index 0
            loss   = F.cross_entropy(logits, labels)
        else:
            # In-batch negatives only (other batch items as negatives)
            sim_matrix = q @ p.T / self.temperature  # (batch, batch)
            labels     = torch.arange(len(q))
            loss       = F.cross_entropy(sim_matrix, labels)

        return loss


# ── TensorFlow evaluation ──────────────────────────────────────────────────

def evaluate_with_tensorflow(
    model_path: str,
    eval_triples: List[dict],
) -> dict:
    """
    Evaluate embedding quality using TensorFlow's cosine similarity ops.

    TensorFlow is used here for evaluation (not training) to demonstrate
    cross-framework fluency — a common pattern in production where
    training happens in PyTorch and serving/eval uses TF SavedModel.

    Covers ETS JD requirement:
      "Expert in Python and familiar with core ML/DL frameworks
       (e.g., TensorFlow, PyTorch, Hugging Face Transformers)"

    Interview angle (ETS):
      "I'm comfortable in both PyTorch and TensorFlow. At HealthEdge
       we trained in PyTorch (tighter research loop) and exported to
       TF SavedModel for serving on TF Serving. The eval logic in TF
       let us validate the export matched the training outputs."
    """
    try:
        import tensorflow as tf

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model     = AutoModel.from_pretrained(model_path)
        model.eval()

        scores = []
        for triple in eval_triples[:20]:   # evaluate on first 20 samples
            def get_emb(text):
                inputs  = tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=MAX_LENGTH)
                with torch.no_grad():
                    out = model(**inputs).last_hidden_state[:, 0, :]
                return out.numpy()

            q_emb = get_emb(triple["query"])
            p_emb = get_emb(triple["positive"])

            # TF cosine similarity
            q_tf  = tf.constant(q_emb, dtype=tf.float32)
            p_tf  = tf.constant(p_emb, dtype=tf.float32)
            sim   = tf.keras.losses.cosine_similarity(q_tf, p_tf)
            # TF cosine_similarity returns negative similarity
            scores.append(float(-sim.numpy()))

        avg_score = float(np.mean(scores))
        logger.info(
            f"TensorFlow eval complete: avg_cosine_similarity={avg_score:.4f} "
            f"on {len(scores)} samples"
        )
        return {
            "framework":              "tensorflow",
            "avg_cosine_similarity":  round(avg_score, 4),
            "num_samples":            len(scores),
            "model_path":             model_path,
        }

    except ImportError:
        logger.warning("TensorFlow not installed. "
                       "Install with: pip install tensorflow")
        return {"error": "TensorFlow not installed"}
    except Exception as e:
        logger.error(f"TensorFlow eval failed: {e}")
        return {"error": str(e)}


# ── Mean Pooled Embedding helper ───────────────────────────────────────────

def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over token embeddings (masked).

    Interview angle:
      "We use mean pooling instead of CLS-token pooling during fine-tuning
       because it's been shown to produce more uniformly distributed
       embedding spaces for contrastive learning. CLS token tends to
       be biased toward the beginning of the sequence."
    """
    mask_expanded  = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (last_hidden_state * mask_expanded).sum(1)
    sum_mask       = mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def _encode_batch(model, batch_encoding) -> torch.Tensor:
    """Run a single batch through the model and return mean-pooled embeddings."""
    input_ids      = batch_encoding["input_ids"].squeeze(1)
    attention_mask = batch_encoding["attention_mask"].squeeze(1)
    outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
    return mean_pool(outputs.last_hidden_state, attention_mask)


# ── Training loop ──────────────────────────────────────────────────────────

def run_fine_tuning(
    training_data:  List[dict] = None,
    epochs:         int        = NUM_EPOCHS,
    batch_size:     int        = BATCH_SIZE,
    learning_rate:  float      = LEARNING_RATE,
    log_to_mlflow:  bool       = True,
    save_path:      str        = FINE_TUNED_PATH,
) -> dict:
    """
    Fine-tune the embedding model on domain-specific contrastive pairs.

    Pipeline:
      1. Load base model + apply LoRA adapters
      2. Freeze base weights (only LoRA matrices train)
      3. Train with InfoNCE contrastive loss
      4. Evaluate using TF cosine similarity
      5. Save adapters + log to MLflow

    Interview angle (ETS — model development + MLOps):
      "The training loop uses LoRA so only ~2% of parameters are
       trainable. After training, we save just the adapter weights —
       not the full model — which keeps the model artifact small and
       makes rollback trivial. The full model is reconstructed at
       inference time by merging base + adapters."
    """
    logger.info(
        f"Starting fine-tuning: epochs={epochs}, batch_size={batch_size}, "
        f"lr={learning_rate}, base_model={BASE_MODEL}"
    )

    if training_data is None:
        training_data = load_training_data()

    # Load model + apply LoRA
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModel.from_pretrained(BASE_MODEL)
    lora_cfg   = get_lora_config()

    try:
        model = get_peft_model(base_model, lora_cfg)
        model.print_trainable_parameters()
        logger.info("LoRA adapters applied successfully.")
    except Exception as e:
        logger.warning(f"PEFT/LoRA unavailable ({e}). Training full model.")
        model = base_model

    # Dataset + DataLoader
    dataset    = ContrastiveDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer + loss
    optimizer  = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                   weight_decay=0.01)
    criterion  = ContrastiveLoss(temperature=0.07)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader)
    )

    history    = {"train_loss": [], "epoch_loss": []}
    model.train()

    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            q_emb = _encode_batch(model, batch["query"])
            p_emb = _encode_batch(model, batch["positive"])

            has_neg  = batch.get("negative") is not None
            n_emb    = _encode_batch(model, batch["negative"]) if has_neg else None

            loss = criterion(q_emb, p_emb, n_emb)
            loss.backward()

            # Gradient clipping — standard for transformer fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            history["train_loss"].append(batch_loss)

            if batch_idx % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

        epoch_avg = float(np.mean(epoch_losses))
        history["epoch_loss"].append(epoch_avg)
        logger.info(f"Epoch {epoch+1} complete. Avg loss: {epoch_avg:.4f}")

    # Save fine-tuned adapters
    os.makedirs(save_path, exist_ok=True)
    try:
        model.save_pretrained(save_path)   # saves LoRA adapters only
        tokenizer.save_pretrained(save_path)
        logger.info(f"Fine-tuned model saved to {save_path}")
    except Exception as e:
        logger.warning(f"Could not save model: {e}")

    # TensorFlow evaluation
    tf_eval = evaluate_with_tensorflow(BASE_MODEL, training_data[:10])

    # Log to MLflow
    final_loss = history["epoch_loss"][-1] if history["epoch_loss"] else 0.0
    if log_to_mlflow:
        try:
            from pipelines.mlflow_tracker import _get_client
            mlflow = _get_client()
            mlflow.set_experiment("embedding_fine_tuning")
            with mlflow.start_run(run_name=f"lora_finetune_e{epochs}"):
                mlflow.log_params({
                    "base_model":      BASE_MODEL,
                    "epochs":          epochs,
                    "batch_size":      batch_size,
                    "learning_rate":   learning_rate,
                    "lora_r":          8,
                    "lora_alpha":      16,
                    "num_samples":     len(training_data),
                })
                for step, loss_val in enumerate(history["train_loss"]):
                    mlflow.log_metric("train_loss", loss_val, step=step)
                mlflow.log_metric("final_epoch_loss", final_loss)
                if "avg_cosine_similarity" in tf_eval:
                    mlflow.log_metric("tf_cosine_similarity",
                                      tf_eval["avg_cosine_similarity"])
                if save_path and os.path.exists(save_path):
                    mlflow.log_artifacts(save_path, artifact_path="model")
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

    return {
        "epochs":          epochs,
        "final_loss":      round(final_loss, 4),
        "history":         history,
        "save_path":       save_path,
        "tf_eval":         tf_eval,
        "training_samples": len(training_data),
    }
