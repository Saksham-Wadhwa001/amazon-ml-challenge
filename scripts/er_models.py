#!/usr/bin/env python3
"""
er_models.py
============

Thin, dependency-tolerant wrappers around the two pretrained models used by the
Business Entity Resolution pipeline:

* ``EmbeddingModel``  -> Qwen3-Embedding (via sentence-transformers) for semantic
  blocking + a similarity feature.
* ``Reranker``        -> Qwen3-Reranker (via transformers, yes/no-logit scoring)
  for a high-precision pairwise relevance feature.

Both are chosen to satisfy the challenge's model rule (Apache-2.0, <= 8B params).

Graceful degradation
--------------------
Neither wrapper raises if the heavy model / GPU / library is missing. Instead it
logs a warning and switches to a cheap CPU fallback (hashing char-ngram vectors
for the embedder; disabled feature for the reranker). This lets the full pipeline
run and be tested anywhere, and unlock its best accuracy on the H100 node.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np


def _log(msg: str) -> None:
    print(f"[er_models] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Embedding model (semantic blocking + feature)
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Sentence embedder. Uses Qwen3-Embedding through sentence-transformers when
    available, otherwise a hashing char n-gram vectorizer (still L2-normalized,
    so cosine == dot product downstream either way).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str | None = None,
        batch_size: int = 128,
        max_seq_length: int | None = 512,
        fallback_dim: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.backend = "none"
        self._model = None
        self._hv = None

        # Explicit request for the cheap CPU backend (no network / no ST import).
        if str(model_name).lower() in {"none", "hashing", "fallback"}:
            self._init_hashing(fallback_dim)
            return

        try:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {}
            # Use fp16 on GPU for the 8B model; harmless elsewhere.
            try:
                import torch

                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                if device.startswith("cuda"):
                    model_kwargs["torch_dtype"] = torch.float16
            except Exception:
                device = device or "cpu"

            self._model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
                model_kwargs=model_kwargs or None,
            )
            if max_seq_length:
                try:
                    self._model.max_seq_length = max_seq_length
                except Exception:
                    pass
            self.backend = "sentence-transformers"
            _log(f"embedding backend=sentence-transformers model={model_name} device={device}")
        except Exception as exc:  # noqa: BLE001
            _log(f"'{model_name}' unavailable ({exc}); using hashing-vectorizer fallback.")
            self._init_hashing(fallback_dim)

    def _init_hashing(self, fallback_dim: int) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._hv = HashingVectorizer(
            n_features=fallback_dim,
            alternate_sign=False,
            norm="l2",
            analyzer="char_wb",
            ngram_range=(3, 5),
        )
        self.backend = "hashing"

    @property
    def available(self) -> bool:
        return self.backend != "none"

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        """Return L2-normalized float32 embeddings, shape (len(texts), dim)."""
        texts = [t if isinstance(t, str) else "" for t in texts]
        if self.backend == "sentence-transformers":
            kwargs = dict(
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            # Qwen3-Embedding applies an instruction prompt to *queries* only.
            if is_query:
                try:
                    emb = self._model.encode(texts, prompt_name="query", **kwargs)
                    return np.ascontiguousarray(emb, dtype="float32")
                except (TypeError, ValueError, KeyError):
                    pass  # model has no named "query" prompt -> fall through
            emb = self._model.encode(texts, **kwargs)
            return np.ascontiguousarray(emb, dtype="float32")

        # Hashing fallback (already L2-normalized by HashingVectorizer).
        mat = self._hv.transform(texts)
        return np.ascontiguousarray(mat.toarray(), dtype="float32")


# ---------------------------------------------------------------------------
# Cross-encoder reranker (high-precision pairwise feature)
# ---------------------------------------------------------------------------

class Reranker:
    """
    Qwen3-Reranker scorer. The model is a causal LM that answers a yes/no
    relevance question; the score is softmax(yes, no)[yes] at the last position
    (the official Qwen3-Reranker recipe). Returns None-scores if unavailable.
    """

    _DEFAULT_INSTRUCTION = (
        "Given a business record, retrieve business records from other sources "
        "that refer to the same real-world business entity."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-8B",
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 1024,
        instruction: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction or self._DEFAULT_INSTRUCTION
        self.available = False
        self._torch = None
        self._tok = None
        self._model = None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

            self._tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype
            ).to(device).eval()
            self._device = next(self._model.parameters()).device

            self._true_id = self._tok.convert_tokens_to_ids("yes")
            self._false_id = self._tok.convert_tokens_to_ids("no")
            self._prefix = (
                "<|im_start|>system\nJudge whether the Document meets the "
                "requirements based on the Query and the Instruct provided. "
                'Note that the answer can only be "yes" or "no".<|im_end|>\n'
                "<|im_start|>user\n"
            )
            self._suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.available = True
            _log(f"reranker backend=transformers model={model_name} device={device}")
        except Exception as exc:  # noqa: BLE001
            _log(f"reranker '{model_name}' unavailable ({exc}); feature disabled.")
            self.available = False

    def _format(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float | None]:
        """Score (query, doc) pairs -> P(match) in [0, 1]; None if unavailable."""
        if not self.available:
            return [None] * len(pairs)
        torch = self._torch
        out: list[float | None] = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i : i + self.batch_size]
            texts = [self._prefix + self._format(q, d) + self._suffix for q, d in chunk]
            enc = self._tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits[:, -1, :]
                true_logit = logits[:, self._true_id]
                false_logit = logits[:, self._false_id]
                stacked = torch.stack([false_logit, true_logit], dim=1)
                prob_yes = torch.log_softmax(stacked, dim=1)[:, 1].exp()
            out.extend(prob_yes.detach().float().cpu().tolist())
        return out
