"""Node embedding computation for the knowledge graph.

Computes 768-dimensional embeddings for KG nodes using SapBERT or
PubMedBERT, based on each node's preferred UMLS concept name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class NodeEmbedder:
    """Compute dense embeddings for KG node surface forms.

    Parameters:
        model_name: HuggingFace model identifier (default: SapBERT).
        device: Torch device string.
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        batch_size: int = 128,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        logger.info("Loading embedding model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Compute embeddings for a list of text strings.

        Args:
            texts: Node surface forms to embed.

        Returns:
            Dict mapping each text to its 768-dim embedding (on CPU).
        """
        embeddings: Dict[str, torch.Tensor] = {}
        unique_texts = list(set(texts))

        for start in range(0, len(unique_texts), self.batch_size):
            batch = unique_texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            # CLS pooling
            cls_embs = outputs.last_hidden_state[:, 0, :]  # (B, D)
            cls_embs = torch.nn.functional.normalize(cls_embs, p=2, dim=-1)

            for text, emb in zip(batch, cls_embs):
                embeddings[text] = emb.cpu()

        logger.info("Embedded %d unique texts", len(embeddings))
        return embeddings

    def embed_graph_nodes(self, graph_data) -> torch.Tensor:
        """Replace graph node features with SapBERT embeddings.

        Args:
            graph_data: A ``torch_geometric.data.Data`` object with
                ``node_texts`` attribute.

        Returns:
            Updated node feature tensor ``(N, 768)``.
        """
        if not hasattr(graph_data, "node_texts"):
            raise ValueError("Graph data must have 'node_texts' attribute")

        texts = graph_data.node_texts
        embeddings = self.embed_texts(texts)

        new_features = []
        for text in texts:
            if text in embeddings:
                new_features.append(embeddings[text])
            else:
                new_features.append(torch.zeros(768))

        x = torch.stack(new_features)
        graph_data.x = x
        return x

    @staticmethod
    def save_embeddings(
        embeddings: Dict[str, torch.Tensor],
        path: Path | str,
    ) -> None:
        """Save embeddings dict to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, str(path))
        logger.info("Saved %d embeddings to %s", len(embeddings), path)

    @staticmethod
    def load_embeddings(path: Path | str) -> Dict[str, torch.Tensor]:
        """Load embeddings dict from disk."""
        return torch.load(str(path), map_location="cpu", weights_only=True)
