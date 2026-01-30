"""RadGraph extraction helpers mirroring the tested notebook pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExtractedEntity:
    text: str
    etype: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class ExtractedTriple:
    head: str
    rel: str
    tail: str


class RadGraphExtractor:
    """Thin wrapper around the radgraph package with deterministic normalization."""

    def __init__(self, model_type: str = "radgraph") -> None:
        try:
            import radgraph  # type: ignore

            self.radgraph = radgraph
        except Exception as exc:
            raise ImportError(
                "radgraph is required. Install with `pip install radgraph`."
            ) from exc

        self.model_type = model_type
        self._model = None

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = self.radgraph.RadGraph(model_type=self.model_type)
        return self._model

    def infer(self, texts: List[str]) -> Dict[str, Dict[str, Any]]:
        """Run RadGraph inference for a batch of reports."""
        model = self._get_model()
        result = model(texts)
        return result

    @staticmethod
    def _normalize_single(
        pred: Dict[str, Any],
    ) -> Tuple[Dict[str, ExtractedEntity], List[ExtractedTriple]]:
        """Normalize one RadGraph prediction to entities and triples."""
        entities: Dict[str, ExtractedEntity] = {}
        raw_relation_tuples: List[Tuple[str, str, str]] = []

        # Get entities from the prediction
        ent_obj = pred.get("entities")
        if ent_obj is None or not isinstance(ent_obj, dict):
            return entities, []

        def _consume_entity(key: str, payload: Dict[str, Any]) -> None:
            # Extract text from 'tokens' field
            text = payload.get("tokens", "")
            if isinstance(text, list):
                text = " ".join(text)
            elif not isinstance(text, str):
                text = str(text)

            # Extract label
            label = payload.get("label", "UNKNOWN")

            # Create entity
            entities[str(key)] = ExtractedEntity(
                text=text.strip(),
                etype=str(label).upper() if label else "UNKNOWN",
                start=payload.get("start_ix"),
                end=payload.get("end_ix"),
            )

            # Process relations: [['relation_type', 'target_entity_id'], ...]
            for rel_item in payload.get("relations", []):
                if isinstance(rel_item, list) and len(rel_item) == 2:
                    rtype_raw, tgt_raw = rel_item
                    raw_relation_tuples.append((str(key), rtype_raw, str(tgt_raw)))

        for k, v in ent_obj.items():
            if isinstance(v, dict):
                _consume_entity(k, v)

        # Build triples
        triples: List[ExtractedTriple] = []

        def rel_type_normalize(x: Any) -> str:
            return str(x).lower().replace(" ", "_").upper()

        for head_id, rtype_raw, tail_id in raw_relation_tuples:
            if head_id in entities and tail_id in entities:
                h_ent = entities[head_id]
                t_ent = entities[tail_id]
                triples.append(
                    ExtractedTriple(
                        h_ent.text, rel_type_normalize(rtype_raw), t_ent.text
                    )
                )

        return entities, triples

    def normalize_batch(
        self, preds: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[Dict[str, ExtractedEntity]], List[ExtractedTriple]]:
        """Normalize a batch of RadGraph outputs."""
        all_entities: List[Dict[str, ExtractedEntity]] = []
        all_triples: List[ExtractedTriple] = []

        # preds is a dict with report IDs as keys
        for report_id, pred_data in preds.items():
            ents, trips = self._normalize_single(pred_data)
            all_entities.append(ents)
            all_triples.extend(trips)

        return all_entities, all_triples
