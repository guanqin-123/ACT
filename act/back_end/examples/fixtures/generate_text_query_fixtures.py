#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers"]
# ///
"""Generate deterministic compact-BERT fixtures for frontend CI coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONFIG: dict[str, object] = {
    "architectures": ["BertForSequenceClassification"],
    "model_type": "bert",
    "vocab_size": 9,
    "hidden_size": 8,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 16,
    "hidden_act": "relu",
    "max_position_embeddings": 32,
    "type_vocab_size": 2,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "num_labels": 2,
}
VOCABULARY = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "good", "film", "."]


def write_checkpoint(root: Path) -> Path:
    """Write the seeded float32 compact-BERT checkpoint beneath ``root``."""
    from act.back_end.examples.fixtures.generate_coverage_ops import check_conversion
    from act.front_end.bert_loader import compact_bert
    from act.util.device_manager import get_default_dtype

    model_dir = root / "buffet/models/model_sst_1"
    checkpoint_dir = model_dir / "ckpt-1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _ = (model_dir / "checkpoint").write_text("1\n", encoding="utf-8")
    _ = (checkpoint_dir / "vocab.txt").write_text(
        "\n".join(VOCABULARY) + "\n",
        encoding="utf-8",
    )
    _ = (checkpoint_dir / "config.json").write_text(
        json.dumps(CONFIG, indent=1) + "\n",
        encoding="utf-8",
    )
    torch.manual_seed(20261025)
    model = compact_bert.CompactBertFromEmbeddings(compact_bert._bert_config(CONFIG))
    state = {
        source: model.state_dict()[target].detach().clone().float()
        for target, source in compact_bert._parameter_key_map(model).items()
    }
    state["bert.embeddings.word_embeddings.weight"] = torch.randn(len(VOCABULARY), 8).float()
    state["bert.embeddings.position_embeddings.weight"] = torch.randn(32, 8).float()
    state["bert.embeddings.token_type_embeddings.weight"] = torch.randn(2, 8).float()
    torch.save(state, checkpoint_dir / "pytorch_model.bin")
    sample = torch.linspace(-0.1, 0.1, 24, dtype=get_default_dtype()).reshape(1, 3, 8)
    for route in ("a", "b"):
        model._act_conversion_route = route
        check_conversion(model.eval(), sample)
    return checkpoint_dir / "pytorch_model.bin"


def write_text_data(root: Path) -> None:
    """Write the exact SST-dev and Yelp query inputs under the fixture root."""
    sst = root / "sst"
    sst.mkdir(exist_ok=True)
    _ = (sst / "dev.txt").write_text(
        "(0 (2 a) (0 (3 good) (2 film)) (2 .))\n",
        encoding="utf-8",
    )
    yelp = root / "yelp"
    yelp.mkdir(exist_ok=True)
    _ = (yelp / "test.csv").write_text(
        "label,text\n" + "1,a good film .\n" * 5,
        encoding="utf-8",
    )


def write_query_index(fixtures: Path, digest: str) -> None:
    """Write the four frontend query rows with the generated checkpoint hash."""
    _ = (fixtures / "bert_query_sst.csv").write_text(
        "dataset,split,depth,example_id,position,p,eps,checkpoint_sha256\n"
        f"sst,dev,1,0,1,inf,0.001,{digest}\n"
        f"sst,dev,1,0,1,1,0.001,{digest}\n"
        "yelp,calibration,1,1,1,1,0.001,\n"
        "yelp,evaluation,1,0,1,inf,0.001,\n",
        encoding="utf-8",
    )


def print_manifest(root: Path) -> None:
    """Print byte-level reproducibility metadata for the generated assets."""
    manifest = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    print(json.dumps(manifest, indent=2))


def main() -> None:
    """Generate every B4 text fixture under the committed fixture directory."""
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    fixtures = Path(__file__).parent
    root = fixtures / "text_query"
    checkpoint = write_checkpoint(root)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _ = (root / "buffet/SHA256SUMS").write_text(
        f"{digest}  buffet/models/model_sst_1/ckpt-1/pytorch_model.bin\n",
        encoding="utf-8",
    )
    write_text_data(root)
    write_query_index(fixtures, digest)
    print_manifest(root)


if __name__ == "__main__":
    main()
