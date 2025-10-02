import os
from pathlib import Path
from datasets import load_dataset
import sentencepiece as spm

def train_sentencepiece(data_dir="data/iwslt14", vocab_size=32000):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load dataset from HF
    print("Loading IWSLT2017 de-en dataset...")
    dataset = load_dataset(
        "iwslt2017",
        "iwslt2017-de-en",
        trust_remote_code=True,
        cache_dir=str(data_dir)
    )

    # Step 2: Dump train split to text file
    all_file = data_dir / "train.all"
    if not all_file.exists():
        print(f"Writing training sentences to {all_file} ...")
        with open(all_file, "w", encoding="utf-8") as f:
            for ex in dataset["train"]:
                f.write(ex["translation"]["de"] + "\n")
                f.write(ex["translation"]["en"] + "\n")
    else:
        print(f"Found existing {all_file}, skipping dump.")

    # Step 3: Train SentencePiece BPE
    model_prefix = data_dir / "iwslt14_bpe"
    if not (data_dir / "iwslt14_bpe.model").exists():
        print("Training SentencePiece model...")
        spm.SentencePieceTrainer.train(
            input=str(all_file),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3
        )
        print(f"Saved model to {model_prefix}.model / {model_prefix}.vocab")
    else:
        print("SPM model already exists, skipping training.")

    return str(model_prefix) + ".model"


if __name__ == "__main__":
    model_path = train_sentencepiece()
    print(f"SentencePiece model ready: {model_path}")