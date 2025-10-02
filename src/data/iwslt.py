import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
from hydra.utils import get_original_cwd
import sentencepiece as spm

class IWSLTDataset(Dataset):
	"""
	Wraps Hugging Face IWSLT'14 De–En dataset.
	Converts raw text into token ID tensors using provided tokenizer.
	Works with SentencePieceProcessor or Hugging Face tokenizers.
	"""

	def __init__(
		self,
		split,
		tokenizer,
		src_lang="de",
		tgt_lang="en",
		pad_idx=0,
		bos_idx=1,
		eos_idx=2,
		max_len=None,
	):
		super().__init__()
		self.split = split
		self.tokenizer = tokenizer
		self.src_lang = src_lang
		self.tgt_lang = tgt_lang
		self.pad_idx = pad_idx
		self.bos_idx = bos_idx
		self.eos_idx = eos_idx
		self.max_len = max_len

	def __len__(self):
		return len(self.split)

	def encode(self, text):
		"""Encode text → list[int] supporting both HF and SentencePiece."""
		if hasattr(self.tokenizer, "encode"):  
			# SentencePieceProcessor and HF both have .encode
			ids = self.tokenizer.encode(text, out_type=int) \
				if "sentencepiece" in str(type(self.tokenizer)).lower() \
				else self.tokenizer.encode(text, add_special_tokens=False)
		else:
			raise ValueError("Tokenizer must have an `encode` method")
		return ids

	def __getitem__(self, idx):
		ex = self.split[idx]
		src_text = ex["translation"][self.src_lang]
		tgt_text = ex["translation"][self.tgt_lang]

		# Encode
		src_ids = self.encode(src_text)
		tgt_ids = self.encode(tgt_text)

		# Truncate
		if self.max_len:
			src_ids = src_ids[: self.max_len]
			tgt_ids = tgt_ids[: self.max_len]

		# Add BOS/EOS
		src_ids = [self.bos_idx] + src_ids + [self.eos_idx]
		tgt_ids = [self.bos_idx] + tgt_ids + [self.eos_idx]

		return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)



# -------------------------------
# Collate function
# -------------------------------
def collate_fn(batch, pad_idx: int):
	src_batch, tgt_batch = zip(*batch)
	max_src_len = max(len(s) for s in src_batch)
	max_tgt_len = max(len(t) for t in tgt_batch)
	batch_size = len(batch)

	src_padded = torch.full((batch_size, max_src_len), pad_idx, dtype=torch.long)
	tgt_padded = torch.full((batch_size, max_tgt_len), pad_idx, dtype=torch.long)

	for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
		src_padded[i, : len(s)] = s
		tgt_padded[i, : len(t)] = t

	src_key_padding_mask = src_padded.eq(pad_idx)
	tgt_key_padding_mask = tgt_padded.eq(pad_idx)

	return src_padded, tgt_padded, src_key_padding_mask, tgt_key_padding_mask


# -------------------------------
# Factory: returns dataloaders
# -------------------------------
def get_iwslt14(cfg):
	"""
	Load IWSLT'14 De→En data via Hugging Face `datasets`.
	cfg must have:
	  - src_lang
	  - tgt_lang
	  - pad_idx, bos_idx, eos_idx
	  - batch_size
	  - num_workers
	  - tokenizer (must support .encode / .ids)
	"""
	# Load HF dataset
	base_dir = Path(get_original_cwd())
	data_dir = base_dir / "data" / "iwslt14"

	config_name = f"iwslt2017-{cfg.src_lang}-{cfg.tgt_lang}"
	dataset = load_dataset("iwslt2017", config_name, trust_remote_code=True, cache_dir=str(data_dir))

	# ---- Subsample for debugging ----
	if getattr(cfg, "subset_fraction", None):
		frac = cfg.subset_fraction
		dataset["train"] = dataset["train"].train_test_split(
			test_size=1-frac, seed=1
		)["train"]
		dataset["validation"] = dataset["validation"].train_test_split(
			test_size=1-frac, seed=1
		)["train"]
		dataset["test"] = dataset["test"].train_test_split(
			test_size=1-frac, seed=1
		)["train"]
		print(f"⚡ Using {frac*100:.0f}% of data for quick testing")
		
	spm_path = Path(cfg.tokenizer_model)
	if not spm_path.is_absolute():
		spm_path = Path(get_original_cwd()) / spm_path

	if spm_path.exists():
		print(f"Loading SentencePiece tokenizer from {spm_path} ...")
		tokenizer = spm.SentencePieceProcessor(model_file=str(spm_path))
		src_vocab_size = tgt_vocab_size = tokenizer.get_piece_size()
	else:
		print("No SentencePiece model provided, falling back to Marian tokenizer...")
		from transformers import AutoTokenizer
		tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
		src_vocab_size = tgt_vocab_size = tokenizer.vocab_size
			
	# make sure vocab sizes in cfg are consistent
	cfg.src_vocab_size = src_vocab_size
	cfg.tgt_vocab_size = tgt_vocab_size

	train_dataset = IWSLTDataset(
		dataset["train"], tokenizer,
		src_lang=cfg.src_lang, tgt_lang=cfg.tgt_lang,
		pad_idx=cfg.pad_idx, bos_idx=cfg.bos_idx, eos_idx=cfg.eos_idx,
		max_len=getattr(cfg, "max_len", None),
	)
	test_dataset = IWSLTDataset(
		dataset["test"], tokenizer,
		src_lang=cfg.src_lang, tgt_lang=cfg.tgt_lang,
		pad_idx=cfg.pad_idx, bos_idx=cfg.bos_idx, eos_idx=cfg.eos_idx,
		max_len=getattr(cfg, "max_len", None),
	)

	collate = partial(collate_fn, pad_idx=cfg.pad_idx)

	train_loader = DataLoader(
		train_dataset, batch_size=cfg.batch_size, shuffle=True,
		num_workers=cfg.num_workers, collate_fn=collate
	)
	test_loader = DataLoader(
		test_dataset, batch_size=cfg.batch_size, shuffle=False,
		num_workers=cfg.num_workers, collate_fn=collate
	)

	return train_loader, test_loader, (cfg.src_vocab_size, cfg.tgt_vocab_size)