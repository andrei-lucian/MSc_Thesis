from src.models.resnet_variants import resnet18_preact, resnet34_preact, basenet18, basenet18_constant
from src.models.cnn import simple_cnn
from src.models.transformer import Seq2SeqTransformer

def get_model(cfg, *args):
	arch = cfg.arch.lower()

	if arch == "transformer":
		src_vocab_size, tgt_vocab_size = args
		from src.models.transformer import Seq2SeqTransformer
		return Seq2SeqTransformer(
			src_vocab_size=src_vocab_size,
			tgt_vocab_size=tgt_vocab_size,
			d_model=cfg.width,
			nhead=cfg.nhead,
			num_encoder_layers=cfg.num_layers,
			num_decoder_layers=cfg.num_layers,
			dropout=cfg.dropout,
			tie_decoder_embeddings=cfg.tie_decoder_embeddings,
			max_len=getattr(cfg, "max_len", 5000)
		)
	
	num_classes, = args
	k = cfg.width

	if arch == "resnet18":
		return resnet18_preact(num_classes=num_classes, k=k)
	elif arch == "resnet34":
		return resnet34_preact(num_classes=num_classes, k=k)
	elif arch == "simplecnn":
		return simple_cnn(num_classes=num_classes, k=k)
	elif arch == "basenet18":
		return basenet18(num_classes=num_classes, first_n_linear=getattr(cfg, "first_n_linear", 0), k=k)
	elif arch == "basenetconstant18":
		return basenet18_constant(num_classes=num_classes, first_n_linear=getattr(cfg, "first_n_linear", 0), k=k)
	else:
		raise ValueError(f"Unknown architecture: {cfg.arch}")
