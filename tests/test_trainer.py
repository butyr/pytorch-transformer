import unittest
from src.transformer.trainer import Trainer
from src.transformer.configurations import Config
from src.transformer.dataset import TextDataset
from src.transformer.transformer import Transformer
from torch.utils.tensorboard import SummaryWriter


class TestTrainer(unittest.TestCase):

    def test_progressbar(self):
        flags = Config(
            nheads=2,
            model_dim=10,
            hidden_dim=10,
            depth=2,
            epochs=1,
        )

        train_dataset = TextDataset(
            path_root='../../ml-datasets/wmt14/',
            path_src="newstest2014.en",
            path_tgt="newstest2014.de",
            path_tokenizer='tokenizer/',
        )

        eval_dataset = TextDataset(
            path_root='../../ml-datasets/wmt14/',
            path_src="newstest2014.en",
            path_tgt="newstest2014.de",
            path_tokenizer='tokenizer/',
        )

        vocab_size = train_dataset.tokenizer.get_vocab_size()
        max_len = max(train_dataset.max_len, eval_dataset.max_len)
        model = Transformer(
            vocab_size=vocab_size,
            model_dim=flags.model_dim,
            hidden_dim=flags.hidden_dim,
            nheads=flags.nheads,
            max_len=max_len,
            depth=flags.depth,
        )

        train_op = Trainer(
            flags=flags,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tb_writer=SummaryWriter(),
            vocab_size=vocab_size,
        )
        train_op.fit()