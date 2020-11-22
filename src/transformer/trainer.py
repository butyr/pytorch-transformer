import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stat_cuda(msg):
    print('--', msg)
    print('allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
    ))


class Trainer:

    def __init__(
            self,
            flags,
            model,
            train_dataset,
            eval_dataset,
            tb_writer,
            vocab_size,
            save_path=None,
            eval_size=1_000,
    ):
        self.flags = flags
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = self._get_dataloader(
            self.train_dataset, self.flags.train_batch_size
        )
        self.eval_dataloader = self._get_dataloader(
            self.eval_dataset, self.flags.eval_batch_size
        )
        self.tb_writer = tb_writer
        self.vocab_size = vocab_size
        self.eval_size = eval_size
        self.save_path = save_path if save_path is not None else '../checkpoints/model.ckp'

        self.optimizer = self._get_optimizer()
        self.loss_fn = self._get_loss_fn()

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), self.flags.lr)

    @staticmethod
    def _get_loss_fn():
        return nn.CrossEntropyLoss()

    def fit(self):
        print("Train on {0} samples, validate on {1} samples".format(
            self.train_dataset.len, self.eval_dataset.len
        ))

        for epoch in range(self.flags.epochs):
            print("Epoch {0}/{1}".format(epoch, self.flags.epochs))

            for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
                batch_src, batch_tgt = batch
                batch_src = batch_src.to(device)
                batch_tgt = batch_tgt.to(device)

                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(batch_src, batch_tgt)
                loss = self.loss_fn(
                    outputs.reshape(-1, self.vocab_size),
                    batch_tgt.reshape(-1)
                )

                loss.backward()
                self.optimizer.step()

                t = (epoch * len(batch)) + batch_idx

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('Train/loss', float(loss), t)
                    self.tb_writer.add_scalar(
                        'Train/bleu', self._get_bleu_score(outputs, batch_tgt), t
                    )
                    self.tb_writer.add_scalar('Train/perplexity', float(torch.exp(loss)), t)

                del outputs
                del batch_src
                del batch_tgt
                torch.cuda.empty_cache()

                if (batch_idx + 1) % self.flags.eval_rate == 0:
                    valid_loss, bleu = self.evaluate()

                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar('Valid/loss', valid_loss, t)
                        self.tb_writer.add_scalar('Valid/bleu', bleu, t)
                        self.tb_writer.add_scalar(
                            'Valid/perplexity', float(torch.exp(valid_loss)), t
                        )

    def predict(self, inputs):
        with torch.no_grad():
            self.model.eval()

            outputs_dummy = torch.zeros(inputs.shape+(self.vocab_size,))
            return self._predict_loop(inputs.to(device), outputs_dummy.to(device))

    def evaluate(self):
        valid_loss = 0
        bleu = 0

        with torch.no_grad():
            self.model.eval()

            for i, batch in enumerate(tqdm(self.eval_dataloader)):
                batch_src, batch_tgt = batch
                batch_src = batch_src.to(device)
                batch_tgt = batch_tgt.to(device)

                outputs = self.model(batch_src, batch_tgt)

                """
                batch_dummy = torch.zeros(
                    batch_tgt.shape+(self.vocab_size,)
                ).to(device)
                outputs = self._predict_loop(batch_src, batch_dummy)
                """

                valid_loss += float(self.loss_fn(
                    outputs.reshape(-1, self.vocab_size),
                    batch_tgt.reshape(-1)
                ))
                bleu += self._get_bleu_score(outputs, batch_tgt)

                if i >= self.eval_size-1:
                    break

        num_batches = min(
            len(self.eval_dataset)//self.flags.eval_batch_size,
            self.eval_size
        )

        return valid_loss/num_batches, bleu/num_batches

    def _predict_loop(self, batch_src, batch_dummy):
        tgt_sentence_len = batch_dummy.shape[1]

        for _ in range(tgt_sentence_len):
            batch_dummy = self.model(
                batch_src,
                torch.argmax(batch_dummy, dim=2)
            )

        return batch_dummy

    def _get_bleu_score(self, outputs, batch_tgt):
        decoded = list(map(self._decode_single, outputs, batch_tgt))
        num_batches = batch_tgt.shape[0]

        del outputs
        del batch_tgt
        torch.cuda.empty_cache()

        candidates, references = list(map(list, zip(*decoded)))
        bleu = sum(map(bleu_score, candidates, references))

        return bleu/num_batches

    def _decode_single(self, output, tgt):
        candidate_corpus = self.train_dataset.tokenizer.decode(
            torch.argmax(output, dim=-1).cpu().detach().numpy()
        ).split()

        references_corpus = self.train_dataset.tokenizer.decode(
            tgt.cpu().detach().numpy()
        ).split(' ')

        return [candidate_corpus], [[references_corpus]]

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.save_path))

    def _get_dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.flags.train_shuffle,
            num_workers=self.flags.num_workers,
            collate_fn=self.train_dataset.pad_collate,
        )

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
