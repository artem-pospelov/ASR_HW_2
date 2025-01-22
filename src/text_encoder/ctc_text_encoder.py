import re
from string import ascii_lowercase
from src.utils.io_utils import read_json
import kenlm
import torch
import numpy as np
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        lm_path: str = None,
        lm_vocab_path: str = None,
        bpe_path: str = None,
        **kwargs,
    ):
        self.lm_path = lm_path
        self.lm_vocab_path = lm_vocab_path

        self.alphabet = list(read_json(bpe_path).keys())
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        kenlm_model = kenlm.Model(lm_path)
        with open(lm_vocab_path, "r") as f:
            unigrams = [t.lower() for t in f.read().strip().split("\n")]

        lm = LanguageModel(
                            kenlm_model,
                            unigrams,
                            alpha=0.7,
                            beta=1,
                            unk_score_offset=-10.0,
                            score_boundary=True
        )
        self.decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, False), language_model=lm)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert isinstance(item, int), "Index must be an integer."
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        last_char_ind = self.char2ind[self.EMPTY_TOK]
        filtered_inds = filter(lambda ind: ind not in (last_char_ind, self.char2ind[self.EMPTY_TOK]), inds)
        return ''.join(self.ind2char[ind] for ind in filtered_inds)

    def ctc_beam_search(self, probs, beam_size=100) -> dict[str, float]:
        if type(probs) is torch.Tensor:
            probs = probs.detach().cpu().numpy()
        else:
            probs = np.array(probs)

        hyp = self.decoder.decode(probs, beam_size) #()
        return {"hyp_text": hyp, "prob": 1}

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text