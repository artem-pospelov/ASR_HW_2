import re

import numpy as np


def calc_cer(target_text, predicted_text) -> float:
    ref_words = [
        w for w in re.sub(r"\s\s+", " ", target_text).lower().strip() if w != ""
    ]
    hyp_words = [
        w for w in re.sub(r"\s\s+", " ", predicted_text).lower().strip() if w != ""
    ]

    cer = _calc_error_rate(ref_words, hyp_words)
    return cer


def calc_wer(target_text, predicted_text):
    ref_words = [
        w for w in re.sub(r"\s\s+", " ", target_text).lower().strip().split() if w != ""
    ]
    hyp_words = [
        w
        for w in re.sub(r"\s\s+", " ", predicted_text).lower().strip().split()
        if w != ""
    ]

    wer = _calc_error_rate(ref_words, hyp_words)
    return wer


def _calc_error_rate(ref_words, hyp_words):
    if len(ref_words) == 0:
        return 0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer
