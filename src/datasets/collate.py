import torch
from torch.nn.utils.rnn import pad_sequence


def process_spectrograms(items):
    """
    Process spectrograms: squeeze, calculate lengths, and pad them.
    """
    spectrograms = [item["spectrogram"].squeeze(0) for item in items]
    lengths = torch.tensor([spec.size(1) for spec in spectrograms])
    max_len = lengths.max().item()

    padded = torch.stack([
        torch.nn.functional.pad(spec, (0, max_len - spec.size(1)), mode="constant", value=0)
        for spec in spectrograms
    ])

    return padded, lengths


def process_text_encoded(items):
    """
    Process encoded text: squeeze, calculate lengths, and pad them.
    """
    encoded_texts = [item["text_encoded"].squeeze(0) for item in items]
    lengths = torch.tensor([enc.size(0) for enc in encoded_texts])

    padded = pad_sequence(encoded_texts, batch_first=True)

    return padded, lengths


def gather_metadata(items):
    """
    Gather metadata like raw text and audio paths.
    """
    texts = [item["text"] for item in items]
    paths = [item["audio_path"] for item in items]

    return texts, paths


def create_batch(items):
    """
    Create a batch from a list of dataset items.
    """
    spectrograms, spec_lengths = process_spectrograms(items)
    text_encoded, text_lengths = process_text_encoded(items)
    texts, audio_paths = gather_metadata(items)

    batch = {
        "spectrogram": spectrograms,
        "spectrogram_length": spec_lengths,
        "text_encoded": text_encoded,
        "text_encoded_length": text_lengths,
        "text": texts,
        "audio_path": audio_paths,
    }

    return batch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version of the tensors.
    """
    return create_batch(dataset_items)
