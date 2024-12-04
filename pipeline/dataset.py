import torch
from torch.utils.data import Dataset


class EmailDataset(Dataset):
    def __init__(self, texts, topic_vectors, char2idx, seq_length):
        self.texts = texts
        self.topic_vectors = topic_vectors
        self.char2idx = char2idx
        self.seq_length = seq_length
        # Add special tokens if they don't exist
        if '<unk>' not in self.char2idx:
            self.char2idx['<unk>'] = len(self.char2idx)
        if '<pad>' not in self.char2idx:
            self.char2idx['<pad>'] = len(self.char2idx)
        self.unk_idx = self.char2idx['<unk>']
        self.pad_idx = self.char2idx['<pad>']
        # Create idx2char mapping
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self._calculate_valid_indices()

    def _calculate_valid_indices(self):
        """Pre-calculate valid text indices and their sequence counts."""
        self.text_mappings = []
        for idx, text in enumerate(self.texts):
            text_len = len(text)
            if text_len >= self.seq_length + 1:
                num_sequences = text_len - self.seq_length
                self.text_mappings.extend([(idx, pos) for pos in range(num_sequences)])

    def __len__(self):
        return len(self.text_mappings)

    def _encode_sequence(self, sequence):
        """Safely encode a sequence of characters to indices."""
        return [self.char2idx.get(char, self.unk_idx) for char in sequence]

    def __getitem__(self, idx):
        text_idx, pos = self.text_mappings[idx]
        text = self.texts[text_idx]
        topic_vec = self.topic_vectors[text_idx]

        # Get the sequence slice
        sequence = text[pos:pos + self.seq_length + 1]
        
        # Encode the sequence with proper handling of unknown characters
        encoded = torch.tensor(
            self._encode_sequence(sequence),
            dtype=torch.long
        )
        
        input_seq = encoded[:-1]
        target_seq = encoded[1:]
        topic_vec = torch.tensor(topic_vec, dtype=torch.float32)

        return input_seq, target_seq, topic_vec
