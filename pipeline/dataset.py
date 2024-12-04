import torch
from torch.utils.data import Dataset


class EmailDataset(Dataset):
    def __init__(self, texts, topic_vectors, char2idx, seq_length):
        self.texts = texts
        self.topic_vectors = topic_vectors
        self.char2idx = char2idx
        self.seq_length = seq_length
        self.data = self.process_texts()

    def process_texts(self):
        data = []
        for text, topic_vec in zip(self.texts, self.topic_vectors):
            encoded = torch.tensor(
                [self.char2idx.get(char, 0) for char in text], dtype=torch.long
            )
            if len(encoded) < self.seq_length + 1:
                continue

            input_seqs = encoded.unfold(0, self.seq_length, 1)
            target_seqs = encoded[1:].unfold(0, self.seq_length, 1)

            topic_vecs = torch.tensor(topic_vec, dtype=torch.float32).repeat(
                len(input_seqs), 1
            )

            data.extend(zip(input_seqs, target_seqs, topic_vecs))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
