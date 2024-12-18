import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_topics):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + num_topics, hidden_dim, batch_first=True)
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.use_checkpointing = True

    def forward(self, x, topic_vec):
        x = self.embedding(x)
        topic_vec_expanded = topic_vec.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, topic_vec_expanded], dim=2)
        
        if self.use_checkpointing and self.training:
            def lstm_function(x):
                return self.lstm(x)[0]
            outputs = checkpoint(lstm_function, x, use_reentrant=False)
            _, (h_n, _) = self.lstm(x)
        else:
            _, (h_n, _) = self.lstm(x)
            
        h_n = h_n.squeeze(0)
        mu = self.hidden_to_mu(h_n)
        logvar = self.hidden_to_logvar(h_n)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_topics):
        super(Decoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim + num_topics, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.outputs2vocab = nn.Linear(hidden_dim, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.use_checkpointing = True

    def forward(self, x, z, topic_vec):
        z = torch.cat([z, topic_vec], dim=1)
        h_0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        c_0 = torch.zeros_like(h_0).to(device)
        x = self.embedding(x)
        
        if self.use_checkpointing and self.training:
            def lstm_function(x, h_0, c_0):
                return self.lstm(x, (h_0, c_0))[0]
            outputs = checkpoint(lstm_function, x, h_0, c_0, use_reentrant=False)
        else:
            outputs, _ = self.lstm(x, (h_0, c_0))
            
        logits = self.outputs2vocab(outputs)
        return logits


class TopicGuidedVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_topics):
        super(TopicGuidedVAE, self).__init__()
        self.encoder = Encoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, num_topics
        )
        self.decoder = Decoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, num_topics
        )

    def forward(self, x, topic_vec):
        mu, logvar = self.encoder(x, topic_vec)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        logits = self.decoder(x, z, topic_vec)
        return logits, mu, logvar

    @torch.no_grad()
    def generate(self, input_seq, topic_vec, max_length=500, temperature=0.7, deterministic=False):
        """
        Generate text sequence given an input sequence and topic vector.
        
        Args:
            input_seq (torch.Tensor): Starting sequence [batch_size, seq_len]
            topic_vec (torch.Tensor): Topic distribution [batch_size, num_topics]
            max_length (int): Maximum length of generated sequence
            temperature (float): Sampling temperature (lower = more conservative)
            deterministic (bool): If True, use mean of latent distribution instead of sampling
            
        Returns:
            torch.Tensor: Generated sequence indices [batch_size, max_length]
        """
        # Get latent representation
        mu, logvar = self.encoder(input_seq, topic_vec)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        
        # Initialize output sequence with input
        generated = input_seq
        
        # Generate one token at a time
        for _ in range(max_length - input_seq.size(1)):
            # Get next token probabilities
            logits = self.decoder(generated, z, topic_vec)
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(next_token_probs, 1)
            
            # Concatenate with generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


def loss_function(logits, targets, mu, logvar, kld_weight=0.01):
    CE = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean"
    )
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return CE + kld_weight * KLD
