import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import joblib
import logging
from typing import List, Dict

# Import the model architecture from main.py
from .model import TopicGuidedVAE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("email_generation")

# Constants
CHECKPOINT_DIR = Path("checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_components():
    """Load all necessary components for inference"""
    # Load character mappings
    char2idx = joblib.load(CHECKPOINT_DIR / "char2idx.pkl")
    idx2char = joblib.load(CHECKPOINT_DIR / "idx2char.pkl")

    # Load LDA model and dictionary
    lda_model = joblib.load(CHECKPOINT_DIR / "lda_model.pkl")
    dictionary = joblib.load(CHECKPOINT_DIR / "dictionary.pkl")

    return char2idx, idx2char, lda_model, dictionary


def initialize_model(vocab_size: int, num_topics: int):
    """Initialize the model with the same architecture as training"""
    model = TopicGuidedVAE(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=32,
        num_topics=num_topics,
    ).to(device)

    # Load the best model checkpoint
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def generate_email(
    model: TopicGuidedVAE,
    topic_vec: np.ndarray,
    char2idx: Dict,
    idx2char: Dict,
    start_text: str = "Dear ",
    length: int = 500,
    temperature: float = 0.7,
    seq_length: int = 100,
    deterministic: bool = False,
) -> str:
    """
    Generate an email with the given topic vector and starting text.

    Args:
        model: The trained VAE model
        topic_vec: Topic distribution vector
        char2idx: Character to index mapping
        idx2char: Index to character mapping
        start_text: Text to start the generation with
        length: Length of text to generate
        temperature: Controls randomness (lower = more conservative)
        seq_length: Sequence length used during training
        deterministic: Whether to use deterministic generation

    Returns:
        str: Generated email text
    """
    with torch.no_grad():
        # Prepare input sequence
        input_seq = [char2idx.get(c, 0) for c in start_text]
        input_seq = (
            torch.tensor(input_seq[-seq_length:], dtype=torch.long)
            .unsqueeze(0)
            .to(device)
        )
        topic_vec = torch.tensor(topic_vec, dtype=torch.float32).unsqueeze(0).to(device)

        # Pad sequence if needed
        if input_seq.size(1) < seq_length:
            pad_size = seq_length - input_seq.size(1)
            input_seq = F.pad(input_seq, (pad_size, 0), "constant", 0)

        # Generate sequence using the model's generate method
        generated_seq = model.generate(
            input_seq,
            topic_vec,
            max_length=length + len(start_text),
            temperature=temperature,
            deterministic=deterministic
        )

        # Convert generated indices to text
        generated_text = ""
        for idx in generated_seq[0].cpu().numpy():
            generated_text += idx2char[idx]

        return generated_text


def generate_emails_for_topics(
    model: TopicGuidedVAE,
    num_topics: int,
    char2idx: Dict,
    idx2char: Dict,
    samples_per_topic: int = 3,
) -> List[str]:
    """Generate multiple emails for each topic"""
    generated_emails = []

    for topic_idx in range(num_topics):
        logger.info(f"\nGenerating emails for Topic {topic_idx}")

        # Create topic vector (one-hot encoding)
        topic_vec = np.zeros(num_topics)
        topic_vec[topic_idx] = 1.0

        # Generate multiple samples for this topic
        for i in range(samples_per_topic):
            email = generate_email(
                model,
                topic_vec,
                char2idx,
                idx2char,
                start_text="Dear ",
                length=500,
                temperature=0.7,
            )
            generated_emails.append((topic_idx, email))
            logger.info(f"\nSample {i+1}:\n{email}\n{'-'*50}")

    return generated_emails


def main():
    # Load components
    logger.info("Loading model components...")
    char2idx, idx2char, lda_model, dictionary = load_model_components()

    # Initialize model
    vocab_size = len(char2idx)
    num_topics = len(lda_model.get_topics())
    logger.info("Initializing model...")
    model = initialize_model(vocab_size, num_topics)

    # Print topic information
    logger.info("\nTopics discovered by LDA:")
    for idx, topic in lda_model.print_topics(-1):
        logger.info(f"Topic {idx}: {topic}")

    # Generate emails
    logger.info("\nGenerating emails...")
    generated_emails = generate_emails_for_topics(
        model, num_topics, char2idx, idx2char, samples_per_topic=3
    )

    # Save generated emails
    output_file = Path("generated_emails.txt")
    with output_file.open("w", encoding="utf-8") as f:
        for topic_idx, email in generated_emails:
            f.write(f"\nTopic {topic_idx}:\n")
            f.write(f"{email}\n")
            f.write("-" * 50 + "\n")

    logger.info(f"\nGenerated emails have been saved to {output_file}")


if __name__ == "__main__":
    main()
