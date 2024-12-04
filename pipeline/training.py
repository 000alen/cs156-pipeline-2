import mailbox
import email

from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gensim import corpora, models
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from bs4 import BeautifulSoup

from torch.utils.tensorboard import SummaryWriter

from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn

from .dataset import EmailDataset
from .model import TopicGuidedVAE, device
from .utils import save_checkpoint, load_checkpoint
from .logging import logger


# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Enable CuDNN benchmark for optimized performance
cudnn.benchmark = True


STOP_WORDS = set(stopwords.words("english"))
GMAIL_CATEGORY_HEADER_MARKER = "Categoría:"


def decode_mime_str(encoded: str) -> str:
    # Decodes a MIME-encoded string to a regular string.
    if not encoded:
        return ""
    fragments = email.header.decode_header(encoded)
    decoded = "".join(
        (
            fragment.decode(charset or "utf-8", errors="ignore")
            if isinstance(fragment, bytes)
            else fragment
        )
        for fragment, charset in fragments
    )
    return decoded


def parse_gmail_labels(gmail_labels_str: str) -> str:
    # Parses Gmail labels to extract the primary category.
    gmail_labels = gmail_labels_str.split(",")
    category_label = "Uncategorized"
    for label in gmail_labels:
        if GMAIL_CATEGORY_HEADER_MARKER in label:
            category_label = label.replace(GMAIL_CATEGORY_HEADER_MARKER, "").strip()
            break
    return category_label


def parse_body(message: mailbox.Message) -> str:
    # Extracts and decodes the body of an email message.
    body_parts = []
    try:
        if message.is_multipart():
            for part in message.walk():
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode("utf-8", errors="ignore"))
        else:
            payload = message.get_payload(decode=True)
            if payload:
                body_parts.append(payload.decode("utf-8", errors="ignore"))
    except Exception as e:
        logger.error(f"Error extracting body: {e}")
    return " ".join(body_parts)


def parse_message(message) -> Optional[Tuple[str, str, str]]:
    # Parses an email message to extract the subject, body, and category.
    try:
        subject = decode_mime_str(message.get("subject", ""))
        body = parse_body(message)
        category = parse_gmail_labels(
            decode_mime_str(message.get("X-Gmail-Labels", ""))
        )
        return subject, body, category
    except Exception as e:
        logger.error(f"Failed to process an email: {e}")
        return None


def load_emails(mbox_file_path: str, max_emails: Optional[int] = None) -> pd.DataFrame:
    # Loads and parses emails from an MBOX file using parallel processing.
    columns = ["Subject", "Body", "Category"]
    data = []
    mbox = mailbox.mbox(mbox_file_path)
    n_processes = max(cpu_count() - 1, 1)
    with Pool(processes=n_processes) as pool:
        for i, result in enumerate(
            tqdm(pool.imap_unordered(parse_message, mbox, chunksize=100))
        ):
            if result:
                data.append(result)
            if max_emails is not None and len(data) >= max_emails:
                break
    return pd.DataFrame(data, columns=columns)


def parse_html(html: str) -> str:
    # Parses HTML content and extracts text.
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator=" ", strip=True)


def transform_parse_html(
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
):
    if columns is None:
        columns = ["Body"]
    for column in columns:
        df[column] = df[column].apply(lambda x: parse_html(x))


def clean_text(text: str) -> str:
    # Normalizes the text by converting to lowercase.
    text = text.lower()
    return text


def transform_clean_text(
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
):
    if columns is None:
        columns = ["Subject", "Body"]
    for column in columns:
        df[column] = df[column].apply(lambda x: clean_text(x))


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Preprocesses the email data by cleaning text.
    transform_parse_html(df)
    transform_clean_text(df)
    df["Text"] = df["Subject"] + " " + df["Body"]
    return df


def preprocess_text_for_topic_modeling(text):
    tokens = simple_preprocess(text, deacc=True)  # deacc=True removes punctuations
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


# Convert topic distributions to fixed-size vectors
def topic_vector(topic_dist, num_topics):
    vec = np.zeros(num_topics)
    for topic_id, prob in topic_dist:
        vec[topic_id] = prob
    return vec


def optimize_model_for_training(model):
    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Move model to GPU
    model = model.to(device)
    
    # Compile model with torch 2.0+
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune')
            logger.info("Successfully compiled model with torch.compile()")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
    
    return model


def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    writer,
    num_epochs=20,
    accumulation_steps=4,
    patience=2,
    save_every: int = 2000,
    log_every: int = 1000,
    sample_every: int = 5000,
    resume_from_checkpoint: Optional[str] = None,
):
    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0
    
    # Create checkpoints directory if it doesn't exist
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Resume from checkpoint if specified
    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        logger.info(f"Loading checkpoint from {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Pre-allocate tensors on GPU
    torch.cuda.empty_cache()
    torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use 95% of available GPU memory
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        total_ce_loss = 0
        total_kld_loss = 0
        model.train()
        
        # Use torch.cuda.amp.autocast context manager for the entire epoch
        with autocast(device_type='cuda', dtype=torch.float16):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (input_seq, target_seq, topic_vec) in enumerate(progress_bar):
                # Efficient data transfer to GPU
                input_seq = input_seq.to(device, non_blocking=True)
                target_seq = target_seq.to(device, non_blocking=True)
                topic_vec = topic_vec.to(device, non_blocking=True)
                
                # Forward pass
                logits, mu, logvar = model(input_seq, topic_vec)
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_seq.view(-1),
                    reduction="mean",
                )
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = ce_loss + kld_loss
                loss = loss / accumulation_steps
                
                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Update metrics
                total_loss += loss.item() * accumulation_steps
                total_ce_loss += ce_loss.item()
                total_kld_loss += kld_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item() * accumulation_steps:.4f}",
                        "ce_loss": f"{ce_loss.item():.4f}",
                        "kld_loss": f"{kld_loss.item():.4f}",
                    }
                )
                
                # Log and save checkpoints
                if batch_idx % log_every == 0:
                    step = epoch * len(dataloader) + batch_idx
                    writer.add_scalar("Loss/total_step", loss.item() * accumulation_steps, step)
                    writer.add_scalar("Loss/ce_step", ce_loss.item(), step)
                    writer.add_scalar("Loss/kld_step", kld_loss.item(), step)
                    
                    # Generate sample text using the model
                    if hasattr(dataloader.dataset, 'idx2char'):
                        model.eval()
                        with torch.no_grad():
                            # Use the first sequence from the batch as seed
                            seed_seq = input_seq[0:1]  # Shape: [1, seq_len]
                            seed_topic = topic_vec[0:1]  # Shape: [1, topic_dim]
                            generated_text = ""
                            
                            # Convert initial sequence to text
                            for idx in seed_seq[0][:10].cpu().numpy():  # Show first 10 chars
                                generated_text += dataloader.dataset.idx2char[idx]
                            
                            generated_text += " -> "  # Separator
                            
                            # Generate 50 new characters
                            current_seq = seed_seq  # Shape: [1, seq_len]
                            # Get latent representation
                            mu, _ = model.encoder(current_seq, seed_topic)
                            z = mu  # Use mean for deterministic output
                            
                            for _ in range(50):
                                logits = model.decoder(current_seq, z, seed_topic)
                                next_char_logits = logits[:, -1, :]  # Shape: [1, vocab_size]
                                next_char_probs = F.softmax(next_char_logits / 0.7, dim=-1)  # Add temperature
                                next_char_idx = torch.multinomial(next_char_probs, 1)  # Shape: [1, 1]
                                next_char = dataloader.dataset.idx2char[next_char_idx.item()]
                                generated_text += next_char
                                
                                # Update sequence for next iteration
                                current_seq = torch.cat([
                                    current_seq[:, 1:],  # Remove first character
                                    next_char_idx,  # Add new character, already shaped [1, 1]
                                ], dim=1)
                            
                            logger.info(f"Sample text: {generated_text}")
                        model.train()
                
                if batch_idx % save_every == 0:
                    torch.cuda.empty_cache()  # Clear cache before saving
                    checkpoint_path = f"checkpoints/model_epoch{epoch}_batch{batch_idx}.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "loss": loss.item() * accumulation_steps,
                            "ce_loss": ce_loss.item(),
                            "kld_loss": kld_loss.item(),
                            "best_loss": best_loss,
                            "patience_counter": patience_counter
                        },
                        checkpoint_path,
                    )
                
                # Generate sample text periodically
                if batch_idx % sample_every == 0:
                    model.eval()
                    with torch.no_grad():
                        # Use the first topic vector from the batch for sampling
                        sample_topic_vec = topic_vec[0].unsqueeze(0)
                        
                        # Initialize input with start token
                        start_text = "Dear "
                        input_seq = torch.tensor([[dataloader.dataset.char2idx.get(c, 0) for c in start_text]], 
                                               dtype=torch.long, device=device)
                        
                        # Generate sample
                        generated_text = start_text
                        for _ in range(200):  # Generate 200 characters
                            logits, _, _ = model(input_seq, sample_topic_vec)
                            next_char_logits = logits[:, -1, :] / 0.7  # temperature = 0.7
                            next_char_probs = F.softmax(next_char_logits, dim=-1)
                            next_char_idx = torch.multinomial(next_char_probs, 1)
                            next_char = dataloader.dataset.idx2char[next_char_idx.item()]
                            generated_text += next_char
                            
                            # Update input sequence
                            input_seq = torch.cat([input_seq[:, 1:], next_char_idx], dim=1)
                        
                        print(f"\nGenerated sample at step {step}:\n{generated_text}\n{'-'*50}")
                        logger.info(f"\nGenerated sample at step {step}:\n{generated_text}\n{'-'*50}")
                    model.train()

        # Handle remaining gradients after last batch
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_ce_loss = total_ce_loss / len(dataloader)
        avg_kld_loss = total_kld_loss / len(dataloader)

        # Update learning rate scheduler
        scheduler.step(avg_loss)

        # Log to tensorboard
        writer.add_scalar("Loss/total", avg_loss, epoch)
        writer.add_scalar("Loss/cross_entropy", avg_ce_loss, epoch)
        writer.add_scalar("Loss/kld", avg_kld_loss, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # Save best model and check for early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": best_loss,
                    "best_loss": best_loss,
                    "patience_counter": patience_counter
                },
                "checkpoints/best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save epoch checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": avg_loss,
                "ce_loss": avg_ce_loss,
                "kld_loss": avg_kld_loss,
                "best_loss": best_loss,
                "patience_counter": patience_counter
            },
            f"checkpoints/model_epoch{epoch}.pt",
        )

        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Cross Entropy Loss: {avg_ce_loss:.4f}")
        logger.info(f"KLD Loss: {avg_kld_loss:.4f}")
        logger.info("-" * 50)

    writer.close()


def main(
    file_path: str,
    max_emails: Optional[int] = None,
    num_topics: int = 5,
    num_epochs: int = 20,
    accumulation_steps: int = 8,
    patience: int = 2,
    seq_length: int = 100,
    resume_from_checkpoint: Optional[str] = None,
):
    # Device configuration (use GPU if available)
    logger.info(f"Using device: {device}")

    # Load the emails
    mbox_path = Path(file_path)

    df = load_checkpoint("preprocessed_df")
    if df is None:
        df = load_emails(str(mbox_path), max_emails=max_emails)
        df = preprocess_data(df)
        save_checkpoint(df, "preprocessed_df")

    # Create dictionary and corpus for LDA
    dictionary = load_checkpoint("dictionary")
    corpus = load_checkpoint("corpus")
    if dictionary is None or corpus is None:
        processed_texts = load_checkpoint("processed_texts")
        if processed_texts is None:
            processed_texts = [
                preprocess_text_for_topic_modeling(text)
                for text in tqdm(df["Text"].tolist(), desc="Processing texts")
            ]
            save_checkpoint(processed_texts, "processed_texts")

        dictionary = corpora.Dictionary(processed_texts)
        save_checkpoint(dictionary, "dictionary")

        corpus = [
            dictionary.doc2bow(text)
            for text in tqdm(processed_texts, desc="Creating corpus")
        ]
        save_checkpoint(corpus, "corpus")

    # Train LDA model
    lda_model = load_checkpoint("lda_model")
    if lda_model is None:
        lda_model = models.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=15,
            random_state=42,
        )
        save_checkpoint(lda_model, "lda_model")

    topic_vectors = load_checkpoint("topic_vectors")
    if topic_vectors is None:
        # Get topic distributions for each document
        topic_distributions = load_checkpoint("topic_distributions")
        if topic_distributions is None:
            topic_distributions = [
                lda_model.get_document_topics(bow)
                for bow in tqdm(corpus, desc="Getting topic distributions")
            ]
            save_checkpoint(topic_distributions, "topic_distributions")

        topic_vectors = [
            topic_vector(td, num_topics)
            for td in tqdm(topic_distributions, desc="Creating topic vectors")
        ]
        save_checkpoint(topic_vectors, "topic_vectors")

    char2idx = load_checkpoint("char2idx")
    if char2idx is None:
        # Create character mapping
        all_text = " ".join(df["Text"])
        chars = sorted(list(set(all_text)))
        char2idx = {char: idx for idx, char in enumerate(chars)}
        # Add special tokens
        char2idx['<unk>'] = len(char2idx)
        char2idx['<pad>'] = len(char2idx)
        save_checkpoint(char2idx, "char2idx")

    idx2char = load_checkpoint("idx2char")
    if idx2char is None:
        idx2char = {idx: char for char, idx in char2idx.items()}
        save_checkpoint(idx2char, "idx2char")

    vocab_size = len(char2idx)
    logger.info(f"Vocabulary size: {vocab_size}")

    # Prepare the dataset and dataloader
    print("Creating dataset...")
    logger.info("Creating dataset...")
    dataset = EmailDataset(df["Text"].tolist(), topic_vectors, char2idx, seq_length)
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Optimize DataLoader settings with smaller batch size
    num_workers = min(4, cpu_count() - 1)  # Reduced number of workers
    dataloader = DataLoader(
        dataset,
        batch_size=32,  # Reduced batch size
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    logger.info(f"Number of batches: {len(dataloader)}")

    # Model hyperparameters
    embedding_dim = 128  # Reduced from 256
    hidden_dim = 256    # Reduced from 512
    latent_dim = 32     # Reduced from 64
    
    logger.info("Model hyperparameters:")
    logger.info(f"- Embedding dimension: {embedding_dim}")
    logger.info(f"- Hidden dimension: {hidden_dim}")
    logger.info(f"- Latent dimension: {latent_dim}")

    # Initialize model with smaller dimensions
    model = TopicGuidedVAE(
        vocab_size, embedding_dim, hidden_dim, latent_dim, num_topics
    ).to(device)
    
    # Use parameter groups with different learning rates
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': 1e-3},
        {'params': decoder_params, 'lr': 1e-3}
    ], weight_decay=1e-5)
    
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6)
    scaler = GradScaler()
    writer = SummaryWriter("runs/topic_guided_vae")

    # Enable memory efficient features
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.memory_stats(device=device)
    
    # Set memory allocation to be more efficient
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of available memory
    
    # Set PyTorch to use cudnn benchmark mode
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 precision where available
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train(
        model,
        dataloader,
        optimizer,
        scheduler,
        scaler,
        writer,
        num_epochs=num_epochs,
        accumulation_steps=accumulation_steps,
        patience=patience,
        resume_from_checkpoint=resume_from_checkpoint,
    )


if __name__ == "__main__":
    main("emails-uni.mbox")
