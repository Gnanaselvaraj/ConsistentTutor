from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "sentence-transformers is required for embeddings. "
        "Install with `pip install sentence-transformers`"
    ) from e

# Fast model with GPU support
EMBED_MODEL = "all-MiniLM-L6-v2"

# Use GPU if available for faster processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate model once per process
_MODEL = SentenceTransformer(EMBED_MODEL, device=DEVICE)


def embed_texts_batched(texts, batch_size=32, workers=1, progress_cb=None):
    """Embed a list of texts using SentenceTransformers with GPU support.

    Returns a numpy array of shape (len(texts), dim) with dtype float32.
    The `workers` parameter controls how many threads encode batches in parallel.
    """
    if not texts:
        return np.zeros((0, _MODEL.get_sentence_embedding_dimension()), dtype="float32")

    # Use larger batch size for GPU
    if DEVICE == "cuda":
        batch_size = max(batch_size, 64)

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    vectors = [None] * len(batches)
    total = len(batches)

    def _encode_batch(idx, batch):
        emb = _MODEL.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return idx, emb

    # Single GPU pass is faster than multi-threading
    if DEVICE == "cuda" or workers is None or workers <= 1 or total == 1:
        done = 0
        for i, b in enumerate(batches):
            _, emb = _encode_batch(i, b)
            vectors[i] = emb
            done += 1
            if progress_cb:
                progress_cb(done, total)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
            futures = [pool.submit(_encode_batch, i, b) for i, b in enumerate(batches)]
            for f in as_completed(futures):
                i, emb = f.result()
                vectors[i] = emb
                done += 1
                if progress_cb:
                    progress_cb(done, total)

    stacked = np.vstack(vectors).astype("float32")
    return stacked
