import datetime
import gc
import hashlib
import json
import os
import re
import warnings

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import hnswlib
import numpy as np
import requests

from bs4 import BeautifulSoup
from pypdf import PdfReader
from rich.console import Console
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

HOME_DIR = Path.home() / ".curlang"
HOME_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "hnsw_index.bin"
MAX_CHUNK_WORDS = 100
MAX_ELEMENTS = 5000
METADATA_FILE = "metadata.json"
SENTENCE_CHUNK_SIZE = 5
VECTOR_DIMENSION = 384


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text)
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


class VectorManager:
    def __init__(self, model_id="all-MiniLM-L6-v2", directory="./data"):
        self.directory = directory

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.index_file = os.path.join(self.directory, INDEX_FILE)
        self.metadata_file = os.path.join(self.directory, METADATA_FILE)
        self.embeddings_file = os.path.join(self.directory, EMBEDDINGS_FILE)

        (HOME_DIR / "cache").mkdir(parents=True, exist_ok=True)

        locks_dir = HOME_DIR / "cache" / ".locks"
        if locks_dir.exists():
            for lock_file in locks_dir.glob("**/*.lock"):
                try:
                    lock_file.unlink()
                except OSError:
                    pass

        self.model = (
            model_id
            if isinstance(model_id, SentenceTransformer)
            else SentenceTransformer(
                model_id, device="cpu", cache_folder=HOME_DIR / "cache"
            )
        )

        self.index = hnswlib.Index(space="cosine", dim=VECTOR_DIMENSION)
        self.metadata, self.hash_set, self.embeddings, self.ids = self._load_metadata_and_embeddings()
        self._initialize_index()

        gc.collect()

    def _initialize_index(self):
        if os.path.exists(self.index_file):
            self.index.load_index(self.index_file, max_elements=MAX_ELEMENTS)
        else:
            self.index.init_index(
                max_elements=MAX_ELEMENTS,
                ef_construction=200,
                M=64
            )

            if self.embeddings is not None and len(self.embeddings) > 0:
                batch_size = 1000

                for i in range(0, len(self.embeddings), batch_size):
                    batch_end = min(i + batch_size, len(self.embeddings))
                    self.index.add_items(
                        self.embeddings[i:batch_end], self.ids[i:batch_end]
                    )
                    gc.collect()

        self.index.set_ef(100)

    def _load_metadata_and_embeddings(self):
        metadata = {}
        hash_set = set()
        embeddings = None
        ids = []

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as file:
                chunk_size = 1024 * 1024
                buffer = ""

                while True:
                    chunk = file.read(chunk_size)

                    if not chunk:
                        break

                    buffer += chunk

                    try:
                        metadata.update(json.loads(buffer))
                        buffer = ""
                    except json.JSONDecodeError:
                        continue

                    hash_set.update(metadata.keys())
                    gc.collect()

        if os.path.exists(self.embeddings_file):
            embeddings = np.load(self.embeddings_file)
            ids = list(map(int, metadata.keys()))

        return metadata, hash_set, embeddings, ids

    def _save_metadata_and_embeddings(self):
        with open(self.metadata_file, "w") as file:
            json.dump(self.metadata, file, indent=2)

        if self.embeddings is not None:
            np.save(
                self.embeddings_file,
                self.embeddings,
                allow_pickle=False
            )

        self.index.save_index(self.index_file)

    def is_index_ready(self):
        return self.index.get_current_count() > 0

    @staticmethod
    def _generate_positive_hash(text):
        hash_object = hashlib.sha256(text.encode())
        return int(hash_object.hexdigest()[:16], 16)

    def add_texts(self, texts: List[str], source_reference: str):
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            batch_embeddings = []
            batch_ids = []
            batch_entries = {}

            hashes = [
                self._generate_positive_hash(text) for text in batch_texts
            ]

            new_texts = [
                (h, text)
                for h, text in zip(hashes, batch_texts)
                if h not in self.hash_set
            ]

            if new_texts:
                batch_embeddings = self.model.encode(
                    [text for _, text in new_texts],
                    normalize_embeddings=True,
                    batch_size=len(new_texts),
                )

                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for idx, (text_hash, text) in enumerate(new_texts):
                    self.hash_set.add(text_hash)
                    batch_ids.append(text_hash)
                    batch_entries[text_hash] = {
                        "hash": text_hash,
                        "source": source_reference,
                        "date": now,
                        "text": text,
                    }

                if len(batch_embeddings) > 0:
                    batch_embeddings = np.array(batch_embeddings)

                    if self.embeddings is None:
                        self.embeddings = batch_embeddings
                    else:
                        self.embeddings = np.vstack(
                            (self.embeddings, batch_embeddings)
                        )

                    self.index.add_items(
                        batch_embeddings,
                        batch_ids
                    )

                    self.metadata.update(batch_entries)
                    self._save_metadata_and_embeddings()

            gc.collect()

    def search_vectors(
            self, query: str,
            top_k: int = 10,
            recency_weight: float = 0.5
    ) -> List[Dict[str, Any]]:

        if not self.is_index_ready():
            return []

        query = query.strip()

        if not query:
            return []

        try:
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False
            )

            actual_k = min(top_k, self.index.get_current_count())

            if actual_k < 1:
                return []

            labels, distances = self.index.knn_query(
                query_embedding,
                k=actual_k
            )

            if len(labels) == 0 or len(labels[0]) == 0:
                return []

            results = []
            now = datetime.datetime.now()

            for idx, distance in zip(labels[0], distances[0]):
                str_idx = str(idx)

                if str_idx in self.metadata:
                    meta = self.metadata[str_idx]

                    doc_date = datetime.datetime.strptime(
                        meta["date"], "%Y-%m-%d %H:%M:%S"
                    )

                    recency_score = 1 / (
                            1 + (now - doc_date).total_seconds() / 86400
                    )

                    combined_score = (
                            recency_weight * recency_score +
                            (1 - recency_weight) * (1 - distance)
                    )

                    results.append({
                        "id": int(idx),
                        "text": meta["text"],
                        "source": meta["source"],
                        "distance": float(distance),
                        "recency_score": recency_score,
                        "combined_score": combined_score,
                    })

            results = sorted(
                results,
                key=lambda x: x["combined_score"],
                reverse=True
            )

            return results[:top_k]
        except Exception:
            return []

    @staticmethod
    def _preprocess_text(text: str) -> str:
        text = " ".join(text.split())
        text = text.replace("•", ".")
        text = text.replace("…", "...")
        text = text.replace("\n", " ")
        return text.strip()

    def _process_text_and_add(self, text: str, source_reference: str):
        if not isinstance(text, str) or not text.strip():
            return

        text = self._preprocess_text(text)
        sentences = split_into_sentences(text)

        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            if sentence_word_count < 3:
                continue

            if current_word_count + sentence_word_count > MAX_CHUNK_WORDS:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)

                    if len(chunk_text.split()) >= 10:
                        self.add_texts([chunk_text], source_reference)

                    current_chunk = []
                    current_word_count = 0

            current_chunk.append(sentence)
            current_word_count += sentence_word_count

            if len(current_chunk) >= SENTENCE_CHUNK_SIZE:
                chunk_text = " ".join(current_chunk)

                if len(chunk_text.split()) >= 10:
                    self.add_texts([chunk_text], source_reference)

                current_chunk = []
                current_word_count = 0

        if current_chunk:
            chunk_text = " ".join(current_chunk)

            if len(chunk_text.split()) >= 10:
                self.add_texts([chunk_text], source_reference)

        gc.collect()

    def add_pdf(self, pdf_path: str):
        is_valid_file = os.path.isfile(pdf_path)
        is_pdf = pdf_path.lower().endswith(".pdf")

        if not is_valid_file or not is_pdf:
            return

        with open(pdf_path, "rb") as file:
            pdf = PdfReader(file)

            page_texts = []
            current_text_length = 0

            for page in pdf.pages:
                text = page.extract_text() or ""

                if text:
                    page_texts.append(text)
                    current_text_length += len(text)

                    if current_text_length >= 10000:
                        combined_text = " ".join(page_texts)

                        self._process_text_and_add(
                            combined_text,
                            pdf_path
                        )

                        page_texts = []
                        current_text_length = 0
                        gc.collect()

            if page_texts:
                combined_text = " ".join(page_texts)
                self._process_text_and_add(combined_text, pdf_path)

    def add_url(self, url: str):
        try:
            with requests.get(url, timeout=10, stream=True) as response:
                response.raise_for_status()
                content = b""

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk

                        if len(content) >= 1_000_000:
                            soup = BeautifulSoup(content, "html.parser")
                            text = soup.get_text(
                                separator=" ",
                                strip=True
                            )
                            self._process_text_and_add(text, url)
                            content = b""
                            gc.collect()

                if content:
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    self._process_text_and_add(text, url)
        except requests.RequestException:
            pass

    @staticmethod
    def get_wikipedia_text(page_title):
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "redirects": 1,
            "format": "json",
        }

        try:
            response = requests.get(
                base_url,
                params=params,
                timeout=10
            )

            response.raise_for_status()

            data = response.json()
            page = next(
                iter(data["query"]["pages"].values())
            )
            return page.get("extract", "")
        except requests.RequestException:
            return ""

    def add_wikipedia_page(self, page_title):
        try:
            text = self.get_wikipedia_text(page_title)

            if text:
                self._process_text_and_add(text, f"wikipedia:{page_title}")
        except requests.RequestException:
            pass
