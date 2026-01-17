from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from wordfreq import top_n_list, zipf_frequency

WORD_RE = re.compile(r"^[a-z]+$")


class ClueEngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClueConfig:
    model_name: str = os.getenv(
        "CODEWORDS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    vocab_size: int = int(os.getenv("CODEWORDS_VOCAB_SIZE", "24000"))
    min_zipf: float = float(os.getenv("CODEWORDS_MIN_ZIPF", "3.3"))
    max_zipf: float = float(os.getenv("CODEWORDS_MAX_ZIPF", "5.6"))
    min_len: int = int(os.getenv("CODEWORDS_MIN_LEN", "3"))
    max_len: int = int(os.getenv("CODEWORDS_MAX_LEN", "12"))
    max_count: int = int(os.getenv("CODEWORDS_MAX_COUNT", "4"))
    team_threshold: float = float(os.getenv("CODEWORDS_TEAM_THRESHOLD", "0.45"))
    min_score: float = float(os.getenv("CODEWORDS_MIN_SCORE", "0.05"))
    team_max_weight: float = float(os.getenv("CODEWORDS_TEAM_MAX_WEIGHT", "0.6"))
    team_mean_weight: float = float(os.getenv("CODEWORDS_TEAM_MEAN_WEIGHT", "0.4"))
    opp_weight: float = float(os.getenv("CODEWORDS_OPP_WEIGHT", "0.8"))
    neutral_weight: float = float(os.getenv("CODEWORDS_NEUTRAL_WEIGHT", "0.35"))
    black_weight: float = float(os.getenv("CODEWORDS_BLACK_WEIGHT", "1.25"))
    opp_gate: float = float(os.getenv("CODEWORDS_OPP_GATE", "0.6"))
    black_gate: float = float(os.getenv("CODEWORDS_BLACK_GATE", "0.45"))


class ClueEngine:
    def __init__(self, config: Optional[ClueConfig] = None) -> None:
        self.config = config or ClueConfig()
        self.model = SentenceTransformer(self.config.model_name)
        cache_path = self._cache_path()
        if not self._load_cache(cache_path):
            self.candidate_words = self._build_vocabulary()
            if len(self.candidate_words) < 500:
                raise ClueEngineError("Vocabulary too small. Check word filters.")
            self.candidate_embeddings = self._encode_words(self.candidate_words)
            self._save_cache(cache_path)
        self.embedding_dim = self.candidate_embeddings.shape[1]

    def sample_board_words(self, count: int) -> List[str]:
        if count > len(self.candidate_words):
            raise ClueEngineError("Not enough words to build the board.")
        return random.sample(self.candidate_words, count)

    def generate_clue(
        self,
        team_words: Sequence[str],
        opp_words: Sequence[str],
        neutral_words: Sequence[str],
        black_words: Sequence[str],
        board_words: Sequence[str],
    ) -> Optional[dict]:
        if not team_words:
            return {"word": "pass", "count": 0}

        board_set = {word.lower() for word in board_words}
        allowed_mask = np.array(
            [not self._violates_rules(word, board_set) for word in self.candidate_words],
            dtype=bool,
        )
        if not allowed_mask.any():
            return {"word": "pass", "count": 0}

        candidate_words = [
            word for word, allowed in zip(self.candidate_words, allowed_mask) if allowed
        ]
        candidate_embeddings = self.candidate_embeddings[allowed_mask]

        team_vecs = self._encode_words(team_words)
        opp_vecs = self._encode_words(opp_words) if opp_words else None
        neutral_vecs = self._encode_words(neutral_words) if neutral_words else None
        black_vecs = self._encode_words(black_words) if black_words else None

        team_sim = candidate_embeddings @ team_vecs.T
        coverage = min(len(team_words), max(1, self.config.max_count))
        topk = np.partition(team_sim, -coverage, axis=1)[:, -coverage:]
        team_mean = np.mean(topk, axis=1)
        team_max = np.max(team_sim, axis=1)
        team_signal = (
            self.config.team_max_weight * team_max
            + self.config.team_mean_weight * team_mean
        )

        if opp_vecs is not None and opp_vecs.size:
            opp_risk = np.max(candidate_embeddings @ opp_vecs.T, axis=1)
        else:
            opp_risk = np.zeros(len(candidate_words), dtype=np.float32)

        if neutral_vecs is not None and neutral_vecs.size:
            neutral_risk = np.max(candidate_embeddings @ neutral_vecs.T, axis=1)
        else:
            neutral_risk = np.zeros(len(candidate_words), dtype=np.float32)

        if black_vecs is not None and black_vecs.size:
            black_risk = np.max(candidate_embeddings @ black_vecs.T, axis=1)
        else:
            black_risk = np.zeros(len(candidate_words), dtype=np.float32)

        score = (
            team_signal
            - self.config.opp_weight * opp_risk
            - self.config.neutral_weight * neutral_risk
            - self.config.black_weight * black_risk
        )
        risk_mask = (black_risk <= self.config.black_gate) & (
            opp_risk <= self.config.opp_gate
        )
        if risk_mask.any():
            masked_score = np.where(risk_mask, score, -np.inf)
            best_idx = int(np.argmax(masked_score))
        else:
            best_idx = int(np.argmax(score))

        if score[best_idx] < self.config.min_score:
            best_idx = int(np.argmax(team_max))

        best_word = candidate_words[best_idx]
        team_sim_best = team_sim[best_idx]
        max_sim = float(np.max(team_sim_best))
        dynamic_threshold = max(self.config.team_threshold, max_sim - 0.08)
        count = int(np.sum(team_sim_best >= dynamic_threshold))
        count = max(1, min(count, self.config.max_count, len(team_words)))
        target_idx = np.argsort(team_sim_best)[-count:][::-1]
        targets = [team_words[i] for i in target_idx]

        return {
            "word": best_word,
            "count": count,
            "analysis": {
                "targets": targets,
                "team_score": round(float(team_signal[best_idx]), 3),
                "risk": {
                    "opponent": round(float(opp_risk[best_idx]), 3),
                    "neutral": round(float(neutral_risk[best_idx]), 3),
                    "black": round(float(black_risk[best_idx]), 3),
                },
                "strategy": "contrastive-embedding",
            },
        }

    def _build_vocabulary(self) -> List[str]:
        words = []
        for word in top_n_list("en", n=self.config.vocab_size):
            word = word.lower()
            if not WORD_RE.match(word):
                continue
            if not (self.config.min_len <= len(word) <= self.config.max_len):
                continue
            freq = zipf_frequency(word, "en")
            if freq < self.config.min_zipf or freq > self.config.max_zipf:
                continue
            words.append(word)
        return list(dict.fromkeys(words))

    def _encode_words(self, words: Sequence[str]) -> np.ndarray:
        if not words:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = self.model.encode(words, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def _cache_path(self) -> Path:
        cache_dir = Path(os.getenv("CODEWORDS_CACHE_DIR", ".cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "model": self.config.model_name,
                "vocab_size": self.config.vocab_size,
                "min_zipf": self.config.min_zipf,
                "max_zipf": self.config.max_zipf,
                "min_len": self.config.min_len,
                "max_len": self.config.max_len,
            },
            sort_keys=True,
        )
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return cache_dir / f"codewords_vocab_{key}.npz"

    def _load_cache(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            data = np.load(path, allow_pickle=False)
            self.candidate_words = data["words"].tolist()
            self.candidate_embeddings = data["embeddings"].astype(np.float32)
            return True
        except Exception:
            return False

    def _save_cache(self, path: Path) -> None:
        np.savez_compressed(
            path,
            words=np.asarray(self.candidate_words),
            embeddings=self.candidate_embeddings,
        )

    def _violates_rules(self, candidate: str, board_words: Iterable[str]) -> bool:
        if candidate in board_words:
            return True
        candidate_root = self._root_form(candidate)
        for word in board_words:
            if candidate in word or word in candidate:
                return True
            if candidate_root == self._root_form(word):
                return True
        return False

    @staticmethod
    def _root_form(word: str) -> str:
        for suffix in ("ing", "ers", "er", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word
