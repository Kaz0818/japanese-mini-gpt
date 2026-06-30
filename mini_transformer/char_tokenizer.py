from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


@dataclass(frozen=True)
class CharTokenizer:
    token_to_id: dict[str, int]

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)

        # Frequency first keeps common Japanese characters at low ids, while the
        # character sort makes ties deterministic across machines.
        text_tokens = sorted(counter, key=lambda token: (-counter[token], token))
        tokens = list(SPECIAL_TOKENS) + [
            token for token in text_tokens if token not in SPECIAL_TOKENS
        ]
        return cls({token: index for index, token in enumerate(tokens)})

    @classmethod
    def from_files(cls, paths: list[Path]) -> "CharTokenizer":
        texts = [path.read_text(encoding="utf-8") for path in paths]
        return cls.from_texts(texts)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = payload["tokens"]
        if tokens[: len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS):
            raise ValueError(f"Unexpected special token order in {path}")
        return cls({token: index for index, token in enumerate(tokens)})

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_id)
        token_ids.extend(
            self.token_to_id.get(character, self.unk_id)
            for character in text
        )
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        id_to_token = self.id_to_token
        special_tokens_to_skip = {"<pad>", "<bos>", "<eos>"}
        pieces: list[str] = []
        for token_id in token_ids:
            token = id_to_token.get(token_id, "<unk>")
            if skip_special_tokens and token in special_tokens_to_skip:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens = [
            token
            for token, _ in sorted(self.token_to_id.items(), key=lambda item: item[1])
        ]
        payload = {
            "type": "char",
            "special_tokens": list(SPECIAL_TOKENS),
            "tokens": tokens,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
