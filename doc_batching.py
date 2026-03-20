from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


class DocumentWindowStream:
    def __init__(self, next_token: Callable[[], int], boundary_token: int):
        self.next_token = next_token
        self.boundary_token = int(boundary_token)
        self.current_doc: list[int] = []
        self.current_pos = 0
        self.pending_boundary = False

    def _load_next_doc(self) -> None:
        doc: list[int] = [self.boundary_token] if self.pending_boundary else []
        self.pending_boundary = False
        while not doc:
            token = int(self.next_token())
            if token == self.boundary_token:
                doc.append(token)
        while True:
            token = int(self.next_token())
            if token == self.boundary_token:
                self.pending_boundary = True
                break
            doc.append(token)
        self.current_doc = doc
        self.current_pos = 0

    def take(
        self,
        num_sequences: int,
        seq_len: int,
        *,
        pad_token: int,
        ignore_index: int,
    ) -> tuple[Tensor, Tensor]:
        x = torch.full((num_sequences, seq_len), int(pad_token), dtype=torch.int64)
        y = torch.full((num_sequences, seq_len), int(ignore_index), dtype=torch.int64)
        for seq_idx in range(num_sequences):
            filled = 0
            while filled < seq_len:
                if not self.current_doc or self.current_pos >= len(self.current_doc) - 1:
                    self._load_next_doc()
                available = len(self.current_doc) - 1 - self.current_pos
                take = min(seq_len - filled, available)
                x[seq_idx, filled : filled + take] = torch.tensor(
                    self.current_doc[self.current_pos : self.current_pos + take], dtype=torch.int64
                )
                y[seq_idx, filled : filled + take] = torch.tensor(
                    self.current_doc[self.current_pos + 1 : self.current_pos + 1 + take], dtype=torch.int64
                )
                self.current_pos += take
                filled += take
                if self.current_pos >= len(self.current_doc) - 1:
                    break
        return x, y
