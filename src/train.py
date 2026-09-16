"""Train and compare ML models for WDBC (legacy entrypoint).

Prefer: python -m src.train_tabular --dataset wdbc
This module remains so older docs/commands keep working.
"""

from __future__ import annotations

from src.train_tabular import train_tabular_dataset


def train_all(random_state: int = 42):
    return train_tabular_dataset("wdbc", random_state=random_state)


if __name__ == "__main__":
    train_all()
