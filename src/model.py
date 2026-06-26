"""
LSTM Sequence Model — WC2026 Predictor
-----------------------------------------
Architecture rationale: form and Elo trajectory are inherently temporal.
Rather than hand-engineering "win rate over last 10 games" as a single
scalar (which throws away ORDER — a team on a 5-game winning streak ending
today is in a different state than one that won 5 of its last 10 scattered
randomly), we feed the raw sequence into an LSTM and let it learn what
patterns of recent results actually predict outcomes.

Two LSTM towers (one per team) process form sequences in parallel, sharing
weights (a team's recent form should be evaluated the same way regardless
of whether they're designated "home" or "away" in the data — symmetry by
design, not an assumption we forgot to make).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TeamFormEncoder(nn.Module):
    """
    Shared-weight LSTM encoder for a single team's recent form sequence.
    Input: sequence of result tokens (0=L, 1=D, 2=W, -1=padding/no-data)
    Output: a fixed-size embedding summarising the team's recent trajectory.
    """
    def __init__(self, n_result_tokens: int = 4, embed_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        # +1 because we shift -1 (padding) to index 0 before embedding
        self.token_embed = nn.Embedding(n_result_tokens, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, form_seq: torch.Tensor) -> torch.Tensor:
        # form_seq: (batch, seq_len) with values in {-1,0,1,2} -> shift to {0,1,2,3}
        shifted = form_seq + 1
        embedded = self.token_embed(shifted)          # (batch, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(embedded)              # h_n: (1, batch, hidden_dim)
        return h_n.squeeze(0)                          # (batch, hidden_dim)


class EloTrajectoryEncoder(nn.Module):
    """
    Shared-weight LSTM encoder for a team's Elo rating trajectory
    (continuous values, not discrete tokens like form).
    """
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(1)

    def forward(self, elo_traj: torch.Tensor) -> torch.Tensor:
        # elo_traj: (batch, n_years) -> normalise and add feature dim
        x = elo_traj.unsqueeze(-1)                     # (batch, n_years, 1)
        # Normalise per-sequence (Elo scale is ~1000-2200, LSTMs prefer small inputs)
        x = (x - 1500.0) / 200.0
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)


class WC2026Predictor(nn.Module):
    """
    Full model: two teams' form + Elo trajectories, combined with static
    features (elo_diff, defending-champion flags, neutral venue), feeding
    into dense layers ending in a 3-way softmax over {home_win, draw, away_win}.
    """
    def __init__(
        self,
        form_hidden_dim: int = 16,
        elo_hidden_dim: int = 16,
        static_feature_dim: int = 7,
        dense_hidden_dim: int = 32,
    ):
        super().__init__()
        # Shared encoders — same weights for home and away team
        self.form_encoder = TeamFormEncoder(hidden_dim=form_hidden_dim)
        self.elo_encoder = EloTrajectoryEncoder(hidden_dim=elo_hidden_dim)

        combined_dim = (form_hidden_dim * 2) + (elo_hidden_dim * 2) + static_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dense_hidden_dim, dense_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dense_hidden_dim // 2, 3),  # home_win, draw, away_win
        )

    def forward(
        self,
        home_form: torch.Tensor,
        away_form: torch.Tensor,
        home_elo_traj: torch.Tensor,
        away_elo_traj: torch.Tensor,
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns raw logits (batch, 3). Apply softmax externally for
        probabilities, or use nn.CrossEntropyLoss directly on logits
        during training.
        """
        home_form_emb = self.form_encoder(home_form)
        away_form_emb = self.form_encoder(away_form)   # SAME weights — shared encoder

        home_elo_emb = self.elo_encoder(home_elo_traj)
        away_elo_emb = self.elo_encoder(away_elo_traj)  # SAME weights — shared encoder

        combined = torch.cat([
            home_form_emb, away_form_emb,
            home_elo_emb, away_elo_emb,
            static_features,
        ], dim=1)

        logits = self.classifier(combined)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test with random data
    model = WC2026Predictor()
    batch_size = 4
    home_form = torch.randint(-1, 3, (batch_size, 12))
    away_form = torch.randint(-1, 3, (batch_size, 12))
    home_elo = torch.rand(batch_size, 5) * 400 + 1300
    away_elo = torch.rand(batch_size, 5) * 400 + 1300
    static = torch.rand(batch_size, 7)

    out = model(home_form, away_form, home_elo, away_elo, static)
    print(f"Output shape: {out.shape}")  # (4, 3)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Sample output (logits):\n{out}")
    print(f"Sample output (probabilities):\n{torch.softmax(out, dim=1)}")
