"""Trajectory data structures for agentic-OPD rollouts.

Each trajectory is a sequence of (assistant generation → tool observation)
pairs, terminating with a `propose` action. Token-level role tags let the
trainer mask out observation tokens (no gradient on them).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TokenRole = Literal["thought", "action", "observation", "proposal", "system", "prompt"]


@dataclass
class TokenSpan:
    """Span of tokens with a single role tag."""
    start: int        # inclusive
    end: int          # exclusive
    role: TokenRole


@dataclass
class ActionStep:
    """One model decision in the trajectory."""
    turn: int
    tool: str                            # "search" / "read" / "propose" / "malformed"
    action_args: dict                    # parsed action JSON
    raw_text: str                        # raw model output for this step (before observation)
    observation_text: str = ""           # text returned by the tool
    parsed_proposal: dict | None = None  # only for propose action


@dataclass
class Trajectory:
    """Full ReAct trajectory for one prompt."""
    prompt_id: str
    topic: str
    discipline: str

    # full token stream as seen by the model
    full_ids: list[int] = field(default_factory=list)
    # role tags aligned to full_ids
    token_roles: list[TokenRole] = field(default_factory=list)
    # generated-token mask (True for tokens π_θ emitted; False for prompt/observation)
    gen_mask: list[bool] = field(default_factory=list)

    # action steps (one per turn)
    actions: list[ActionStep] = field(default_factory=list)

    # final state
    final_proposal: dict | None = None         # parsed gene_genome (or None)
    read_paper_ids: list[str] = field(default_factory=list)  # for R_lineage
    search_count: int = 0
    read_count: int = 0
    extract_count: int = 0                     # v2: extract_genome calls
    diff_count: int = 0                        # v2: genome_diff calls
    novelty_count: int = 0                     # v2: novelty_check calls
    propose_emitted: bool = False
    truncated: bool = False                    # ran out of turns or token budget
    malformed_count: int = 0                   # tool-call parse failures

    # diagnostics
    raw_text: str = ""                         # full assistant trace (joined)
    n_generated_tokens: int = 0
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "topic": self.topic,
            "n_actions": len(self.actions),
            "search_count": self.search_count,
            "read_count": self.read_count,
            "extract_count": self.extract_count,
            "diff_count": self.diff_count,
            "novelty_count": self.novelty_count,
            "read_paper_ids": self.read_paper_ids,
            "propose_emitted": self.propose_emitted,
            "truncated": self.truncated,
            "malformed_count": self.malformed_count,
            "final_proposal": self.final_proposal,
            "n_generated_tokens": self.n_generated_tokens,
            "wall_time_s": self.wall_time_s,
        }
