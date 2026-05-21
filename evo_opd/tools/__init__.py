"""Tool wrappers for agentic-OPD: search / read / propose.

All tools are pure local: no API calls, deterministic, fast (<10ms each).
This is essential for fast rollouts in RL — a single tool call per turn must
not bottleneck the loop.
"""
from .search import SearchTool, SearchResult
from .read import ReadTool
from .propose import parse_propose_action

__all__ = ["SearchTool", "SearchResult", "ReadTool", "parse_propose_action"]
