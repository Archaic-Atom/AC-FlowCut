import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReductionTraceEvent:
    stage: str
    layer_idx: int
    num_tokens_before: int
    num_tokens_after: int
    target_num: int
    prune_num: int
    sample_id: Optional[int] = None
    entropy: Optional[float] = None
    keep_indices: Optional[List[int]] = None
    score_summary: Dict[str, Any] = field(default_factory=dict)


class ReductionTrace:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.events = []

    def clear(self):
        self.events = []

    def add(self, event):
        if not self.enabled:
            return
        if isinstance(event, ReductionTraceEvent):
            self.events.append(event)
        else:
            self.events.append(ReductionTraceEvent(**event))

    def to_list(self):
        return [asdict(event) for event in self.events]

    def save_jsonl(self, path):
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for event in self.to_list():
                f.write(json.dumps(event) + "\n")
