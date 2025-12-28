# wc3kin/wc3mdl/parse_mdl.py
"""
Warcraft 3 MDL text parser.

Produces a raw AST-like structure preserving:
- Block hierarchy
- Identifiers
- Numeric literals
- Keyframe times and values (absolute)
- Flags and interpolation modes

NO semantic interpretation is performed here.
"""

from dataclasses import dataclass
from typing import Any, List, Dict, Union
import re

Token = Union[str, float, int]

@dataclass
class Block:
    type: str
    name: str | None
    body: List[Any]

class MDLParser:
    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.i = 0

    def parse(self) -> List[Block]:
        blocks = []
        while self.i < len(self.lines):
            line = self._clean(self.lines[self.i])
            if not line:
                self.i += 1
                continue
            if "{" in line:
                blocks.append(self._parse_block())
            else:
                self.i += 1
        return blocks

    def _parse_block(self) -> Block:
        header = self._clean(self.lines[self.i])
        self.i += 1

        m = re.match(r'(\w+)(?:\s+"([^"]+)")?', header)
        block_type = m.group(1)
        name = m.group(2)

        body = []
        while self.i < len(self.lines):
            line = self._clean(self.lines[self.i])
            if line == "}":
                self.i += 1
                break
            if "{" in line:
                body.append(self._parse_block())
            else:
                body.append(self._parse_statement(line))
                self.i += 1
        return Block(block_type, name, body)

    def _parse_statement(self, line: str):
        if ":" in line:
            t, rest = line.split(":", 1)
            return ("key", int(t.strip()), self._parse_value(rest))
        return ("stmt", line)

    def _parse_value(self, text: str):
        nums = re.findall(r"-?\d+\.?\d*", text)
        if len(nums) == 1:
            return float(nums[0])
        return tuple(float(n) for n in nums)

    def _clean(self, line: str) -> str:
        return line.strip().rstrip(",")

def parse_mdl(path: str) -> List[Block]:
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        return MDLParser(f.read()).parse()
