# wc3kin/wc3mdl/parse_mdl.py
"""
Warcraft 3 MDL text parser.

Produces a raw AST-like structure preserving:
- Block hierarchy (Type ["Name"] { ... })
- Statement lines (including inline-braced statements like Interval { ... })
- Keyframe lines (t: { ... } or t: number)
- Tuple-like data lines ({ x, y, z }) as plain statements (builder can interpret)

NO semantic interpretation is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import re

_HDR_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*(?:"([^"]*)")?\s*(?:(-?\d+(?:\.\d+)?))?\s*\{\s*$')
_HDR_NAME_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*(?:"([^"]*)")?\s*(?:(-?\d+(?:\.\d+)?))?\s*\{')
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Block:
    type: str
    name: Optional[str]
    body: List[Any]
    tag: Optional[float] = None   # NEW: numeric tag after type/name (e.g. Alpha 1 {)


class MDLParser:
    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.i = 0

    def _is_block_header(self, line: str) -> bool:
        # True only for "Type {", 'Type "Name" {'
        return _HDR_RE.match(line.rstrip()) is not None

    def parse(self) -> List[Block]:
        blocks: List[Block] = []
        while self.i < len(self.lines):
            line = self._clean(self.lines[self.i])
            if not line:
                self.i += 1
                continue

            if self._is_block_header(line):
                blocks.append(self._parse_block())
            else:
                # top-level stray stmt lines are ignored
                self.i += 1
        return blocks

    def _parse_block(self) -> Block:
        header = self._clean(self.lines[self.i])
        self.i += 1

        m = _HDR_NAME_RE.match(header)
        if not m:
            raise ValueError(f"Bad block header: {header!r} \n Invalid block header at line {self.i}: {header!r}")
        if not m:
            raise ValueError(f"Invalid block header at line {self.i}: {header!r}")

        block_type = m.group(1)
        name = m.group(2)
        tag_s = m.group(3)
        tag = float(tag_s) if tag_s is not None else None
        
        body: List[Any] = []
        while self.i < len(self.lines):
            line = self._clean(self.lines[self.i])
            if not line:
                self.i += 1
                continue

            if line == "}":
                self.i += 1
                break

            # IMPORTANT: only recurse on TRUE block headers.
            if self._is_block_header(line):
                body.append(self._parse_block())
            else:
                body.append(self._parse_statement(line))
                self.i += 1

        return Block(block_type, name, body,tag=tag)

    def _parse_statement(self, line: str):
        # Keyframe line examples:
        #   0: { 0, 0, 0 }
        #   1600: 1
        # We keep values raw-ish (float or tuple[float,...])
        if ":" in line:
            t_str, rest = line.split(":", 1)
            t_str = t_str.strip()
            if t_str.isdigit() or (t_str.startswith("-") and t_str[1:].isdigit()):
                return ("key", int(t_str), self._parse_value(rest))
        return ("stmt", line)

    def _parse_value(self, text: str):
        nums = _NUM_RE.findall(text)
        if not nums:
            return None
        if len(nums) == 1:
            # keep ints as ints when possible (useful for flags/vis)
            n = nums[0]
            return int(n) if n.isdigit() or (n.startswith("-") and n[1:].isdigit()) else float(n)
        return tuple(float(n) for n in nums)

    def _clean(self, line: str) -> str:
        # Strip comments (MDL commonly uses //)
        if "//" in line:
            line = line.split("//", 1)[0]
        return line.strip().rstrip(",")


def parse_mdl(path: str) -> List[Block]:
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        return MDLParser(f.read()).parse()
