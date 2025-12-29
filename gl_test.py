from __future__ import annotations

import os
from pathlib import Path

import pytest

from wc3kin.wc3mdl.import_mdl import import_mdl
from wc3kin.wc3mdl.parse_mdl import parse_mdl


def _archer_path() -> Path:
    p = r"D:\wc3_all_assets\Units\NightElf\Archer\Archer.mdl"
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def test_importer_sequences_restored():
    p = _archer_path()
    m = import_mdl(str(p))

    assert len(m.sequences) == 13
    names = [s.name for s in m.sequences]
    for n in ["Death", "Decay Flesh", "Decay Bone"]:
        assert n in names

    assert len(m.geosets) == 4
    # geoset_anims is a dict[int, GeosetAnim]
    assert len(m.geoset_anims) == 4


def test_geosetanim_alpha_key_ranges():
    p = _archer_path()
    m = import_mdl(str(p))

    # expected (from your confirmed output)
    expected = {
        0: (1, 199333, 199333, 1),
        1: (1, 199333, 199333, 1),
        2: (19, 167, 199333, 19),
        3: (15, 167, 259333, 15),
    }

    for gid, (nkeys, tmin, tmax, gsid) in expected.items():
        ga = m.geoset_anims[gid]
        keys = ga.alpha.keys if ga.alpha else []
        ts = [k.t for k in keys]
        assert len(keys) == nkeys
        assert (min(ts), max(ts)) == (tmin, tmax)
        assert ga.global_seq_id == gsid


def test_queries_death_decay_semantics():
    p = _archer_path()
    m = import_mdl(str(p))

    from wc3kin.wc3mdl import query

    assert query.geosets_affected_in_sequence(m, "Death") == [2, 3]
    assert query.geosets_affected_in_sequence(m, "Decay Flesh") == [2, 3]
    assert query.geosets_affected_in_sequence(m, "Decay Bone") == [0, 1, 2, 3]

    assert query.geosets_hidden_at_end(m, "Death") == [2, 3]
    assert query.geosets_hidden_at_end(m, "Decay Flesh") == [2]
    assert query.geosets_hidden_at_end(m, "Decay Bone") == [0, 1, 2, 3]


def test_parser_sequences_block_present():
    p = _archer_path()
    ast = parse_mdl(str(p))

    seq_blocks = [b for b in ast if b.type == "Sequences"]
    assert len(seq_blocks) >= 1

    # Expect 13 Anim blocks under the first Sequences block
    sb = seq_blocks[0]
    anims = [x for x in sb.body if getattr(x, "type", None) == "Anim"]
    assert len(anims) == 13