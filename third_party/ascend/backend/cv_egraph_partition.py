# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# E-graph based block partition generator for Ascend DynamicCVPipeline.
# Compatible with egglog >= 13.2.0 (Python 3.11+).
# Requires: pip install egglog

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

CORE_OTHER = -1
CORE_CUBE = 1
CORE_VECTOR = 2

try:
    from egglog import (
        EGraph,
        Expr,
        birewrite,
        eq,
        function,
        i64,
        i64Like,
        ne,
        relation,
        rewrite,
        rule,
        ruleset,
        set_,
    )
    _HAS_EGGLOG = True
except ImportError:
    _HAS_EGGLOG = False

if _HAS_EGGLOG:
    class Seg(Expr):
        @classmethod
        def op(cls, idx: i64Like, core: i64Like) -> Seg: ...

        @classmethod
        def seq(cls, lhs: Seg, rhs: Seg) -> Seg: ...

        @classmethod
        def fuse(cls, lhs: Seg, rhs: Seg) -> Seg: ...

    @function
    def seg_core(s: Seg) -> i64: ...

    indep = relation("indep", i64, i64)

    @ruleset
    def analysis_rules(s: Seg, a: Seg, b: Seg, i: i64, c1: i64):
        yield rule(eq(s).to(Seg.op(i, c1))).then(set_(seg_core(s)).to(c1))
        yield rule(
            eq(s).to(Seg.fuse(a, b)),
            eq(seg_core(a)).to(c1),
            eq(seg_core(b)).to(c1),
        ).then(set_(seg_core(s)).to(c1))

    @ruleset
    def partition_rules(a: Seg, b: Seg, c: Seg, i: i64, j: i64, c1: i64, c2: i64):
        yield birewrite(Seg.seq(Seg.seq(a, b), c)).to(Seg.seq(a, Seg.seq(b, c)))
        yield birewrite(Seg.fuse(Seg.fuse(a, b), c)).to(Seg.fuse(a, Seg.fuse(b, c)))
        yield rewrite(Seg.seq(Seg.op(i, c1), Seg.op(j, c2))).to(
            Seg.seq(Seg.op(j, c2), Seg.op(i, c1)),
            indep(i, j),
        )
        yield birewrite(Seg.seq(a, b)).to(
            Seg.fuse(a, b),
            eq(seg_core(a)).to(c1),
            eq(seg_core(b)).to(c1),
            ne(c1).to(i64(CORE_OTHER)),
        )


def _compute_independent_pairs(num_nodes: int, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return all node pairs (i, j) with i < j that have no directed path between them."""
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            adj[u].add(v)

    reach: List[Optional[Set[int]]] = [None] * num_nodes

    def dfs(u: int, visiting: Set[int]) -> Set[int]:
        if reach[u] is not None:
            return reach[u]
        visiting.add(u)
        r: Set[int] = set(adj[u])
        for nxt in adj[u]:
            if nxt not in visiting:
                r |= dfs(nxt, visiting)
        visiting.remove(u)
        reach[u] = r
        return r

    for idx in range(num_nodes):
        if reach[idx] is None:
            dfs(idx, set())

    pairs: List[Tuple[int, int]] = []
    for i in range(num_nodes):
        ri = reach[i] or set()
        for j in range(i + 1, num_nodes):
            rj = reach[j] or set()
            if j not in ri and i not in rj:
                pairs.append((i, j))
    return pairs


def _build_macro_dag(
    ops: List[Dict[str, Any]],
    edges: List[Tuple[int, int]],
    max_nodes: int,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]], List[List[int]]]:
    """
    When a loop body has more operations than `max_nodes`, compress contiguous
    same-block runs into macro-nodes so equality saturation stays fast.
    If there is room under `max_nodes`, split the largest multi-op macro-nodes
    in half so the E-graph can still explore re-fusing / re-splitting them.
    """
    if len(ops) <= max_nodes:
        macro_ops = [
            {
                "idx": i,
                "core": int(op["core"]),
                "block_id": int(op["block_id"]),
                "bytes": int(op.get("bytes", 0)),
            }
            for i, op in enumerate(ops)
        ]
        groups = [[i] for i in range(len(ops))]
        return macro_ops, edges, groups

    groups: List[List[int]] = []
    for i, op in enumerate(ops):
        core = int(op["core"])
        bid = int(op["block_id"])
        if (
            groups
            and core != CORE_OTHER
            and int(ops[groups[-1][-1]]["core"]) == core
            and int(ops[groups[-1][-1]]["block_id"]) == bid
        ):
            groups[-1].append(i)
        else:
            groups.append([i])

    while len(groups) < max_nodes:
        best_g = -1
        best_len = 1
        for gi, grp in enumerate(groups):
            if len(grp) > best_len and int(ops[grp[0]]["core"]) != CORE_OTHER:
                best_len = len(grp)
                best_g = gi
        if best_g < 0:
            break
        grp = groups[best_g]
        mid = len(grp) // 2
        groups[best_g : best_g + 1] = [grp[:mid], grp[mid:]]

    op_to_macro: Dict[int, int] = {}
    macro_ops: List[Dict[str, Any]] = []
    for mi, grp in enumerate(groups):
        for op_idx in grp:
            op_to_macro[op_idx] = mi
        macro_ops.append(
            {
                "idx": mi,
                "core": int(ops[grp[0]]["core"]),
                "block_id": int(ops[grp[0]]["block_id"]),
                "bytes": sum(int(ops[k].get("bytes", 0)) for k in grp),
            }
        )

    macro_edge_set: Set[Tuple[int, int]] = set()
    for u, v in edges:
        mu = op_to_macro.get(u)
        mv = op_to_macro.get(v)
        if mu is not None and mv is not None and mu != mv:
            macro_edge_set.add((mu, mv))

    return macro_ops, sorted(macro_edge_set), groups


def _parse_seg_to_block_map(expr_str: str) -> Dict[int, int]:
    """
    Parse an extracted egglog Seg expression such as:
      Seg.seq(Seg.fuse(Seg.op(0, 2), Seg.op(2, 2)), Seg.op(1, 1))
    into a {node_idx: new_block_id} mapping.
    """
    tokens = re.findall(r"Seg\.\w+|-?\d+|[(),]", expr_str)
    pos = 0

    def parse():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")
        tok = tokens[pos]
        pos += 1
        if tok == "Seg.op":
            if tokens[pos] != "(":
                raise ValueError(f"Expected '(' after Seg.op, got {tokens[pos]}")
            pos += 1
            idx = int(tokens[pos])
            pos += 1
            if tokens[pos] != ",":
                raise ValueError(f"Expected ',' in Seg.op, got {tokens[pos]}")
            pos += 1
            _core = int(tokens[pos])
            pos += 1
            if tokens[pos] != ")":
                raise ValueError(f"Expected ')' after Seg.op, got {tokens[pos]}")
            pos += 1
            return ("op", idx)
        if tok in ("Seg.seq", "Seg.fuse"):
            if tokens[pos] != "(":
                raise ValueError(f"Expected '(' after {tok}, got {tokens[pos]}")
            pos += 1
            lhs = parse()
            if tokens[pos] != ",":
                raise ValueError(f"Expected ',' in {tok}, got {tokens[pos]}")
            pos += 1
            rhs = parse()
            if tokens[pos] != ")":
                raise ValueError(f"Expected ')' after {tok}, got {tokens[pos]}")
            pos += 1
            return (tok, lhs, rhs)
        raise ValueError(f"Unexpected token in Seg expression: {tok}")

    tree = parse()
    block_map: Dict[int, int] = {}
    next_bid = 0

    def collect_ops(node, out: List[int]) -> None:
        if node[0] == "op":
            out.append(node[1])
        else:
            collect_ops(node[1], out)
            collect_ops(node[2], out)

    def assign(node) -> None:
        nonlocal next_bid
        if node[0] == "Seg.seq":
            assign(node[1])
            assign(node[2])
        else:
            members: List[int] = []
            collect_ops(node, members)
            bid = next_bid
            next_bid += 1
            for node_idx in members:
                block_map[node_idx] = bid

    assign(tree)
    return block_map


def _has_block_cycle(block_map: Dict[int, int], edges: List[Tuple[int, int]]) -> bool:
    """Return True if grouping nodes by `block_map` introduces a cycle in `edges`."""
    bids = set(block_map.values())
    adj: Dict[int, Set[int]] = {b: set() for b in bids}
    indeg: Dict[int, int] = {b: 0 for b in bids}
    for u, v in edges:
        bu = block_map.get(u)
        bv = block_map.get(v)
        if bu is not None and bv is not None and bu != bv and bv not in adj[bu]:
            adj[bu].add(bv)
            indeg[bv] += 1

    ready = [b for b, d in indeg.items() if d == 0]
    visited = 0
    while ready:
        cur = ready.pop()
        visited += 1
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    return visited != len(bids)


def _canonicalize_block_map(op_block_map: Dict[int, int], num_ops: int) -> str:
    """Normalize block IDs by first-appearance order and encode as '0:0,1:1,...'."""
    remap: Dict[int, int] = {}
    next_id = 0
    parts: List[str] = []
    for idx in range(num_ops):
        raw_bid = op_block_map[idx]
        if raw_bid not in remap:
            remap[raw_bid] = next_id
            next_id += 1
        parts.append(f"{idx}:{remap[raw_bid]}")
    return ",".join(parts)


def generate_egraph_block_maps(
    dag_json_path: str,
    max_candidates: int = 16,
    sat_iters: Optional[int] = None,
    max_nodes: Optional[int] = None,
) -> List[str]:
    """
    Build an E-graph from the dumped main mixed-loop DAG JSON and extract up to
    `max_candidates` unique, acyclic block_id maps for TRITON_ASCEND_CV_BLOCK_MAP.
    """
    if not _HAS_EGGLOG:
        print("[AddDynamicCVPipeline] egglog is not installed (pip install egglog); skipping E-graph search.")
        return []

    if sat_iters is None:
        try:
            sat_iters = max(1, int(os.environ.get("TRITON_ASCEND_CV_EGRAPH_ITERS", "4")))
        except ValueError:
            sat_iters = 4
    if max_nodes is None:
        try:
            max_nodes = max(4, int(os.environ.get("TRITON_ASCEND_CV_EGRAPH_MAX_NODES", "24")))
        except ValueError:
            max_nodes = 24

    with open(dag_json_path, "r", encoding="utf-8") as f:
        dag = json.load(f)

    raw_ops = dag.get("ops", [])
    raw_edges = [(int(e[0]), int(e[1])) for e in dag.get("edges", [])]
    if len(raw_ops) < 2:
        return []

    baseline_map = {i: int(op["block_id"]) for i, op in enumerate(raw_ops)}
    baseline_encoded = _canonicalize_block_map(baseline_map, len(raw_ops))

    nodes, edges, groups = _build_macro_dag(raw_ops, raw_edges, max_nodes=max_nodes)
    if len(nodes) < 2:
        return []

    egraph = EGraph()
    for i, j in _compute_independent_pairs(len(nodes), edges):
        egraph.register(indep(i64(i), i64(j)), indep(i64(j), i64(i)))

    root = Seg.op(nodes[0]["idx"], nodes[0]["core"])
    for k in range(1, len(nodes)):
        leaf = Seg.op(nodes[k]["idx"], nodes[k]["core"])
        if nodes[k]["block_id"] == nodes[k - 1]["block_id"] and nodes[k]["core"] != CORE_OTHER:
            root = Seg.fuse(root, leaf)
        else:
            root = Seg.seq(root, leaf)
    root_expr = egraph.let("root", root)

    # Saturate core-type analysis before each partition rewrite iteration.
    egraph.run((analysis_rules.saturate() + partition_rules) * sat_iters)

    extract_limit = max(max_candidates * 3, 16)
    if hasattr(egraph, "extract_multiple"):
        extracted = egraph.extract_multiple(root_expr, extract_limit)
    else:
        extracted = [egraph.extract(root_expr)]

    seen: Set[str] = {baseline_encoded}
    valid_maps: List[str] = []
    for expr in extracted:
        try:
            macro_bmap = _parse_seg_to_block_map(str(expr))
        except Exception:
            continue
        if len(macro_bmap) != len(nodes):
            continue

        op_bmap: Dict[int, int] = {}
        for mi, grp in enumerate(groups):
            bid = macro_bmap[mi]
            for op_idx in grp:
                op_bmap[op_idx] = bid

        if len(op_bmap) != len(raw_ops) or _has_block_cycle(op_bmap, raw_edges):
            continue

        encoded = _canonicalize_block_map(op_bmap, len(raw_ops))
        if encoded not in seen:
            seen.add(encoded)
            valid_maps.append(encoded)
            if len(valid_maps) >= max_candidates:
                break

    return valid_maps
