"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the route_package function in this module.

*****IMPORTANT*****
Team name: Team67
Email address: rpannu@uwaterloo.ca
*******************
"""

from typing import Optional, Dict, List, Tuple, Iterable, Any, Set
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package
import heapq
from collections import defaultdict

# Cache placeholder (unused but kept)
path_cache: Dict[Tuple[str, str], List[str]] = {}
# Track package congestion per timestep
congestion_tracker: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))


# ---------------- Schema-tolerant helpers ----------------

def _as_iter(obj) -> Iterable:
    """Return an iterable over obj whether it is a dict or list."""
    if isinstance(obj, dict):
        return obj.values()
    return obj if obj is not None else []


def _iter_connections(state: GameState) -> Iterable[Any]:
    """Yield connection objects from state.connections (list or dict)."""
    return _as_iter(getattr(state, "connections", []))


def _iter_packages(state: GameState) -> Iterable[Tuple[str, Any]]:
    """
    Yield (pkg_id, pkg) from state.packages whether it's a dict or a list.
    """
    pkgs = getattr(state, "packages", {})
    if isinstance(pkgs, dict):
        for k, v in pkgs.items():
            yield str(k), v
    else:
        for p in pkgs:
            pid = getattr(p, "id", None)
            if pid is None and isinstance(p, dict):
                pid = p.get("id")
            if pid is None:
                pid = str(id(p))  # fallback stable-ish key
            yield str(pid), p


def _pkg_get(pkg: Any, *names, default=None):
    """Get first existing attribute/key from names on a package object/dict."""
    for n in names:
        if hasattr(pkg, n):
            return getattr(pkg, n)
        if isinstance(pkg, dict) and n in pkg:
            return pkg[n]
    return default


def _pkg_current_fc(pkg: Any) -> Optional[str]:
    return _pkg_get(pkg, "current_fc", "current", "location", "at_fc")


def _pkg_destination_fc(pkg: Any) -> Optional[str]:
    return _pkg_get(pkg, "destination_fc", "dest_fc", "destination", "target")


def _pkg_planned_next(pkg: Any) -> Optional[str]:
    return _pkg_get(pkg, "_planned_next", default=None)


def _pkg_set_planned_next(pkg: Any, fc: Optional[str]) -> None:
    try:
        setattr(pkg, "_planned_next", fc)
    except Exception:
        # If it's a dict, set key
        if isinstance(pkg, dict):
            pkg["_planned_next"] = fc


def _conn_get(conn: Any, *names, default=None):
    """Get first existing attribute/key from names on a connection object/dict."""
    for n in names:
        if hasattr(conn, n):
            return getattr(conn, n)
        if isinstance(conn, dict) and n in conn:
            return conn[n]
    return default


def _conn_from(conn: Any) -> str:
    return _conn_get(conn, "from_fc", "from_id", "source", "from", "origin")


def _conn_to(conn: Any) -> str:
    return _conn_get(conn, "to_fc", "to_id", "target", "to", "destination")


def _conn_bandwidth(conn: Any) -> int:
    # Prefer explicit bandwidth-like fields; default to 1 to avoid div-by-zero
    bw = _conn_get(conn, "bandwidth", "capacity", "max_packages", default=1)
    try:
        bw = int(bw)
    except Exception:
        bw = 1
    return max(bw, 1)


def _conn_weight(conn: Any) -> float:
    """
    Base weight/cost for the edge. Accepts multiple field names:
    base_weight, weight, time, cost, latency. Defaults to 1.0.
    """
    w = _conn_get(conn, "base_weight", "weight", "time", "cost", "latency", default=1.0)
    try:
        return float(w)
    except Exception:
        return 1.0


def _conn_key(from_fc: str, to_fc: str) -> str:
    return f"{from_fc}-{to_fc}"


def _conn_lookup(state: GameState) -> Dict[str, Any]:
    lut: Dict[str, Any] = {}
    for c in _iter_connections(state):
        f = _conn_from(c)
        t = _conn_to(c)
        if f and t:
            lut[_conn_key(f, t)] = c
    return lut


def _fc_ids(state: GameState) -> Set[str]:
    """
    Collect FC ids from fulfillment_centers (list or dict), connections, and packages.
    """
    ids: Set[str] = set()

    # From fulfillment_centers
    fcs = getattr(state, "fulfillment_centers", None)
    if isinstance(fcs, dict):
        ids.update(fcs.keys())
    else:
        for fc in _as_iter(fcs or []):
            # Allow objects with .id or dicts with ["id"]
            if hasattr(fc, "id"):
                ids.add(getattr(fc, "id"))
            elif isinstance(fc, dict) and "id" in fc:
                ids.add(fc["id"])

    # From connections
    for c in _iter_connections(state):
        f = _conn_from(c)
        t = _conn_to(c)
        if f:
            ids.add(f)
        if t:
            ids.add(t)

    # From packages
    for _, pkg in _iter_packages(state):
        cur = _pkg_current_fc(pkg)
        dst = _pkg_destination_fc(pkg)
        if cur:
            ids.add(cur)
        if dst:
            ids.add(dst)

    return ids


# ---------------- Core logic ----------------

def dijkstra(state: GameState, source: str, destination: str, current_time: int) -> List[str]:
    """
    Shortest path using Dijkstra with dynamic congestion-aware weights.
    """
    # Build adjacency
    graph: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
    for conn in _iter_connections(state):
        f = _conn_from(conn)
        t = _conn_to(conn)
        if not f or not t:
            continue

        base_weight = _conn_weight(conn)
        bw = _conn_bandwidth(conn)

        # current utilization: count packages planning to use f->t
        packages_on_edge = sum(
            1
            for _, pkg in _iter_packages(state)
            if _pkg_current_fc(pkg) == f and _pkg_planned_next(pkg) == t
        )
        utilization = packages_on_edge / max(bw, 1)

        congestion_factor = 1.0 + (utilization ** 2) * 3.0
        time_penalty = congestion_tracker[current_time].get(_conn_key(f, t), 0) * 0.1
        dynamic_weight = base_weight * congestion_factor + time_penalty

        graph[f].append((t, dynamic_weight, bw))

    nodes = _fc_ids(state)
    if source not in nodes or destination not in nodes:
        return []

    distances = {fc: float("inf") for fc in nodes}
    previous: Dict[str, str] = {}
    distances[source] = 0.0

    pq: List[Tuple[float, str]] = [(0.0, source)]
    visited: Set[str] = set()

    while pq:
        current_dist, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        if current == destination:
            break

        for neighbor, weight, _bw in graph.get(current, []):
            if neighbor in visited:
                continue
            nd = current_dist + weight
            if nd < distances.get(neighbor, float("inf")):
                distances[neighbor] = nd
                previous[neighbor] = current
                heapq.heappush(pq, (nd, neighbor))

    if source == destination:
        return []
    if destination not in previous:
        return []

    # Reconstruct
    path: List[str] = []
    cur = destination
    while cur != source:
        path.append(cur)
        cur = previous.get(cur)
        if cur is None:
            return []
    path.reverse()
    return path


def get_connection_utilization(state: GameState, from_fc: str, to_fc: str) -> float:
    """
    Fraction of planned packages on the edge relative to capacity.
    """
    lut = _conn_lookup(state)
    key = _conn_key(from_fc, to_fc)
    conn = lut.get(key)
    if conn is None:
        return 1.0  # treat missing edge as saturated

    bw = _conn_bandwidth(conn)
    count = 0
    for _, pkg in _iter_packages(state):
        if _pkg_current_fc(pkg) == from_fc and _pkg_planned_next(pkg) == to_fc:
            count += 1
    return count / max(bw, 1)


def route_package(state: GameState, package: Package) -> Optional[str]:
    """
    Adaptive next-hop selection with congestion avoidance.
    """
    cur = _pkg_current_fc(package)
    dst = _pkg_destination_fc(package)
    if cur is None or dst is None:
        return None
    if cur == dst:
        return None

    current_time = getattr(state, "current_time", 0)

    # Primary path
    path = dijkstra(state, cur, dst, current_time)

    # Fallback: pick any outgoing with headroom if no path
    if not path:
        for conn in _iter_connections(state):
            if _conn_from(conn) == cur:
                util = get_connection_utilization(state, cur, _conn_to(conn))
                if util < 0.9:
                    next_fc = _conn_to(conn)
                    _pkg_set_planned_next(package, next_fc)
                    congestion_tracker[current_time][_conn_key(cur, next_fc)] += 1
                    return next_fc
        return None

    # Choose first hop
    next_fc = path[0]

    # If chosen hop is congested, try alternative from same source
    util_main = get_connection_utilization(state, cur, next_fc)
    if util_main > 0.8:
        for conn in _iter_connections(state):
            if _conn_from(conn) == cur and _conn_to(conn) != next_fc:
                alt_to = _conn_to(conn)
                alt_util = get_connection_utilization(state, cur, alt_to)
                if alt_util < 0.5:
                    # ensure alternative is viable
                    alt_path = dijkstra(state, alt_to, dst, current_time)
                    if alt_path:
                        next_fc = alt_to
                        break

    # Record congestion + plan
    congestion_tracker[current_time][_conn_key(cur, next_fc)] += 1
    _pkg_set_planned_next(package, next_fc)
    return next_fc
