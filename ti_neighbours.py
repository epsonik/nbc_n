import numpy as np

from distance_metrics import distance
from point import Point

from typing import Set, Dict, Tuple, List

from sortedcontainers import SortedSet
import time

def ti_k_neighbourhood(vec, k, ref_point):
    knb, r_knb = _init(vec)

    points = _ti(vec, ref_point)
    for point in points:
        idx = point.idx
        neighbours = _ti_neighbours(point, k)
        _fill(knb, r_knb, idx, neighbours)

    return knb, r_knb

def _ti_neighbours(point, k):
    bp = point
    fp = point
    back_search = bp.preceding
    bp = bp.preceding
    forw_search = fp.following
    fp = fp.following

    neighbour_cand = SortedSet(key=lambda x: x[1])

    bp, fp, back_search, forw_search = _cand_nbs(back_search,
                                                 forw_search,
                                                 neighbour_cand,
                                                 k,
                                                 point,
                                                 bp,
                                                 fp)
    eps = neighbour_cand[-1][1]
    eps = _ver_back(point, bp, back_search, neighbour_cand, k, eps)
    eps = _verify_forw(point, fp, forw_search, neighbour_cand, k, eps)
    return [n[0].idx for n in neighbour_cand]


def _cand_nbs(back_search,
              forw_search,
              neighbour_cand: SortedSet,
              k,
              p,
              bp,
              fp):
    i = 0
    while forw_search and back_search and i < k:
        if p.dist - bp.dist < fp.dist - p.dist:
            dist = distance(bp.vec, p.vec)
            neighbour_cand.add((bp, dist))
            back_search = bp.preceding
            bp = bp.preceding
        else:
            dist = distance(fp.vec, p.vec)
            neighbour_cand.add((fp, dist))
            forw_search = fp.following
            fp = fp.following
        i += 1
    while forw_search and i < k:
        dist = distance(fp.vec, p.vec)
        i += 1
        neighbour_cand.add((fp, dist))
        forw_search = fp.following
        fp = fp.following
    while back_search and i < k:
        dist = distance(bp.vec, p.vec)
        i += 1
        neighbour_cand.add((bp, dist))
        back_search = bp.preceding
        bp = bp.preceding
    return bp, fp, back_search, forw_search


def _ti(vectors,
        ref_point):
    rp_dist = []
    for idx, v in enumerate(vectors):
        dist = distance(v, ref_point)
        rp_dist.append(dist)
    sort_start_time = time.time()
    arg_sorted_rp_dist = np.argsort(rp_dist)
    print("Sorting time TI--- %s seconds ---" % (time.time() - sort_start_time))
    points = []
    for i, vector_id in enumerate(arg_sorted_rp_dist):
        if i == 0:
            points.append(Point(vector_id, vectors[vector_id], rp_dist[vector_id]))
        else:
            point = Point(vector_id, vectors[vector_id], rp_dist[vector_id], preceding=points[i - 1])
            points.append(point)
            points[i - 1].following = point
    return points




def _verify_forw(p, fp, forw_search, neighbour_cand: SortedSet, k, eps):
    while forw_search and (p.dist - fp.dist) <= eps:
        dist = distance(fp.vec, p.vec)
        if dist < eps:
            i = len([e for e in neighbour_cand if e[1] == eps])
            if len(neighbour_cand) - i >= k - 1:
                for e in neighbour_cand:
                    if e[1] == eps:
                        neighbour_cand.remove(e)
                neighbour_cand.add((fp, dist))
                eps = neighbour_cand[-1][1]
            else:
                neighbour_cand.add((fp, dist))
        elif dist == eps:
            neighbour_cand.add((fp, dist))
        forw_search = fp.following
        fp = fp.following
    return eps


def _ver_back(p, bp, back_search, neighbour_cand: SortedSet, k, eps):
    while back_search and (p.dist - bp.dist) <= eps:
        dist = distance(bp.vec, p.vec)
        if dist < eps:
            i = len([e for e in neighbour_cand if e[1] == eps])
            if len(neighbour_cand) - i >= k - 1:
                for e in neighbour_cand:
                    if e[1] == eps:
                        neighbour_cand.remove(e)
                neighbour_cand.add((bp, dist))
                eps = neighbour_cand[-1][1]
            else:
                neighbour_cand.add((bp, dist))
        elif dist == eps:
            neighbour_cand.add((bp, dist))
        back_search = bp.preceding
        bp = bp.preceding
    return eps

def _fill(knb, r_knb, vec_idx, neighbours):
    knb[vec_idx] = neighbours
    for n in neighbours:
        r_knb[n].add(vec_idx)


def _init(vec):
    knb = {}
    r_knb = {}
    for i in range(vec.shape[0]):
        r_knb[i] = set()
    return knb, r_knb