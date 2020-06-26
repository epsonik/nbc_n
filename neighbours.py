import time

from distance_metrics import distance


def k_neighbourhood(vec, k):
    knb, r_knb = _init(vec)
    sort_time = 0
    for idx1, v1 in enumerate(vec):
        neighbour_cand = []
        for idx2, v2 in enumerate(vec):
            if idx1 != idx2:
                dist = distance(v1, v2)
                neighbour_cand.append((idx2, dist))
        sort_start_time = time.time()
        neighbour_cand.sort(key=lambda t: t[1])
        sort_time += (time.time() - sort_start_time)

        eps = neighbour_cand[:k][-1][1]

        neighbours = set()
        for (i, d) in neighbour_cand:
            if d > eps:
                break
            neighbours.add(i)
        _fill(knb, r_knb, idx1, neighbours)
    print("Sorting time --- %s seconds ---" % (sort_time))
    return knb, r_knb


def ndf(knb, r_knb):
    ndfs = {}
    for k in knb.keys():
        ndfs[k] = len(r_knb[k]) / len(knb[k])
    return ndfs

def _init(vec):
    knb = {}
    r_knb = {}
    for i in range(vec.shape[0]):
        r_knb[i] = set()
    return knb, r_knb

def _fill(knb, r_knb, vec_idx, neighbours):
    knb[vec_idx] = neighbours
    for n in neighbours:
        r_knb[n].add(vec_idx)
