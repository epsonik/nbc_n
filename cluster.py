import time

import neighbours
import ti_neighbours
import pandas as pd


def save_to_file(output_path, points, clusters):
    noise_points_count = 0
    for _, value in sorted(clusters.items(), key=lambda kv: kv[0]):
        if value == -1:
            noise_points_count += 1
    print("Number of noise point {}".format(noise_points_count))
    output_file = pd.DataFrame({"X": points[0], "Y": points[1],"CLUSTER": list(clusters.values())})
    output_file.to_csv(output_path, index=False)

def nbc(vectors, k, ref_point) :
    clusters = {}
    for idx, _ in enumerate(vectors):
        clusters[idx] = -1
    if ref_point is not None:
        group_start_time = time.time()
        knb, r_knb = ti_neighbours.ti_k_neighbourhood(vectors, k, ref_point)
        grouping_time = (time.time() - group_start_time)
    else:
        group_start_time = time.time()
        knb, r_knb = neighbours.k_neighbourhood(vectors, k)
        grouping_time = (time.time() - group_start_time)
    print("Grouping time --- %s seconds ---" % (grouping_time))

    ndf = neighbours.ndf(knb, r_knb)

    current_cluster_id = 0
    for idx, v in enumerate(vectors):
        if _has_cluster(idx, clusters) or not _check_is_dp(idx, ndf):
            continue
        clusters[idx] = current_cluster_id
        d_points = set()

        for n_idx in knb[idx]:
            clusters[n_idx] = current_cluster_id
            if _check_is_dp(n_idx, ndf):
                d_points.add(n_idx)

        while d_points:
            dp = d_points.pop()
            for n_idx in knb[dp]:
                if _has_cluster(n_idx, clusters):
                    continue
                clusters[n_idx] = current_cluster_id
                if _check_is_dp(n_idx, ndf):
                    d_points.add(n_idx)

        current_cluster_id += 1

    return clusters, current_cluster_id

# check if point is dense point
def _check_is_dp(idx, ndf):
    return ndf[idx] >= 1


def _has_cluster(idx, clusters):
    return clusters[idx] != -1