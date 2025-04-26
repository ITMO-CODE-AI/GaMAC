extern "C" {

    float mm_dist(
        float* x,
        float* y,
        unsigned int D
    ) {
        float d_acc = 0.0f;
        for (unsigned idx = 0; idx < D; ++idx) {
            float diff = x[idx] - y[idx];
            d_acc += diff * diff;
        }
        return sqrt(d_acc);
    }

    __global__ void get_centroids(
        float* data,
        int* labels,
        unsigned int N,
        unsigned int D,
        unsigned int K,
        float* centroids
    ) {
        unsigned int k_idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int d_idx = blockDim.y * blockIdx.y + threadIdx.y;

        if (k_idx < K && d_idx < D) {
            float c_val = 0.0f;
            unsigned int c_num = 0;

            for (unsigned int x_idx = 0; x_idx < N; ++x_idx) {
                int x_label = labels[x_idx];
                float x_val = data[x_idx * D + d_idx];
                if (x_label == k_idx) {
                    c_val += x_val;
                    c_num++;
                }
            }

            centroids[k_idx * D + d_idx] = c_val / c_num;
        }
    }

    __global__ void get_cent_dists(
        float* cluster,
        unsigned int cl_n,
        unsigned int D,
        float* centroids,
        unsigned int k_idx,
        float* cent_dists
    ) {
        unsigned int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        float* centroid = centroids + k_idx * D;
        if (x_idx < cl_n) {
            float* x_obj = cluster + D * x_idx;
            float xc_dist = mm_dist(x_obj, centroid, D);
            cent_dists[x_idx] = xc_dist;
        }
    }

    __global__ void get_sym_data(
        float* cluster,
        unsigned int cl_n,
        unsigned int D,
        float* centroids,
        unsigned int k_idx,
        float* sym_data
    ) {
        unsigned int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int d_idx = blockDim.y * blockIdx.y + threadIdx.y;
        if (x_idx < cl_n && d_idx < D) {
            float x_val = cluster[x_idx * D + d_idx];
            float c_val = centroids[k_idx * D + d_idx];
            float sym_val = 2 * c_val - x_val;
            sym_data[x_idx * D + d_idx] = sym_val;
        }
    }

    __global__ void get_sym_dists(
        float* cluster,
        unsigned int cl_n,
        unsigned int D,
        float* cent_dists,
        float* sym_data,
        float* sym_dists
    ) {
        unsigned int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (x_idx < cl_n) {
            float d_near = 1e38f;
            float* x_sym = sym_data + D * x_idx;
            float xc_dist = cent_dists[x_idx];

            for (unsigned int y_idx = 0; y_idx < cl_n; ++y_idx) {
                float* y_obj = cluster + D * y_idx;
                float xy_dist = mm_dist(x_sym, y_obj, D);
                d_near = min(d_near, xy_dist);
            }

            float sym_d_val = (d_near + 1e-6) * xc_dist;
            sym_dists[x_idx] = sym_d_val;
        }
    }

    __global__ void get_cent_matrix(
        float* centroids,
        unsigned int K,
        unsigned int D,
        float* cent_matrix
    ) {
        unsigned int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int y_idx = blockDim.y * blockIdx.y + threadIdx.y;
        if (x_idx < K && y_idx < K) {
            float* x_cent = centroids + x_idx * D;
            float* y_cent = centroids + y_idx * D;
            float xy_dist = mm_dist(x_cent, y_cent, D);
            cent_matrix[x_idx * K + y_idx] = xy_dist;
        }
    }

    __global__ void mcr(
        float* data,
        unsigned int N,
        unsigned int D,
        int* labels,
        float* s_w,
        float* s_b
    ) {
        unsigned int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (x_idx < N) {
            float* x_obj = data + x_idx * D;
            int x_label = labels[x_idx];

            float s_w_val = 0.0f, s_b_val = 0.0f;

            for (unsigned int y_idx = 0; y_idx < x_idx; ++y_idx) {
                float* y_obj = data + y_idx * D;
                int y_label = labels[y_idx];

                float xy_dist = mm_dist(x_obj, y_obj, D);

                if (x_label == y_label) {
                    s_w_val += xy_dist;
                } else {
                    s_b_val += xy_dist;
                }
            }

            s_w[x_idx] = s_w_val;
            s_b[x_idx] = s_b_val;
        }
    }

    uint2 get_pair_indices(
        unsigned int N,
        unsigned int lin_idx
    ) {
        unsigned int l = 1;
        unsigned int r = N;
        unsigned int lin_val = lin_idx + 1;
        while (l <= r) {
            unsigned int m = (l + r) / 2;
            unsigned int m_val = m * (m + 1) / 2;
            if (m_val < lin_val) {
                l = m + 1;
            } else if (m_val > lin_val) {
                r = m - 1;
            } else {
                return make_uint2(m, m - 1);
            }
        }
        unsigned int lower = l * (l - 1) / 2;
        return make_uint2(l, lin_idx - lower);
    }

    __global__ void c_index(
        float* data,
        unsigned int N,
        unsigned int D,
        unsigned int pairs,
        unsigned int batch_start,
        int* labels,
        unsigned int s_min_idx,
        float* s_min,
        unsigned int s_max_idx,
        float* s_max,
        float* s_c
    ) {
        unsigned int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int lin_idx = batch_start + t_idx;
        if (lin_idx < pairs) {
            bool is_reducer = lin_idx == 0;
            uint2 pair_idx = get_pair_indices(N, lin_idx);

            float* x_obj = data + pair_idx.x * D;
            float* y_obj = data + pair_idx.y * D;
            float xy_dist = mm_dist(x_obj, y_obj, D);

            unsigned int gt_count = 0;
            unsigned int p_count = 0;
            float intra_dist_acc = 0.0f;

            for (unsigned int o1_idx = 1; o1_idx < N; ++o1_idx) {
                float* o1_obj = data + o1_idx * D;

                for (unsigned int o2_idx = 0; o2_idx < o1_idx; ++o2_idx, ++p_count) {
                    float* o2_obj = data + o2_idx * D;
                    float o_dist = mm_dist(o1_obj, o2_obj, D);
                    if (o_dist < xy_dist) {
                        gt_count++;
                    } else if (p_count < lin_idx && o_dist == xy_dist) {
                        gt_count++;
                    }
                   if (is_reducer) {
                        int o1_label = labels[o1_idx];
                        int o2_label = labels[o2_idx];
                        if (o1_label == o2_label) {
                            intra_dist_acc += o_dist;
                        }
                   }
                }
            }

            if (gt_count < s_min_idx) {
                s_min[t_idx] = xy_dist;
            }
            if (gt_count >= s_max_idx) {
                s_max[t_idx] = xy_dist;
            }
            if (is_reducer) {
                s_c[0] = intra_dist_acc;
            }
        }
    }

    __global__ void crosstab(
        unsigned int N,
        int* classes,
        unsigned int classes_k,
        int* labels,
        unsigned int labels_k,
        unsigned int* crosstab_matrix
    ) {
        unsigned int thread_class = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int thread_label = blockDim.y * blockIdx.y + threadIdx.y;
        if (thread_label < labels_k && thread_class < classes_k) {
            unsigned int crosstab_count = 0;
            for (unsigned int x_idx = 0; x_idx < N; ++x_idx) {
                int x_class = classes[x_idx];
                int x_label = labels[x_idx];
                if (x_label == thread_label && x_class == thread_class) {
                    crosstab_count++;
                }
            }
            unsigned int crosstab_idx = thread_class * labels_k + thread_label;
            crosstab_matrix[crosstab_idx] = crosstab_count;
        }
    }
}