extern "C" {

    #define NUM_BUCKETS 128

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

    __global__ void meta_dist_partial(
        unsigned int N,
        unsigned int D,
        float* data,
        unsigned int batch_start,
        unsigned int batch_size,
        float* partial_dists
    ) {
        unsigned int b_idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int x_idx = batch_start + b_idx;
        unsigned int y_idx = blockDim.y * blockIdx.y + threadIdx.y;

        if (x_idx < N && y_idx < N && b_idx < batch_size) {
            float* x_obj = data + x_idx * D;
            float* y_obj = data + y_idx * D;
            float xy_dist = mm_dist(x_obj, y_obj, D);
            partial_dists[b_idx * N + y_idx] = xy_dist;
        }
    }

    __global__ void meta_dist_stat(
        unsigned int Q, // N div BUCKETS
        unsigned int R, // N mod BUCKETS
        unsigned int N,
        float* sorted_dists,
        unsigned int batch_size,
        float* dist_stats
    ) {
        unsigned int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int bucket_idx = blockDim.y * blockIdx.y + threadIdx.y;

        if (batch_idx < batch_size && bucket_idx < NUM_BUCKETS) {
            unsigned int bucket_start, bucket_size;
            if (bucket_idx < R) {
                bucket_start = (Q + 1) * bucket_idx;
                bucket_size = Q + 1;
            } else {
                bucket_start = (Q + 1) * R + (bucket_idx - R) * Q;
                bucket_size = Q;
            }
            unsigned int bucket_end = bucket_start + bucket_size;
            unsigned int row_idx = batch_idx * N;

            float diff_sum = 0.0f, diff_max = 0.0f, dist_sum = 0.0f;
            float first_val = sorted_dists[row_idx + bucket_start];
            float last_val = sorted_dists[row_idx + bucket_end - 1];

            for (unsigned int d_idx = bucket_start + 1; d_idx < bucket_end; ++d_idx) {
                float prev_val = sorted_dists[row_idx + d_idx - 1];
                float cur_val = sorted_dists[row_idx + d_idx];
                float diff_val = cur_val - prev_val;

                diff_sum += diff_val;
                diff_max = max(diff_max, diff_val);
                dist_sum += cur_val;
            }

            unsigned int stats_idx = batch_idx * NUM_BUCKETS * 4 + bucket_idx * 4;
            dist_stats[stats_idx + 0] = (dist_sum + first_val) / bucket_size;
            dist_stats[stats_idx + 1] = last_val - first_val;
            dist_stats[stats_idx + 2] = diff_sum / (bucket_size - 1);
            dist_stats[stats_idx + 3] = diff_max;
        }

    }
}