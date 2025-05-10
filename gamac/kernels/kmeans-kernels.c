extern "C" {

    __global__ void kmeans_labels(
        const float* X,
        const float* centers,
        unsigned int N,
        unsigned int K,
        unsigned int D,
        int* labels
    ) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        float min_dist = 1e30;
        int min_label = -1;

        for (int c = 0; c < K; ++c) {
            float dist = 0.0;
            for (int d = 0; d < D; ++d) {
                float diff = X[idx * D + d] - centers[c * D + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                min_label = c;
            }
        }
        labels[idx] = min_label;
    }

    __global__ void kmeans_sse(
        const float* X,
        const float* centers,
        const int* labels,
        float* sse,
        unsigned int N,
        unsigned int D
    ) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        int label = labels[idx];
        float sum = 0.0;
        for (int d = 0; d < D; ++d) {
            float diff = X[idx * D + d] - centers[label * D + d];
            sum += diff * diff;
        }
        atomicAdd(sse, sum);
    }

}
