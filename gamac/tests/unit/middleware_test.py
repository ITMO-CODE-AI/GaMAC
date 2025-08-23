import pytest
import cupy as cp

# Import the module containing the Middleware class
from gamac.kernels.middleware import Middleware, KernelInvocation


class TestMiddleware:
    @pytest.fixture
    def mock_raw_module(self):
        # Create a mock RawModule with a get_function method
        mock_module = type('MockRawModule', (), {})()
        mock_module.get_function = lambda x: f"mock_kernel_{x}"
        return mock_module

    @pytest.fixture
    def middleware(self, monkeypatch, mock_raw_module):
        # Patch the load_module function to return our mock RawModule
        def mock_load_module(file):
            return mock_raw_module

        monkeypatch.setattr('gamac.kernels.middleware.load_module', mock_load_module)
        return Middleware()

    @pytest.fixture
    def sample_arrays(self):
        # Create some sample arrays for testing
        return {
            'data': cp.array([[1.0, 2.0], [3.0, 4.0]]),
            'partial_dists': cp.array([2.0, 3.0, 1.0]),
            'sorted_dists': cp.array([1.0, 2.0, 3.0]),
            'max_dists': cp.array([1.0, 2.0]),
            'dist_stats': cp.array([0.0, 0.0]),
            'labels': cp.array([0, 1]),
            'uniq_labels': cp.array([0, 1]),
            'centroids': cp.array([[1.5, 2.5]]),
            'cent_dists': cp.array([0.0, 0.0]),
            'cent_matrix': cp.array([[0.0, 1.0], [1.0, 0.0]]),
            'sym_data': cp.array([[0.0, 0.0]]),
            'sym_dists': cp.array([0.0, 0.0]),
            's_w': cp.array([0.0]),
            's_b': cp.array([0.0]),
            's_min': cp.array([0.0]),
            's_max': cp.array([0.0]),
            's_c': cp.array([0.0]),
            'o_val': cp.array([0.0]),
            'crosstab_matrix': cp.array([[0, 0], [0, 0]]),
            'sse': cp.array([0.0]),
            'tp_val': cp.array([0, 0]),
            'fp_val': cp.array([0, 0]),
            'fn_val': cp.array([0, 0]),
        }

    def test_init(self, middleware):
        assert hasattr(middleware, '_meta')
        assert hasattr(middleware, '_cvi')
        assert hasattr(middleware, '_kmeans')

    def test_meta_dist_partial(self, middleware, sample_arrays):
        invocation = middleware.meta_dist_partial(
            N=2,
            D=2,
            data=sample_arrays['data'],
            batch_start=0,
            batch_size=2,
            partial_dists=sample_arrays['sorted_dists'],
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_meta_dist_partial"
        assert len(invocation.args) == 6
        assert invocation.args[0] == 2  # N
        assert invocation.args[1] == 2  # D
        assert cp.array_equal(invocation.args[2], sample_arrays['data'])

    def test_meta_dist_stat(self, middleware, sample_arrays):
        invocation = middleware.meta_dist_stat(
            Q=1,
            R=1,
            N=2,
            sorted_dists=sample_arrays['sorted_dists'],
            batch_size=2,
            dist_stats=sample_arrays['dist_stats']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_meta_dist_stat"
        assert len(invocation.args) == 6
        assert invocation.args[0] == 1  # Q
        assert invocation.args[1] == 1  # R

    def test_get_centroids(self, middleware, sample_arrays):
        invocation = middleware.get_centroids(
            data=sample_arrays['data'],
            labels=sample_arrays['labels'],
            N=2,
            D=2,
            K=1,
            uniq_labels=sample_arrays['uniq_labels'],
            centroids=sample_arrays['centroids']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_get_centroids"
        assert len(invocation.args) == 7
        assert invocation.args[2] == 2  # N
        assert invocation.args[3] == 2  # D

    def test_get_cent_dists(self, middleware, sample_arrays):
        invocation = middleware.get_cent_dists(
            cluster=sample_arrays['data'],
            cl_n=2,
            D=2,
            centroids=sample_arrays['centroids'],
            k_idx=0,
            cent_dists=sample_arrays['cent_dists']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_get_cent_dists"
        assert len(invocation.args) == 6
        assert invocation.args[1] == 2  # cl_n
        assert invocation.args[2] == 2  # D

    def test_get_cent_matrix(self, middleware, sample_arrays):
        invocation = middleware.get_cent_matrix(
            centroids=sample_arrays['centroids'],
            K=1,
            D=2,
            cent_matrix=sample_arrays['cent_matrix']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_get_cent_matrix"
        assert len(invocation.args) == 4
        assert invocation.args[1] == 1  # K
        assert invocation.args[2] == 2  # D

    def test_get_sym_data(self, middleware, sample_arrays):
        invocation = middleware.get_sym_data(
            cluster=sample_arrays['data'],
            cl_n=2,
            D=2,
            centroids=sample_arrays['centroids'],
            k_idx=0,
            sym_data=sample_arrays['sym_data']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_get_sym_data"
        assert len(invocation.args) == 6
        assert invocation.args[1] == 2  # cl_n

    def test_get_sym_dists(self, middleware, sample_arrays):
        invocation = middleware.get_sym_dists(
            cluster=sample_arrays['data'],
            cl_n=2,
            D=2,
            cent_dists=sample_arrays['cent_dists'],
            sym_data=sample_arrays['sym_data'],
            sym_dists=sample_arrays['sym_dists']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_get_sym_dists"
        assert len(invocation.args) == 6
        assert invocation.args[1] == 2  # cl_n

    def test_mcr(self, middleware, sample_arrays):
        invocation = middleware.mcr(
            data=sample_arrays['data'],
            N=2,
            D=2,
            labels=sample_arrays['labels'],
            s_w=sample_arrays['s_w'],
            s_b=sample_arrays['s_b']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_mcr"
        assert len(invocation.args) == 6
        assert invocation.args[1] == 2  # N

    def test_c_index(self, middleware, sample_arrays):
        invocation = middleware.c_index(
            data=sample_arrays['data'],
            N=2,
            D=2,
            pairs=1,
            labels=sample_arrays['labels'],
            s_min_idx=0,
            s_min=sample_arrays['s_min'],
            s_max_idx=0,
            s_max=sample_arrays['s_max'],
            s_c=sample_arrays['s_c']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_c_index"
        assert len(invocation.args) == 10
        assert invocation.args[1] == 2  # N

    def test_os(self, middleware, sample_arrays):
        invocation = middleware.os(
            data=sample_arrays['data'],
            N=2,
            D=2,
            centroids=sample_arrays['centroids'],
            K=1,
            labels=sample_arrays['labels'],
            uniq_labels=sample_arrays['uniq_labels'],
            o_val=sample_arrays['o_val']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_os"
        assert len(invocation.args) == 8
        assert invocation.args[1] == 2  # N

    def test_external_crosstab(self, middleware, sample_arrays):
        invocation = middleware.external_crosstab(
            N=2,
            uniq_classes=sample_arrays['uniq_labels'],
            classes=sample_arrays['labels'],
            classes_k=2,
            uniq_labels=sample_arrays['uniq_labels'],
            labels=sample_arrays['labels'],
            labels_k=2,
            crosstab_matrix=sample_arrays['crosstab_matrix']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_external_crosstab"
        assert len(invocation.args) == 8
        assert invocation.args[0] == 2  # N

    def test_external_pairwise(self, middleware, sample_arrays):
        invocation = middleware.external_pairwise(
            N=2,
            classes=sample_arrays['labels'],
            labels=sample_arrays['labels'],
            tp_val=sample_arrays['tp_val'],
            fp_val=sample_arrays['fp_val'],
            fn_val=sample_arrays['fn_val'],
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_external_pairwise"
        assert len(invocation.args) == 6
        assert invocation.args[0] == 2  # N

    def test_kmeans_labels(self, middleware, sample_arrays):
        invocation = middleware.kmeans_labels(
            X=sample_arrays['data'],
            centers=sample_arrays['centroids'],
            N=2,
            K=1,
            D=2,
            labels=sample_arrays['labels']
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_kmeans_labels"
        assert len(invocation.args) == 6
        assert invocation.args[2] == 2  # N

    def test_kmeans_sse(self, middleware, sample_arrays):
        invocation = middleware.kmeans_sse(
            X=sample_arrays['data'],
            centers=sample_arrays['centroids'],
            labels=sample_arrays['labels'],
            sse=sample_arrays['sse'],
            N=2,
            D=2
        )

        assert isinstance(invocation, KernelInvocation)
        assert invocation.kernel == "mock_kernel_kmeans_sse"
        assert len(invocation.args) == 6
        assert invocation.args[4] == 2  # N
