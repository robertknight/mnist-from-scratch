from numpy.testing import assert_array_equal
from neuralnet.util import onehot


class TestOneHot:
    def test_returns_onehot_vec(self):
        assert_array_equal(onehot(2, 5), [0.0, 0.0, 1.0, 0.0, 0.0])

        length = 10
        for i in range(0, length):
            vec = onehot(i, length)
            for k in range(0, length):
                assert vec[k] == (1.0 if i == k else 0.0)
