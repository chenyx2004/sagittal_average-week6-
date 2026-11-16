import numpy as np
from sagittal_brain import run_averages


input = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])


expected = np.array([2.5, 6.5, 10.5, 14.5])
# average of each column [7.0, 8.0, 9.0, 10.0]


def test_run_averages(tmp_path):
    input_file = tmp_path / "brain_sample.csv"
    output_file = tmp_path / "brain_average.csv"

    np.savetxt(input_file, input, fmt='%d', delimiter=',')

    run_averages(str(input_file), str(output_file))

    output = np.loadtxt(output_file, delimiter=',')

    np.testing.assert_allclose(output, expected)
