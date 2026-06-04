import unittest
import warnings

import numpy as np

import byxtal.integer_manipulations as int_man


class IntegerManipulationGuardrailTests(unittest.TestCase):
    def test_bounded_integer_direction_recovers_low_index_direction(self):
        direction = np.array([3.0, 2.0, -5.0])
        direction = direction/np.linalg.norm(direction)
        direction = direction + np.array([1e-6, -2e-6, 1e-6])
        direction = direction/np.linalg.norm(direction)

        int_direction, _, angular_error_deg = \
            int_man.bounded_integer_direction(
                direction, angular_tolerance_deg=0.01)

        np.testing.assert_array_equal(int_direction, [3, 2, -5])
        self.assertLess(angular_error_deg, 0.01)

    def test_bounded_integer_direction_floors_max_index(self):
        direction = np.array([3.0, 2.0, -5.0])
        direction = direction/np.linalg.norm(direction)

        result = int_man.bounded_integer_direction(
            direction, max_index=5.9, return_diagnostics=True)
        int_direction, _, _, diagnostics = result

        np.testing.assert_array_equal(int_direction, [3, 2, -5])
        self.assertEqual(diagnostics['max_index'], [5, 5, 5])

    def test_int_finder_columns_returns_multipliers(self):
        mat = np.array([
            [0.5, 0.0],
            [0.0, 1.0/3.0],
            [0.0, 0.0],
        ])

        int_mat, multipliers = int_man.int_finder(
            mat, order='columns', return_multipliers=True)

        np.testing.assert_array_equal(int_mat, [[1, 0], [0, 1], [0, 0]])
        np.testing.assert_allclose(multipliers, [2.0, 3.0])

    def test_int_finder_falls_back_from_large_indices(self):
        direction = np.array([1.0, 1.0, 1.0])
        direction = direction/np.linalg.norm(direction)
        direction = direction + np.array([1e-6, -1e-6, 0.0])
        direction = direction/np.linalg.norm(direction)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            int_direction, _, diagnostics = int_man.int_finder(
                direction, return_multipliers=True,
                return_diagnostics=True)

        np.testing.assert_array_equal(int_direction, [1, 1, 1])
        self.assertTrue(diagnostics[0]['used_bounded_fallback'])
        self.assertTrue(any(
            'angular_tolerance_deg' in str(item.message) and
            'max_index' in str(item.message)
            for item in caught))

    def test_int_finder_default_return_is_unchanged(self):
        direction = np.array([0.5, 0.5, 0.5])

        int_direction = int_man.int_finder(direction)

        self.assertIsInstance(int_direction, np.ndarray)
        np.testing.assert_array_equal(int_direction, [1, 1, 1])


if __name__ == '__main__':
    unittest.main()
