import numpy as np

import byxtal.boundary_plane_sampling as bps
import byxtal.csl_utility_functions as cuf
import byxtal.lattice as gbl


def test_canonicalize_plane_index_treats_opposites_as_same():
    idx1 = bps._canonicalize_plane_index([2, 2, 0])
    idx2 = bps._canonicalize_plane_index([-4, -4, 0])
    assert np.array_equal(idx1, np.array([1, 1, 0]))
    assert np.array_equal(idx1, idx2)


def test_pareto_filter_keeps_tradeoff_options():
    options = [
        {'area': 1.0, 'angle_error_deg': 10.0, 'aspect_ratio': 3.0},
        {'area': 2.0, 'angle_error_deg': 1.0, 'aspect_ratio': 1.2},
        {'area': 3.0, 'angle_error_deg': 5.0, 'aspect_ratio': 2.0},
    ]
    pareto = bps._pareto_filter(options)
    assert options[0] in pareto
    assert options[1] in pareto
    assert options[2] not in pareto


def test_boundary_plane_group_merges_source_candidates():
    def option(area, angle_error, aspect_ratio, score, transform):
        return {
            'basis': np.eye(3, 2),
            'transform_2x2': np.array(transform, dtype='int64'),
            'area': area,
            'area_multiplier': 1,
            'lengths': np.array([1.0, aspect_ratio]),
            'angle_deg': 90.0-angle_error,
            'angle_error_deg': angle_error,
            'aspect_ratio': aspect_ratio,
            'score': score,
        }

    def candidate(csl_index, grain1_hkl, best_option):
        return {
            'csl_reciprocal_index': np.array(csl_index, dtype='int64'),
            'grain1_miller_conventional': np.array(
                grain1_hkl, dtype='int64'),
            'normal_po': np.array([1.0, 0.0, 0.0]),
            'target_angle_error_deg': 0.0,
            'best_primitive': best_option,
            'best_by_area': best_option,
            'best_by_angle_error': best_option,
            'best_balanced': best_option,
            'pareto_candidates': [best_option],
        }

    source1 = candidate(
        [1, 0, 0], [1, 0, 0],
        option(10.0, 5.0, 1.2, 12.0, [[1, 0], [0, 1]]))
    source2 = candidate(
        [0, 1, 0], [0, 1, 0],
        option(8.0, 1.0, 1.1, 9.0, [[1, 1], [0, 1]]))

    representatives, groups = bps._canonicalize_by_bp_symmetry(
        [source1, source2], csl_bp_props=None)

    assert len(groups) == 1
    assert len(representatives) == 1

    representative = representatives[0]
    assert representative['source_count'] == 2
    assert np.array_equal(
        representative['best_by_area']['source_csl_reciprocal_index'],
        np.array([0, 1, 0]))
    assert np.array_equal(
        representative['best_balanced']['source_csl_reciprocal_index'],
        np.array([0, 1, 0]))
    assert len(representative['source_candidates']) == 2


def test_angular_search_sigma13_smoke():
    lat_type = gbl.Lattice()
    csl_record = cuf.csl_record_from_sig_id('13a', 'common', lat_type)
    assert csl_record['sig_id'] == '13a'

    result = bps.search_boundary_plane(
        csl_record, lat_type, [1, 1, 1], max_area=8.0,
        angle_radius_deg=25.0, max_transform_index=1,
        max_area_multiplier=4, n_results=5)

    assert result['recommended'] is not None
    assert len(result['candidates']) > 0

    candidate = result['recommended']
    assert 'grain1_miller_conventional' in candidate
    assert 'csl_reciprocal_index' in candidate
    assert candidate['primitive_metrics']['area'] > 0
    assert candidate['best_by_area']['area'] > 0
    assert 0 <= candidate['best_by_angle_error']['angle_deg'] <= 180
    assert candidate['best_by_angle_error']['aspect_ratio'] >= 1
    assert len(candidate['pareto_candidates']) > 0


def test_max_area_normal_generation_sigma651_smoke():
    lat_type = gbl.Lattice()
    csl_props = cuf.enumerate_csl_props(651, 'common', lat_type)
    csl_record = cuf.csl_record_from_props(csl_props, '651a')

    normals = bps.generate_normals_by_area(
        csl_record['csl_mat'], lat_type.l_p_po, max_area=100.0)

    assert len(normals) > 0
    assert normals[0]['primitive_area'] > 0
    assert np.array_equal(
        bps._canonicalize_plane_index(normals[0]['csl_reciprocal_index']),
        normals[0]['csl_reciprocal_index'])


if __name__ == '__main__':
    test_canonicalize_plane_index_treats_opposites_as_same()
    test_pareto_filter_keeps_tradeoff_options()
    test_boundary_plane_group_merges_source_candidates()
    test_angular_search_sigma13_smoke()
    test_max_area_normal_generation_sigma651_smoke()
