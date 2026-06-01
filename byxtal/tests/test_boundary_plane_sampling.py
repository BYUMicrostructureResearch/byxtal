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


def test_pareto_filter_ignores_aspect_ratio():
    compact_but_dominated = {
        'area': 2.0, 'angle_error_deg': 2.0, 'aspect_ratio': 1.0}
    elongated_but_dominant = {
        'area': 1.0, 'angle_error_deg': 1.0, 'aspect_ratio': 10.0}

    pareto = bps._pareto_filter([
        compact_but_dominated,
        elongated_but_dominant,
    ])

    assert elongated_but_dominant in pareto
    assert compact_but_dominated not in pareto


def test_effective_area_scales_area_by_angle_error():
    options = [
        {'area': 2.0, 'angle_error_deg': 0.0},
        {'area': 4.0, 'angle_error_deg': 45.0},
    ]
    bps._assign_effective_area(options)

    assert options[0]['effective_area'] == 1.0
    assert options[1]['effective_area'] == 4.0


def test_search_2d_supercells_reports_best_by_effective_area():
    basis = np.array([
        [1.0, 0.2],
        [0.0, 1.0],
        [0.0, 0.0],
    ])
    cell_options = bps._search_2d_supercells(
        basis, np.eye(3), max_transform_index=1,
        max_area_multiplier=4)

    expected = min(cell_options['cell_options'], key=lambda option: (
        option['effective_area'], option['area'],
        option['angle_error_deg'], option['aspect_ratio']))
    assert cell_options['best_by_effective_area'] is expected


def test_plot_effective_area_uses_common_minimum_area():
    planes = [
        {
            'best_by_effective_area': {
                'area': 2.0,
                'angle_error_deg': 0.0,
            },
        },
        {
            'best_by_effective_area': {
                'area': 4.0,
                'angle_error_deg': 45.0,
            },
        },
    ]

    values = bps.plot_effective_area_values(planes)

    assert np.allclose(values, np.array([1.0, 4.0]))


def test_fz_boundary_segments_include_expected_symmetry_shapes():
    d4h_segments = bps._fz_boundary_segments('D4h', num=5)
    assert len(d4h_segments) == 3
    assert np.allclose(d4h_segments[0][0], np.array([0.0, 0.0]))
    assert np.allclose(d4h_segments[0][-1], np.array([1.0, 0.0]))
    assert np.allclose(
        d4h_segments[2][-1],
        np.array([np.sqrt(0.5), np.sqrt(0.5)]))

    oh_segments = bps._fz_boundary_segments('Oh', num=5)
    assert len(oh_segments) == 3
    assert np.allclose(oh_segments[0][-1], np.array([
        1.0/np.sqrt(2.0), 0.0]))
    assert np.allclose(oh_segments[2][-1], np.array([
        1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]))


def test_plot_boundary_plane_fz_draws_boundary_and_symmetry_title():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    result = {
        'method': 'synthetic',
        'planes': [
            _simple_plane([1.0, 0.0, 0.0], [1, 0, 0], area=1.0),
            _simple_plane([0.0, 1.0, 0.0], [0, 1, 0], area=2.0),
        ],
        'csl_bp_props': {
            'bp_symm_grp': 'D4h',
        },
    }

    fig, ax = plt.subplots()
    bps.plot_boundary_plane_fz(result, ax=ax, title='test title')

    assert ax.get_title() == 'test title (D4h)'
    assert len(ax.lines) == 3
    assert len(ax.collections) == 1
    assert all(line.get_zorder() == 0 for line in ax.lines)
    plt.close(fig)


def _simple_plane(normal, csl_index, area=1.0, angle_error=0.0):
    option = {
        'area': area,
        'angle_error_deg': angle_error,
        'angle_deg': 90.0-angle_error,
        'aspect_ratio': 1.0,
        'effective_area': area,
    }
    return {
        'csl_reciprocal_index': np.array(csl_index, dtype='int64'),
        'grain1_miller_conventional': np.array(csl_index, dtype='int64'),
        'normal_po': np.array(normal, dtype='double'),
        'normal_fz_po': np.array(normal, dtype='double'),
        'normal_fz_stereo': np.array(normal, dtype='double'),
        'best_by_effective_area': option,
    }


def test_sample_boundary_plane_fz_selects_quality_ordered_spaced_points():
    result = {
        'method': 'synthetic',
        'planes': [
            _simple_plane([1.0, 0.0, 0.0], [1, 0, 0], area=1.0),
            _simple_plane([0.99, 0.1, 0.0], [1, 1, 0], area=1.5),
            _simple_plane([0.0, 1.0, 0.0], [0, 1, 0], area=2.0),
            _simple_plane([0.0, 0.0, 1.0], [0, 0, 1], area=3.0),
        ],
    }

    sampled = bps.sample_boundary_plane_fz(
        result, min_spacing_deg=20.0)

    assert sampled['method'] == 'fz_quality_spacing'
    assert sampled['diagnostics']['selected_indices'] == [0, 2, 3]
    assert sampled['diagnostics']['selected_count'] == 3
    assert sampled['diagnostics']['rejected_by_min_spacing_count'] == 1
    assert sampled['diagnostics']['stopped_by'] == 'candidate_pool_exhausted'


def test_sample_boundary_plane_fz_stops_when_max_spacing_met():
    result = {
        'method': 'synthetic',
        'planes': [
            _simple_plane([1.0, 0.0, 0.0], [1, 0, 0], area=1.0),
            _simple_plane([0.0, 1.0, 0.0], [0, 1, 0], area=2.0),
            _simple_plane([0.0, 0.0, 1.0], [0, 0, 1], area=3.0),
        ],
    }

    loose = bps.sample_boundary_plane_fz(
        result, min_spacing_deg=0.0, max_spacing_deg=100.0)
    tight = bps.sample_boundary_plane_fz(
        result, min_spacing_deg=100.0, max_spacing_deg=10.0)

    assert loose['diagnostics']['selected_count'] == 1
    assert loose['diagnostics']['coverage_complete']
    assert loose['diagnostics']['stopped_by'] == 'max_spacing_met'
    assert tight['diagnostics']['selected_count'] == 1
    assert not tight['diagnostics']['coverage_complete']
    assert tight['diagnostics']['stopped_by'] == 'candidate_pool_exhausted'


def test_sample_boundary_plane_fz_rejects_boundary_seeding_for_now():
    result = {
        'method': 'synthetic',
        'planes': [
            _simple_plane([1.0, 0.0, 0.0], [1, 0, 0], area=1.0),
        ],
    }

    try:
        bps.sample_boundary_plane_fz(
            result, seed_boundary=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError('seed_boundary should not be implemented yet.')


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
            'effective_area': area,
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
            'best_by_effective_area': best_option,
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
    assert np.array_equal(
        representative['best_by_effective_area'][
            'source_csl_reciprocal_index'],
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
    assert candidate['best_by_effective_area']['effective_area'] >= 1
    assert len(candidate['pareto_candidates']) > 0


def test_max_area_normal_generation_sigma651_smoke():
    lat_type = gbl.Lattice()
    csl_props = cuf.enumerate_csl_props(651, 'common', lat_type)
    csl_record = cuf.csl_record_from_props(csl_props, '651a')

    normals = bps.generate_normals_by_area(
        csl_record['csl_mat'], lat_type.l_p_po, max_area=100.0)

    assert len(normals) > 0
    assert 'cell_options' not in normals[0]
    assert normals[0]['primitive_area'] > 0
    assert np.array_equal(
        bps._canonicalize_plane_index(normals[0]['csl_reciprocal_index']),
        normals[0]['csl_reciprocal_index'])


def test_max_area_enumeration_sigma13_smoke():
    lat_type = gbl.Lattice()
    csl_record = cuf.csl_record_from_sig_id('13a', 'common', lat_type)

    result = bps.enumerate_boundary_planes_by_area(
        csl_record, lat_type, max_area=8.0,
        max_transform_index=1, max_area_multiplier=4)

    assert result['method'] == 'max_area'
    assert result['recommended'] is None
    assert len(result['planes']) > 0

    plane = result['planes'][0]
    common_fields = [
        'csl_reciprocal_index',
        'grain1_miller_conventional',
        'normal_po',
        'normal_fz_po',
        'normal_fz_stereo',
        'primitive_basis',
        'primitive_metrics',
        'cell_options',
        'pareto_candidates',
        'best_primitive',
        'best_by_area',
        'best_by_angle_error',
        'best_by_effective_area',
    ]
    for field in common_fields:
        assert field in plane
    assert len(plane['cell_options']) > 0
    assert len(plane['pareto_candidates']) > 0


def test_fz_quality_spacing_sigma13_smoke():
    lat_type = gbl.Lattice()
    csl_record = cuf.csl_record_from_sig_id('13a', 'common', lat_type)
    area_result = bps.enumerate_boundary_planes_by_area(
        csl_record, lat_type, max_area=8.0,
        max_transform_index=1, max_area_multiplier=4)

    sampled = bps.sample_boundary_plane_fz(
        area_result, min_spacing_deg=10.0, max_spacing_deg=35.0)

    assert sampled['method'] == 'fz_quality_spacing'
    assert len(sampled['planes']) > 0
    assert sampled['diagnostics']['selected_count'] == len(
        sampled['planes'])
    assert sampled['diagnostics']['candidate_count'] == len(
        area_result['planes'])
    assert sampled['diagnostics']['max_gap_deg'] >= 0


def test_target_search_and_area_enumeration_share_plane_fields():
    lat_type = gbl.Lattice()
    csl_record = cuf.csl_record_from_sig_id('13a', 'common', lat_type)

    target_result = bps.search_boundary_plane(
        csl_record, lat_type, [1, 1, 1], max_area=8.0,
        angle_radius_deg=25.0, max_transform_index=1,
        max_area_multiplier=4, n_results=5)
    area_result = bps.enumerate_boundary_planes_by_area(
        csl_record, lat_type, max_area=8.0,
        max_transform_index=1, max_area_multiplier=4)

    target_fields = set(target_result['planes'][0])
    area_fields = set(area_result['planes'][0])

    for field in [
            'csl_reciprocal_index',
            'grain1_miller_conventional',
            'cell_options',
            'pareto_candidates',
            'best_by_effective_area']:
        assert field in target_fields
        assert field in area_fields


if __name__ == '__main__':
    test_canonicalize_plane_index_treats_opposites_as_same()
    test_pareto_filter_keeps_tradeoff_options()
    test_pareto_filter_ignores_aspect_ratio()
    test_effective_area_scales_area_by_angle_error()
    test_search_2d_supercells_reports_best_by_effective_area()
    test_plot_effective_area_uses_common_minimum_area()
    test_fz_boundary_segments_include_expected_symmetry_shapes()
    test_plot_boundary_plane_fz_draws_boundary_and_symmetry_title()
    test_sample_boundary_plane_fz_selects_quality_ordered_spaced_points()
    test_sample_boundary_plane_fz_stops_when_max_spacing_met()
    test_sample_boundary_plane_fz_rejects_boundary_seeding_for_now()
    test_boundary_plane_group_merges_source_candidates()
    test_angular_search_sigma13_smoke()
    test_max_area_normal_generation_sigma651_smoke()
    test_max_area_enumeration_sigma13_smoke()
    test_fz_quality_spacing_sigma13_smoke()
    test_target_search_and_area_enumeration_share_plane_fields()
