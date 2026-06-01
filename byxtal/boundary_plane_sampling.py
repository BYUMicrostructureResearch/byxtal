"""
Boundary-plane sampling helpers for CSL lattices.

This module starts with the single-target workflow: given one CSL and a
grain-1 conventional Miller normal, find nearby CSL plane normals and rank
their 2D CSL cell options.
"""

import itertools
import math

import numpy as np
import numpy.linalg as nla

from . import bp_basis as bpb
from . import find_csl_dsc as fcd
from . import integer_manipulations as int_man
from . import pick_fz_bpl as pfb
from . import reduce_po_lat as rpl


def search_boundary_plane(csl_record, lat_type, target_miller, max_area,
                          angle_radius_deg=5.0, max_index=None,
                          max_transform_index=2, max_area_multiplier=8,
                          n_results=20, tol=1e-6):
    """
    Search for CSL boundary planes near a target grain-1 Miller normal.

    Parameters
    ----------
    csl_record : dict
        Single-CSL record. The minimal required keys are ``csl_mat`` and
        ``sig_mat``; ``sig_id``, ``csl_rotation_id``, and ``csl_bp_props`` are
        copied into the returned candidate records when present.
    lat_type : class
        byxtal lattice object for grain 1.
    target_miller : array_like
        Grain-1 conventional Miller normal.
    max_area : float
        Maximum primitive 2D CSL area to include in the candidate pool.
    angle_radius_deg : float, optional
        Maximum unoriented angular distance from the target normal.
    max_index : int, optional
        Explicit CSL reciprocal index bound. By default, a bound is derived
        from ``max_area`` and the CSL reciprocal metric.
    max_transform_index : int, optional
        Integer entry bound for 2x2 supercell transforms.
    max_area_multiplier : int, optional
        Largest accepted absolute determinant for 2x2 supercell transforms.
    n_results : int, optional
        Number of ranked candidates to return in ``candidates``.
    tol : float, optional
        Numerical tolerance for integer approximations and reductions.

    Returns
    -------
    dict
        ``recommended`` contains the first ranked candidate or ``None``.
        ``candidates`` contains the first ``n_results`` group representatives.
        ``all_candidates`` contains every group representative.
        ``source_candidates`` contains every matching source normal record.
        ``groups`` contains source records grouped by boundary-plane FZ key.
    """
    _validate_csl_record(csl_record)
    normal_records = _generate_normals_near_target(
        csl_record['csl_mat'], lat_type.l_p_po,
        _normal_from_conventional_miller(target_miller, lat_type), max_area,
        angle_radius_deg, max_index=max_index, tol=tol)

    evaluation = evaluate_boundary_plane_normals(
        csl_record, lat_type, normal_records,
        target_miller=target_miller,
        max_transform_index=max_transform_index,
        max_area_multiplier=max_area_multiplier,
        reduce_by_symmetry=True,
        tol=tol)
    planes = evaluation['planes']
    planes.sort(key=_candidate_rank_key)

    return {
        'method': 'target_search',
        'query': {
            'target_miller': np.asarray(target_miller, dtype='int64'),
            'max_area': float(max_area),
            'angle_radius_deg': float(angle_radius_deg),
            'max_index': max_index,
            'max_transform_index': int(max_transform_index),
            'max_area_multiplier': int(max_area_multiplier),
        },
        'recommended': planes[0] if planes else None,
        'planes': planes,
        'candidates': planes[:n_results],
        'all_candidates': planes,
        'source_candidates': evaluation['source_candidates'],
        'groups': evaluation['groups'],
    }


def enumerate_boundary_planes_by_area(csl_record, lat_type, max_area,
                                      max_index=None,
                                      max_transform_index=2,
                                      max_area_multiplier=8,
                                      reduce_by_symmetry=True, tol=1e-6):
    """
    Enumerate full boundary-plane records from a max-area normal pool.

    ``generate_normals_by_area`` remains the low-level generator for CSL
    reciprocal normals. This function evaluates each generated normal into the
    common plane-record shape used by ``search_boundary_plane``.
    """
    _validate_csl_record(csl_record)
    normal_records = generate_normals_by_area(
        csl_record['csl_mat'], lat_type.l_p_po, max_area,
        max_index=max_index, tol=tol)

    evaluation = evaluate_boundary_plane_normals(
        csl_record, lat_type, normal_records,
        max_transform_index=max_transform_index,
        max_area_multiplier=max_area_multiplier,
        reduce_by_symmetry=reduce_by_symmetry,
        tol=tol)
    planes = evaluation['planes']
    planes.sort(key=_max_area_plane_rank_key)

    return {
        'method': 'max_area',
        'query': {
            'max_area': float(max_area),
            'max_index': max_index,
            'max_transform_index': int(max_transform_index),
            'max_area_multiplier': int(max_area_multiplier),
            'reduce_by_symmetry': bool(reduce_by_symmetry),
        },
        'recommended': None,
        'planes': planes,
        'candidates': planes,
        'all_candidates': planes,
        'source_candidates': evaluation['source_candidates'],
        'groups': evaluation['groups'],
    }


def evaluate_boundary_plane_normals(csl_record, lat_type, normal_records,
                                    target_miller=None,
                                    target_normal_po=None,
                                    max_transform_index=2,
                                    max_area_multiplier=8,
                                    reduce_by_symmetry=True, tol=1e-6):
    """
    Evaluate CSL reciprocal normals into common boundary-plane records.
    """
    _validate_csl_record(csl_record)
    if target_miller is not None:
        target_normal_po = _normal_from_conventional_miller(
            target_miller, lat_type)
    elif target_normal_po is not None:
        target_normal_po = _unit(target_normal_po)

    candidates = []
    for normal_record in normal_records:
        csl_index = _normal_record_index(normal_record)
        candidate = _compute_2d_csl_metrics(
            csl_index,
            csl_record['csl_mat'],
            lat_type,
            target_normal_po=target_normal_po,
            max_transform_index=max_transform_index,
            max_area_multiplier=max_area_multiplier,
            tol=tol)
        _add_csl_metadata(candidate, csl_record)
        if target_miller is not None:
            candidate['target_miller'] = np.asarray(
                target_miller, dtype='int64')
        candidates.append(candidate)

    source_candidates = candidates
    if reduce_by_symmetry:
        planes, groups = _canonicalize_by_bp_symmetry(
            candidates, csl_record.get('csl_bp_props'))
    else:
        planes, groups = _identity_boundary_plane_groups(candidates)

    return {
        'method': 'evaluate_normals',
        'query': {
            'target_miller': (
                None if target_miller is None
                else np.asarray(target_miller, dtype='int64')),
            'max_transform_index': int(max_transform_index),
            'max_area_multiplier': int(max_area_multiplier),
            'reduce_by_symmetry': bool(reduce_by_symmetry),
        },
        'planes': planes,
        'source_candidates': source_candidates,
        'groups': groups,
    }


def sample_boundary_plane_fz(*args, **kwargs):
    """
    Placeholder for full boundary-plane FZ coverage sampling.

    The first implementation milestone is single-target search. The FZ
    coverage workflow will build on ``generate_normals_by_area`` and
    ``_canonicalize_by_bp_symmetry`` after the candidate metrics are validated.
    """
    raise NotImplementedError(
        'Boundary-plane FZ coverage sampling is planned after '
        'single-target search is validated.')


def _validate_csl_record(csl_record):
    for key in ('csl_mat', 'sig_mat'):
        if key not in csl_record:
            raise ValueError('csl_record is missing required key: '+key)


def _normal_record_index(normal_record):
    if isinstance(normal_record, dict):
        return normal_record['csl_reciprocal_index']
    return normal_record


def _add_csl_metadata(candidate, csl_record):
    for key in (
            'sig_id', 'csl_rotation_id', 'sig_mat', 'dsc_mat',
            'dis_quat', 'dis_axis_angle'):
        if key in csl_record:
            candidate[key] = csl_record[key]


def _generate_normals_near_target(csl_mat, l_p_po, target_normal_po, max_area,
                                  angle_radius_deg, max_index=None,
                                  tol=1e-9):
    """
    Generate primitive CSL reciprocal normals within an angular target cone.
    """
    target_normal_po = _unit(target_normal_po)
    candidates = generate_normals_by_area(
        csl_mat, l_p_po, max_area, max_index=max_index, tol=tol)

    records = []
    for candidate in candidates:
        normal_po = candidate['normal_po']
        angle_error = _unoriented_angle_deg(normal_po, target_normal_po)
        if angle_error <= angle_radius_deg + tol:
            aligned_normal_po = _align_vector(normal_po, target_normal_po)
            record = candidate.copy()
            record['normal_po'] = aligned_normal_po
            record['target_angle_error_deg'] = angle_error
            records.append(record)

    records.sort(key=lambda record: (
        record['target_angle_error_deg'],
        record['primitive_area'],
        tuple(record['csl_reciprocal_index'])))
    return records


def generate_normals_by_area(csl_mat, l_p_po, max_area, max_index=None,
                             tol=1e-9):
    """
    Generate primitive CSL reciprocal normals with 2D area <= ``max_area``.
    """
    if max_area <= 0:
        raise ValueError('max_area must be positive.')

    l_csl_po = np.dot(l_p_po, csl_mat)
    l_cslR_po = fcd.reciprocal_mat(l_csl_po)
    csl_volume = abs(nla.det(l_csl_po))
    max_g_norm = max_area/csl_volume

    if max_index is None:
        min_sing_val = np.min(nla.svd(l_cslR_po, compute_uv=False))
        max_index = int(np.ceil(max_g_norm/min_sing_val)) + 1
    if max_index < 1:
        return []

    records = []
    seen = set()
    index_range = range(-max_index, max_index+1)
    for idx in itertools.product(index_range, repeat=3):
        if idx == (0, 0, 0):
            continue
        primitive_idx = _canonicalize_plane_index(idx)
        idx_key = tuple(primitive_idx)
        if idx_key in seen:
            continue
        seen.add(idx_key)

        normal_po = np.dot(l_cslR_po, primitive_idx)
        normal_norm = nla.norm(normal_po)
        if normal_norm < tol:
            continue
        primitive_area = csl_volume*normal_norm
        if primitive_area <= max_area + tol:
            records.append({
                'csl_reciprocal_index': primitive_idx,
                'normal_po': normal_po/normal_norm,
                'primitive_area': float(primitive_area),
            })

    records.sort(key=lambda record: (
        record['primitive_area'],
        tuple(record['csl_reciprocal_index'])))
    return records


def _convert_csl_reciprocal_to_grain1_miller(csl_index, csl_mat, lat_type,
                                             align_to=None, tol=1e-6):
    """
    Convert a CSL reciprocal normal index to grain-1 conventional Miller form.
    """
    l_csl_po = np.dot(lat_type.l_p_po, csl_mat)
    l_cslR_po = fcd.reciprocal_mat(l_csl_po)
    normal_po = np.dot(l_cslR_po, csl_index)
    if align_to is not None:
        target_po = _normal_from_conventional_miller(align_to, lat_type)
        normal_po = _align_vector(normal_po, target_po)

    l_conv_po = _conventional_basis_po(lat_type)
    l_convR_po = fcd.reciprocal_mat(l_conv_po)
    conv_hkl = np.dot(nla.inv(l_convR_po), normal_po)
    conv_hkl, _ = int_man.int_approx(conv_hkl, tol)
    conv_hkl = _primitive_int_vector(conv_hkl)

    if align_to is None:
        conv_hkl = _canonicalize_plane_index(conv_hkl)
    return conv_hkl


def _compute_2d_csl_metrics(csl_index, csl_mat, lat_type,
                            target_normal_po=None, max_transform_index=2,
                            max_area_multiplier=8, tol=1e-6):
    """
    Compute primitive 2D CSL basis, metrics, and supercell options.
    """
    csl_index = _canonicalize_plane_index(csl_index)
    l_p_po = lat_type.l_p_po

    primitive_basis_csl = bpb.bp_basis(csl_index)
    primitive_basis_p1 = np.dot(csl_mat, primitive_basis_csl)
    l_sig2_sig1 = rpl.reduce_po_lat(primitive_basis_p1, l_p_po, tol)
    primitive_basis_p1 = np.dot(primitive_basis_p1, l_sig2_sig1)

    primitive_metrics = _basis_metrics(primitive_basis_p1, l_p_po)
    cell_options = _search_2d_supercells(
        primitive_basis_p1, l_p_po,
        max_transform_index=max_transform_index,
        max_area_multiplier=max_area_multiplier)

    target_miller = None
    if target_normal_po is not None:
        target_miller = _convert_normal_to_conventional_miller(
            target_normal_po, lat_type, tol)

    grain1_miller = _convert_csl_reciprocal_to_grain1_miller(
        csl_index, csl_mat, lat_type, align_to=target_miller, tol=tol)

    normal_po = _normal_from_conventional_miller(grain1_miller, lat_type)
    target_angle_error = None
    if target_normal_po is not None:
        normal_po = _align_vector(normal_po, target_normal_po)
        target_angle_error = _unoriented_angle_deg(
            normal_po, target_normal_po)

    return {
        'csl_reciprocal_index': csl_index,
        'grain1_miller_conventional': grain1_miller,
        'grain1_miller_primitive': grain1_miller,
        'normal_po': normal_po,
        'normal_fz_po': None,
        'normal_fz_stereo': None,
        'target_angle_error_deg': target_angle_error,
        'primitive_basis': primitive_basis_p1,
        'primitive_metrics': primitive_metrics,
        'best_primitive': cell_options['best_primitive'],
        'best_by_area': cell_options['best_by_area'],
        'best_by_angle_error': cell_options['best_by_angle_error'],
        'best_by_effective_area': cell_options['best_by_effective_area'],
        'best_balanced': cell_options['best_balanced'],
        'cell_options': cell_options['cell_options'],
        'pareto_candidates': cell_options['pareto_candidates'],
    }


def _search_2d_supercells(primitive_basis_p1, l_p_po, max_transform_index=2,
                          max_area_multiplier=8):
    """
    Search small integer 2x2 transforms of a primitive 2D CSL basis.
    """
    if max_transform_index < 1:
        raise ValueError('max_transform_index must be at least 1.')
    if max_area_multiplier < 1:
        raise ValueError('max_area_multiplier must be at least 1.')

    identity = np.eye(2, dtype='int64')
    best_primitive = _cell_option(primitive_basis_p1, l_p_po, identity)

    options = []
    seen = set()
    entry_range = range(-max_transform_index, max_transform_index+1)
    for vals in itertools.product(entry_range, repeat=4):
        transform = np.array(vals, dtype='int64').reshape((2, 2))
        det_val = int(round(nla.det(transform)))
        area_multiplier = abs(det_val)
        if area_multiplier < 1 or area_multiplier > max_area_multiplier:
            continue

        option = _cell_option(primitive_basis_p1, l_p_po, transform)
        key = (
            round(option['area'], 10),
            round(option['angle_error_deg'], 10),
            round(option['aspect_ratio'], 10),
            tuple(transform.flatten()))
        if key in seen:
            continue
        seen.add(key)
        options.append(option)

    min_area = _assign_effective_area(options)
    _set_effective_area(best_primitive, min_area)

    options.sort(key=lambda option: (
        option['area'],
        option['angle_error_deg'],
        option['aspect_ratio']))

    return {
        'best_primitive': best_primitive,
        'best_by_area': min(options, key=lambda option: (
            option['area'], option['angle_error_deg'],
            option['aspect_ratio'])),
        'best_by_angle_error': min(options, key=lambda option: (
            option['angle_error_deg'], option['area'],
            option['aspect_ratio'])),
        'best_by_effective_area': min(options, key=lambda option: (
            option['effective_area'], option['area'],
            option['angle_error_deg'], option['aspect_ratio'])),
        'best_balanced': min(options, key=lambda option: option['score']),
        'cell_options': options,
        'pareto_candidates': _pareto_filter(options),
    }


def _canonicalize_by_bp_symmetry(candidates, csl_bp_props=None, decimals=10):
    """
    Map normals into the boundary-plane FZ and merge equivalent candidates.
    """
    if not candidates:
        return candidates, {}

    if csl_bp_props is None:
        groups = {}
        for candidate in candidates:
            normal_po = _unit(candidate['normal_po'])
            candidate['normal_fz_po'] = normal_po
            candidate['normal_fz_stereo'] = normal_po
            key = _direction_key(normal_po, decimals)
            candidate['fz_group_key'] = key
            groups.setdefault(key, []).append(candidate)
        return _representatives_from_groups(groups), groups

    normals = np.array([_unit(candidate['normal_po'])
                        for candidate in candidates])
    fz_norms, fz_stereo = pfb.pick_fz_bpl(
        normals,
        csl_bp_props['bp_symm_grp'],
        csl_bp_props['symm_grp_ax'])

    groups = {}
    for candidate, fz_normal, fz_stereo_vec in zip(
            candidates, fz_norms, fz_stereo):
        fz_normal = _unit(fz_normal)
        candidate['normal_fz_po'] = fz_normal
        candidate['normal_fz_stereo'] = np.asarray(
            fz_stereo_vec, dtype='double')
        key = _direction_key(fz_normal, decimals)
        candidate['fz_group_key'] = key
        groups.setdefault(key, []).append(candidate)

    return _representatives_from_groups(groups), groups


def _identity_boundary_plane_groups(candidates, decimals=10):
    groups = {}
    for candidate in candidates:
        normal_po = _unit(candidate['normal_po'])
        candidate['normal_fz_po'] = normal_po
        candidate['normal_fz_stereo'] = normal_po
        key = _direction_key(normal_po, decimals)
        candidate['fz_group_key'] = key
        candidate['source_candidates'] = [candidate]
        candidate['source_count'] = 1
        candidate['source_csl_reciprocal_indices'] = [
            candidate['csl_reciprocal_index']]
        candidate['source_grain1_miller_conventionals'] = [
            candidate['grain1_miller_conventional']]
        groups.setdefault(key, []).append(candidate)
    return candidates, groups


def _representatives_from_groups(groups):
    representatives = []
    for key, source_candidates in groups.items():
        sources = sorted(source_candidates, key=_candidate_rank_key)
        merged_options = _merge_group_cell_options(sources)
        representative = _source_for_option(
            sources, merged_options['best_by_effective_area']).copy()

        representative['best_primitive'] = merged_options['best_primitive']
        representative['best_by_area'] = merged_options['best_by_area']
        representative['best_by_angle_error'] = (
            merged_options['best_by_angle_error'])
        representative['best_by_effective_area'] = (
            merged_options['best_by_effective_area'])
        representative['best_balanced'] = merged_options['best_balanced']
        representative['cell_options'] = merged_options['cell_options']
        representative['pareto_candidates'] = (
            merged_options['pareto_candidates'])
        representative['source_candidates'] = sources
        representative['source_count'] = len(sources)
        representative['source_csl_reciprocal_indices'] = [
            source['csl_reciprocal_index'] for source in sources]
        representative['source_grain1_miller_conventionals'] = [
            source['grain1_miller_conventional'] for source in sources]
        representative['fz_group_key'] = key
        representatives.append(representative)
    return representatives


def _merge_group_cell_options(source_candidates):
    options = []
    primitive_options = []
    seen = set()
    for source in source_candidates:
        primitive_option = _annotate_cell_option(
            source['best_primitive'], source)
        primitive_options.append(primitive_option)

        source_options = source.get('cell_options')
        if source_options is None:
            source_options = [
                source['best_by_area'],
                source['best_by_angle_error'],
                source['best_by_effective_area'],
                source['best_balanced']] + source['pareto_candidates']

        for option in source_options:
            option = _annotate_cell_option(option, source)
            key = _cell_option_key(option)
            if key in seen:
                continue
            seen.add(key)
            options.append(option)

    min_area = _assign_effective_area(options)
    for option in primitive_options:
        _set_effective_area(option, min_area)

    return {
        'best_primitive': min(primitive_options, key=lambda option: (
            option['area'], option['angle_error_deg'],
            option['aspect_ratio'])),
        'best_by_area': min(options, key=lambda option: (
            option['area'], option['angle_error_deg'],
            option['aspect_ratio'])),
        'best_by_angle_error': min(options, key=lambda option: (
            option['angle_error_deg'], option['area'],
            option['aspect_ratio'])),
        'best_by_effective_area': min(options, key=lambda option: (
            option['effective_area'], option['area'],
            option['angle_error_deg'], option['aspect_ratio'])),
        'best_balanced': min(options, key=lambda option: option['score']),
        'cell_options': options,
        'pareto_candidates': _pareto_filter(options),
    }


def _annotate_cell_option(option, source):
    option_copy = option.copy()
    option_copy['source_csl_reciprocal_index'] = (
        source['csl_reciprocal_index'])
    option_copy['source_grain1_miller_conventional'] = (
        source['grain1_miller_conventional'])
    option_copy['source_target_angle_error_deg'] = (
        source['target_angle_error_deg'])
    option_copy['source_fz_group_key'] = source['fz_group_key']
    return option_copy


def _source_for_option(sources, option):
    option_source = tuple(np.asarray(
        option['source_csl_reciprocal_index'], dtype='int64').reshape(3,))
    for source in sources:
        source_index = tuple(np.asarray(
            source['csl_reciprocal_index'], dtype='int64').reshape(3,))
        if source_index == option_source:
            return source
    return sources[0]


def _cell_option_key(option):
    source_index = tuple(np.asarray(
        option['source_csl_reciprocal_index'], dtype='int64').reshape(3,))
    transform = tuple(np.asarray(
        option['transform_2x2'], dtype='int64').reshape(4,))
    return source_index + transform


def _pareto_filter(options, keys=('area', 'angle_error_deg')):
    """
    Return options that are not dominated across the selected metrics.
    """
    pareto_options = []
    values = np.array([[option[key] for key in keys] for option in options],
                      dtype='double')

    for idx, option in enumerate(options):
        dominated = False
        for jdx in range(len(options)):
            if idx == jdx:
                continue
            no_worse = np.all(values[jdx] <= values[idx] + 1e-12)
            strictly_better = np.any(values[jdx] < values[idx] - 1e-12)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_options.append(option)

    pareto_options.sort(key=lambda option: (
        option['area'], option['angle_error_deg'], option['aspect_ratio']))
    return pareto_options


def _cell_option(primitive_basis_p1, l_p_po, transform):
    basis_p1 = np.dot(primitive_basis_p1, transform)
    metrics = _basis_metrics(basis_p1, l_p_po)
    area_multiplier = int(abs(round(nla.det(transform))))
    score = (
        metrics['area']
        + metrics['angle_error_deg']/90.0*metrics['area']
        + (metrics['aspect_ratio'] - 1.0)*metrics['area'])
    return {
        'basis': basis_p1,
        'transform_2x2': np.array(transform, dtype='int64'),
        'area': metrics['area'],
        'area_multiplier': area_multiplier,
        'lengths': metrics['lengths'],
        'angle_deg': metrics['angle_deg'],
        'angle_error_deg': metrics['angle_error_deg'],
        'aspect_ratio': metrics['aspect_ratio'],
        'score': float(score),
    }


def plot_pareto_front(candidate_or_options, ax=None, color_by='effective_area',
                      title=None, annotate=False):
    """
    Plot cell-option area versus orthogonality error.

    Parameters
    ----------
    candidate_or_options : dict or list
        Candidate record with ``cell_options`` and ``pareto_candidates``, or a
        list of cell-option records.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. A new axes is created when omitted.
    color_by : str or None, optional
        Cell-option key used to color all candidate points. Useful values are
        ``'effective_area'`` and ``'aspect_ratio'``.
    title : str, optional
        Plot title.
    annotate : bool, optional
        If True, annotate Pareto points by their source CSL reciprocal index.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    import matplotlib.pyplot as plt

    if isinstance(candidate_or_options, dict):
        options = candidate_or_options.get('cell_options')
        if options is None:
            options = candidate_or_options['pareto_candidates']
        pareto_options = candidate_or_options['pareto_candidates']
    else:
        options = list(candidate_or_options)
        pareto_options = _pareto_filter(options)

    if ax is None:
        _, ax = plt.subplots()

    area = np.array([option['area'] for option in options], dtype='double')
    angle_error = np.array(
        [option['angle_error_deg'] for option in options], dtype='double')
    pareto_options = sorted(
        pareto_options,
        key=lambda option: (option['area'], option['angle_error_deg']))
    pareto_area = np.array(
        [option['area'] for option in pareto_options], dtype='double')
    pareto_angle_error = np.array(
        [option['angle_error_deg'] for option in pareto_options],
        dtype='double')

    if len(pareto_options) > 1:
        ax.plot(pareto_area, pareto_angle_error, color='tab:red',
                linewidth=1.5, label='Pareto front', zorder=1)

    scatter_kwargs = {
        's': 24,
        'alpha': 0.55,
        'label': 'candidate cell options',
        'zorder': 2,
    }
    if color_by is None:
        scatter_kwargs['color'] = '0.65'
        scatter = ax.scatter(area, angle_error, **scatter_kwargs)
    else:
        color_values = np.array([option[color_by] for option in options],
                                dtype='double')
        scatter = ax.scatter(area, angle_error, c=color_values,
                             **scatter_kwargs)
        cbar = ax.figure.colorbar(scatter, ax=ax)
        cbar.set_label(color_by.replace('_', ' '))

    if len(pareto_options) == 1:
        ax.scatter(pareto_area, pareto_angle_error, s=180,
                   facecolors='none', edgecolors='tab:red',
                   linewidths=2.0, label='Pareto point', zorder=3)

    if annotate:
        for option in pareto_options:
            source_index = option.get('source_csl_reciprocal_index')
            if source_index is None:
                label = ''
            else:
                label = str(np.asarray(source_index, dtype='int64')
                            .reshape(3,).tolist())
            ax.annotate(label, (option['area'], option['angle_error_deg']),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=8, zorder=4)

    ax.set_xlabel('2D CSL area')
    ax.set_ylabel('angle error from 90 deg')
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_effective_area_values(planes, cell_key='best_by_effective_area',
                               angle_reference_deg=45.0):
    """
    Compute plot-level effective area values across several plane records.

    The returned values are normalized by the minimum selected-cell area among
    all supplied planes, so they can be compared across different normals.
    """
    planes = _planes_from_result(planes)
    if not planes:
        return np.array([], dtype='double')

    options = [_selected_cell_option(plane, cell_key) for plane in planes]
    min_area = min(option['area'] for option in options)
    return np.array([
        _effective_area(
            option['area'],
            option['angle_error_deg'],
            min_area,
            angle_reference_deg=angle_reference_deg)
        for option in options], dtype='double')


def plot_boundary_plane_fz(planes, ax=None, cell_key='best_by_effective_area',
                           color_by='plot_effective_area',
                           min_marker_size=24, max_marker_size=160,
                           title=None, annotate=False,
                           angle_reference_deg=45.0):
    """
    Plot boundary-plane FZ points with marker size scaled by cell quality.
    """
    import matplotlib.pyplot as plt

    planes = _planes_from_result(planes)
    if ax is None:
        _, ax = plt.subplots()

    if not planes:
        ax.set_xlabel('FZ stereographic x')
        ax.set_ylabel('FZ stereographic y')
        return ax

    points = np.array([_plane_stereo_point(plane) for plane in planes],
                      dtype='double')
    plot_eff_area = plot_effective_area_values(
        planes, cell_key=cell_key, angle_reference_deg=angle_reference_deg)
    sizes = _inverse_marker_sizes(
        plot_eff_area, min_marker_size, max_marker_size)

    if color_by == 'plot_effective_area':
        color_values = plot_eff_area
        color_label = 'plot effective area'
    elif color_by is None:
        color_values = None
        color_label = None
    else:
        color_values = np.array([
            _selected_cell_option(plane, cell_key)[color_by]
            for plane in planes], dtype='double')
        color_label = color_by.replace('_', ' ')

    scatter_kwargs = {
        's': sizes,
        'alpha': 0.75,
        'edgecolors': '0.2',
        'linewidths': 0.5,
    }
    if color_values is None:
        scatter = ax.scatter(points[:, 0], points[:, 1],
                             color='tab:blue', **scatter_kwargs)
    else:
        scatter = ax.scatter(points[:, 0], points[:, 1], c=color_values,
                             **scatter_kwargs)
        cbar = ax.figure.colorbar(scatter, ax=ax)
        cbar.set_label(color_label)

    if annotate:
        for point, plane in zip(points, planes):
            label = str(np.asarray(
                plane['grain1_miller_conventional'],
                dtype='int64').reshape(3,).tolist())
            ax.annotate(label, point, textcoords='offset points',
                        xytext=(4, 4), fontsize=8)

    ax.set_xlabel('FZ stereographic x')
    ax.set_ylabel('FZ stereographic y')
    if title is not None:
        ax.set_title(title)
    ax.set_aspect('equal', adjustable='datalim')
    return ax


def _planes_from_result(planes_or_result):
    if isinstance(planes_or_result, dict) and 'planes' in planes_or_result:
        return list(planes_or_result['planes'])
    return list(planes_or_result)


def _selected_cell_option(plane, cell_key):
    try:
        return plane[cell_key]
    except KeyError as exc:
        raise ValueError('Plane record is missing cell option: '+cell_key) \
            from exc


def _plane_stereo_point(plane):
    point = plane.get('normal_fz_stereo')
    if point is None:
        point = plane.get('normal_fz_po', plane['normal_po'])
    point = np.asarray(point, dtype='double').reshape(-1,)
    if point.size < 2:
        raise ValueError('Plane stereographic point must have at least 2 values.')
    return point[:2]


def _inverse_marker_sizes(values, min_marker_size, max_marker_size):
    values = np.asarray(values, dtype='double')
    if min_marker_size <= 0 or max_marker_size <= 0:
        raise ValueError('Marker sizes must be positive.')
    if max_marker_size < min_marker_size:
        raise ValueError('max_marker_size must be >= min_marker_size.')

    quality = 1.0/np.maximum(values, np.finfo(float).eps)
    q_min = np.min(quality)
    q_max = np.max(quality)
    if np.isclose(q_min, q_max):
        return np.full(values.shape, (min_marker_size+max_marker_size)/2.0)
    scaled = (quality-q_min)/(q_max-q_min)
    return min_marker_size + scaled*(max_marker_size-min_marker_size)


def _assign_effective_area(options):
    min_area = min(option['area'] for option in options)
    for option in options:
        _set_effective_area(option, min_area)
    return min_area


def _set_effective_area(option, min_area):
    option['effective_area'] = _effective_area(
        option['area'], option['angle_error_deg'], min_area)


def _effective_area(area, angle_error_deg, min_area,
                    angle_reference_deg=45.0):
    if min_area <= 0:
        raise ValueError('min_area must be positive.')
    angle_factor = angle_error_deg/angle_reference_deg
    return float((1.0 + angle_factor*angle_factor)*(area/min_area))


def _basis_metrics(basis_p1, l_p_po):
    basis_po = np.dot(l_p_po, basis_p1)
    v1 = basis_po[:, 0]
    v2 = basis_po[:, 1]
    len1 = nla.norm(v1)
    len2 = nla.norm(v2)
    if len1 == 0 or len2 == 0:
        raise ValueError('2D basis vectors must be nonzero.')

    cos_ang = np.dot(v1, v2)/(len1*len2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_ang))
    area = nla.norm(np.cross(v1, v2))
    aspect_ratio = max(len1, len2)/min(len1, len2)
    return {
        'area': float(area),
        'lengths': np.array([len1, len2], dtype='double'),
        'angle_deg': float(angle_deg),
        'angle_error_deg': float(abs(90.0-angle_deg)),
        'aspect_ratio': float(aspect_ratio),
    }


def _normal_from_conventional_miller(miller, lat_type):
    miller = np.asarray(miller, dtype='double').reshape(3,)
    if nla.norm(miller) == 0:
        raise ValueError('Miller normal cannot be [0, 0, 0].')
    l_conv_po = _conventional_basis_po(lat_type)
    l_convR_po = fcd.reciprocal_mat(l_conv_po)
    return _unit(np.dot(l_convR_po, miller))


def _convert_normal_to_conventional_miller(normal_po, lat_type, tol=1e-6):
    l_conv_po = _conventional_basis_po(lat_type)
    l_convR_po = fcd.reciprocal_mat(l_conv_po)
    conv_hkl = np.dot(nla.inv(l_convR_po), normal_po)
    conv_hkl, _ = int_man.int_approx(conv_hkl, tol)
    return _primitive_int_vector(conv_hkl)


def _conventional_basis_po(lat_type):
    """
    Return conventional direct basis vectors in the orthogonal frame.
    """
    if not hasattr(lat_type, 'lat_params'):
        raise ValueError('lat_type must define lat_params.')

    params = lat_type.lat_params
    a = params['a']
    b = params['b']
    c = params['c']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']

    b1 = np.array([a, 0.0, 0.0], dtype='double')
    b2 = np.array([b*np.cos(gamma), b*np.sin(gamma), 0.0],
                  dtype='double')
    cx = c*np.cos(beta)
    cy = c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma)
    cz_sq = c*c - cx*cx - cy*cy
    cz = math.sqrt(max(cz_sq, 0.0))
    b3 = np.array([cx, cy, cz], dtype='double')
    return np.column_stack((b1, b2, b3))


def _candidate_rank_key(candidate):
    target_angle_error = candidate.get('target_angle_error_deg')
    if target_angle_error is None:
        target_angle_error = 0.0
    return (
        target_angle_error,
        candidate['best_by_effective_area']['effective_area'],
        candidate['best_by_area']['area'],
        candidate['best_by_angle_error']['angle_error_deg'],
        tuple(candidate['csl_reciprocal_index']))


def _max_area_plane_rank_key(candidate):
    return (
        candidate['best_by_effective_area']['effective_area'],
        candidate['best_by_area']['area'],
        candidate['best_by_angle_error']['angle_error_deg'],
        tuple(candidate['csl_reciprocal_index']))


def _direction_key(normal, decimals):
    normal = _unit(normal)
    return tuple(np.round(normal, decimals))


def _unoriented_angle_deg(vec1, vec2):
    vec1 = _unit(vec1)
    vec2 = _unit(vec2)
    cos_ang = abs(np.dot(vec1, vec2))
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def _align_vector(vec, target_vec):
    vec = _unit(vec)
    target_vec = _unit(target_vec)
    if np.dot(vec, target_vec) < 0:
        return -vec
    return vec


def _unit(vec):
    vec = np.asarray(vec, dtype='double').reshape(3,)
    norm_val = nla.norm(vec)
    if norm_val == 0:
        raise ValueError('Cannot normalize a zero vector.')
    return vec/norm_val


def _canonicalize_plane_index(vec):
    vec = _primitive_int_vector(vec)
    for value in vec:
        if value < 0:
            return -vec
        if value > 0:
            return vec
    raise ValueError('Plane index cannot be [0, 0, 0].')


def _primitive_int_vector(vec):
    vec = np.asarray(vec, dtype='int64').reshape(3,)
    if np.all(vec == 0):
        raise ValueError('Integer vector cannot be [0, 0, 0].')

    divisor = 0
    for value in vec:
        divisor = math.gcd(divisor, int(abs(value)))
    if divisor > 1:
        vec = vec//divisor
    return vec
