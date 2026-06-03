"""
Boundary-plane sampling helpers for CSL lattices.

This module starts with the single-target workflow: given one CSL and a
grain-1 conventional Miller normal, find nearby CSL plane normals and rank
their 2D CSL cell options.
"""

import itertools
import math
import os

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
        'csl_bp_props': evaluation['csl_bp_props'],
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
        'csl_bp_props': evaluation['csl_bp_props'],
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
        'csl_bp_props': csl_record.get('csl_bp_props'),
    }


def sample_boundary_plane_fz(candidate_result, min_spacing_deg=0.0,
                             max_spacing_deg=None,
                             seed_indices=None, seed_planes=None,
                             seed_boundary=False,
                             cell_key='best_by_effective_area'):
    """
    Select high-quality boundary-plane records with angular spacing controls.

    Parameters
    ----------
    candidate_result : dict or list
        Result from ``enumerate_boundary_planes_by_area`` or a list of plane
        records.
    min_spacing_deg : float, optional
        Reject a candidate if it is closer than this to any selected plane.
    max_spacing_deg : float, optional
        Stop once every candidate in the finite pool is within this angular
        distance of a selected plane. If omitted, the function walks the whole
        quality-sorted list and keeps every candidate satisfying
        ``min_spacing_deg``.
    seed_indices : sequence of int, optional
        Candidate-pool indices to include before quality-ordered selection.
    seed_planes : sequence of dict, optional
        Plane records to include before quality-ordered selection.
    seed_boundary : bool, optional
        Reserved for future FZ-vertex/boundary seeding.
    cell_key : str, optional
        Cell option used as the practical-quality tie breaker.

    Returns
    -------
    dict
        Common result envelope with selected planes in ``planes`` and
        diagnostics describing the remaining coverage gap.
    """
    if seed_boundary:
        raise NotImplementedError(
            'Boundary seeding requires the planned FZ vertex/boundary '
            'finder. Pass seed_indices or seed_planes for manual seeds.')

    planes = _planes_from_result(candidate_result)
    csl_bp_props = _csl_bp_props_from_result(candidate_result)
    selected_indices, diagnostics = _quality_spaced_select(
        planes,
        csl_bp_props=csl_bp_props,
        min_spacing_deg=min_spacing_deg,
        max_spacing_deg=max_spacing_deg,
        seed_indices=seed_indices,
        seed_planes=seed_planes,
        cell_key=cell_key)

    selected_set = set(selected_indices)
    selected_planes = [planes[idx] for idx in selected_indices]
    unselected_planes = [
        plane for idx, plane in enumerate(planes)
        if idx not in selected_set]

    return {
        'method': 'fz_quality_spacing',
        'query': {
            'source_method': (
                candidate_result.get('method')
                if isinstance(candidate_result, dict) else None),
            'min_spacing_deg': float(min_spacing_deg),
            'max_spacing_deg': (
                None if max_spacing_deg is None
                else float(max_spacing_deg)),
            'seed_indices': (
                None if seed_indices is None else list(seed_indices)),
            'seed_boundary': bool(seed_boundary),
            'cell_key': cell_key,
        },
        'recommended': selected_planes[0] if selected_planes else None,
        'planes': selected_planes,
        'candidates': selected_planes,
        'all_candidates': planes,
        'source_candidates': (
            candidate_result.get('source_candidates', planes)
            if isinstance(candidate_result, dict) else planes),
        'groups': (
            candidate_result.get('groups', {})
            if isinstance(candidate_result, dict) else {}),
        'unselected_planes': unselected_planes,
        'diagnostics': diagnostics,
        'csl_bp_props': csl_bp_props,
    }


def export_boundary_plane_record(plane, csl_record, lat_type,
                                 cell_key='best_by_effective_area',
                                 completion_search_radius=2,
                                 completion_strategy=(
                                     'balanced_orthogonality_volume'),
                                 max_completion_skew_deg=60.0,
                                 tol=1e-6):
    """
    Export one evaluated boundary plane as a plain Python record.

    The exported record is intended as a stable interchange shape for
    atomistic workflows. Numerical work remains NumPy-based inside byxtal,
    but the returned record contains only built-in Python containers and
    scalar types so it can be stored in JSON-like metadata or ASE DB row data.
    """
    _validate_csl_record(csl_record)
    if cell_key not in plane:
        raise ValueError('Plane record is missing cell option: '+cell_key)

    csl_cell_spec = _export_csl_cell_spec(
        plane, csl_record, lat_type, cell_key,
        completion_search_radius=completion_search_radius,
        completion_strategy=completion_strategy,
        max_completion_skew_deg=max_completion_skew_deg,
        tol=tol)

    return _to_builtin({
        'record_type': 'boundary_plane',
        'sig_id': plane.get('sig_id', csl_record.get('sig_id')),
        'csl_rotation_id': plane.get(
            'csl_rotation_id', csl_record.get('csl_rotation_id')),
        'lattice_type': getattr(lat_type, 'elem_type', None),
        'plane_normals': _export_plane_normals(
            plane, csl_record, lat_type, tol=tol),
        'bp_2d_csl_cell': _export_bp_2d_csl_cell(
            plane, csl_record, lat_type, cell_key, tol=tol),
        'orientation_spec': _orientation_spec_from_csl_cell(csl_cell_spec),
        'csl_cell_spec': csl_cell_spec,
        'byxtal_provenance': _export_boundary_plane_provenance(
            plane, csl_record, cell_key, completion_search_radius,
            completion_strategy, max_completion_skew_deg),
    })


def export_boundary_plane_records(result, csl_record, lat_type,
                                  cell_key='best_by_effective_area',
                                  completion_search_radius=2,
                                  completion_strategy=(
                                      'balanced_orthogonality_volume'),
                                  max_completion_skew_deg=60.0,
                                  tol=1e-6):
    """
    Export every plane in a byxtal boundary-plane result envelope.
    """
    planes = _planes_from_result(result)
    return [
        export_boundary_plane_record(
            plane, csl_record, lat_type, cell_key=cell_key,
            completion_search_radius=completion_search_radius,
            completion_strategy=completion_strategy,
            max_completion_skew_deg=max_completion_skew_deg,
            tol=tol)
        for plane in planes
    ]


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


def _quality_spaced_select(planes, csl_bp_props=None, min_spacing_deg=0.0,
                           max_spacing_deg=None, seed_indices=None,
                           seed_planes=None,
                           cell_key='best_by_effective_area'):
    """
    Quality-ordered selection with minimum and maximum spacing diagnostics.
    """
    plane_count = len(planes)
    if plane_count == 0:
        return [], _coverage_diagnostics(
            [], [], None, None, 'empty_pool', 0,
            min_spacing_deg=min_spacing_deg,
            max_spacing_deg=max_spacing_deg,
            rejected_by_min_spacing_count=0)

    if min_spacing_deg < 0:
        raise ValueError('min_spacing_deg must be nonnegative.')
    if max_spacing_deg is not None and max_spacing_deg < 0:
        raise ValueError('max_spacing_deg must be nonnegative.')

    distance_matrix = _coverage_distance_matrix(planes, csl_bp_props)
    selected = _initial_seed_indices(
        planes, seed_indices=seed_indices, seed_planes=seed_planes,
        cell_key=cell_key)

    rejected_by_min_spacing = 0
    quality_order = sorted(
        range(plane_count),
        key=lambda idx: _plane_quality_key(planes[idx], cell_key))

    stopped_by = 'candidate_pool_exhausted'
    for idx in quality_order:
        if idx in selected:
            max_gap_deg, max_gap_index, nearest = _coverage_gap(
                distance_matrix, selected)
            if (max_spacing_deg is not None
                    and max_gap_deg <= max_spacing_deg):
                stopped_by = 'max_spacing_met'
                break
            continue

        nearest_to_selected = np.min(distance_matrix[idx, selected])
        if nearest_to_selected < min_spacing_deg:
            rejected_by_min_spacing += 1
            continue

        selected.append(idx)
        max_gap_deg, max_gap_index, nearest = _coverage_gap(
            distance_matrix, selected)
        if (max_spacing_deg is not None
                and max_gap_deg <= max_spacing_deg):
            stopped_by = 'max_spacing_met'
            break

    max_gap_deg, max_gap_index, nearest = _coverage_gap(
        distance_matrix, selected)
    diagnostics = _coverage_diagnostics(
        selected, nearest, max_gap_deg, max_gap_index, stopped_by,
        plane_count,
        min_spacing_deg=min_spacing_deg,
        max_spacing_deg=max_spacing_deg,
        rejected_by_min_spacing_count=rejected_by_min_spacing)
    return selected, diagnostics


def _initial_seed_indices(planes, seed_indices=None, seed_planes=None,
                          cell_key='best_by_effective_area'):
    selected = []
    if seed_indices is not None:
        for idx in seed_indices:
            idx = int(idx)
            if idx < 0 or idx >= len(planes):
                raise ValueError('seed index out of range: '+str(idx))
            if idx not in selected:
                selected.append(idx)

    if seed_planes is not None:
        plane_key_to_index = {
            _plane_identity_key(plane): idx
            for idx, plane in enumerate(planes)}
        for seed_plane in seed_planes:
            key = _plane_identity_key(seed_plane)
            if key not in plane_key_to_index:
                raise ValueError('seed plane is not in candidate pool.')
            idx = plane_key_to_index[key]
            if idx not in selected:
                selected.append(idx)

    if not selected:
        selected.append(min(
            range(len(planes)),
            key=lambda idx: _plane_quality_key(planes[idx], cell_key)))
    return selected


def _coverage_gap(distance_matrix, selected):
    if not selected:
        return math.inf, None, np.full(
            distance_matrix.shape[0], math.inf, dtype='double')

    nearest = np.min(distance_matrix[:, selected], axis=1)
    selected_set = set(selected)
    unselected = [
        idx for idx in range(distance_matrix.shape[0])
        if idx not in selected_set]
    if not unselected:
        return 0.0, None, nearest

    max_gap_index = max(unselected, key=lambda idx: nearest[idx])
    return float(nearest[max_gap_index]), max_gap_index, nearest


def _coverage_diagnostics(selected, nearest, max_gap_deg, max_gap_index,
                          stopped_by, plane_count, min_spacing_deg=0.0,
                          max_spacing_deg=None,
                          rejected_by_min_spacing_count=0):
    if max_gap_deg is None:
        max_gap_deg = 0.0
    coverage_complete = None
    if max_spacing_deg is not None:
        coverage_complete = max_gap_deg <= max_spacing_deg

    return {
        'selected_indices': list(selected),
        'selected_count': len(selected),
        'unselected_count': max(plane_count-len(selected), 0),
        'candidate_count': plane_count,
        'min_spacing_deg': float(min_spacing_deg),
        'max_spacing_deg': max_spacing_deg,
        'max_gap_deg': float(max_gap_deg),
        'max_gap_index': max_gap_index,
        'coverage_complete': coverage_complete,
        'stopped_by': stopped_by,
        'rejected_by_min_spacing_count': rejected_by_min_spacing_count,
        'nearest_selected_distance_deg': np.asarray(nearest, dtype='double'),
    }


def _coverage_distance_matrix(planes, csl_bp_props=None):
    normals = np.array([
        _coverage_normal_from_plane(plane) for plane in planes],
        dtype='double')
    copies = [
        _symmetry_copies_for_normal(normal, csl_bp_props)
        for normal in normals]
    plane_count = len(planes)
    distances = np.zeros((plane_count, plane_count), dtype='double')
    for idx in range(plane_count):
        for jdx in range(idx+1, plane_count):
            dist_ij = _min_unoriented_angle_deg(normals[idx], copies[jdx])
            dist_ji = _min_unoriented_angle_deg(normals[jdx], copies[idx])
            distance = min(dist_ij, dist_ji)
            distances[idx, jdx] = distance
            distances[jdx, idx] = distance
    return distances


def _coverage_normal_from_plane(plane):
    normal = plane.get('normal_fz_po')
    if normal is None:
        normal = plane['normal_po']
    return _unit(normal)


def _symmetry_aware_angle_deg(normal1, normal2, csl_bp_props=None):
    copies = _symmetry_copies_for_normal(normal2, csl_bp_props)
    return _min_unoriented_angle_deg(normal1, copies)


def _symmetry_copies_for_normal(normal, csl_bp_props=None):
    normal = _unit(normal)
    if csl_bp_props is None:
        return normal.reshape((1, 3))

    file_path = _bp_symmetry_file_path(csl_bp_props['bp_symm_grp'])
    copies = pfb.rot_symm(
        csl_bp_props['symm_grp_ax'],
        normal.reshape((1, 3)),
        file_path)
    return np.array([_unit(copy) for copy in copies[:, 0, :]],
                    dtype='double')


def _bp_symmetry_file_path(bp_symm_grp):
    file_names = {
        'Cs': 'symm_mats_Cs.pkl',
        'C2h': 'symm_mats_C2h.pkl',
        'D3d': 'symm_mats_D3d.pkl',
        'D2h': 'symm_mats_D2h.pkl',
        'D4h': 'symm_mats_D4h.pkl',
        'D6h': 'symm_mats_D6h.pkl',
        'D8h': 'symm_mats_D8h.pkl',
        'Oh': 'symm_mats_Oh.pkl',
    }
    if bp_symm_grp not in file_names:
        raise ValueError('Unsupported boundary-plane symmetry group: '
                         + str(bp_symm_grp))
    return os.path.join(
        os.path.dirname(os.path.realpath(pfb.__file__)),
        'data_files',
        file_names[bp_symm_grp])


def _min_unoriented_angle_deg(normal, copies):
    normal = _unit(normal)
    copies = np.asarray(copies, dtype='double').reshape((-1, 3))
    dots = np.abs(np.dot(copies, normal))
    dots = np.clip(dots, -1.0, 1.0)
    return float(np.degrees(np.arccos(np.max(dots))))


def _plane_quality_key(plane, cell_key):
    option = _selected_cell_option(plane, cell_key)
    return (
        _cell_effective_area_cost(option),
        option['area'],
        option['angle_error_deg'],
        option.get('aspect_ratio', 1.0),
        tuple(np.asarray(
            plane.get('csl_reciprocal_index', [0, 0, 0]),
            dtype='int64').reshape(3,)))


def _cell_effective_area_cost(option, angle_reference_deg=45.0):
    angle_factor = option['angle_error_deg']/angle_reference_deg
    return float((1.0 + angle_factor*angle_factor)*option['area'])


def _plane_identity_key(plane):
    if 'fz_group_key' in plane:
        return ('fz', tuple(plane['fz_group_key']))
    return ('csl', tuple(np.asarray(
        plane['csl_reciprocal_index'], dtype='int64').reshape(3,)))


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

    csl_bp_props = _csl_bp_props_from_result(planes)
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
    _plot_fz_boundary(ax, csl_bp_props)

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
        ax.set_title(_title_with_bp_symmetry(title, csl_bp_props))
    ax.set_aspect('equal', adjustable='datalim')
    return ax


def _planes_from_result(planes_or_result):
    if isinstance(planes_or_result, dict) and 'planes' in planes_or_result:
        return list(planes_or_result['planes'])
    return list(planes_or_result)


def _csl_bp_props_from_result(planes_or_result):
    if isinstance(planes_or_result, dict):
        return planes_or_result.get('csl_bp_props')
    return None


def _title_with_bp_symmetry(title, csl_bp_props):
    bp_symm_grp = _bp_symmetry_group(csl_bp_props)
    if bp_symm_grp is None:
        return title
    return f'{title} ({bp_symm_grp})'


def _plot_fz_boundary(ax, csl_bp_props):
    bp_symm_grp = _bp_symmetry_group(csl_bp_props)
    if bp_symm_grp is None:
        return
    for segment in _fz_boundary_segments(bp_symm_grp):
        ax.plot(segment[:, 0], segment[:, 1], color='0.25',
                linewidth=1.0, zorder=0)


def _bp_symmetry_group(csl_bp_props):
    if csl_bp_props is None:
        return None
    return csl_bp_props.get('bp_symm_grp')


def _fz_boundary_segments(bp_symm_grp, num=181):
    """
    Return BP FZ boundary segments in the x-y plotting coordinates.
    """
    bp_symm_grp = _normalize_bp_symmetry_group(bp_symm_grp)
    if bp_symm_grp == 'Cs':
        return [_unit_circle_arc(-math.pi, math.pi, num)]
    if bp_symm_grp == 'C2h':
        return _wedge_boundary_segments(0.0, math.pi, num)
    if bp_symm_grp == 'D2h':
        return _wedge_boundary_segments(0.0, math.pi/2.0, num)
    if bp_symm_grp == 'D3d':
        alpha = math.pi/6.0
        return _wedge_boundary_segments(-alpha, alpha, num)
    if bp_symm_grp == 'D4h':
        return _wedge_boundary_segments(0.0, math.pi/4.0, num)
    if bp_symm_grp == 'D6h':
        return _wedge_boundary_segments(0.0, math.pi/6.0, num)
    if bp_symm_grp == 'D8h':
        return _wedge_boundary_segments(0.0, math.pi/8.0, num)
    if bp_symm_grp == 'Oh':
        return _oh_boundary_segments(num)
    raise ValueError('Unsupported boundary-plane symmetry group: '
                     + str(bp_symm_grp))


def _normalize_bp_symmetry_group(bp_symm_grp):
    aliases = {
        'C_s': 'Cs',
        'C_2h': 'C2h',
        'D_3d': 'D3d',
        'D_2h': 'D2h',
        'D_4h': 'D4h',
        'D_6h': 'D6h',
        'D_8h': 'D8h',
        'O_h': 'Oh',
    }
    return aliases.get(bp_symm_grp, bp_symm_grp)


def _wedge_boundary_segments(theta_min, theta_max, num):
    return [
        _radial_segment(theta_min, 0.0, num),
        _unit_circle_arc(theta_min, theta_max, num),
        _radial_segment(theta_max, 0.0, num),
    ]


def _unit_circle_arc(theta_min, theta_max, num):
    theta = np.linspace(theta_min, theta_max, num)
    return np.column_stack((np.cos(theta), np.sin(theta)))


def _radial_segment(theta, start_radius, num):
    radius = np.linspace(start_radius, 1.0, num)
    return np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))


def _oh_boundary_segments(num):
    x_axis = np.column_stack((
        np.linspace(0.0, 1.0/math.sqrt(2.0), num),
        np.zeros(num)))
    diagonal_radius = np.linspace(0.0, 1.0/math.sqrt(3.0), num)
    diagonal = np.column_stack((diagonal_radius, diagonal_radius))
    x_vals = np.linspace(
        1.0/math.sqrt(3.0), 1.0/math.sqrt(2.0), num)
    curve = np.column_stack((x_vals, np.sqrt(1.0-2.0*x_vals*x_vals)))
    return [x_axis, curve, diagonal]


def _selected_cell_option(plane, cell_key):
    try:
        return plane[cell_key]
    except KeyError as exc:
        raise ValueError('Plane record is missing cell option: '+cell_key) \
            from exc


def _export_plane_normals(plane, csl_record, lat_type, tol=1e-6):
    normal_g1_po = _unit(plane['normal_po'])
    normal_g2_po = _grain2_normal_from_grain1_po(
        normal_g1_po, csl_record['sig_mat'], lat_type)

    bp_fz_g1_po = _unit(plane['normal_fz_po'])
    bp_fz_g2_po = _grain2_normal_from_grain1_po(
        bp_fz_g1_po, csl_record['sig_mat'], lat_type)

    return {
        'grain1_miller': plane['grain1_miller_conventional'],
        'grain2_miller': _convert_normal_to_conventional_miller(
            normal_g2_po, lat_type, tol=tol),
        'bp_fz_miller_grain1': _convert_normal_to_conventional_miller(
            bp_fz_g1_po, lat_type, tol=tol),
        'bp_fz_miller_grain2': _convert_normal_to_conventional_miller(
            bp_fz_g2_po, lat_type, tol=tol),
        'bp_fz_normal_grain1_po': bp_fz_g1_po,
        'bp_fz_normal_grain2_po': bp_fz_g2_po,
        'bp_fz_stereographic_grain1': plane.get('normal_fz_stereo'),
        'basis_convention': 'column_vectors',
    }


def _export_bp_2d_csl_cell(plane, csl_record, lat_type, cell_key, tol=1e-6):
    option = plane[cell_key]
    basis_g1_primitive = np.asarray(option['basis'], dtype='double')
    basis_g1_primitive, _ = _int_approx_columns(basis_g1_primitive, tol)
    basis_g2_primitive = _grain2_direct_from_grain1_p1(
        basis_g1_primitive, csl_record['sig_mat'], tol=tol)
    conv_g1, conv_dirs_g1, conv_mults_g1 = \
        _direct_basis_conventional_views(basis_g1_primitive, lat_type, tol)
    conv_g2, conv_dirs_g2, conv_mults_g2 = \
        _direct_basis_conventional_views(basis_g2_primitive, lat_type, tol)

    return {
        'basis_convention': 'column_vectors',
        'basis_grain1_primitive': basis_g1_primitive,
        'basis_grain2_primitive': basis_g2_primitive,
        'basis_grain1_conventional': conv_g1,
        'basis_grain2_conventional': conv_g2,
        'basis_grain1_conventional_directions': conv_dirs_g1,
        'basis_grain2_conventional_directions': conv_dirs_g2,
        'basis_grain1_conventional_direction_multipliers': conv_mults_g1,
        'basis_grain2_conventional_direction_multipliers': conv_mults_g2,
        'basis_cartesian': np.dot(lat_type.l_p_po, basis_g1_primitive),
        'area': option['area'],
        'lengths': option['lengths'],
        'angle_deg': option['angle_deg'],
        'aspect_ratio': option['aspect_ratio'],
    }


def _export_csl_cell_spec(plane, csl_record, lat_type, cell_key,
                          completion_search_radius=2,
                          completion_strategy=(
                              'balanced_orthogonality_volume'),
                          max_completion_skew_deg=60.0,
                          tol=1e-6):
    option = plane[cell_key]
    basis_2d_g1_primitive = np.asarray(option['basis'], dtype='double')
    basis_2d_g1_primitive, _ = _int_approx_columns(
        basis_2d_g1_primitive, tol)
    completion = _best_csl_completion_vector(
        basis_2d_g1_primitive,
        csl_record['csl_mat'],
        lat_type.l_p_po,
        max_index=completion_search_radius,
        strategy=completion_strategy,
        max_skew_deg=max_completion_skew_deg,
        tol=tol)
    basis_g1_primitive = np.column_stack((
        basis_2d_g1_primitive,
        completion['vector_grain1_primitive']))
    basis_g1_primitive, _ = _int_approx_columns(basis_g1_primitive, tol)
    basis_g2_primitive = _grain2_direct_from_grain1_p1(
        basis_g1_primitive, csl_record['sig_mat'], tol=tol)
    basis_cartesian = np.dot(lat_type.l_p_po, basis_g1_primitive)

    conv_g1, conv_dirs_g1, conv_mults_g1 = \
        _direct_basis_conventional_views(basis_g1_primitive, lat_type, tol)
    conv_g2, conv_dirs_g2, conv_mults_g2 = \
        _direct_basis_conventional_views(basis_g2_primitive, lat_type, tol)
    volume = float(abs(nla.det(basis_cartesian)))
    primitive_lattice_volume = float(abs(nla.det(lat_type.l_p_po)))
    primitive_csl_volume = float(abs(nla.det(
        np.dot(lat_type.l_p_po, np.asarray(csl_record['csl_mat'],
                                           dtype='double')))))
    sigma_from_csl_volume = (
        None if primitive_lattice_volume < tol
        else primitive_csl_volume/primitive_lattice_volume)
    volume_multiplier = (
        None if primitive_csl_volume < tol
        else volume/primitive_csl_volume)

    return {
        'cell_type': 'periodic_csl',
        'basis_convention': 'column_vectors',
        'basis_grain1_primitive': basis_g1_primitive,
        'basis_grain2_primitive': basis_g2_primitive,
        'basis_grain1_conventional': conv_g1,
        'basis_grain2_conventional': conv_g2,
        'basis_grain1_conventional_directions': conv_dirs_g1,
        'basis_grain2_conventional_directions': conv_dirs_g2,
        'basis_grain1_conventional_direction_multipliers': conv_mults_g1,
        'basis_grain2_conventional_direction_multipliers': conv_mults_g2,
        'basis_cartesian': basis_cartesian,
        'two_d_basis_columns': [0, 1],
        'completion_vector_column': 2,
        'area': option['area'],
        'volume': volume,
        'primitive_lattice_volume': primitive_lattice_volume,
        'primitive_csl_volume': primitive_csl_volume,
        'sigma_from_csl_volume': sigma_from_csl_volume,
        'volume_multiplier_over_primitive_csl': volume_multiplier,
        'lengths': _basis_lengths(basis_cartesian),
        'angles_deg': _cell_angles_deg(basis_cartesian),
        'completion_vector': completion,
    }


def _orientation_spec_from_csl_cell(csl_cell_spec):
    return {
        'basis_convention': csl_cell_spec['basis_convention'],
        'coordinate_frame': 'crystal_direction_indices',
        'grain1': {
            'primitive_directions':
                csl_cell_spec['basis_grain1_primitive'],
            'conventional_directions':
                csl_cell_spec['basis_grain1_conventional_directions'],
            'conventional_direction_multipliers':
                csl_cell_spec[
                    'basis_grain1_conventional_direction_multipliers'],
        },
        'grain2': {
            'primitive_directions':
                csl_cell_spec['basis_grain2_primitive'],
            'conventional_directions':
                csl_cell_spec['basis_grain2_conventional_directions'],
            'conventional_direction_multipliers':
                csl_cell_spec[
                    'basis_grain2_conventional_direction_multipliers'],
        },
    }


def _export_boundary_plane_provenance(plane, csl_record, cell_key,
                                      completion_search_radius,
                                      completion_strategy,
                                      max_completion_skew_deg):
    return {
        'source': 'byxtal.boundary_plane_sampling',
        'source_cell_key': cell_key,
        'completion_search_radius': completion_search_radius,
        'completion_strategy': completion_strategy,
        'max_completion_skew_deg': max_completion_skew_deg,
        'csl_reciprocal_index': plane.get('csl_reciprocal_index'),
        'source_count': plane.get('source_count'),
        'source_csl_reciprocal_indices':
            plane.get('source_csl_reciprocal_indices', []),
        'source_grain1_miller_conventionals':
            plane.get('source_grain1_miller_conventionals', []),
        'target_angle_error_deg': plane.get('target_angle_error_deg'),
        'fz_group_key': plane.get('fz_group_key'),
        'sig_mat': csl_record.get('sig_mat'),
        'csl_mat': csl_record.get('csl_mat'),
        'dsc_mat': csl_record.get('dsc_mat'),
        'dis_quat': csl_record.get('dis_quat'),
        'dis_axis_angle': csl_record.get('dis_axis_angle'),
        'csl_bp_props': csl_record.get('csl_bp_props'),
    }


def _best_csl_completion_vector(two_d_basis_g1_primitive, csl_mat, l_p_po,
                                max_index=2,
                                strategy='balanced_orthogonality_volume',
                                max_skew_deg=60.0,
                                tol=1e-6):
    if max_index < 1:
        raise ValueError('max_index must be at least 1.')
    strategy = _normalize_completion_strategy(strategy)
    max_skew_deg = float(max_skew_deg)
    if max_skew_deg < 0.0 or max_skew_deg > 90.0:
        raise ValueError('max_skew_deg must be between 0 and 90 degrees.')

    two_d_basis_g1_primitive = np.asarray(
        two_d_basis_g1_primitive, dtype='double')
    csl_mat = np.asarray(csl_mat, dtype='double')
    two_d_basis_po = np.dot(l_p_po, two_d_basis_g1_primitive)
    plane_cross = np.cross(two_d_basis_po[:, 0], two_d_basis_po[:, 1])
    area = nla.norm(plane_cross)
    if area < tol:
        raise ValueError('2D CSL basis is degenerate.')
    plane_normal = plane_cross/area

    candidates = []
    search_range = range(-max_index, max_index+1)
    for coeff_tuple in itertools.product(search_range, repeat=3):
        if coeff_tuple == (0, 0, 0):
            continue
        coeff = np.asarray(coeff_tuple, dtype='int64')
        candidate_g1_primitive = np.dot(csl_mat, coeff)
        candidate_g1_primitive, _ = int_man.int_approx(
            candidate_g1_primitive, tol)
        candidate_po = np.dot(l_p_po, candidate_g1_primitive)
        candidate_length = nla.norm(candidate_po)
        if candidate_length < tol:
            continue
        basis_po = np.column_stack((two_d_basis_po, candidate_po))
        signed_volume = nla.det(basis_po)
        volume = abs(signed_volume)
        if volume < tol:
            continue
        if signed_volume < 0:
            coeff = -coeff
            candidate_g1_primitive = -candidate_g1_primitive
            candidate_po = -candidate_po
            signed_volume = -signed_volume
        angle_deg = _unoriented_angle_deg(candidate_po, plane_normal)
        candidates.append({
            'csl_coefficients': coeff,
            'vector_grain1_primitive': candidate_g1_primitive,
            'vector_cartesian': candidate_po,
            'volume': float(signed_volume),
            'length': float(candidate_length),
            'angle_to_plane_normal_deg': float(angle_deg),
        })

    if not candidates:
        raise ValueError('Could not find a non-coplanar CSL completion vector.')

    strategy_used = strategy
    fallback_reason = None
    selection_candidates = candidates
    if strategy == 'balanced_orthogonality_volume':
        selection_candidates = [
            candidate for candidate in candidates
            if candidate['angle_to_plane_normal_deg'] <=
            max_skew_deg + tol]
        if selection_candidates:
            key_name = 'volume_angle_length_coefficients'
        else:
            strategy_used = 'close_to_orthogonal'
            fallback_reason = (
                'No completion vector was found within max_skew_deg.')
            selection_candidates = candidates
            key_name = 'angle_volume_length_coefficients'
    elif strategy == 'min_volume':
        key_name = 'volume_angle_length_coefficients'
    else:
        key_name = 'angle_volume_length_coefficients'

    key_func = _completion_selection_key_func(key_name)
    best = min(selection_candidates, key=key_func)
    selection_key = key_func(best)

    return {
        'search_algorithm': 'bounded_csl_vector_search',
        'selection_strategy': strategy,
        'strategy_used': strategy_used,
        'selection_rule': _completion_selection_rule(
            strategy, max_skew_deg, strategy_used),
        'selection_key_name': key_name,
        'selection_key': selection_key,
        'fallback_reason': fallback_reason,
        'search_radius': int(max_index),
        'max_skew_deg': max_skew_deg,
        'candidate_count': len(candidates),
        'eligible_candidate_count': len(selection_candidates),
        'csl_coefficients': best['csl_coefficients'],
        'vector_grain1_primitive': best['vector_grain1_primitive'],
        'vector_cartesian': best['vector_cartesian'],
        'volume': best['volume'],
        'length': best['length'],
        'angle_to_plane_normal_deg':
            best['angle_to_plane_normal_deg'],
    }


def _normalize_completion_strategy(strategy):
    strategy = str(strategy).strip().lower().replace('-', '_')
    aliases = {
        'balanced': 'balanced_orthogonality_volume',
        'balanced_orthogonality': 'balanced_orthogonality_volume',
        'balanced_orthogonality_volume': 'balanced_orthogonality_volume',
        'default': 'balanced_orthogonality_volume',
        'orthogonal': 'close_to_orthogonal',
        'close_to_orthogonal': 'close_to_orthogonal',
        'min': 'min_volume',
        'minimum_volume': 'min_volume',
        'min_volume': 'min_volume',
    }
    if strategy not in aliases:
        raise ValueError('Unknown CSL completion strategy: '+str(strategy))
    return aliases[strategy]


def _completion_selection_key_func(key_name):
    if key_name == 'volume_angle_length_coefficients':
        return lambda candidate: (
            candidate['volume'],
            candidate['angle_to_plane_normal_deg'],
            candidate['length'],
            tuple(np.abs(candidate['csl_coefficients'])),
            tuple(candidate['csl_coefficients']))
    if key_name == 'angle_volume_length_coefficients':
        return lambda candidate: (
            candidate['angle_to_plane_normal_deg'],
            candidate['volume'],
            candidate['length'],
            tuple(np.abs(candidate['csl_coefficients'])),
            tuple(candidate['csl_coefficients']))
    raise ValueError('Unknown completion selection key: '+str(key_name))


def _completion_selection_rule(strategy, max_skew_deg, strategy_used):
    if strategy == 'balanced_orthogonality_volume':
        if strategy_used == 'balanced_orthogonality_volume':
            return (
                'Choose the smallest-volume CSL completion vector with '
                'angle_to_plane_normal_deg <= max_skew_deg; ties prefer '
                'smaller angle, shorter length, and smaller coefficients.')
        return (
            'Requested balanced_orthogonality_volume, but no vector met '
            'max_skew_deg; fell back to close_to_orthogonal.')
    if strategy == 'min_volume':
        return (
            'Choose the smallest-volume CSL completion vector; ties prefer '
            'smaller angle, shorter length, and smaller coefficients.')
    return (
        'Choose the CSL completion vector closest to the boundary-plane '
        'normal; ties prefer smaller volume, shorter length, and smaller '
        'coefficients.')


def _grain2_normal_from_grain1_po(normal_g1_po, sig_mat, lat_type):
    l_po2_po1 = _orthogonal_rotation_from_primitive(sig_mat, lat_type)
    l_po1_po2 = nla.inv(l_po2_po1)
    return _unit(-np.dot(l_po1_po2, normal_g1_po))


def _grain2_direct_from_grain1_p1(basis_g1_p1, sig_mat, tol=1e-6):
    basis_g2_p2 = np.dot(nla.inv(sig_mat), basis_g1_p1)
    basis_g2_p2, _ = _int_approx_columns(basis_g2_p2, tol)
    return basis_g2_p2


def _orthogonal_rotation_from_primitive(sig_mat, lat_type):
    l_p_po = lat_type.l_p_po
    l_po_p = nla.inv(l_p_po)
    return np.dot(l_p_po, np.dot(sig_mat, l_po_p))


def _direct_basis_conventional_views(basis_p, lat_type, tol):
    conv_coords = _direct_basis_to_conventional_coords(basis_p, lat_type)
    conv_dirs, multipliers = _int_approx_columns(conv_coords, tol)
    return conv_coords, conv_dirs, multipliers


def _direct_basis_to_conventional_coords(basis_p, lat_type):
    basis_po = np.dot(lat_type.l_p_po, basis_p)
    return np.dot(nla.inv(_conventional_basis_po(lat_type)), basis_po)


def _int_approx_columns(mat, tol):
    """
    Apply byxtal's integer approximation helper to each matrix column.
    """
    mat = np.asarray(mat, dtype='double')
    out = np.zeros(mat.shape, dtype='int64')
    multipliers = []
    for idx in range(mat.shape[1]):
        out[:, idx], multiplier = int_man.int_approx(mat[:, idx], tol)
        multipliers.append(float(multiplier))
    return out, np.asarray(multipliers, dtype='double')


def _basis_lengths(basis_po):
    return np.array([
        nla.norm(basis_po[:, idx]) for idx in range(basis_po.shape[1])
    ], dtype='double')


def _cell_angles_deg(basis_po):
    a_vec = basis_po[:, 0]
    b_vec = basis_po[:, 1]
    c_vec = basis_po[:, 2]
    return {
        'alpha': _vector_angle_deg(b_vec, c_vec),
        'beta': _vector_angle_deg(a_vec, c_vec),
        'gamma': _vector_angle_deg(a_vec, b_vec),
    }


def _vector_angle_deg(vec1, vec2):
    len1 = nla.norm(vec1)
    len2 = nla.norm(vec2)
    if len1 == 0 or len2 == 0:
        raise ValueError('Cannot compute an angle with a zero vector.')
    cos_ang = np.dot(vec1, vec2)/(len1*len2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def _to_builtin(value):
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    return value


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
