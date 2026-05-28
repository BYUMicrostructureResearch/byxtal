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
        ``candidates`` contains the first ``n_results`` records.
        ``all_candidates`` contains every matching record.
        ``groups`` contains records grouped by boundary-plane FZ key.
    """
    _validate_csl_record(csl_record)
    target_normal_po = _normal_from_conventional_miller(
        target_miller, lat_type)

    normal_records = _generate_normals_near_target(
        csl_record['csl_mat'], lat_type.l_p_po, target_normal_po, max_area,
        angle_radius_deg, max_index=max_index, tol=tol)

    candidates = []
    for normal_record in normal_records:
        candidate = _compute_2d_csl_metrics(
            normal_record['csl_reciprocal_index'],
            csl_record['csl_mat'],
            lat_type,
            target_normal_po=target_normal_po,
            max_transform_index=max_transform_index,
            max_area_multiplier=max_area_multiplier,
            tol=tol)

        candidate['sig_id'] = csl_record.get('sig_id')
        candidate['csl_rotation_id'] = csl_record.get('csl_rotation_id')
        candidate['sig_mat'] = csl_record.get('sig_mat')
        candidate['target_miller'] = np.asarray(target_miller, dtype='int64')
        candidates.append(candidate)

    candidates, groups = _canonicalize_by_bp_symmetry(
        candidates, csl_record.get('csl_bp_props'))
    candidates.sort(key=_candidate_rank_key)

    return {
        'recommended': candidates[0] if candidates else None,
        'candidates': candidates[:n_results],
        'all_candidates': candidates,
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
        'best_balanced': cell_options['best_balanced'],
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
        'best_balanced': min(options, key=lambda option: option['score']),
        'pareto_candidates': _pareto_filter(options),
    }


def _canonicalize_by_bp_symmetry(candidates, csl_bp_props=None, decimals=10):
    """
    Map normals into the boundary-plane FZ and group equivalent candidates.
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
        return candidates, groups

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

    return candidates, groups


def _pareto_filter(options, keys=('area', 'angle_error_deg', 'aspect_ratio')):
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
    return (
        candidate['best_balanced']['score'],
        candidate['target_angle_error_deg'],
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
