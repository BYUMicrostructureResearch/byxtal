import numpy as np
from sympy import Rational
import sympy as spy
from fractions import Fraction
import numpy.linalg as nla
import warnings


def gcd_vec(int_mat):
    """
    Compute the greatest common divisor of a matrix.

    This function is intended for integer arrays or vectors, including arrays
    with a single value. The implementation expects the input object to provide
    ``flatten()``.

    Parameters
    ----------
    int_mat: int (array-like object with integer entries)
        Input integer number; the input is flattened before computing the GCD.

    Returns
    -------
    gcd1: int
        The greatest common divisor of the input number; computed over all
        flattened entries.
        
    """
    input1 = int_mat.flatten()
    Sz = input1.shape
    gcd1 = 0
    for ct1 in range(Sz[0]):
        gcd1 = spy.gcd(gcd1, input1[ct1])

    return int(gcd1)


def gcd_array(input, order='all'):
    """
    The function computes the GCD of an array of numbers.

    This is the row/column-aware version of ``gcd_vec``. It is useful for
    reducing batches of integer directions either all together or one row/column
    at a time.

    Parameters
    ----------
    input : numpy.array or list
        Input n-D array of integers (most suitable for 1D and 2D arrays)
    order : {'rows', 'columns', 'cols', 'all'}, optional

    Returns
    -------
    Agcd: numpy.array (int for ``order='all'``)
        An array of greatest common divisors of the input; one GCD for
        ``order='all'``, otherwise an array of row or column GCDs.

    Notes
    -------
    * If order = **all**, the input array is flattened and the GCD is calculated
    * If order = **rows**, GCD of elements in each row is calculated
    * If order = **columns** or **cols**, GCD of elements in each column is calculated

    See Also
    --------
    gcd_vec: from fractions module for computing gcd of two integers

    """

    input = np.array(input)
    if not np.issubdtype(input.dtype, np.integer):
        raise Exception("Inputs must be real integers.")

    order_options = ('rows', 'columns', 'cols', 'all')
    try:
        Keys = (order_options.index(order))
    except:
        raise Exception(err_msg)

    if (Keys == 3):
        Agcd = gcd_vec(input)
    if (Keys == 0):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Agcd = np.zeros((sz1, 1))
        for ct1 in range(sz1):
            tmp_row = input[ct1, :]
            Agcd[ct1] = gcd_vec(tmp_row)
    if ((Keys == 1) or (Keys == 2)):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Agcd = np.zeros((1, sz2))
        for ct1 in range(sz2):
            tmp_row = input[:, ct1]
            Agcd[0, ct1] = gcd_vec(tmp_row)
    return Agcd


def lcm_vec(Dmat):
    """
    The function computes the least common multiple (LCM).

    The input is flattened first. This is commonly used to clear a set of
    rational denominators with one multiplier.

    Parameters
    ----------
    Dmat: int (array-like object with integer entries)
        The input number; the input is flattened before computing the LCM.

    Returns
    -------
    lcm1: int
        The least common multiple of the input number; computed over all
        flattened entries.
    """
    input1 = Dmat.flatten()
    Sz = input1.shape
    lcm1 = 1
    for ct1 in range(Sz[0]):
        lcm1 = spy.lcm(lcm1, input1[ct1])

    return int(lcm1)


def lcm_array(input, order='all'):
    """
    The function computes the LCM of an array of numbers.

    This is the row/column-aware version of ``lcm_vec``. It is commonly used
    to clear rational denominators either all together or one row/column at a
    time.

    Parameters
    ----------
    input: numpy.array or list of intgers
        Input n-D array of integers (most suitable for 1D and 2D arrays)
    order: {'rows', 'columns', 'cols', 'all'}, optional

    Returns
    -------
    Alcm: numpy.array (int for ``order='all'``)
        An array of least common multiples of the input; one LCM for
        ``order='all'``, otherwise an array of row or column LCMs.

    Notes
    -------
    * If order = **all**, the input array is flattened and the LCM is calculated
    * If order = **rows**, LCM of elements in each row is calculated
    * If order = **columns** or **cols**, LCM of elements in each column is calculated

    See Also
    --------
    lcm_vec: helper for computing the LCM after flattening an integer input
    """

    input = np.array(input)
    # Only integer values are allowed
    # if input.dtype.name != 'int64':
    if not np.issubdtype(input.dtype, np.integer):
        raise Exception("Inputs must be real integers.")

    order_options = ('rows', 'columns', 'cols', 'all')
    try:
        Keys = (order_options.index(order))
    except:
        raise Exception(err_msg)

    if (Keys == 3):
        Alcm = lcm_vec(input)
    if (Keys == 0):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Alcm = np.zeros((sz1, 1))
        for ct1 in range(sz1):
            tmp_row = input[ct1, :]
            Alcm[ct1] = lcm_vec(tmp_row)
    if ((Keys == 1) or (Keys == 2)):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Alcm = np.zeros((1, sz2))
        for ct1 in range(sz2):
            tmp_row = input[:, ct1]
            Alcm[0, ct1] = lcm_vec(tmp_row)
    return Alcm


def check_int_mat(T, tol1):
    """
    The function checks whether matrix is integer.

    Every entry is compared with its nearest rounded integer. This is a global
    all-entry check, not a row/column-wise check.

    Parameters
    ----------
    T: numpy.array
        Input matrix; ``sympy.Matrix`` inputs are converted to NumPy arrays.
    tol1: float
        Tolerance with default value 0.01
    Returns
    -------
    Boolean
        True: If the matrix has integer elements.
        False: If the matrix does not have integer elements.
    """
    if isinstance(T, spy.Matrix):
        T = np.array(T, dtype='double')
    return (np.max(np.abs(T - np.around(T))) < tol1)


def rat_approx(Tmat, tol1=0.01):
    """
    The function approximates the input with a rational number.

    Each entry is approximated independently using SymPy
    ``Rational(...).limit_denominator(1/tol1)``. The returned numerator and
    denominator arrays keep the original input shape.

    Parameters
    ----------
    Tmat: numpy.array
        Input; scalar, vector, matrix, and higher-dimensional array shapes are
        preserved in the output.
    tol1: float
        Tolerance with default value 0.01
    Returns
    -------
    Nmat1: int
        The nominator of the approximated rational number; returned as an
        integer array with the same shape as ``Tmat``.
    Dmat1: int
        The nominator of the approximated rational number; returned as an
        integer denominator array with the same shape as ``Tmat``.
    """
    Tmat = np.array(Tmat)
    input1 = Tmat.flatten()
    nshape = np.shape(Tmat)
    denum_max = 1/tol1
    Sz = input1.shape
    Nmat = np.zeros(np.shape(input1), dtype='int64')
    Dmat = np.zeros(np.shape(input1), dtype='int64')
    for ct1 in range(Sz[0]):
        num1 = (Rational(input1[ct1]).limit_denominator(denum_max))
        Nmat[ct1] = num1.p
        Dmat[ct1] = num1.q

    Nmat1 = np.reshape(Nmat, nshape)
    Dmat1 = np.reshape(Dmat, nshape)

    Nmat1 = np.array(Nmat1, dtype='int64')
    Dmat1 = np.array(Dmat1, dtype='int64')

    return Nmat1, Dmat1


def int_approx(Tmat, tol1=0.01):
    """
    The function finds an integer representation of one coupled object.

    It seeks an integer array and one global multiplier such that
    ``Tmat ≈ int_mat1/t1_mult``. For matrices, the multiplier is shared by all
    entries, which is appropriate for transformation matrices or rotation
    matrices where a common rational scale matters.

    Parameters
    ----------
    Tmat: numpy.array
        Transformation matrix; the whole input is treated as one coupled
        object with one returned multiplier.
    tol1: float
        Tolerance with default value 0.01
    Returns
    -------
    int_mat1:
        Integer array that approximates ``Tmat*t1_mult``.
    t1_mult:
        Global multiplier satisfying ``Tmat ≈ int_mat1/t1_mult``.
    """
    Tmat = np.array(Tmat)
    tct1 = np.max(np.abs(Tmat))
    tct2 = np.min(np.abs(Tmat))

    mult1 = 1/((tct1 + tct2)/2)
    mult2 = 1/np.max(np.abs(Tmat))

    int_mat1, t1_mult, err1 = mult_fac_err(Tmat, mult1, tol1)
    int_mat2, t2_mult, err2 = mult_fac_err(Tmat, mult2, tol1)

    if err1 == err2:
        tnorm1 = nla.norm(int_mat1)
        tnorm2 = nla.norm(int_mat2)
        if (tnorm1 > tnorm2):
            return int_mat2, t2_mult
        else:
            return int_mat1, t1_mult
    else:
        if err1 > err2:
            return int_mat2, t2_mult
        else:
            return int_mat1, t1_mult


def int_mult_approx(Tmat, tol1=0.01):
    """
    The function applies integer approximation with initial multiplier 1.

    This is useful when the input already has meaningful absolute scale and the
    caller wants the denominator-clearing multiplier found by ``mult_fac_err``
    without trying the alternate initial multipliers used by ``int_approx``.

    Parameters
    ----------
    Tmat: numpy.array
        Transformation matrix; the whole input is treated as one coupled
        object with one returned multiplier.
    tol1: float
        Tolerance with default value 0.01
    Returns
    -------
    int_mat1:
        Integer array that approximates ``Tmat*t1_mult``.
    t1_mult:
        Global multiplier satisfying ``Tmat ≈ int_mat1/t1_mult``.
    """
    Tmat = np.array(Tmat)
    int_mat1, t1_mult, err1 = mult_fac_err(Tmat, 1, tol1)
    return int_mat1, t1_mult


def mult_fac_err(Tmat, mult1, tol1):
    """
    The function computes the integer approximation for one trial multiplier.

    It rationalizes ``Tmat*mult1``, clears denominators with an LCM, divides
    out the global GCD, and reports the reconstruction error. This is the core
    helper behind ``int_approx`` and ``int_mult_approx``.

    Parameters
    ----------
    Tmat: numpy.array
        Transformation matrix; treated as one coupled object.
    mult1:
        Trial multiplier applied before rational approximation.
    tol1: float
        Tolerance
    Returns
    -------
    int_mat1:
        Integer array after clearing denominators and dividing by the GCD.
    t1_mult:
        Final multiplier satisfying ``Tmat ≈ int_mat1/t1_mult``.
    err1:
        Maximum absolute reconstruction error.
    """
    Tmat1 = Tmat*mult1
    N1, D1 = rat_approx(Tmat1, tol1)

    lcm1 = lcm_array(D1)
    N1 = np.array(N1, dtype='double')
    D1 = np.array(D1, dtype='double')

    int_mat1 = np.array((N1/D1)*lcm1, dtype='double')

    cond1 = check_int_mat(int_mat1, tol1*0.01)
    if cond1:
        int_mat1 = np.around(int_mat1)
        int_mat1 = np.array(int_mat1, dtype='int64')
    else:
        raise Exception("int_mat1 is not an integer matrix")
    gcd1 = gcd_vec(int_mat1)
    int_mat1 = int_mat1/gcd1

    int_mat1 = np.array(int_mat1, dtype='int64')
    t1_mult = mult1*lcm1/gcd1
    err1 = np.max(np.abs(Tmat - int_mat1/t1_mult))
    return int_mat1, t1_mult, err1


def int_finder(input_v, rational_tolerance=1e-6, order='all',
               zero_tolerance=1e-6, angular_tolerance_deg=0.001,
               max_index=1000, warn=True, return_multipliers=False,
               return_diagnostics=False):
    """
    The function computes the scaling factor required to multiply the
    given input array to obtain an integer array. The integer array is
    returned.

    The original method rationalizes component ratios and clears
    denominators. The new guardrail checks then verify angular error and
    maximum integer index for each requested group. If a group fails those
    checks, ``bounded_integer_direction`` is tried as a low-index angular
    fallback.

    Parameters
    ----------
    input1: numpy.array
        input array
    rational_tolerance: float
        Tolerance used for rational approximation, Default = 1e-06; this is
        the renamed form of the old ``tol`` argument.
    order: str
        choices are 'rows', 'columns', 'col', 'all'.
        If order = 'all', the input array is flattened and then scaled. This is default value.
        If order = 'rows', elements in each row are scaled
        If order = 'columns' or 'cols'', elements in each column are scaled
    zero_tolerance: float
        Tolerance used to treat components as zero, Default = 1e-06; this is
        the renamed form of the old ``tol1`` argument.
    angular_tolerance_deg: float
        Angular tolerance in degrees used for guardrail checks, Default = 0.001; groups with
        higher angular error try the bounded fallback.
    max_index: int
        Maximum allowed absolute integer index before bounded fallback is
        attempted, Default = 1000. Float inputs are floored to an integer.
    warn: bool
        If True, warn when guardrails fail and fallback is used or unavailable, Default = True.
    return_multipliers: bool
        If True, also return multiplier(s) satisfying
        ``input*multiplier ≈ output`` for each group, Default = False.
    return_diagnostics: bool
        If True, also return per-group guardrail and fallback diagnostics, Default = False.

    Returns
    -------
    output: numpy.array
        An array of integers obtained by scaling input
    multipliers: float or numpy.array
        Optional; returned when ``return_multipliers=True``. Scalar for
        ``order='all'`` or 1D input; otherwise one multiplier per requested
        row or column group, satisfying ``input*multiplier`` is approximately
        ``output``.
    diagnostics: list of dict
        Optional; returned when ``return_diagnostics=True``. Contains
        per-group guardrail and fallback information, including angular error,
        max index checks, multipliers, and whether bounded fallback was used.

    Notes
    -----
    If both optional return flags are True, values are returned as
    ``output, multipliers, diagnostics``.
    """

    input_original = np.array(input_v, dtype=np.float64)
    input1 = np.array(input_v)
    Sz = input1.shape
    if np.ndim(input1) == 1:
        input1 = np.reshape(input1, (1, input1.shape[0]))

    if int_check(input1, 15).all():
        input1 = np.around(input1)
        # Divide by LCM (rows, cols, all) <--- To Do
        tmult = gcd_array(input1.astype(dtype='int64'), order)
        if (order == 'all'):
            input1 = input1 / tmult
        elif (order == 'rows'):
            tmult = np.tile(tmult, (1, np.shape(input1)[1]))
            input1 = input1 / tmult
        elif (order == 'col' or order == 'cols' or order == 'columns'):
            tmult = np.tile(tmult, (np.shape(input1)[0], 1))
            input1 = input1 / tmult
        output_v = input1
        if len(Sz) == 1:
            output_v = np.reshape(output_v, (np.size(output_v),))
        output_ok, multipliers, diagnostics = _check_int_finder_output(
            input_original, output_v, order, angular_tolerance_deg, max_index)
        if not output_ok:
            output_v, multipliers, diagnostics = \
                _replace_failed_int_finder_groups_with_bounded_search(
                    input_original, output_v, order, angular_tolerance_deg,
                    max_index, diagnostics, warn)
        if return_multipliers and return_diagnostics:
            return output_v, multipliers, diagnostics
        if return_multipliers:
            return output_v, multipliers
        if return_diagnostics:
            return output_v, diagnostics
        return output_v
    else:
        #   By default it flattens the array (if nargin < 3)
        if order.lower() == 'all':
            if len(Sz) != 1:
                input1.shape = (1, Sz[0]*Sz[1])
        else:
            Switch = 0
            err_msg = "Not a valid input. For the third argument please"+ \
                      " choose either \"rows\" or \"columns\" keys for this function."
            order_options = ('rows', 'columns', 'col')
            try:
                Keys = (order_options.index(order.lower()))
            except:
                raise Exception(err_msg)

            if (Keys == 1) or (Keys == 2):
                if input1.shape[0] != 1:
                    # Handling the case of asking a row vector
                    # with the 'column' key by mistake.
                    input1 = input1.T
                    Switch = 1
            # Handling the case of asking a column
            # vector with the 'row' key by mistake.
            if (Keys == 0) and (input1.shape[1] == 1):
                input1 = input1.T
                Switch = 1

        if (abs(input1) < rational_tolerance).all():
            excep1 = 'All the input components cannot' \
                     + 'be smaller than rational_tolerance.'
            raise Exception(excep1)

        tmp = np.array((abs(input1) > zero_tolerance))
        Vec = 2 * abs(input1[::]).max() * np.ones(
            (input1.shape[0], input1.shape[1]))
        Vec[tmp] = input1[tmp]
        MIN = abs(Vec).min(axis=1)
        # Transposing a row to a column
        MIN.shape = (len(MIN), 1)
        input1 = input1 / np.tile(MIN, (1, input1.shape[1]))
        N, D = rat(input1, rational_tolerance)
        N[~tmp] = 0 # <---- added
        D[~tmp] = 1 # <---- added
        lcm_rows = lcm_array(D, 'rows')
        lcm_mat = np.tile(lcm_rows, (1, input1.shape[1]))
        Rounded = (N * lcm_mat) / D
        output_v = Rounded

        # --------------------------
        if order.lower() == 'all':
            if len(Sz) != 1:
                output_v.shape = (Sz[0], Sz[1])
        else:
            if (Keys) == 1 or (Keys) == 2:
                output_v = output_v.T
            if Keys == 0 and Switch == 1:
                output_v = output_v.T

        if len(Sz) == 1:
            output_v = np.reshape(output_v, (np.size(output_v), ))

        output_ok, multipliers, diagnostics = _check_int_finder_output(
            input_original, output_v, order, angular_tolerance_deg, max_index)
        if not output_ok:
            output_v, multipliers, diagnostics = \
                _replace_failed_int_finder_groups_with_bounded_search(
                    input_original, output_v, order, angular_tolerance_deg,
                    max_index, diagnostics, warn)
        if return_multipliers and return_diagnostics:
            return output_v, multipliers, diagnostics
        if return_multipliers:
            return output_v, multipliers
        if return_diagnostics:
            return output_v, diagnostics
        return output_v


def bounded_integer_direction(direction, max_index=1000,
                              angular_tolerance_deg=0.001,
                              warn=True, return_multiplier=False,
                              return_diagnostics=False):
    """
    Find a bounded integer direction closest to an input direction.

    The search follows the integer candidates obtained by rounding
    ``k*direction`` for positive integer ``k`` values bounded by
    ``max_index``. It returns the best angular match, reduced by the GCD of
    its entries.

    Parameters
    ----------
    direction: numpy.array or list
        Input direction vector; only the direction matters, not the magnitude.
    max_index: int
        Maximum absolute value allowed in each integer component (default: 1000); float inputs
        are floored before use.
    angular_tolerance_deg: float
        Angular tolerance in degrees (default: 0.001) used to decide whether the best candidate
        met the requested tolerance.
    warn: bool
        If True (default: True), warn when no candidate is found within
        ``angular_tolerance_deg``.
    return_multiplier: bool
        If True (default: False), also return the least-squares scalar multiplier satisfying
        ``direction*multiplier ≈ int_direction``.
    return_diagnostics: bool
        If True (default: False), also return a diagnostics dictionary with angular error,
        candidate count, tolerance, max index, and multiplier information.

    Returns
    -------
    int_direction: numpy.array
        Primitive integer direction with GCD divided out.
    err: float
        Dot-product error, ``1 - dot(unit(direction), unit(int_direction))``.
    angular_error_deg: float
        Angular error between the input direction and integer direction.
    multiplier: float
        Optional; returned when ``return_multiplier=True``. Least-squares
        scalar satisfying ``direction*multiplier`` is approximately
        ``int_direction``.
    diagnostics: dict
        Optional; returned when ``return_diagnostics=True``. Contains
        ``max_index``, ``angular_tolerance_deg``, ``found_within_tolerance``,
        ``dot_value``, ``err``, ``angular_error_deg``, ``multiplier``, and
        ``candidate_count``.

    Notes
    -----
    Optional values are appended after ``angular_error_deg`` in the order
    ``multiplier``, then ``diagnostics``.
    """
    direction = np.asarray(direction, dtype='double').reshape(-1,)
    if direction.size == 0:
        raise ValueError('direction must contain at least one component.')
    direction_norm = nla.norm(direction)
    if direction_norm == 0:
        raise ValueError('direction must not be the zero vector.')

    max_index_arr = _max_index_array(max_index, direction.size)
    angular_tolerance_deg = float(angular_tolerance_deg)
    dot_value_tolerance = 1.0 - np.cos(np.deg2rad(angular_tolerance_deg))

    unit_direction = direction/direction_norm
    nonzero = np.abs(unit_direction) > np.finfo(float).eps
    max_steps = int(np.floor(np.min(
        np.abs(max_index_arr[nonzero]/unit_direction[nonzero]))))
    if max_steps < 1:
        raise ValueError('max_index is too small for the input direction.')

    candidates = np.around(
        np.arange(1, max_steps+1, dtype='double')[:, None]*unit_direction)
    candidates = candidates.astype(dtype='int64')
    candidate_norms = nla.norm(candidates, axis=1)
    nonzero_candidates = candidate_norms > 0
    candidates = candidates[nonzero_candidates]
    candidate_norms = candidate_norms[nonzero_candidates]
    if candidates.size == 0:
        raise ValueError('No nonzero integer direction candidates found.')

    candidate_units = candidates/candidate_norms[:, None]
    dot_values = np.dot(candidate_units, unit_direction)
    best_index = int(np.argmax(dot_values))
    best_dot = float(np.clip(dot_values[best_index], -1.0, 1.0))
    found_within_tolerance = bool(
        np.any((1.0 - dot_values) <= dot_value_tolerance))
    if warn and not found_within_tolerance:
        warnings.warn(
            'bounded_integer_direction did not find a candidate within '
            'angular_tolerance_deg; returning the best candidate '
            '(angular_tolerance_deg=' + str(angular_tolerance_deg) +
            ', max_index=' +
            str(_to_builtin_number_or_list(max_index_arr)) + ').',
            RuntimeWarning,
            stacklevel=2)

    int_direction = _primitive_int_vector(candidates[best_index])
    if np.dot(int_direction, direction) < 0:
        int_direction = -int_direction

    err = 1.0 - best_dot
    if err < 0:
        err = 0.0
    angular_error_deg = float(np.rad2deg(np.arccos(1.0 - err)))
    multiplier = _direction_multiplier(direction, int_direction)
    diagnostics = {
        'max_index': _to_builtin_number_or_list(max_index_arr),
        'angular_tolerance_deg': angular_tolerance_deg,
        'found_within_tolerance': found_within_tolerance,
        'dot_value': best_dot,
        'err': float(err),
        'angular_error_deg': angular_error_deg,
        'multiplier': multiplier,
        'candidate_count': int(candidates.shape[0]),
    }

    result = (int_direction, float(err), angular_error_deg)
    if return_multiplier:
        result = result + (multiplier,)
    if return_diagnostics:
        result = result + (diagnostics,)
    return result


def _check_int_finder_output(input_v, output_v, order,
                             angular_tolerance_deg, max_index):
    """
    Check whether int_finder's rationalized output is a usable direction set.

    This is the guardrail layer around the original int_finder algorithm. It
    leaves the original integer approximation untouched, then measures each
    requested direction group for angular error, maximum index size, and the
    scalar multiplier relating the input direction to the integer output.

    Parameters
    ----------
    input_v: numpy.array
        Original input direction data.
    output_v: numpy.array
        Integer-like output produced by int_finder before fallback checks.
    order: str
        Direction grouping convention, matching int_finder's order argument.
    angular_tolerance_deg: float
        Maximum allowed angular error for each direction group.
    max_index: int or array-like
        Maximum allowed absolute integer index.

    Returns
    -------
    output_ok: bool
        True when every direction group satisfies the angular and index checks.
    multipliers: float or numpy.array
        Scalar multiplier(s) satisfying input*multiplier is approximately output.
    diagnostics: list of dict
        Per-group measurements and pass/fail flags used by the fallback step.
    """
    diagnostics = []
    multipliers = []
    output_ok = True
    for group in _int_finder_direction_groups(input_v, output_v, order):
        input_direction = group['input_direction']
        output_direction = group['output_direction']
        max_index_arr = _max_index_array(max_index, output_direction.size)
        angular_error_deg = _direction_angular_error_deg(
            input_direction, output_direction)
        max_abs_index = (
            0 if output_direction.size == 0
            else int(np.max(np.abs(output_direction))))
        max_index_ok = bool(np.all(np.abs(output_direction) <= max_index_arr))
        angular_ok = bool(angular_error_deg <= angular_tolerance_deg)
        multiplier = _direction_multiplier(input_direction, output_direction)
        diagnostics.append({
            'group_index': group['group_index'],
            'group_axis': group['group_axis'],
            'angular_error_deg': angular_error_deg,
            'angular_tolerance_deg': float(angular_tolerance_deg),
            'angular_ok': angular_ok,
            'max_abs_index': max_abs_index,
            'max_index': _to_builtin_number_or_list(max_index_arr),
            'max_index_ok': max_index_ok,
            'multiplier': multiplier,
            'used_bounded_fallback': False,
        })
        multipliers.append(multiplier)
        if (not angular_ok) or (not max_index_ok):
            output_ok = False
    return output_ok, _format_int_finder_group_values(
        multipliers, input_v, order), diagnostics


def _replace_failed_int_finder_groups_with_bounded_search(
        input_v, output_v, order, angular_tolerance_deg, max_index,
        diagnostics, warn):
    """
    Replace failed int_finder direction groups with bounded low-index searches.

    This helper is only called after _check_int_finder_output finds at least
    one direction group that fails the angular or index guardrails. It tries
    bounded_integer_direction for the failed groups, keeps the replacement when
    it satisfies the guardrails or improves the angular error, and leaves
    passing groups unchanged.

    Parameters
    ----------
    input_v: numpy.array
        Original input direction data.
    output_v: numpy.array
        Integer-like output produced by int_finder before fallback replacement.
    order: str
        Direction grouping convention, matching int_finder's order argument.
    angular_tolerance_deg: float
        Maximum allowed angular error for each direction group.
    max_index: int or array-like
        Maximum allowed absolute integer index.
    diagnostics: list of dict
        Per-group diagnostics from _check_int_finder_output.
    warn: bool
        If True, warn when fallback is used or when no better fallback is found.

    Returns
    -------
    output_arr: numpy.array
        Output array after replacing any failed groups accepted from fallback.
    multipliers: float or numpy.array
        Updated scalar multiplier(s) for the returned output.
    new_diagnostics: list of dict
        Updated per-group diagnostics, including bounded fallback details.
    """
    output_arr = np.asarray(output_v, dtype='double').copy()
    multipliers = []
    new_diagnostics = []
    groups = _int_finder_direction_groups(input_v, output_arr, order)
    for group, diagnostic in zip(groups, diagnostics):
        if diagnostic['angular_ok'] and diagnostic['max_index_ok']:
            multipliers.append(diagnostic['multiplier'])
            new_diagnostics.append(diagnostic)
            continue

        fallback_used = False
        try:
            bounded_result = bounded_integer_direction(
                group['input_direction'], max_index=max_index,
                angular_tolerance_deg=angular_tolerance_deg, warn=False,
                return_multiplier=True, return_diagnostics=True)
            bounded_v, _, _, bounded_multiplier, bounded_diagnostic = \
                bounded_result
            bounded_angle = bounded_diagnostic['angular_error_deg']
            bounded_max = int(np.max(np.abs(bounded_v)))
            bounded_max_index = _max_index_array(max_index, bounded_v.size)
            bounded_ok = (
                bounded_angle <= angular_tolerance_deg and
                np.all(np.abs(bounded_v) <= bounded_max_index))
            old_ok = diagnostic['angular_ok'] and diagnostic['max_index_ok']
            if bounded_ok or (
                    not old_ok and
                    bounded_angle <= diagnostic['angular_error_deg']):
                _set_int_finder_direction(output_arr, group, bounded_v)
                diagnostic = {
                    **diagnostic,
                    'angular_error_deg': bounded_angle,
                    'angular_ok': bool(
                        bounded_angle <= angular_tolerance_deg),
                    'max_abs_index': bounded_max,
                    'max_index_ok': bool(
                        np.all(np.abs(bounded_v) <= bounded_max_index)),
                    'multiplier': bounded_multiplier,
                    'used_bounded_fallback': True,
                    'bounded_fallback': bounded_diagnostic,
                }
                fallback_used = True
        except ValueError as exc:
            diagnostic = {
                **diagnostic,
                'bounded_fallback_error': str(exc),
            }

        if warn:
            if fallback_used:
                warnings.warn(
                    'int_finder rational output failed guardrails and was '
                    'replaced by bounded_integer_direction '
                    '(angular_tolerance_deg=' +
                    str(float(angular_tolerance_deg)) + ', max_index=' +
                    str(_to_builtin_number_or_list(
                        _max_index_array(max_index,
                                         group['output_direction'].size))) +
                    ').',
                    RuntimeWarning,
                    stacklevel=2)
            else:
                warnings.warn(
                    'int_finder rational output failed guardrails, but '
                    'bounded_integer_direction did not provide a better '
                    'replacement; keeping rational output '
                    '(angular_tolerance_deg=' +
                    str(float(angular_tolerance_deg)) + ', max_index=' +
                    str(_to_builtin_number_or_list(
                        _max_index_array(max_index,
                                         group['output_direction'].size))) +
                    ').',
                    RuntimeWarning,
                    stacklevel=2)

        multipliers.append(diagnostic['multiplier'])
        new_diagnostics.append(diagnostic)

    if np.issubdtype(np.asarray(output_v).dtype, np.integer):
        output_arr = output_arr.astype(dtype='int64')
    return (
        output_arr,
        _format_int_finder_group_values(multipliers, input_v, order),
        new_diagnostics)


def _int_finder_direction_groups(input_v, output_v, order):
    """
    Split int_finder input and output arrays into comparable direction groups.

    The original int_finder can treat the whole array, each row, or each column
    as the unit to rationalize. The guardrail checks need the same grouping so
    each independent direction is checked and, if needed, replaced without
    changing unrelated rows or columns.

    Parameters
    ----------
    input_v: numpy.array
        Original input direction data.
    output_v: numpy.array
        Integer-like output produced from the same input.
    order: str
        Direction grouping convention, matching int_finder's order argument.

    Returns
    -------
    groups: list of dict
        Direction group records containing input/output vectors, group labels,
        group index, and the slice needed to write replacements back.
    """
    input_arr = np.asarray(input_v, dtype='double')
    output_arr = np.asarray(output_v, dtype='double')
    order = order.lower()
    if input_arr.ndim == 1 or order == 'all':
        return [{
            'group_index': 0,
            'group_axis': 'all',
            'input_direction': input_arr.reshape(-1,),
            'output_direction': output_arr.reshape(-1,),
            'slice': None,
        }]
    if order == 'rows':
        return [
            {
                'group_index': idx,
                'group_axis': 'rows',
                'input_direction': input_arr[idx, :],
                'output_direction': output_arr[idx, :],
                'slice': (idx, slice(None)),
            }
            for idx in range(input_arr.shape[0])
        ]
    if order in ('columns', 'cols', 'col'):
        return [
            {
                'group_index': idx,
                'group_axis': 'columns',
                'input_direction': input_arr[:, idx],
                'output_direction': output_arr[:, idx],
                'slice': (slice(None), idx),
            }
            for idx in range(input_arr.shape[1])
        ]
    raise ValueError('order must be all, rows, columns, cols, or col.')


def _set_int_finder_direction(output_arr, group, direction):
    """
    Write one fallback direction back into the correct int_finder output group.

    Parameters
    ----------
    output_arr: numpy.array
        Mutable output array being updated.
    group: dict
        Direction group record from _int_finder_direction_groups.
    direction: numpy.array
        Replacement integer direction for the selected group.
    """
    if group['slice'] is None:
        output_arr[...] = np.asarray(direction).reshape(output_arr.shape)
    else:
        output_arr[group['slice']] = direction


def _format_int_finder_group_values(values, input_v, order):
    """
    Match optional int_finder return values to the requested grouping style.

    The public int_finder return should be scalar-like for a single direction
    and array-like when rows or columns were treated independently. This helper
    keeps multiplier returns consistent with that convention.

    Parameters
    ----------
    values: sequence
        Per-group values to format.
    input_v: numpy.array
        Original input direction data.
    order: str
        Direction grouping convention, matching int_finder's order argument.

    Returns
    -------
    formatted_values: float or numpy.array
        Scalar for all/1D grouping, otherwise an array of per-group values.
    """
    values = np.asarray(values, dtype='double')
    if np.asarray(input_v).ndim == 1 or order.lower() == 'all':
        return values[0]
    return values


def _direction_angular_error_deg(direction, integer_direction):
    """
    Compute unsigned angular error between a direction and an integer direction.

    The sign of a crystallographic direction can be equivalent in these checks,
    so the absolute dot product is used. A zero direction returns infinity so
    the guardrails fail conservatively instead of silently accepting it.

    Parameters
    ----------
    direction: numpy.array
        Input direction.
    integer_direction: numpy.array
        Candidate integer direction.

    Returns
    -------
    angular_error_deg: float
        Unsigned angular error in degrees.
    """
    direction = np.asarray(direction, dtype='double').reshape(-1,)
    integer_direction = np.asarray(
        integer_direction, dtype='double').reshape(-1,)
    if nla.norm(direction) == 0 or nla.norm(integer_direction) == 0:
        return np.inf
    dot_val = np.dot(direction, integer_direction)/(
        nla.norm(direction)*nla.norm(integer_direction))
    dot_val = np.clip(abs(dot_val), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(dot_val)))


def _direction_multiplier(direction, integer_direction):
    """
    Estimate the scalar multiplier connecting a direction to an integer vector.

    The multiplier is a least-squares scale factor for
    direction*multiplier approximately equal to integer_direction. It is used
    for diagnostics and optional int_finder multiplier returns.

    Parameters
    ----------
    direction: numpy.array
        Input direction.
    integer_direction: numpy.array
        Integer direction associated with the input.

    Returns
    -------
    multiplier: float
        Least-squares scalar multiplier, or NaN for a zero input direction.
    """
    direction = np.asarray(direction, dtype='double').reshape(-1,)
    integer_direction = np.asarray(
        integer_direction, dtype='double').reshape(-1,)
    denom = np.dot(direction, direction)
    if denom == 0:
        return np.nan
    return float(np.dot(direction, integer_direction)/denom)


def _primitive_int_vector(vector):
    """
    Reduce an integer vector to primitive integer indices.

    This divides all components by their greatest common divisor so equivalent
    integer directions are reported in their smallest crystallographic form.

    Parameters
    ----------
    vector: numpy.array
        Integer vector to reduce.

    Returns
    -------
    primitive_vector: numpy.array
        Integer vector divided by the component GCD when possible.
    """
    vector = np.asarray(vector, dtype='int64').reshape(-1,)
    gcd1 = gcd_vec(vector)
    if gcd1 == 0:
        return vector
    return (vector/gcd1).astype(dtype='int64')


def _max_index_array(max_index, size):
    """
    Normalize scalar or per-component max_index input for guardrail checks.

    The bounded search and guardrails accept either one maximum index for every
    component or one value per component. Float values are floored so callers
    cannot accidentally expand the allowed search range with non-integer input.

    Parameters
    ----------
    max_index: int, float, or array-like
        Scalar or per-component maximum absolute integer index.
    size: int
        Number of components in the direction being checked.

    Returns
    -------
    max_index_arr: numpy.array
        Integer array with one maximum index value per direction component.
    """
    max_index_arr = np.asarray(max_index, dtype='double')
    if max_index_arr.ndim == 0:
        max_index_arr = np.full((size,), np.floor(max_index_arr))
    else:
        max_index_arr = np.floor(max_index_arr.reshape(-1,))
        if max_index_arr.size != size:
            raise ValueError('max_index must be scalar or match vector size.')
    if np.any(max_index_arr < 1):
        raise ValueError('max_index values must be at least 1 after flooring.')
    return max_index_arr.astype(dtype='int64')


def _to_builtin_number_or_list(value):
    """
    Convert NumPy scalar or array values into warning-friendly Python objects.

    Diagnostics and warning messages should not expose NumPy scalar reprs when
    a plain Python number or list is easier to read. This helper keeps those
    messages compact without changing the numerical checks.

    Parameters
    ----------
    value: numpy.array
        Scalar or array-like value to format.

    Returns
    -------
    formatted_value: int, float, or list
        Python scalar for single values, otherwise a Python list.
    """
    value = np.asarray(value)
    if value.ndim == 0 or value.size == 1:
        return value.reshape(-1,)[0].item()
    return value.tolist()


def int_check(input, precis=6):
    """
    Checks whether the input variable (arrays) is an interger or not.
    A precision value is specified and the integer check is performed
    up to that decimal point.

    This is an elementwise check. It returns a Boolean array with the same
    shape as the input rather than one global pass/fail value.

    Parameters
    ----------
    input : numpy.array or list
        Input n-D array of floats; checked element by element.
    precis : int
        Default = 6.
        A value that specifies the precision to which the number is an
        integer. **precis = 6** implies a precision of :math:`10^{-6}`.

    Returns
    -------
    cond: Boolean
        'True' if the element is an integer to a certain precision,
        'False' otherwise; returned elementwise with the input shape.
    """

    var = np.array(input)
    tval = 10 ** -precis
    t1 = abs(var)
    cond = (abs(t1 - np.around(t1)) < tval)
    return cond


def rat(input, tol=1e-06):
    """
    The function returns a rational (p/q) approximation of a given
    floating point array to a given precision

    This uses Python ``Fraction.from_float(...).limit_denominator(1/tol)`` and
    is mostly oriented toward 1D and 2D inputs. It is used by ``int_finder`` for
    ratio rationalization.

    Parameters
    ----------
    input : numpy.array or list
        input which is real numbers; 1D inputs are treated as one row.
    tol : float
        Tolerance, Default = 1e-06

    Returns
    -------
    N: numpy.array
        N contain the numerators (p) and denominators (q) of the
        rational approximations; specifically the numerator array.
    D: numpy.array
        D contain the numerators (p) and denominators (q) of the
        rational approximations; specifically the denominator array.
    """
    input1 = np.array(input)
    if np.ndim(input1) == 1:
        input1 = np.reshape(input1, (1, input1.shape[0]))

    ## Why is this case necessary?
    if input1.ndim == 0:
        input1 = np.reshape(input1, (1, 1))

    Sz = input1.shape
    N = np.zeros((Sz[0], Sz[1]), dtype='int64')
    D = np.zeros((Sz[0], Sz[1]), dtype='int64')
    nDec = int(1/tol)
    for i in range(Sz[0]):
        for j in range(Sz[1]):
            N[i, j] = (Fraction.from_float(input1[i, j]).
                       limit_denominator(nDec).numerator)
            D[i, j] = (Fraction.from_float(input1[i, j]).
                       limit_denominator(nDec).denominator)
    return N, D
