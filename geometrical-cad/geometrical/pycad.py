from dataclasses import dataclass
from typing import List, Literal, Optional
import copy
import numpy as np

LinearConstraint = np.array
ConstraintList = np.ndarray
Section = LinearConstraint | Literal['+inf'] | Literal['-inf']

@dataclass
class Cell:
    """
    A cell of our decomposition.
    """
    lower_bound: Section
    upper_bound: Section
    stack_above: List['Cell']

    # We keep a reference to the parent so we can reconstruct our corner points
    # when given a leaf cell. Not set for R0
    parent: Optional['Cell']

    # At the top level, we want to associate a function of one dimension higher
    # with each cell; the component function of that node.
    component_function: Optional[LinearConstraint] = None

    # We cache a sample point, since it is used to both lift the cells and to
    # check containment
    sample_point: np.array = None

    # Cached values used by absorb_component_functions().
    _stack_above_upper_bounds: Optional[np.ndarray] = None
    _stack_above_upper_bounds_key: Optional[tuple] = None
    _stack_above_upper_bounds_values: Optional[np.ndarray] = None

@dataclass
class PWL:
    """
    Represents the PWL of a node (hidden or output).
    """
    # We have to keep a list of constraints so we can merge them with another
    # PWL in the sum step.
    constraints: ConstraintList

    # We keep the decomposition so we can search for the PWL on each leaf cell
    # of the CAD.
    cad: Cell

#
# Calculating the CAD.
#

def project_dimension(constraints: np.ndarray):
    """
    Projects the highest dimension away using vectorized 2D array operations.
    `constraints` is a 2D np.array of shape (num_constraints, dim + 1)
    """
    if constraints.size == 0:
        return np.array([])

    dim = constraints.shape[1] - 1

    mask_nonzero = constraints[:, dim] != 0
    to_intersect = constraints[mask_nonzero]
    lower_dims = constraints[~mask_nonzero, :-1]

    if to_intersect.shape[0] < 2:
        return np.unique(lower_dims, axis=0) if lower_dims.size > 0 else np.array([])

    # Vectorized FM elimination.
    # (a[:-1] / a[dim]) - (b[:-1] / b[dim]) for all pairs
    coeffs = to_intersect[:, :-1]
    divisors = to_intersect[:, [dim]] # Keep dims for broadcasting
    normalized = coeffs / divisors

    # Basically itertools.combinations.
    all_pairs = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]

    # Extract only the upper triangle to avoid redundant (a-b) and (b-a) or (a-a)
    tri_indices = np.triu_indices(all_pairs.shape[0], k=1)
    projections = all_pairs[tri_indices]

    # Combine and unique
    if lower_dims.size > 0:
        result = np.vstack([projections, lower_dims])
    else:
        result = projections

    # Drop all constraints that state a0=0.
    result = result[np.any(result[:, 1:] != 0, axis=1)]

    # Normalize to make the now highest dim equal to 1 (if it is nonzero).
    divisors = result[:, -1].copy()
    divisors[divisors == 0] = 1
    result = result / divisors[:, None]

    return np.unique(result, axis=0)


def project_to_r0(constraints: ConstraintList, highest_dim=None):
    highest_dim = highest_dim if highest_dim else constraints.shape[1] - 1
    constraints_per_step = {highest_dim: constraints}

    for d in range(highest_dim, 0, -1):
        constraints_per_step[d-1] = project_dimension(constraints_per_step[d])

    return constraints_per_step


def calculate_xn(constraints: ConstraintList, var_values: np.array):
    """
    Given a linear constraint (list of coefficients) and the values of all
    variable values except for the last one, calculates the variable value.

    That is, we want a0 + a1*x1 + ... + an*xn for all combinations of
    constraints and values.
    """
    # coeffs can be 1D (one equation) or 2D (multiple equations)
    a0 = constraints[..., 0]
    an = constraints[..., -1]
    ai = constraints[..., 1:-1]

    if ai.size == 0:
        aixi_sums = 0.0
    else:
        if constraints.ndim == 1:
            aixi_sums = float(np.dot(ai, var_values))
        else:
            aixi_sums = np.tensordot(ai, var_values, axes=([-1], [0]))

    return -(a0 + aixi_sums) / an


def vertical_constraint_of_dim(dim: int, value):
    constraint = np.zeros(shape=(dim+1,))
    constraint[0] = value
    constraint[-1] = -1

    return constraint


def lift_recursive(
    dim: int,
    parent_cell: Cell,
    constraints_per_dim: dict,
    sample_point_buf: np.ndarray,
    max_dim: int,
):
    """
    Lifts cells using the actual linear equations as boundaries. Requires to
    keep a list of previous sample point so we can calculate the new points and
    order the constraints.
    """
    # sample_point_buf is a preallocated buffer of length = max_dim.
    # The current prefix is sample_point_buf[:dim-1].
    sample_point = sample_point_buf[: dim - 1]
    current_constraints = constraints_per_dim[dim]

    # If a dimension contains no constraints, the interval will be from -inf to
    # +inf.
    if current_constraints.size == 0:
        sample_point_buf[dim - 1] = 0.0
        sector_cell = Cell(
            lower_bound=vertical_constraint_of_dim(dim, -np.inf),
            upper_bound=vertical_constraint_of_dim(dim, np.inf),
            stack_above=[],
            parent=parent_cell,
            sample_point=sample_point_buf[:dim].copy()
        )
        parent_cell.stack_above.append(sector_cell)

        if dim < max_dim:
            lift_recursive(dim + 1, sector_cell, constraints_per_dim, sample_point_buf, max_dim)

        return

    # Only constraints that involve the current dimension.
    mask_active = current_constraints[:, -1] != 0
    active_constraints = current_constraints[mask_active]

    if active_constraints.size > 0:
        # Calculate the sample points to order the constraints
        a0 = active_constraints[:, 0]
        ai = active_constraints[:, 1:-1]
        an = active_constraints[:, -1]
        vals = -(a0 + ai @ sample_point) / an

        sorted_indices = np.argsort(vals)
        sorted_constraints = active_constraints[sorted_indices]
        sorted_vals = vals[sorted_indices]
    else:
        sorted_constraints = np.array([])
        sorted_vals = np.array([])

    # Define the sequence of sections and sectors
    num_boundaries = len(sorted_constraints)
    for i in range(num_boundaries + 1):
        lower_eqn = sorted_constraints[i-1] if i > 0 else vertical_constraint_of_dim(dim, -np.inf)
        upper_eqn = sorted_constraints[i] if i < num_boundaries else vertical_constraint_of_dim(dim, np.inf)

        # Generate a sample point for the next dimension
        l_val = sorted_vals[i-1] if i > 0 else (sorted_vals[i] - 1 if num_boundaries > 0 else 0)
        u_val = sorted_vals[i] if i < num_boundaries else (sorted_vals[i-1] + 1 if num_boundaries > 0 else 0)
        mid_val = (l_val + u_val) / 2
        sample_point_buf[dim - 1] = mid_val
        new_sample_point = sample_point_buf[:dim].copy()

        sector_cell = Cell(
            lower_bound=lower_eqn,
            upper_bound=upper_eqn,
            stack_above=[],
            parent=parent_cell,
            sample_point=new_sample_point
        )
        parent_cell.stack_above.append(sector_cell)

        # Generate a sample point for the next dimension
        if dim < max_dim:
            lift_recursive(dim + 1, sector_cell, constraints_per_dim, sample_point_buf, max_dim)

        if i < num_boundaries:
            sample_point_buf[dim - 1] = sorted_vals[i]
            new_sample_point = sample_point_buf[:dim].copy()
            boundary_eqn = sorted_constraints[i]
            section_cell = Cell(
                lower_bound=boundary_eqn,
                upper_bound=boundary_eqn,
                stack_above=[],
                parent=parent_cell,
                sample_point=new_sample_point
            )
            parent_cell.stack_above.append(section_cell)

            if dim < max_dim:
                lift_recursive(dim + 1, section_cell, constraints_per_dim, sample_point_buf, max_dim)


def create_root_cell():
    return Cell(lower_bound=None, upper_bound=None, stack_above=[], parent=None, sample_point=None)


def construct_cad(constraints: ConstraintList, highest_dim=None):
    """
    highest_dim can be passed e.g. in the case of an empty constraint list.
    """
    constraints = np.unique(constraints, axis=0)
    constraints_per_dimension = project_to_r0(constraints, highest_dim)
    max_dim = max(constraints_per_dimension.keys())

    root = create_root_cell()
    sample_point_buf = np.empty(shape=(max_dim,), dtype=np.float64)
    lift_recursive(1, root, constraints_per_dimension, sample_point_buf, max_dim)

    return root

def empty_1d_cad():
    root = Cell(lower_bound=None, upper_bound=None, stack_above=[], parent=None)
    child = Cell(lower_bound=[-np.inf, -1], upper_bound=[np.inf, 1], stack_above=[], parent=root)
    root.stack_above.append(child)

    return root

#
# Operations on the CAD cells.
#

def _format_bound(bound):
    if bound is None:
        return None

    bound = bound.copy()
    bound /= -bound[-1]
    ai = bound[1:-1]

    if ai.shape[0] > 0 and bound[1:-1].any():
        return bound

    return bound[0]

def pretty_print_cad(cell: Cell, indent=0):
    lower_bound = _format_bound(cell.lower_bound)
    upper_bound = _format_bound(cell.upper_bound)

    print(f"Bounds: {lower_bound} to {upper_bound}")
    if cell.component_function is not None:
        print(" " * indent, end="")
        print(f"** Component function: {cell.component_function}")

    for child_cell in cell.stack_above:
        print(" " * (indent + 2), end="")
        pretty_print_cad(child_cell, indent + 2)


def get_parent_chain(cell: Cell):
    current_cell = cell
    parents = [cell]

    while current_cell.parent:
        parents.append(current_cell.parent)
        current_cell = current_cell.parent

    return list(reversed(parents))


def bound_2d(pwl: PWL, min_val, max_val):
    """
    Adds a bounding box around a 2D decomposition and removes the cells that
    fall outside of these bounds.

    Not efficient but can improve later if actually used.
    """

    all_constraints = np.vstack([
        pwl.constraints,
        [min_val, -1, 0],
        [max_val, -1, 0],
        [min_val, 0, -1],
        [max_val, 0, -1],
    ])

    def in_bounds(cell: Cell) -> bool:
        if not cell.parent:
            # Root cell.
            return True

        vertices = np.array(get_cell_vertices(cell))
        epsilon = 1e-9

        return (
            not (vertices < (min_val - epsilon)).any()
            and not (vertices > (max_val + epsilon)).any()
        )

    bounded_pwl_cad = construct_cad(all_constraints)
    bounded_pwl_cad = filter_cad(bounded_pwl_cad, in_bounds)
    bounded_pwl = PWL(all_constraints, bounded_pwl_cad)
    absorb_component_functions(bounded_pwl, pwl)

    return bounded_pwl


def get_cell_vertices(cell: Cell):
    parents = get_parent_chain(cell)

    # R0 / root cell.
    if len(parents) == 1:
        return []

    parents = parents[1:]
    all_vertex_coeffs = [[]]

    # TODO: see if we can convert this numpy instead of a basic loop.
    # TODO: also, maybe we should cache this. At least for leaf cells, but also
    # in the algorithm itself it can be useful that each parent cell already has
    # its vertices calculated. Could maybe just be part of the lifting phase.
    for parent in parents:
        constraints = np.array([parent.lower_bound, parent.upper_bound])

        new_coeffs = []
        for vertex_coeffs in all_vertex_coeffs:
            new_xns = calculate_xn(constraints, np.array(vertex_coeffs))

            for xn in new_xns:
                new_coeffs.append(vertex_coeffs + [xn.item()])

        all_vertex_coeffs = new_coeffs

    return all_vertex_coeffs


def get_leaf_cells(cell: Cell):
    if not cell.stack_above:
        yield cell

    for child_cell in cell.stack_above:
        yield from get_leaf_cells(child_cell)


def filter_cad(cell: Cell, predicate):
    if not predicate(cell):
        return None

    if not cell.stack_above:
        return Cell(
            lower_bound=cell.lower_bound,
            upper_bound=cell.upper_bound,
            stack_above=[],
            parent=cell.parent,
            sample_point=cell.sample_point
        )

    filtered_children = [
        filter_cad(child, predicate)
        for child in cell.stack_above
    ]

    valid_children = [c for c in filtered_children if c is not None]
    if not valid_children:
        return None

    return Cell(
        lower_bound=cell.lower_bound,
        upper_bound=cell.upper_bound,
        stack_above=valid_children,
        parent=cell.parent,
        sample_point=cell.sample_point
    )

#
# PWL
#

def get_pwl_dimension(pwl: PWL):
    first_leaf_cell = next(get_leaf_cells(pwl.cad))

    # -1 because of the a0 dimension.
    return first_leaf_cell.component_function.size - 1

def get_pwl_cad_dimension(pwl: PWL):
    return get_pwl_dimension(pwl) - 1

def set_component_function_on_all_leaf_cells(cell: Cell, component_function: LinearConstraint):
    for leaf_cell in get_leaf_cells(cell):
        leaf_cell.component_function = component_function


def point_position_wrt_cell(cell: Cell, point: np.array) -> bool:
    """
    0 if point in cell; -1 if point below; 1 if point above.
    """

    x = point[-1]

    bounds = np.array([
        cell.lower_bound,
        cell.upper_bound
    ])

    results = calculate_xn(bounds, point[:-1])
    [lower, upper] = results

    if np.isclose(lower, upper) and np.isclose(lower, x):
        return 0
    if np.isinf(lower):
        return 0 if x < upper else 1
    if np.isinf(upper):
        return 0 if x > lower else -1

    if x < lower:
        return -1

    if x > upper:
        return 1

    return 0


def get_cell_containing_point(cell: Cell, point: np.array, dimension=1):
    """
    Given a point, returns the cell in which it is contained.
    """

    if dimension == 1 and cell.parent:
        raise Exception("Non-root cell given, aborting")

    if not cell.stack_above:
        return cell

    children = cell.stack_above
    left = 0
    right = len(children) - 1

    # Binary search in the stack above.
    while left <= right:
        mid = (left + right) // 2
        child = children[mid]

        pos = point_position_wrt_cell(child, point[:dimension])
        if pos == 0:
            return get_cell_containing_point(child, point, dimension + 1)
        elif pos < 0:
            right = mid - 1
        else:
            left = mid + 1

    raise Exception("No cell found for point")

def get_midpoint(lower_bound, upper_bound):
    if np.isinf(lower_bound) and np.isinf(upper_bound):
        return 0
    if np.isinf(lower_bound):
        return upper_bound - 1
    if np.isinf(upper_bound):
        return lower_bound + 1

    return (lower_bound + upper_bound) / 2

def generate_sample_point(cell: Cell):
    """
    Given a cell, generates a sample point within the cell.
    """
    return cell.sample_point


def get_sample_component_function_value(cell: Cell):
    if cell.component_function is None:
        raise Exception("Cell has no component function")

    sample_point = generate_sample_point(cell)

    return calculate_xn(cell.component_function, sample_point)

def create_pwl_from_constraints(constraints: ConstraintList, component_function: LinearConstraint, dimensions=None):
    cad = construct_cad(constraints, dimensions)
    set_component_function_on_all_leaf_cells(cad, component_function)

    return PWL(constraints, cad)

def absorb_component_functions(finer_pwl: PWL, coarser_pwl: PWL):
    """
    Combines 2 PWLs by summing the component function. assumes the first is
    "finer" than the second, i.e., the first PWL constains at least all
    constraints of the second PWL.
    """

    # To prevent extra calculations, we loop over the cells of the finer PWL's
    # CAD. For each child cell, we know that the sample point is on the same
    # vertical. Thus, we can calculate the bounds for that sample point of the
    # entire stack above of the coarser PWL. We keep a pointer to the cell in
    # the coarser PWL we are currently in, and only advance as needed (since
    # there are 1+ finer cells for each coarser cell).

    class StackAbovePointer:
        """
        Holds a pointer to which cell in the stack above we are currently in (of
        the coarser PWL).
        """

        def __init__(self, cell: Cell, sample_point: np.array):
            # If the coarser cell has no stack above, it is a leaf in this
            # dimension. In that case the same coarser cell applies for all
            # finer descendants, so we can skip bound evaluation entirely.
            if not cell.stack_above:
                self.stack_above = []
                self.all_upper_bounds = None
                self.current_cell_index = 0
                self._leaf_cell = cell
                return

            self._leaf_cell = None

            # Cache the stacked upper-bound constraints matrix per cell.
            if cell._stack_above_upper_bounds is None:
                cell._stack_above_upper_bounds = np.stack(
                    [child_cell.upper_bound for child_cell in cell.stack_above], axis=0
                )

            self.stack_above = cell.stack_above

            # Cache the evaluated upper-bound values for the (small)
            # sample_point.
            key = tuple(float(x) for x in sample_point)
            if key != cell._stack_above_upper_bounds_key:
                cell._stack_above_upper_bounds_key = key
                cell._stack_above_upper_bounds_values = calculate_xn(
                    cell._stack_above_upper_bounds, sample_point
                )

            self.all_upper_bounds = cell._stack_above_upper_bounds_values
            self.current_cell_index = 0

        def get_cell_for_point(self, point):
            if self._leaf_cell is not None:
                return self._leaf_cell

            current_upper_bound = self.all_upper_bounds[self.current_cell_index]
            if point < current_upper_bound:
                return self.stack_above[self.current_cell_index]

            if point > current_upper_bound or self.current_cell_index == 0:
                self.current_cell_index += 1
                return self.get_cell_for_point(point)

            prev_bound = self.all_upper_bounds[self.current_cell_index - 1]

            if current_upper_bound == prev_bound:
                return self.stack_above[self.current_cell_index]
            else:
                self.current_cell_index += 1
                return self.get_cell_for_point(point)

    def recurse_child_cells(finer_cell: Cell, coarser_stack: StackAbovePointer):
        for child in finer_cell.stack_above:
            sample_point = child.sample_point
            coarser_cell = coarser_stack.get_cell_for_point(sample_point[-1])
            child_stack = StackAbovePointer(coarser_cell, sample_point)

            # Leaf cell: add the component functions.
            if not child.stack_above:
                coarse_component_function = coarser_cell.component_function
                if child.component_function is None:
                    child.component_function = coarse_component_function.copy()
                else:
                    child.component_function = coarse_component_function + child.component_function
                    child.component_function[-1] = -1

            recurse_child_cells(child, child_stack)

    recurse_child_cells(finer_pwl.cad, StackAbovePointer(coarser_pwl.cad, np.array([])))


def sum_pwls(pwls: List[PWL]):
    # In the case of empty decompositions, we can't check the constraints for
    # the highest dimension, so we will need to pass it manually.
    dimension = get_pwl_cad_dimension(pwls[0])
    all_constraints = np.vstack([pwl.constraints for pwl in pwls])
    all_constraints = np.unique(all_constraints, axis=0)
    combined_cad = construct_cad(all_constraints, dimension)
    combined_pwl = PWL(all_constraints, combined_cad)

    for pwl in pwls:
        absorb_component_functions(combined_pwl, pwl)

    return combined_pwl

def copy_cad(cad: Cell, component_function_map_fn):
    def go(cell: Cell, parent_cell=None):
        new_cell = Cell(
            lower_bound=cell.lower_bound,
            upper_bound=cell.upper_bound,
            stack_above=[],
            parent=parent_cell,
            component_function=None
        )
        if cell.component_function is not None:
            new_cell.component_function = component_function_map_fn(
                cell.component_function.copy()
            )

        child_cells = [go(c, new_cell) for c in cell.stack_above]
        new_cell.stack_above = child_cells

        return new_cell

    return go(cad)


def scale_in_place(pwl: PWL, weight: float):
    for cell in get_leaf_cells(pwl.cad):
        function = cell.component_function.copy()
        function[:-1] *= weight
        cell.component_function = function


def scale(pwl: PWL, weight: float):
    def scale_component_function(component_function):
        component_function[:-1] *= weight

        return component_function

    new_cad = copy_cad(pwl.cad, scale_component_function)
    scaled_pwl = PWL(pwl.constraints, new_cad)

    return scaled_pwl


def add_bias(pwl: PWL, bias: float):
    new_pwl = copy.deepcopy(pwl)
    add_bias_in_place(new_pwl, bias)

    return new_pwl


def add_bias_in_place(pwl: PWL, bias: float):
    for cell in get_leaf_cells(pwl.cad):
        function = cell.component_function.copy()
        function[0] += bias
        cell.component_function = function


def get_component_function_zero_sets(cad: Cell):
    zero_sets = []

    for cell in get_leaf_cells(cad):
        component_function = cell.component_function

        zero_set = component_function[:-1]
        zero_sets.append(zero_set)

    return np.unique(zero_sets, axis=0)


def apply_relu_to_component_function(cad: Cell):
    """
    TODO: should rename this to something else to avoid confusion with
    `apply_relu`, which does everything at once for PWL instances.
    """
    new_cad = copy.deepcopy(cad)
    apply_relu_to_component_function_in_place(new_cad)

    return new_cad


def apply_relu_to_component_function_in_place(cad: Cell):
    for cell in get_leaf_cells(cad):
        value = get_sample_component_function_value(cell)
        if value <= 0:
            zero = np.zeros_like(cell.component_function)
            zero[-1] = -1
            cell.component_function = zero


def merge_component_function_zero_sets(pwl: PWL):
    all_constraints = get_component_function_zero_sets(pwl.cad)
    if pwl.constraints.any():
        all_constraints = np.append(all_constraints, pwl.constraints, axis=0)

    all_constraints = np.unique(all_constraints, axis=0)
    new_cad = construct_cad(all_constraints)
    new_pwl = PWL(all_constraints, new_cad)

    absorb_component_functions(new_pwl, pwl)

    return new_pwl


def apply_relu(pwl: PWL):
    pwl = merge_component_function_zero_sets(pwl)
    apply_relu_to_component_function_in_place(pwl.cad)

    return pwl


def merge_vertically_adjacent_leaf_cells(cad: Cell):
    previous_leaf_cell = None

    for leaf_cell in get_leaf_cells(cad):
        if previous_leaf_cell is None:
            previous_leaf_cell = leaf_cell
            continue

        if leaf_cell.parent is not previous_leaf_cell.parent:
            yield previous_leaf_cell
            previous_leaf_cell = leaf_cell
            continue

        if np.array_equal(leaf_cell.component_function, previous_leaf_cell.component_function):
            previous_leaf_cell.upper_bound = leaf_cell.upper_bound
            continue
        else:
            yield previous_leaf_cell
            previous_leaf_cell = leaf_cell
            continue

    yield previous_leaf_cell


# TODO: when adding the relu constraint, we could only add it for component
# functions that actually reach 0 in their polytope? To make the constraint
# space smaller.

if __name__ == "__main__":
    # cd_tussentabellen_joins met dims=3
    constraints = np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, -1],
        [-3, 1, 0, 0, 0],
        [-3, 0, 1, 0, 0],
        [-3, 0, 0, 1, 0],
        [-3, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ])

    constraints1 = np.array([
        [-3, 0, 1],
        [-2, 0, 1],
    ])
    constraints2 = np.array([
        [-2, 0, 1],
        [-1, 0, 1],
    ])

    pwl1 = create_pwl_from_constraints(constraints1, np.array([1, 2, 3, -1]))
    pwl2 = create_pwl_from_constraints(constraints2, np.array([5, 5, 5, 1]))

    combined = sum_pwls([pwl1, pwl2])
    pretty_print_cad(combined.cad)
