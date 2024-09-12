from dataclasses import dataclass
from typing import Any, Deque, Iterable, List, Tuple
from minitorch.operators import sum
from collections import defaultdict, deque

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    values = list(vals)
    delta1 = vals[arg] + epsilon / 2
    delta2 = vals[arg] - epsilon / 2

    values[arg] = delta1
    y1 = f(*values)
    values[arg] = delta2
    y2 = f(*values)
    print(y1, y2)

    return (y1 - y2) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # check: 1.4
    sorted = []
    queue: Deque[Variable] = deque()
    nodes = {}
    queue.append(variable)
    nodes[variable.unique_id] = variable
    m = defaultdict(int)

    m[variable.unique_id] = 0

    while queue:
        node = queue.popleft()
        nodes[node.unique_id] = node
        for dep in node.parents:
            queue.append(dep)

    for node in nodes.values():
        for dep in node.parents:
            m[dep.unique_id] += 1

    queue = deque([nodes[k] for k, v in m.items() if v == 0])
    # print("nodes", nodes, m.items(), queue)

    while queue:
        node = queue.popleft()
        sorted.append(node)

        for dep in node.parents:
            # print("dep for ", node.unique_id, dep.unique_id, "cnt", m[dep.unique_id])
            m[dep.unique_id] -= 1
            if m[dep.unique_id] == 0:
                queue.append(dep)
            # print(
            #     "dep for ", node.unique_id, dep.unique_id, "after cnt", m[dep.unique_id]
            # )

    if len(sorted) != len(m):
        # print("deps ring", sorted, m.items(), len(sorted), len(m))
        raise Exception("deps ring", sorted, m.items(), len(sorted), len(m))

    return sorted


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for 1.4
    sorted: Deque[Variable] = deque(topological_sort(variable))
    midvals = defaultdict(lambda: 0)
    midvals[variable.unique_id] = deriv

    while sorted:
        node = sorted.popleft()
        if node.is_leaf():
            continue
        raw = node.chain_rule(midvals[node.unique_id])  # d_y/d_input

        for var_der in raw:
            val, der = var_der[0], var_der[1]
            # print("cur_deri", der)
            if val.is_leaf():
                val.accumulate_derivative(der)
            else:
                midvals[val.unique_id] += der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
