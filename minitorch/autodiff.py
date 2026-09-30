from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

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
    new_vals = [0] * len(vals)
    for idx, val in enumerate(vals):
        if idx == arg:
            new_vals[idx] = val + epsilon
        else:
            new_vals[idx] = val

    print(vals)
    print(new_vals)
    derivative_value = (f(*new_vals) - f(*vals)) / epsilon
    return derivative_value


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

    def chain_rule(self, d_output: Any) -> Iterable[tuple["Variable", Any]]:
        pass


def generate_indegree(graph):
    indegree = {node: 0 for node in graph}
    for n in graph:
        for neighbor in n:
            indegree[neighbor] += 1
    return indegree


def regular_topoligcal_sort(graph):
    q = deque()
    res = []
    indegree = generate_indegree(graph)

    for n in graph:
        if indegree[n] == 0:
            q.append(n)

    while len(q) > 0:
        node = q.popleft()
        res.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)

    return res if len(res) == len(graph) else 0


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # Graph discovery
    res = []
    graph = [variable]
    q = deque()
    children = defaultdict(list)

    for var in graph:
        for parent in var.parents:
            children[parent].append(var)
            if parent.is_leaf():
                q.append(parent)
            graph.append(parent)

    # Filling indegrees
    indegree = {node: 0 for node in graph}
    for node in graph:
        indegree[node] = len(node.parents)

    # Traversal
    while len(q) > 0:
        node = q.popleft()
        res.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                q.append(child)

    return res


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> tuple[Any, ...]:
        return self.saved_values
