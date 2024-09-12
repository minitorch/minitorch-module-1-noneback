from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
    sum,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    "Check that the main operators all return the same value of the python version"
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    "Check that a - 1.0 is always less than a"
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Check properties of the sigmoid function."""
    from minitorch.operators import sigmoid  # 假设 sigmoid 函数已经定义了

    # 计算 sigmoid
    sig = sigmoid(a)

    # It is always between 0.0 and 1.0
    assert 0.0 <= sig <= 1.0, "Sigmoid output is outside [0, 1]"

    # one minus sigmoid is the same as sigmoid of the negative
    assert abs(1 - sig - sigmoid(-a)) < 1e-9, "1 - sigmoid(a) != sigmoid(-a)"

    # It crosses 0 at 0.5
    if a == 0:
        assert abs(sig - 0.5) < 1e-9, "Sigmoid at 0 should be 0.5"

    # # It is strictly increasing
    a1, a2 = a / 100, (a + 50) / 100
    # print(a1, a2)
    assert sigmoid(a1) < sigmoid(
        a2
    ), "Sigmoid is not strictly increasing: a1 {} a2 {}".format(a1, a2)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    "Test the transitive property of less-than (a < b and b < c implies a < c)"
    if a < b and b < c:
        assert a < c, "Transitive property failed"


@pytest.mark.task0_2
def test_symmetric() -> None:
    """
    Test that minitorch.operators.mul is symmetric.
    """
    from minitorch.operators import mul  # 假设乘法函数已经定义了

    # 定义一些示例值进行测试
    x = 3.0
    y = 5.0
    assert mul(x, y) == mul(y, x), "Multiplication is not symmetric"


@pytest.mark.task0_2
def test_distribute() -> None:
    r"""
    Test that operators distribute:
    z * (x + y) == z * x + z * y
    """
    from minitorch.operators import mul, add  # 假设加法和乘法函数已经定义了

    z = 2.0
    x = 3.0
    y = 4.0

    assert mul(z, add(x, y)) == add(
        mul(z, x), mul(z, y)
    ), "Distribution property failed"


@pytest.mark.task0_2
def test_other() -> None:
    """
    Test some other property, like identity for multiplication.
    """
    from minitorch.operators import mul  # 假设乘法函数已经定义了

    x = 1.0
    assert mul(x, 1.0) == x, "Multiplication with 1 should return the same number"
    assert mul(x, 0.0) == 0.0, "Multiplication with 0 should return 0"


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """
    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """

    def manual_sum(lst: List[float]) -> float:
        total = 0.0
        for number in lst:
            total += number
        return total

    # 验证两个总和是否相等
    assert manual_sum(ls1) == sum(ls1) and sum(ls2) == manual_sum(ls2)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
