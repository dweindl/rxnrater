from sympy.abc import _clash
import sympy as sp


def sympify(expr):
    """Sympify a string expression.

    Raise ValueError if it cannot be parsed.
    """

    try:
        return sp.sympify(expr, locals=_clash)
    except sp.SympifyError as e:
        raise ValueError(f"Cannot parse {expr} as a sympy expression") from e


def gcd(*args) -> sp.Expr:
    # sympy only supports binary gcd
    assert len(args) > 0
    result = args[0]
    for arg in args[1:]:
        result = sp.gcd(result, arg)
    return result
