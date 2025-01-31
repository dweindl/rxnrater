"""Test enzyme reactions.

Test cases are mostly based on:

* W. W. Cleland, “The kinetics of enzyme-catalyzed reactions with two or more
  substrates or products: I. Nomenclature and rate equations,”
  Biochimica et Biophysica Acta (BBA), vol. 67, pp. 104–137, 1963,
  doi: https://doi.org/10.1016/0926-6569(63)90211-6.

* F. Wu, F. Yang, K. C. Vinnakota, and D. A. Beard,
  “Computer Modeling of Mitochondrial Tricarboxylic Acid Cycle,
  Oxidative Phosphorylation, Metabolite Transport, and Electrophysiology”
  Journal of Biological Chemistry, vol. 282, no. 34, pp. 24525–24537,
   Aug. 2007, doi: 10.1074/jbc.M701024200.
"""

import pytest
from warnings import warn
from rxnrater.enzyme_rxn import EnzymeReaction, CoefficientFormRateEquation
from rxnrater._utils import sympify
import sympy as sp


def check_kinetic_flux_expr(
    er: EnzymeReaction, flux_exp: sp.Expr, pars_exp: dict[str, sp.Expr]
):
    flux_act, pars = er.simplify_flux()
    _check_kinetic_flux_expr(
        flux_act, pars, flux_exp, pars_exp, er.micro_rate_constants
    )


def _check_kinetic_flux_expr(
    flux_act: sp.Expr,
    pars_act: dict[str, sp.Expr],
    flux_exp: sp.Expr,
    pars_exp: dict[str, sp.Expr],
    micro_rate_constants: set[sp.Symbol],
):
    pars = pars_act
    for key, val in pars_exp.items():
        assert pars[key].equals(val), f"{key}: expected {val}, got {pars[key]}"

    for key, val in pars.items():
        if key not in pars_exp:
            warn(f"Unexpected kinetic parameter: {key} = {val}")

    assert flux_exp.free_symbols.isdisjoint(micro_rate_constants)
    assert flux_act.free_symbols.isdisjoint(micro_rate_constants)

    # for now, it's good enough if we have a valid simplification,
    #  it doesn't have to be exactly the same as the expected expression
    pars_sym = {sp.Symbol(k): v for k, v in pars.items()}
    # replace twice, to ensure that all symbols are replaced
    #  (alternatively, toposort the expressions)
    assert (
        (flux_exp - flux_act).subs(pars_sym).subs(pars_sym).simplify().is_zero
    ), f"{flux_exp} != {flux_act}"

    if not flux_act.equals(flux_exp):
        warn(f"Flux expressions differ: {flux_act} != {flux_exp}")


def check_exhaustive(er: EnzymeReaction, solutions_exp):
    solutions_act = er.simplify_flux(exhaustive=True)
    assert len(solutions_act) == len(solutions_exp)

    for flux_exp, pars_exp in solutions_exp:
        for flux_act, pars_act in solutions_act:
            try:
                _check_kinetic_flux_expr(
                    flux_act, pars_act, flux_exp, pars_exp, er.micro_rate_constants
                )
                break
            except AssertionError:
                pass
        else:
            raise AssertionError(f"Missing expected solution: {flux_exp}, {pars_exp}")


def test_cleland1963_ordered_uni_bi():
    """Test ordered uni-bi (Cleland1963, mech. 14)."""
    rxn_str = """
    E + A -- E:A , k1 , k2
    E:A -- E:Q + P , k3 , k4
    E:Q -- E + Q , k5 , k6
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(A -- P + Q)>"
    res = er.get_kinetic_parameters()
    assert res["flux"].equals(
        sympify(
            "E0 * (k1 * k3 * k5 * A - k2 * k4 * k6 * P * Q) "
            "/ ((k2 + k3) * k5 + k1 * (k3 + k5) * A + k2 * k4 * P "
            "+ (k2 + k3) * k6 * Q + k1 * k4 * A * P + k4 * k6 * P * Q)"
        )
    )

    assert res["keq_micro"] == sympify("k1*k3*k5/(k2*k4*k6)")

    pars_exp = {
        "Km_A": sympify("(k2 * k5 + k3 * k5) / (k1 * k3 + k1 * k5)"),
        "Km_P": sympify("(k2 + k3)/k4"),
        "Km_Q": sympify("(k2 / k6)"),
        "Ki_A": sympify("k2/k1"),
        "Ki_P": sympify("(k3 + k5) / k4"),
        "Ki_Q": sympify("k5/k6"),
        "V_mr": sympify("Ki_Q*Km_P*V_mf/(Keq*Km_A)"),
    }
    E0 = sp.Symbol("E0")
    assert sp.Symbol("k1") == res["V_mr"] / (pars_exp["Ki_A"] * E0)
    assert sp.Symbol("k2") == res["V_mr"] / (E0)
    assert (sp.Integer(1) / sp.Symbol("k3")).equals(
        E0 / res["V_mf"] - sp.Integer(1) / sp.Symbol("k5")
    )
    assert sp.Symbol("k4") == sp.sympify("k2 + k3") / res["Km_P"]
    assert sp.Symbol("k5") == pars_exp["Ki_Q"] * res["V_mr"] / (res["Km_Q"] * E0)
    assert sp.Symbol("k6") == res["V_mr"] / (res["Km_Q"] * E0)

    flux_exp = sympify(
        "V_mf*V_mr*(A - P*Q/Keq)/("
        "Km_A*V_mr + A*V_mr + Km_Q*P*V_mf/Keq + Km_P*Q*V_mf/Keq"
        " + P*Q*V_mf/Keq  + V_mr / Ki_P * A * P )"
    )
    check_kinetic_flux_expr(er, flux_exp, pars_exp)


def test_cleland1963_ordered_bi_bi():
    """Test ordered bi-bi (Cleland1963, mech. 6)."""
    rxn_str = """
    E + A -- E:A , k1 , k2
    E:A + B -- E:A:B , k3 , k4
    E:A:B -- E:Q + P , k5 , k6
    E:Q -- E + Q , k7 , k8
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(A + B -- P + Q)>"
    res = er.get_kinetic_parameters()

    k1, k2, k3, k4, k5, k6, k7, k8 = sp.symbols("k1 k2 k3 k4 k5 k6 k7 k8")
    A, B, P, Q = sp.symbols("A B P Q")
    E0 = sp.Symbol("E0")

    flux_exp = (
        E0
        * (k1 * k3 * k5 * k7 * A * B - k2 * k4 * k6 * k8 * P * Q)
        / (
            (k4 + k5) * k2 * k7
            + k1 * (k4 + k5) * k7 * A
            + k2 * k8 * (k4 + k5) * Q
            + k3 * k5 * k7 * B
            + k2 * k4 * k6 * P
            + k1 * k3 * (k5 + k7) * A * B
            + (k2 + k4) * k6 * k8 * P * Q
            + k1 * k4 * k6 * A * P
            + k1 * k3 * k6 * A * B * P
            + k3 * k5 * k8 * B * Q
            + k3 * k6 * k8 * B * P * Q
        )
    )

    assert res["flux"].equals(flux_exp)

    E0 = sp.Symbol("E0")
    res["Ki_A"] = sympify("k2/k1")
    assert res["keq_micro"] == sympify("k1*k3*k5*k7/(k2*k4*k6*k8)")
    assert sp.Symbol("k1") == res["V_mf"] / (res["Km_A"] * E0)
    assert sp.Symbol("k2") == res["V_mf"] * res["Ki_A"] / (res["Km_A"] * E0)
    assert sp.Symbol("k3").equals(
        res["V_mf"] / (res["Km_B"] * E0) * sympify("1 + k4 / k5")
    )
    assert (1 / sp.Symbol("k4")).equals(E0 / res["V_mr"] - sympify("1 / k2"))
    assert (1 / sp.Symbol("k5")).equals(E0 / res["V_mf"] - sympify("1 / k7"))
    assert sp.Symbol("k6").equals(
        res["V_mr"] / (res["Km_P"] * E0) * sympify("1 + k5 / k4")
    )

    pars_exp = {
        "Km_A": sympify("k5*k7/(k1*k5 + k1*k7)"),
        "Km_B": sympify("(k4*k7 + k5*k7)/(k3*k5 + k3*k7)"),
        "Km_P": sympify("(k2*k4 + k2*k5)/(k2*k6 + k4*k6)"),
        "Km_Q": sympify("k2*k4/(k2*k8 + k4*k8)"),
        "Ki_A": sympify("k2/k1"),
        "Ki_B": sympify("(k2 + k4)/k3"),
        "Ki_P": sympify("(k5 + k7)/k6"),
        "Ki_Q": sympify("k7/k8"),
    }
    solutions_exp = [
        (
            sympify(
                "V_mf*V_mr*(A * B - P*Q/Keq)/("
                "Ki_A * Km_B * V_mr + Km_B * V_mr * A + Km_A * V_mr * B"
                "+ V_mr * A * B + Km_Q * V_mf * P / Keq + Km_P * V_mf * Q / Keq"
                "+ V_mf * P * Q / Keq + Km_Q * V_mf * A * P / (Keq * Ki_A)"
                # last two terms differ, depending on the choice of Ki_B, Ki_P
                "+ Km_A * V_mr * B * Q / Ki_Q + V_mr * A * B * P / Ki_P "
                "+ V_mf * B * P * Q / (Ki_B * Keq))"
            ),
            pars_exp,
        ),
        (
            sympify(
                "V_mf*V_mr*(A*B - P*Q/Keq)/("
                "Ki_A*Km_B*V_mr + A*Km_B*V_mr + B*Km_A*V_mr"
                "+ A*B*V_mr + Km_Q*P*V_mf/Keq +  Km_P*Q*V_mf/Keq   "
                "+ P*Q*V_mf/Keq + A*Km_Q*P*V_mf/(Keq*Ki_A)  "
                "+ B*Km_A*Q*V_mr/Ki_Q + A*B*Km_Q*P*V_mf/(Keq*Ki_A*Ki_B) "
                "+ B*Km_A*P*Q*V_mr/(Ki_P*Ki_Q) "
                ")"
            ),
            pars_exp
            | {
                "Ki_B": sympify("k4/k3"),
                "Ki_P": sympify("k5/k6"),
            },
        ),
    ]
    check_exhaustive(er, solutions_exp)


def test_cleland1963_theorell_chance():
    """Test Theorell-Chance (Cleland1963, mech. 8)."""
    rxn_str = """
    E + A -- E:A , k1 , k2
    E:A + B -- E:Q + P , k3 , k4
    E:Q -- E + Q , k5 , k6
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(A + B -- P + Q)>"

    k1, k2, k3, k4, k5, k6 = sp.symbols("k1 k2 k3 k4 k5 k6")
    A, B, P, Q = sp.symbols("A B P Q")

    cf = er.coeff_form()
    assert cf.coefficients == {
        (): k2 * k5,
        (A,): k1 * k5,
        (B,): k3 * k5,
        (A, B): k1 * k3,
        (P,): k2 * k4,
        (Q,): k2 * k6,
        (P, Q): k4 * k6,
        (A, P): k1 * k4,
        (B, Q): k3 * k6,
    }

    res = er.get_kinetic_parameters()
    assert res["flux"].equals(
        sympify(
            "E0 * (k1 * k3 * k5 * A * B - k2 * k4 * k6 * P * Q) "
            "/ ( k2 * k5 + k1 * k5 * A + k3 * k5 * B + k1 * k3 * A * B "
            "+ k2 * k4 * P + k2 * k6 * Q + k4 * k6 * P * Q "
            "+ k1 * k4 * A * P + k3 * k6 * B * Q)"
        )
    )

    flux_exp = sympify(
        "V_mf*V_mr*(A * B - P*Q/Keq)/("
        "Ki_A * Km_B * V_mr + Km_B * V_mr * A + Km_A * V_mr * B"
        "+ V_mr * A * B + Km_Q * V_mf * P / Keq + Km_P * V_mf * Q / Keq"
        "+ V_mf * P * Q / Keq "
        # FIXME:
        # exp: A*Km_B*P*V_mr/Ki_P
        # act: A*Km_Q*P*V_mf/(Keq*Ki_A)
        # both are valid, but we prefer the simpler version
        "+ Km_Q * V_mf * A * P / (Keq * Ki_A)"
        "+ Km_A * V_mr * B * Q / Ki_Q)"
    )

    pars_exp = {
        "Km_A": sympify("k5 / k1"),
        "Km_B": sympify("k5 / k3"),
        "Km_P": sympify("k2 / k4"),
        "Km_Q": sympify("k2 / k6"),
        "Ki_A": sympify("k2/k1"),
        "Ki_B": sympify("k2/k3"),
        "Ki_P": sympify("k5/k4"),
        "Ki_Q": sympify("k5/k6"),
    }

    check_kinetic_flux_expr(er, flux_exp, pars_exp)


def test_cleland1963_ping_pong_bi_bi():
    """Test ping-pong bi-bi

    (Cleland1963, mech. 10)

    Cleland, Enzyme Kinetics and Mechanism, p.383.
    """
    # FIXME we currently cannot handle enzyme state "F", so they are
    #  prefixed with "E_" instead
    rxn_str = """
    E + A -- E:A , k1 , k2
    E:A -- E_F + P , k3 , k4
    E_F + B -- E_F:B , k5 , k6
    E_F:B -- E + Q , k7 , k8
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(A + B -- P + Q)>"
    assert set(er.enzyme_states) == set(sp.symbols("E, E_A, E_F, E_F_B"))
    res = er.get_kinetic_parameters()

    k1, k2, k3, k4, k5, k6, k7, k8 = sp.symbols("k1 k2 k3 k4 k5 k6 k7 k8")
    A, B, P, Q = sp.symbols("A B P Q")
    E0 = sp.Symbol("E0")

    flux_exp = (
        E0
        * (A * B * k1 * k3 * k5 * k7 - P * Q * k2 * k4 * k6 * k8)
        / (
            A * k1 * k3 * (k6 + k7)
            + B * k5 * k7 * (k2 + k3)
            + k1 * k5 * (k3 + k7) * A * B
            + k6 * k8 * (k2 + k3) * Q
            + k2 * k4 * P * (k6 + k7)
            + k4 * k8 * (k2 + k6) * P * Q
            + k1 * k4 * A * P * (k6 + k7)
            + k5 * k8 * B * Q * (k2 + k3)
        )
    )
    assert res["flux"].equals(flux_exp)
    res_exp = {
        "flux": flux_exp,
        "Km_A": k7 * (k2 + k3) / (k1 * (k3 + k7)),
        "Km_B": k3 * (k6 + k7) / (k5 * (k3 + k7)),
        "Km_P": k6 * (k2 + k3) / (k4 * (k2 + k6)),
        "Km_Q": k2 * (k6 + k7) / (k8 * (k2 + k6)),
        "V_mf": E0 * k3 * k7 / (k3 + k7),
        "V_mr": E0 * k2 * k6 / (k2 + k6),
        "keq_micro": k1 * k3 * k5 * k7 / (k2 * k4 * k6 * k8),
    }
    for key, val in res_exp.items():
        assert res[key].equals(val), f"{key}: expected {val}, got {res[key]}"

    pars_exp = {
        "Km_A": k7 * (k2 + k3) / (k1 * (k3 + k7)),
        "Km_B": k3 * (k6 + k7) / (k5 * (k3 + k7)),
        "Km_P": k6 * (k2 + k3) / (k4 * (k2 + k6)),
        "Km_Q": k2 * (k6 + k7) / (k8 * (k2 + k6)),
        "Ki_A": k2 / k1,
        "Ki_B": k6 / k5,
        "Ki_P": k3 / k4,
        "Ki_Q": k7 / k8,
    }

    flux_exp = sympify(
        "V_mf*V_mr*(A*B - P*Q/Keq)/("
        "A*Km_B*V_mr + B*Km_A*V_mr + A*B*V_mr "
        "+ Km_Q*P*V_mf/Keq + Km_P*Q*V_mf/Keq + P*Q*V_mf/Keq"
        "+ A*Km_B*P*V_mr/Ki_P  + B*Km_A*Q*V_mr/Ki_Q )"
        # last term is chosen differently in the paper:
        #    Km_P * V_mf * B * Q / (Ki_B * Keq)
        # but both are valid solutions
    )
    check_kinetic_flux_expr(er, flux_exp, pars_exp)


@pytest.mark.xfail(reason="Not implemented yet")
def test_cleland1963_random_bi_bi():
    rxn_str = """
    E + A -- E:A , k1 , k2
    E:A + B -- E:A:B , k3 , k4
    E + B -- E:B , k5 , k6
    E:B + A -- E:A:B , k7 , k8
    E:A:B -- E:P:Q , k9 , k10
    E:P:Q -- E:Q + P , k11 , k12
    E:Q -- E + Q , k13 , k14
    E:P:Q -- E:P + Q , k15 , k16
    E:P -- E + P , k17 , k18
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(A + B -- P + Q)>"
    # res = er.get_kinetic_parameters()
    # TODO: continue here


def test_WuYan2007_fumarase():
    """Test reversible uni-uni."""
    rxn_str = """
    E + FUM -- E:FUM , k1f , k1r
    E:FUM -- E + MAL , k2f , k2r
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(FUM -- MAL)>"

    res = er.get_kinetic_parameters()
    assert res["V_mf"] == sympify("E0 * k2f")
    assert res["V_mr"] == sympify("E0 * k1r")
    assert res["flux"] == sympify(
        "E0 * (FUM * k1f * k2f - MAL * k1r * k2r) / (FUM * k1f + MAL * k2r + k1r + k2f)"
    )
    assert res["Km_FUM"] == sympify("(k1r + k2f) / k1f")
    assert res["Km_MAL"] == sympify("(k1r + k2f) / k2r")
    assert res["keq_micro"] == sympify("k1f*k2f/(k1r*k2r)")

    flux_exp = sympify(
        "V_mf*V_mr*(FUM - MAL/Keq)/(FUM*V_mr + Km_FUM*V_mr + MAL*V_mf/Keq)"
    )
    check_kinetic_flux_expr(
        er,
        flux_exp,
        {
            "Km_FUM": sympify("(k1r + k2f) / k1f"),
            "Km_MAL": sympify("(k1r + k2f) / k2r"),
        },
    )


def test_WuYan2007_mdh():
    """Test reversible ordered bi-bi."""
    rxn_str = """
    E + NAD -- E:NAD , k1f , k1r
    E:NAD + MAL -- E:NAD:MAL , k2f , k2r
    E:NAD:MAL -- E:NADH + OAA , k3f , k3r
    E:NADH -- E + NADH , k4f , k4r
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(MAL + NAD -- NADH + OAA)>"

    res = er.get_kinetic_parameters()
    assert res["V_mf"] == sympify("E0*k3f*k4f/(k3f + k4f)")
    assert res["V_mr"] == sympify("E0*k1r*k2r/(k1r + k2r)")
    assert res["Km_NAD"] == sympify("k3f*k4f/(k1f*(k3f + k4f))")
    assert res["Km_MAL"] == sympify("k4f*(k2r + k3f)/(k2f*(k3f + k4f))")
    assert res["Km_OAA"] == sympify("k1r*(k2r + k3f)/(k3r*(k1r + k2r))")
    assert res["Km_NADH"] == sympify("k1r*k2r/(k4r*(k1r + k2r))")
    assert res["keq_micro"] == sympify("k1f*k2f*k3f*k4f/(k1r*k2r*k3r*k4r)")

    pars_exp = {
        "Km_NAD": sympify("k3f*k4f/(k1f*(k3f + k4f))"),
        "Km_MAL": sympify("k4f*(k2r + k3f)/(k2f*(k3f + k4f))"),
        "Km_OAA": sympify("k1r*(k2r + k3f)/(k3r*(k1r + k2r))"),
        "Km_NADH": sympify("k1r*k2r/(k4r*(k1r + k2r))"),
        "Ki_NAD": sympify("k1r/k1f"),
        # different, but equivalent solution (k2r/k2f) possible
        "Ki_MAL": sympify("(k1r+k2r)/k2f"),
        # different, but equivalent solution (k3f/k3r) possible
        "Ki_OAA": sympify("(k3f + k4f)/k3r"),
        "Ki_NADH": sympify("k4f / k4r"),
    }

    solutions_exp = [
        (
            sympify(
                "V_mf*V_mr*(MAL*NAD - NADH*OAA/Keq)/(Ki_NAD*Km_MAL*V_mr "
                "+ Km_MAL*NAD*V_mr + Km_NAD*MAL*V_mr + MAL*NAD*V_mr "
                "+ MAL*NAD*OAA*V_mr/Ki_OAA + Km_NAD*MAL*NADH*V_mr/Ki_NADH "
                # ^^^^^^^^^^^^^^^^^^^^^
                "+ Km_NADH*OAA*V_mf/Keq + Km_OAA*NADH*V_mf/Keq + NADH*OAA*V_mf/Keq "
                "+ Km_NADH*NAD*OAA*V_mf/(Keq*Ki_NAD) + MAL*NADH*OAA*V_mf/(Keq*Ki_MAL))"
                #                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            ),
            pars_exp,
        ),
        (
            sympify(
                "V_mf*V_mr*(MAL*NAD - NADH*OAA/Keq)/(Ki_NAD*Km_MAL*V_mr "
                "+ Km_MAL*NAD*V_mr + Km_NAD*MAL*V_mr + MAL*NAD*V_mr "
                "+ Km_NADH*MAL*NAD*OAA*V_mf/(Keq*Ki_MAL*Ki_NAD) + Km_NAD*MAL*NADH*V_mr/Ki_NADH "
                # ^^^^^^^^^^^^^^^^^^^^^
                "+ Km_NADH*OAA*V_mf/Keq + Km_OAA*NADH*V_mf/Keq + NADH*OAA*V_mf/Keq "
                "+ Km_NADH*NAD*OAA*V_mf/(Keq*Ki_NAD) + Km_NAD*MAL*NADH*OAA*V_mr/(Ki_NADH*Ki_OAA))"
                #                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            ),
            pars_exp
            | {
                "Ki_MAL": sympify("k2r/k2f"),
                "Ki_OAA": sympify("k3f/k3r"),
            },
        ),
    ]
    check_exhaustive(er, solutions_exp)


def test_WuYan2007_citrate_synthase():
    rxn_str = """
    E + OAA -- E:OAA , k1f , k1r
    E:OAA + ACCOA -- E:OAA:ACCOA , k2f , k2r
    E:OAA:ACCOA -> E:HCIT + COASH , k3
    E:HCIT -> E + CIT , k4
    """
    er = EnzymeReaction(rxn_str)
    res = er.get_kinetic_parameters()
    assert res["V_mf"] == sympify("E0*k3*k4/(k3 + k4)")
    assert res["V_mr"] == 0
    assert res["Km_OAA"] == sympify("k3*k4/(k1f*(k3 + k4))")
    assert res["Km_ACCOA"] == sympify("k4*(k2r + k3)/(k2f*(k3 + k4))")
    assert res["keq_micro"] == sp.oo

    pars_exp = {
        "Ki_OAA": sympify("k1r/k1f"),
        "Km_OAA": sympify("k3*k4/(k1f*(k3 + k4))"),
        "Km_ACCOA": sympify("k4*(k2r + k3)/(k2f*(k3 + k4))"),
    }
    flux_exp = sympify(
        "ACCOA*OAA*V_mf/(ACCOA*Km_OAA + ACCOA*OAA + Ki_OAA*Km_ACCOA + Km_ACCOA*OAA)"
    )
    check_kinetic_flux_expr(er, flux_exp, pars_exp)


def test_WuYan2007_pdh():
    rxn_str = """
    E + PYR -- E:PYR , k1f , k1r
    E:PYR -> E:CHOCH3 + CO2 , k2
    E:CHOCH3 + COASH -- E:ACCOA , k3f , k3r
    E:ACCOA -> E:m + ACCOA , k4
    E:m + NAD -- E:NAD , k5f , k5r
    E:NAD -> E + NADH , k6
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(COASH + NAD + PYR -- ACCOA + CO2 + NADH)>"
    res = er.get_kinetic_parameters()
    assert res["V_mf"] == sympify("E0*k2*k4*k6/(k2*k4 + k2*k6 + k4*k6)")
    assert res["V_mr"] == 0
    assert res["Km_PYR"] == sympify("k4*k6*(k1r + k2)/(k1f*(k2*k4 + k2*k6 + k4*k6))")
    assert res["Km_COASH"] == sympify("k2*k6*(k3r + k4)/(k3f*(k2*k4 + k2*k6 + k4*k6))")
    assert res["Km_NAD"] == sympify("k2*k4*(k5r + k6)/(k5f*(k2*k4 + k2*k6 + k4*k6))")
    assert res["keq_micro"] == sp.oo

    pars_exp = {
        "Km_PYR": sympify("k4*k6*(k1r + k2)/(k1f*(k2*k4 + k2*k6 + k4*k6))"),
        "Km_COASH": sympify("k2*k6*(k3r + k4)/(k3f*(k2*k4 + k2*k6 + k4*k6))"),
        "Km_NAD": sympify("k2*k4*(k5r + k6)/(k5f*(k2*k4 + k2*k6 + k4*k6))"),
    }

    flux_exp = sympify(
        "COASH*NAD*PYR*V_mf/(COASH*Km_NAD*PYR + COASH*Km_PYR*NAD + COASH*NAD*PYR + Km_COASH*NAD*PYR)"
    )

    check_kinetic_flux_expr(er, flux_exp, pars_exp)


WuYan2007_succinyl_coa_synthetase_flux_str = "E0*(-COASH*MgGTP*SUC*k1r*k2r*k3r*k4r*k5r*k6r + MgGDP*PI*SCOA*k1f*k2f*k3f*k4f*k5f*k6f)/(COASH*MgGDP*PI*SCOA*SUC*k1f*k2f*k3f*k4r*k5r + COASH*MgGDP*PI*SCOA*k1f*k2f*k3f*k4r*k6f + COASH*MgGDP*SCOA*SUC*k1f*k2f*k3r*k4r*k5r + COASH*MgGDP*SCOA*k1f*k2f*k3r*k4r*k6f + COASH*MgGDP*SUC*k1f*k2r*k3r*k4r*k5r + COASH*MgGDP*k1f*k2r*k3r*k4r*k6f + COASH*MgGTP*PI*SCOA*SUC*k2f*k3f*k4r*k5r*k6r + COASH*MgGTP*PI*SUC*k1r*k3f*k4r*k5r*k6r + COASH*MgGTP*SCOA*SUC*k2f*k3r*k4r*k5r*k6r + COASH*MgGTP*SUC*k1r*k2r*k4r*k5r*k6r + COASH*MgGTP*SUC*k1r*k3r*k4r*k5r*k6r + COASH*MgGTP*SUC*k2r*k3r*k4r*k5r*k6r + COASH*MgGTP*k1r*k2r*k3r*k4r*k6r + COASH*SUC*k1r*k2r*k3r*k4r*k5r + COASH*k1r*k2r*k3r*k4r*k6f + MgGDP*PI*SCOA*SUC*k1f*k2f*k3f*k4f*k5r + MgGDP*PI*SCOA*k1f*k2f*k3f*k4f*k5f + MgGDP*PI*SCOA*k1f*k2f*k3f*k4f*k6f + MgGDP*PI*SCOA*k1f*k2f*k3f*k5f*k6f + MgGDP*PI*k1f*k3f*k4f*k5f*k6f + MgGDP*SCOA*k1f*k2f*k3r*k5f*k6f + MgGDP*SCOA*k1f*k2f*k4f*k5f*k6f + MgGDP*k1f*k2r*k3r*k5f*k6f + MgGDP*k1f*k2r*k4f*k5f*k6f + MgGTP*PI*SCOA*SUC*k2f*k3f*k4f*k5r*k6r + MgGTP*PI*SCOA*k2f*k3f*k4f*k5f*k6r + MgGTP*PI*SUC*k1r*k3f*k4f*k5r*k6r + MgGTP*PI*k1r*k3f*k4f*k5f*k6r + MgGTP*SUC*k1r*k2r*k3r*k5r*k6r + MgGTP*SUC*k1r*k2r*k4f*k5r*k6r + MgGTP*k1r*k2r*k3r*k5f*k6r + MgGTP*k1r*k2r*k4f*k5f*k6r + PI*SCOA*k2f*k3f*k4f*k5f*k6f + PI*k1r*k3f*k4f*k5f*k6f + k1r*k2r*k3r*k5f*k6f + k1r*k2r*k4f*k5f*k6f)"


def test_WuYan2007_succinyl_coa_synthetase_fast():
    """Test ordered ter-ter.

    See also Cook & Cleland, Enzyme kinetics and mechanism, p387f
    """
    # pre-computed steadystate flux expression
    flux = sympify(WuYan2007_succinyl_coa_synthetase_flux_str)
    substrates = sp.symbols("MgGDP SCOA PI")
    products = sp.symbols("COASH SUC MgGTP")

    cf = CoefficientFormRateEquation(flux, substrates, products, True)

    pars_exp = {
        "Km_MgGDP": sympify("k4f*k5f*k6f/(k1f*(k4f*k5f + k4f*k6f + k5f*k6f))"),
        "Km_SCOA": sympify("k4f*k5f*k6f/(k2f*(k4f*k5f + k4f*k6f + k5f*k6f))"),
        "Km_PI": sympify("k5f*k6f*(k3r + k4f)/(k3f*(k4f*k5f + k4f*k6f + k5f*k6f))"),
        "Km_COASH": sympify("k1r*k2r*(k3r + k4f)/(k4r*(k1r*k2r + k1r*k3r + k2r*k3r))"),
        "Km_SUC": sympify("k1r*k2r*k3r/(k5r*(k1r*k2r + k1r*k3r + k2r*k3r))"),
        "Km_MgGTP": sympify("k1r*k2r*k3r/(k6r*(k1r*k2r + k1r*k3r + k2r*k3r))"),
        "Ki_MgGDP": sympify("k1r/k1f"),
        "Ki_SCOA": sympify("k2r/k2f"),
        "Ki_PI": sympify("k3r/k3f"),
        "Ki_COASH": sympify("k4f/k4r"),
        "Ki_SUC": sympify("k5f/k5r"),
        "Ki_MgGTP": sympify("k6f/k6r"),
    }
    # A: GDP, B: SCOA, C: PI
    # P: COASH, Q: SUC, R: GTP
    flux_exp = sympify(
        "V_mf*V_mr*(MgGDP*PI*SCOA - COASH*MgGTP*SUC/Keq)/("
        "Ki_MgGDP*Ki_SCOA*Km_PI*V_mr "
        "+ Ki_SCOA*Km_PI*MgGDP*V_mr "
        "+ Ki_MgGDP*Km_SCOA*PI*V_mr "
        "+ Km_PI*MgGDP*SCOA*V_mr "
        "+ Km_SCOA*MgGDP*PI*V_mr "
        "+ Km_MgGDP*PI*SCOA*V_mr "
        "+ MgGDP*PI*SCOA*V_mr "
        "+ COASH*Ki_MgGTP*Km_SUC*V_mf/Keq "
        "+ Ki_SUC*Km_COASH*MgGTP*V_mf/Keq "
        "+ COASH*Km_MgGTP*SUC*V_mf/Keq "
        "+ COASH*Km_SUC*MgGTP*V_mf/Keq "  # PR
        "+ Km_COASH*MgGTP*SUC*V_mf/Keq"  # QR
        "+ COASH*MgGTP*SUC*V_mf/Keq "  # PQR
        "+ COASH*Ki_MgGTP*Km_SUC*MgGDP*V_mf/(Keq*Ki_MgGDP) "  # AP
        "+ Ki_MgGDP*Km_SCOA*MgGTP*PI*V_mr/Ki_MgGTP "  # C R
        "+ COASH*Ki_MgGTP*Km_SUC*MgGDP*SCOA*V_mf/(Keq*Ki_MgGDP*Ki_SCOA) "  # ABP (incorrectly "ABQ" in the paper)
        "+ Km_MgGDP*MgGTP*PI*SCOA*V_mr/Ki_MgGTP "  # BCR
        "+ COASH*Km_MgGTP*MgGDP*SUC*V_mf/(Keq*Ki_MgGDP) "  # APQ
        "+ Ki_MgGDP*Km_SCOA*MgGTP*PI*SUC*V_mr/(Ki_MgGTP*Ki_SUC) "  # CQR
        "+ COASH*Ki_MgGTP*Km_SUC*MgGDP*PI*SCOA*V_mf/(Keq*Ki_MgGDP*Ki_PI*Ki_SCOA) "  # ABCP
        "+ Ki_COASH*Km_MgGTP*MgGDP*PI*SCOA*SUC*V_mf/(Keq*Ki_MgGDP*Ki_PI*Ki_SCOA) "  # ABCQ
        "+ COASH*Km_MgGTP*MgGDP*SCOA*SUC*V_mf/(Keq*Ki_MgGDP*Ki_SCOA) "  # ABPQ
        "+ Km_MgGDP*MgGTP*PI*SCOA*SUC*V_mr/(Ki_MgGTP*Ki_SUC) "  # BCQR
        "+ COASH*Ki_PI*Km_MgGDP*MgGTP*SCOA*SUC*V_mr/(Ki_COASH*Ki_MgGTP*Ki_SUC) "  # BPQR
        "+ COASH*Ki_MgGDP*Km_SCOA*MgGTP*PI*SUC*V_mr/(Ki_COASH*Ki_MgGTP*Ki_SUC) "  # CPQR
        "+ COASH*Km_MgGTP*MgGDP*PI*SCOA*SUC*V_mf/(Keq*Ki_MgGDP*Ki_PI*Ki_SCOA) "  # ABCPQ
        "+ COASH*Km_MgGDP*MgGTP*PI*SCOA*SUC*V_mr/(Ki_COASH*Ki_MgGTP*Ki_SUC) "  # BCPQR
        ")"
    )
    flux_act, pars_act = cf.simplify()
    micro_rate_constants = set(
        sp.symbols("k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, k5f, k5r, k6f, k6r")
    )
    _check_kinetic_flux_expr(
        flux_act, pars_act, flux_exp, pars_exp, micro_rate_constants
    )


@pytest.mark.skip(">15min")
def test_WuYan2007_succinyl_coa_synthetase_slow():
    """Test ordered ter-ter.

    See also Cook & Cleland, Enzyme kinetics and mechanism, p387f

    Test only computing the flux expression.
    The rest is tested in the fast test.
    """
    # NOTE: slow (~15min)
    rxn_str = """
    E + MgGDP -- E:MgGDP , k1f , k1r
    E:MgGDP + SCOA -- E:MgGDP:SCOA , k2f , k2r
    E:MgGDP:SCOA + PI -- E:MgGDP:SCOA:PI , k3f , k3r
    E:MgGDP:SCOA:PI -- E:MgGTP:SUC + COASH , k4f , k4r
    E:MgGTP:SUC -- E:MgGTP + SUC , k5f , k5r
    E:MgGTP -- E + MgGTP , k6f , k6r
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(MgGDP + PI + SCOA -- COASH + MgGTP + SUC)>"
    assert er.net_flux.equals(sympify(WuYan2007_succinyl_coa_synthetase_flux_str))
    res = er.get_kinetic_parameters()
    assert res["V_mf"] == sympify("E0*k4f*k5f*k6f/(k4f*k5f + k4f*k6f + k5f*k6f)")
    assert res["V_mr"] == sympify("E0*k1r*k2r*k3r/(k1r*k2r + k1r*k3r + k2r*k3r)")
    assert res["Km_MgGDP"] == sympify("k4f*k5f*k6f/(k1f*(k4f*k5f + k4f*k6f + k5f*k6f))")
    assert res["Km_SCOA"] == sympify("k4f*k5f*k6f/(k2f*(k4f*k5f + k4f*k6f + k5f*k6f))")
    assert res["Km_PI"] == sympify(
        "k5f*k6f*(k3r + k4f)/(k3f*(k4f*k5f + k4f*k6f + k5f*k6f))"
    )
    assert res["Km_COASH"] == sympify(
        "k1r*k2r*(k3r + k4f)/(k4r*(k1r*k2r + k1r*k3r + k2r*k3r))"
    )
    assert res["Km_SUC"] == sympify("k1r*k2r*k3r/(k5r*(k1r*k2r + k1r*k3r + k2r*k3r))")
    assert res["Km_MgGTP"] == sympify("k1r*k2r*k3r/(k6r*(k1r*k2r + k1r*k3r + k2r*k3r))")
    assert res["keq_micro"] == sympify(
        "k1f*k2f*k3f*k4f*k5f*k6f/(k1r*k2r*k3r*k4r*k5r*k6r)"
    )


def test_cleland1963_ordered_ter_bi():
    """Ordered ter-bi Cleland 1963a p.130 (mech. 17)"""
    rxn_str = """
    E + A -- E:A , k1 , k2
    E:A + B -- E:A:B , k3 , k4
    E:A:B + C -- E:A:B:C , k5 , k6
    E:A:B:C -- E:Q + P , k7 , k8
    E:Q -- E + Q , k9 , k10
    """
    er = EnzymeReaction(rxn_str)
    assert repr(er) == "<EnzymeReaction(A + B + C -- P + Q)>"

    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = sp.symbols(
        "k1 k2 k3 k4 k5 k6 k7 k8 k9 k10"
    )
    A, B, C, P, Q, E0 = sp.symbols("A B C P Q E0")

    flux_expected = (
        E0
        * (k1 * k3 * k5 * k7 * k9 * A * B * C - k2 * k4 * k6 * k8 * k10 * P * Q)
        / (
            k2 * k4 * (k6 + k7) * k9
            + k1 * k4 * (k6 + k7) * k9 * A
            + k2 * k5 * k7 * k9 * C
            + k1 * k3 * (k6 + k7) * k9 * A * B
            + k1 * k5 * k7 * k9 * A * C
            + k3 * k5 * k7 * k9 * B * C
            + k1 * k3 * k5 * (k7 + k9) * A * B * C
            + k1 * k4 * k6 * k8 * A * P
            + k2 * k5 * k7 * k10 * C * Q
            + k2 * k4 * (k6 + k7) * k10 * Q
            + k2 * k4 * k6 * k8 * P
            + (k2 * k4 + k4 * k6 + k2 * k6) * k8 * k10 * P * Q
            + k1 * k3 * k6 * k8 * A * B * P
            + k3 * k5 * k7 * k10 * B * C * Q
            + k3 * k6 * k8 * k10 * B * P * Q
            + k2 * k5 * k8 * k10 * C * P * Q
            + k1 * k3 * k5 * k8 * A * B * C * P
            + k3 * k5 * k8 * k10 * B * C * P * Q
        )
    )
    assert er.net_flux.equals(flux_expected)
    res = er.get_kinetic_parameters()
    assert res["keq_micro"] == k1 * k3 * k5 * k7 * k9 / (k10 * k2 * k4 * k6 * k8), res[
        "keq_micro"
    ]

    assert res["flux"].equals(flux_expected), (res["flux"] - flux_expected).cancel()

    pars_exp = {
        "Km_A": k7 * k9 / (k1 * (k7 + k9)),
        "Km_B": k7 * k9 / (k3 * k7 + k3 * k9),
        "Km_C": (k6 * k9 + k7 * k9) / (k5 * k7 + k5 * k9),
        "Km_P": (k2 * k4 * k6 + k2 * k4 * k7)
        / (k2 * k4 * k8 + k2 * k6 * k8 + k4 * k6 * k8),
        "Km_Q": k2 * k4 * k6 / (k10 * k2 * k4 + k10 * k2 * k6 + k10 * k4 * k6),
        "Ki_A": k2 / k1,
        "Ki_B": k4 / k3,
        "Ki_C": k6 / k5,
        "Ki_P": k7 / k8,
        "Ki_Q": k9 / k10,
    }

    (
        V_mf,
        V_mr,
        Km_Q,
        Km_P,
        Km_C,
        Km_B,
        Km_A,
        Ki_C,
        Ki_B,
        Ki_A,
        Ki_P,
        Ki_Q,
        Keq,
    ) = sp.symbols("V_mf V_mr Km_Q Km_P Km_C Km_B Km_A Ki_C Ki_B Ki_A Ki_P Ki_Q Keq")

    flux_exp = (
        V_mf
        * V_mr
        * (A * B * C - P * Q / Keq)
        / (
            Ki_A * Ki_B * Km_C * V_mr
            + A * Ki_B * Km_C * V_mr
            + C * Ki_A * Km_B * V_mr
            + A * B * Km_C * V_mr
            + A * C * Km_B * V_mr
            + B * C * Km_A * V_mr
            + A * B * C * V_mr
            + Km_P * Q * V_mf / Keq
            + Km_Q * P * V_mf / Keq
            + P * Q * V_mf / Keq
            + A * Km_Q * P * V_mf / (Keq * Ki_A)
            + C * Ki_A * Km_B * Q * V_mr / Ki_Q
            + A * B * Km_Q * P * V_mf / (Keq * Ki_A * Ki_B)
            + B * C * Km_A * Q * V_mr / Ki_Q
            + B * Ki_C * Km_A * P * Q * V_mr / (Ki_P * Ki_Q)
            + C * Ki_A * Km_B * P * Q * V_mr / (Ki_P * Ki_Q)
            + A * B * C * Km_Q * P * V_mf / (Keq * Ki_A * Ki_B * Ki_C)
            + B * C * Km_A * P * Q * V_mr / (Ki_P * Ki_Q)
        )
    )

    check_kinetic_flux_expr(er, flux_exp, pars_exp)
