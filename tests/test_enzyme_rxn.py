from rxnrater.enzyme_rxn import EnzymeReaction
from rxnrater._utils import sympify
import sympy as sp


def test_WuYan2007_fumarase():
    rxn_str = """
    E + FUM -- E:FUM , k1f , k1r
    E:FUM -- E + MAL , k2f , k2r
    """
    er = EnzymeReaction(rxn_str)
    res = er.get_kinetic_parameters()
    assert res["V_mf"] == sympify("E0 * k2f")
    assert res["V_mr"] == sympify("E0 * k1r")
    assert res["flux"] == sympify(
        "E0 * (FUM * k1f * k2f - MAL * k1r * k2r) / (FUM * k1f + MAL * k2r + k1r + k2f)"
    )
    assert res["Km_FUM"] == sympify("(k1r + k2f) / k1f")
    assert res["Km_MAL"] == sympify("(k1r + k2f) / k2r")
    assert res["keq_micro"] == sympify("k1f*k2f/(k1r*k2r)")


def test_WuYan2007_mdh():
    rxn_str = """
    E + NAD -- E:NAD , k1f , k1r
    E:NAD + MAL -- E:NAD:MAL , k2f , k2r
    E:NAD:MAL -- E:NADH + OAA , k3f , k3r
    E:NADH -- E + NADH , k4f , k4r
    """
    er = EnzymeReaction(rxn_str)
    res = er.get_kinetic_parameters()
    assert res["V_mf"] == sympify("E0*k3f*k4f/(k3f + k4f)")
    assert res["V_mr"] == sympify("E0*k1r*k2r/(k1r + k2r)")
    assert res["Km_NAD"] == sympify("k3f*k4f/(k1f*(k3f + k4f))")
    assert res["Km_MAL"] == sympify("k4f*(k2r + k3f)/(k2f*(k3f + k4f))")
    assert res["Km_OAA"] == sympify("k1r*(k2r + k3f)/(k3r*(k1r + k2r))")
    assert res["Km_NADH"] == sympify("k1r*k2r/(k4r*(k1r + k2r))")
    assert res["keq_micro"] == sympify("k1f*k2f*k3f*k4f/(k1r*k2r*k3r*k4r)")


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
    res = er.get_kinetic_parameters()
    assert res["V_mf"] == sympify("E0*k2*k4*k6/(k2*k4 + k2*k6 + k4*k6)")
    assert res["V_mr"] == 0
    assert res["Km_PYR"] == sympify("k4*k6*(k1r + k2)/(k1f*(k2*k4 + k2*k6 + k4*k6))")
    assert res["Km_COASH"] == sympify("k2*k6*(k3r + k4)/(k3f*(k2*k4 + k2*k6 + k4*k6))")
    assert res["Km_NAD"] == sympify("k2*k4*(k5r + k6)/(k5f*(k2*k4 + k2*k6 + k4*k6))")
    assert res["keq_micro"] == sp.oo
