from rxnrater.enzyme_rxn import EnzymeReaction
from rxnrater._utils import sympify
import sympy as sp


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
    # Should Ki be computed in this case?
    assert res["Ki_FUM"] == sympify("k1r/k1f")
    assert er.simplify_flux() == sympify(
        "V_mf*V_mr*(FUM - MAL/Keq)/(FUM*V_mr + Km_FUM*V_mr + MAL*V_mf/Keq)"
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
    assert res["Ki_NAD"] == sympify("k1r/k1f")
    assert res["Ki_MAL"] == sympify("(k1r+k2r)/k2f")
    assert res["Ki_OAA"] == sympify("(k3f + k4f)/k3r")
    assert res["Ki_NADH"] == sympify("k4f / k4r")
    assert er.simplify_flux() == sympify(
        "V_mf*V_mr*(MAL*NAD - NADH*OAA/Keq)/(Ki_NAD*Km_MAL*V_mr "
        "+ Km_MAL*NAD*V_mr + Km_NAD*MAL*V_mr + MAL*NAD*V_mr "
        "+ MAL*NAD*OAA*V_mr/Ki_OAA + Km_NAD*MAL*NADH*V_mr/Ki_NADH "
        "+ Km_NADH*OAA*V_mf/Keq + Km_OAA*NADH*V_mf/Keq + NADH*OAA*V_mf/Keq "
        "+ Km_NADH*NAD*OAA*V_mf/(Keq*Ki_NAD) + MAL*NADH*OAA*V_mf/(Keq*Ki_MAL))"
    )


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
    assert res["Ki_OAA"] == sympify("k1r/k1f")
    assert er.simplify_flux() == sympify(
        "ACCOA*OAA*V_mf/(ACCOA*Km_OAA + ACCOA*OAA + Ki_OAA*Km_ACCOA + Km_ACCOA*OAA)"
    )


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
    assert er.simplify_flux() == sympify(
        "COASH*NAD*PYR*V_mf/(COASH*Km_NAD*PYR + COASH*Km_PYR*NAD + COASH*NAD*PYR + Km_COASH*NAD*PYR)"
    )


def test_WuYan2007_succinyl_coa_synthetase():
    """Test ordered ter-ter.

    See also Cook & Cleland, Enzyme kinetics and mechanism, p387f
    """
    # NOTE: slow
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
    assert res["Ki_MgGDP"] == sympify("k1r/k1f")
    # FIXME: there fail, but might still be correct as there are
    #  potentially multiple ways to express those Ki values
    #  (needs verification)
    # assert res["Ki_SCOA"] == sympify("k2r/k2f")
    # assert res["Ki_PI"] == sympify("k3r/k3f")
    # assert res["Ki_COASH"] == sympify("k4f/k4r")
    # assert res["Ki_SUC"] == sympify("k5f/k5r")
    # assert res["Ki_MgGTP"] == sympify("k6f/k6r")

    # TODO er.simplify_flux()
