from rxnrater.micro_rxn import Rxn
from sympy import Symbol
from rxnrater._utils import sympify


def test_parse_irreversible():
    rxn = Rxn.from_string(
        """
        A + B -> C + D , k1
    """
    )
    assert rxn.substrates == [Symbol("A"), Symbol("B")]
    assert rxn.products == [Symbol("C"), Symbol("D")]
    assert rxn.reversible is False
    assert rxn.enzyme_states == []
    assert rxn.rates == [sympify("k1 * A * B")]


def test_parse_reversible():
    rxn = Rxn.from_string(
        """
        A1 + B2 + E -- E:B2 + A , k1 , k2
    """
    )
    assert rxn.substrates == [Symbol("A1"), Symbol("B2"), Symbol("E")]
    assert rxn.products == [Symbol("E_B2"), Symbol("A")]
    assert rxn.reversible is True
    assert rxn.enzyme_states == [Symbol("E"), Symbol("E_B2")]
    assert len(rxn.rates) == 2
    assert rxn.rates[0] == sympify("k1 * A1 * B2 * E")
    assert rxn.rates[1] == sympify("k2 * E_B2 * A")
