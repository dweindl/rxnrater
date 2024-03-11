from __future__ import annotations
from dataclasses import dataclass
from itertools import chain
import sympy as sp
from ._utils import sympify


@dataclass
class Rxn:
    substrates: list[sp.Symbol]
    products: list[sp.Symbol]
    rates: list[sp.Symbol | sp.Expr]

    @staticmethod
    def from_string(rxn_str: str) -> Rxn:
        rxn_str = rxn_str.strip()
        tokens = rxn_str.split(" ")
        substrates = []
        products = []
        rates = []
        reversible = True
        cur_list = substrates
        for token in tokens:
            if not token or token == "+":
                continue

            if "," in token and token != ",":
                raise ValueError("Comma in token, forgot a blank?")

            if token in {"--", "->"}:
                cur_list = products
                if token == "->":
                    reversible = False
                continue

            # TODO currently requires space-separated commas
            if token == ",":
                cur_list = rates
                continue

            # ":" not supported by sympify
            expr = token.replace(":", "_")
            sym = sympify(expr)
            if cur_list is rates:
                sym *= sp.Mul(*(substrates if len(rates) == 0 else products))

            cur_list.append(sym)

        if reversible and len(rates) != 2:
            raise ValueError(
                f"A reversible reaction requires exactly two rate constants, got {rates}"
            )
        if not reversible and len(rates) != 1:
            raise ValueError(
                f"An irreversible reaction requires exactly one rate constant, got {rates}"
            )

        return Rxn(substrates, products, rates)

    @property
    def enzyme_states(self):
        # could be determined from conserved quantities
        return [
            reactant
            for reactant in chain(self.substrates, self.products)
            if reactant.name == "E" or reactant.name.startswith("E_")
        ]

    @property
    def reversible(self):
        return len(self.rates) == 2
