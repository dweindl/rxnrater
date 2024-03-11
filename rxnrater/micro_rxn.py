from __future__ import annotations
from dataclasses import dataclass
from itertools import chain
import sympy as sp
from ._utils import sympify


@dataclass
class Rxn:
    substrates: list[sp.Symbol]
    products: list[sp.Symbol]
    rate_constants: list[sp.Symbol]

    @property
    def rates(self) -> list[sp.Symbol | sp.Expr]:
        return [
            rate_constant * sp.Mul(*reactants)
            for rate_constant, reactants in zip(
                self.rate_constants, [self.substrates, self.products]
            )
        ]

    @staticmethod
    def from_string(rxn_str: str) -> Rxn:
        rxn_str = rxn_str.strip()
        tokens = rxn_str.split(" ")
        substrates = []
        products = []
        rate_constants = []
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
                cur_list = rate_constants
                continue

            # ":" not supported by sympify
            expr = token.replace(":", "_")
            sym = sympify(expr)
            cur_list.append(sym)

        # TODO warn if rate constants are reused
        if reversible and len(rate_constants) != 2:
            raise ValueError(
                f"A reversible reaction requires exactly two rate constants, got {rate_constants}"
            )
        if not reversible and len(rate_constants) != 1:
            raise ValueError(
                f"An irreversible reaction requires exactly one rate constant, got {rate_constants}"
            )

        return Rxn(substrates, products, rate_constants)

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
