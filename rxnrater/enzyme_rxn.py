from .micro_rxn import Rxn
from itertools import chain
import sympy as sp
from ._utils import gcd


class EnzymeReaction:
    """An enzyme-catalyzed reaction.

    Contains the micro-reactions and computes the rate expression of the
    net reaction under the steady state assumption of the enzyme complexes.

    See https://doi.org/10.1021/ja01104a045 for more details.
    """

    def __init__(self, rxn_str: str):
        self.reactions = self._parse_reaction(rxn_str)
        self.enzyme_states = []

        for rxn in self.reactions:
            self.enzyme_states.extend(rxn.enzyme_states)
        self.enzyme_states = list(set(self.enzyme_states))

        self.substrates: list[sp.Symbol] = []
        self.products: list[sp.Symbol] = []
        self._set_net_reaction()
        self.stoichiometric_matrix: sp.Matrix = self._create_stoichiometric_matrix()
        self.micro_fluxes: sp.Matrix = self._get_micro_fluxes()
        self.steadystate_concentrations: dict[
            sp.Symbol, sp.Expr
        ] = self._compute_steadystate_concentrations()
        self.net_flux = self._compute_flux()

    def __repr__(self):
        substrates = " + ".join(str(s) for s in self.substrates)
        products = " + ".join(str(p) for p in self.products)
        return f"<EnzymeReaction({substrates} -- {products})>"

    @staticmethod
    def _parse_reaction(rxn_str: str) -> list[Rxn]:
        """Parse the micro-reaction string."""
        res = []
        for line in rxn_str.split("\n"):
            if not line.strip():
                continue
            res.append(Rxn.from_string(line))
        return res

    def _set_net_reaction(self):
        """Determine the net reaction of the micro-reactions."""
        net_reaction = sp.Add(
            *chain.from_iterable(rxn.products for rxn in self.reactions)
        ) - sp.Add(*chain.from_iterable(rxn.substrates for rxn in self.reactions))
        self.substrates = []
        self.products = []
        for sym in net_reaction.free_symbols:
            deriv = sp.diff(net_reaction, sym)
            if deriv > 0:
                self.products.append(sym)
            elif deriv < 0:
                self.substrates.append(sym)
            else:
                raise ValueError(f"Cannot determine if {sym} is a substrate or product")

    def _create_stoichiometric_matrix(self) -> sp.Matrix:
        """Create the stoichiometric matrix of the net reaction.

        Includes only the enzyme states as rows.
        """
        s = sp.zeros(
            len(self.enzyme_states), sum(len(rxn.rates) for rxn in self.reactions)
        )
        i = 0
        for rxn in self.reactions:
            for j, state in enumerate(self.enzyme_states):
                if state in rxn.products:
                    s[j, i] = 1
                elif state in rxn.substrates:
                    s[j, i] = -1
            if rxn.reversible:
                i += 1
                for j, state in enumerate(self.enzyme_states):
                    if state in rxn.products:
                        s[j, i] = -1
                    elif state in rxn.substrates:
                        s[j, i] = 1
            i += 1
        return s

    def _get_micro_fluxes(self) -> sp.Matrix:
        """Get the fluxes of the micro-reactions."""
        rates = sp.Matrix([rate for rxn in self.reactions for rate in rxn.rates])
        return rates

    def _xdot(self) -> sp.Matrix:
        """Compute the time derivative of the enzyme states."""
        return self.stoichiometric_matrix * self.micro_fluxes

    def _compute_steadystate_concentrations(self) -> dict[sp.Symbol, sp.Expr]:
        """Compute the steady state concentrations of the enzyme complexes.

        See also https://doi.org/10.1021/ja01104a045.
        """
        xdot = self._xdot()
        # Add the conservation of enzyme amounts
        xdot = xdot.row_insert(
            xdot.shape[0], sp.Matrix([[sp.Add(*self.enzyme_states) - 1]])
        )
        res = sp.solve(xdot, self.enzyme_states)

        # drop denominator which is the total enzyme concentration
        # ... this is slow
        # ... there is probably a smarter way to do that?!
        d = gcd(*res.values())
        return {k: (v / d).simplify() for k, v in res.items()}

    def _compute_flux(self) -> sp.Expr:
        """Compute the net flux of the reaction."""
        # choose some (net) product
        # should be the same result for all products?! (mass balance)
        chosen_product = self.products[0]
        # find all reactions producing / consuming it
        # sum up their rates
        flux = sp.Float(0)
        for rxn in self.reactions:
            if chosen_product in rxn.products:
                flux += rxn.rates[0]
            elif chosen_product in rxn.substrates:
                flux -= rxn.rates[1]
            if rxn.reversible:
                if chosen_product in rxn.products:
                    flux -= rxn.rates[1]
                elif chosen_product in rxn.substrates:
                    flux += rxn.rates[0]

        # substitute steady state concentrations of the enzyme complexes
        ss_conc = self.steadystate_concentrations
        flux = flux.subs(ss_conc)
        # divide by total enzyme concentration
        ret = sp.Symbol("E0") * flux / sp.Add(*ss_conc.values())
        return ret.simplify()

    def get_vmax(self) -> tuple[sp.Expr, sp.Expr]:
        """Get the maximum reaction rates.

        I.e., the limit of the net flux of the forward (backward) reaction
        for infinite substrate (product) and zero product (substrate)
        concentrations.
        """
        net_flux = self.net_flux

        # set products to 0, replace substrates by a single substrate,
        # as sympy doesn't support multivariate limits
        flux_tmp = net_flux.subs(
            {p: 0 for p in self.products}
            | {s: self.substrates[0] for s in self.substrates}
        )
        v_max_f = sp.limit(flux_tmp, self.substrates[0], sp.oo)

        flux_tmp = -net_flux.subs(
            {p: 0 for p in self.substrates}
            | {s: self.products[0] for s in self.products}
        )
        v_max_r = sp.limit(flux_tmp, self.products[0], sp.oo)
        return v_max_f.simplify(), v_max_r.simplify()

    def get_kinetic_parameters(self):
        """Get the kinetic parameters of the reaction.

        Get the maximum reaction rates and the half-saturation (K_m) constants.

        Half-saturation constants are the substrate (product) concentration at
        which the reaction rate is half of the maximum forward (backward) rate.
        """
        flux = self.net_flux
        v_max_f, v_max_r = self.get_vmax()

        pars = {}
        pars["V_mf"] = v_max_f
        pars["V_mr"] = v_max_r
        pars["flux"] = flux
        pars["keq_micro"] = sp.oo

        for s in self.substrates:
            flux2 = flux.subs({p: 0 for p in self.products})
            tmp = sp.solve(sp.Eq(flux2, 1 / 2 * v_max_f), s)
            assert len(tmp) == 1, (tmp, flux2)
            tmp = tmp[0]
            if len(self.substrates) > 1:
                # let other substrates approach inf
                other_substrates = self.substrates.copy()
                other_substrates.remove(s)
                tmp = tmp.subs(
                    {_s: other_substrates[0] for _s in self.substrates if _s != s}
                )
                tmp = sp.limit(tmp, other_substrates[0], sp.oo)
            pars[f"Km_{s}"] = tmp.simplify()
        if v_max_r.is_zero:
            return pars

        for p in self.products:
            flux2 = flux.subs({s: 0 for s in self.substrates})
            tmp = sp.solve(sp.Eq(-flux2, 1 / 2 * v_max_r), p)
            assert len(tmp) == 1, (tmp, flux2)
            tmp = tmp[0]
            if len(self.products) > 1:
                # let other substrates approach inf
                other_products = self.products.copy()
                other_products.remove(p)
                tmp = tmp.subs(
                    {_s: other_products[0] for _s in self.products if _s != p}
                )
                tmp = sp.limit(tmp, other_products[0], sp.oo)
            pars[f"Km_{p}"] = tmp.simplify()

        # equilibrium constant of net reaction in terms of microscopic rate
        # constants
        # keq = \prod products_ss / \prod substrates_ss
        # solve flux = 0 for first product, then multiply by all others,
        # divide by all substrates
        # exclude E0 = 0 solution
        flux_tmp = flux.subs(sp.Symbol("E0"), sp.Symbol("E0", positive=True))
        ss_prod_0 = sp.solve(flux_tmp, self.products[0])
        assert len(ss_prod_0) == 1
        ss_prod_0 = ss_prod_0[0]
        keq = ss_prod_0 * sp.Mul(*self.products[1:]) / sp.Mul(*self.substrates)
        pars["keq_micro"] = keq

        return pars
