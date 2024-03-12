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
        # symbol for the total enzyme concentration
        self.e0 = sp.Symbol("E0")
        # symbol for free enzyme
        self.e_free = sp.Symbol("E")

        self.enzyme_states = []

        for rxn in self.reactions:
            self.enzyme_states.extend(rxn.enzyme_states)
        self.enzyme_states = list(set(self.enzyme_states))

        self.substrates: list[sp.Symbol] = []
        self.products: list[sp.Symbol] = []
        self._set_net_reaction()

        self.kinetic_parameters = None
        # TODO compute on demand
        self.stoichiometric_matrix: sp.Matrix = self._create_stoichiometric_matrix()
        self.micro_fluxes: sp.Matrix = self._get_micro_fluxes()
        self.steadystate_concentrations: dict[
            sp.Symbol, sp.Expr
        ] = self._compute_steadystate_concentrations()
        self.net_flux = self._compute_flux()
        self.vmax_f, self.vmax_r = self._compute_vmax()

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

        # TODO: warn if rate constants are reused
        return res

    def _set_net_reaction(self):
        """Determine the net reaction of the micro-reactions."""
        net_reaction = sp.Add(
            *chain.from_iterable(rxn.products for rxn in self.reactions)
        ) - sp.Add(*chain.from_iterable(rxn.substrates for rxn in self.reactions))
        self.substrates = []
        self.products = []
        for sym in sorted(net_reaction.free_symbols, key=str):
            deriv = sp.diff(net_reaction, sym)
            if deriv > 0:
                self.products.append(sym)
            elif deriv < 0:
                self.substrates.append(sym)
            else:
                raise ValueError(f"Cannot determine if {sym} is a substrate or product")

        net_reaction_enzyme_states = (set(self.products) | set(self.substrates)) & set(
            self.enzyme_states
        )
        if net_reaction_enzyme_states:
            raise ValueError(
                f"Enzyme states must not be substrates or products of the net reaction: {net_reaction_enzyme_states}"
            )

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
        ret = self.e0 * flux / sp.Add(*ss_conc.values())
        return ret.simplify()

    def _compute_vmax(self) -> tuple[sp.Expr, sp.Expr]:
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

    def _compute_kms(self) -> dict[str, sp.Expr]:
        """Compute the half-saturation constants of the reaction."""
        kms = {}

        for substrates, products, net_flux, vmax_f in zip(
            [self.substrates, self.products],
            [self.products, self.substrates],
            [self.net_flux, -self.net_flux],
            [self.vmax_f, self.vmax_r],
        ):
            for s in substrates:
                # set other substrates to 0
                flux2 = net_flux.subs({p: 0 for p in products})
                # compute concentration of `s` for half-maximum rate
                tmp = sp.solve(sp.Eq(flux2, 1 / 2 * vmax_f), s)
                assert len(tmp) == 1, (tmp, flux2)
                tmp = tmp[0]
                if len(substrates) > 1:
                    # let other substrates approach inf
                    other_substrates = substrates.copy()
                    other_substrates.remove(s)
                    tmp = tmp.subs(
                        {_s: other_substrates[0] for _s in substrates if _s != s}
                    )
                    tmp = sp.limit(tmp, other_substrates[0], sp.oo)
                kms[f"Km_{s}"] = tmp.simplify()

            if not self.reversible:
                break

        return kms

    def _compute_kis(self) -> dict[str, sp.Expr]:
        """Compute the inhibition constants of the reaction."""
        # TODO Consider computing (all possible) Ki parameters based on the rate
        # TODO: missing Kii parameters
        # expression denominator term coefficients
        # see https://doi.org/10.4324/9780203833575 p41ff
        kis: dict[str, sp.Expr] = {}

        for substrates, products, net_flux, vmax in zip(
            [self.substrates, self.products],
            [self.products, self.substrates],
            [-self.net_flux, self.net_flux],
            [self.vmax_r, self.vmax_f],
        ):
            for s in substrates:
                # set other substrates to 0
                limit_flux = net_flux.subs({s_: 0 for s_ in substrates if s_ != s})
                # let products approach inf
                for p in products:
                    limit_flux = sp.limit(limit_flux, p, sp.oo).cancel()

                # not completely sure this is generally valid
                if limit_flux.has(s):
                    # uncompetitive product inhibition
                    # compute substrate concentration for half-maximum backward rate
                    tmp = sp.solve(sp.Eq(limit_flux, 1 / 2 * vmax), s)
                    assert len(tmp) == 1, (tmp, limit_flux)
                    kis[f"Ki_{s}"] = tmp[0]
                    continue

                # competitive product inhibition
                # Ki is just the dissociation constant of the enzyme-substrate
                # complex
                # find micro-reaction S + E -> E_S
                num = denom = None
                # enzyme-substrate complex
                for reaction in self.reactions:
                    if set(reaction.substrates) == {self.e_free, s}:
                        # S + E -> E_S
                        assert (
                            denom is None
                        ), "Found multiple S + E -- E_S micro-reactions"
                        denom = reaction.rate_constants[0]
                        assert len(reaction.products) == 1
                        assert reaction.products[0] in self.enzyme_states
                        if reaction.reversible:
                            num = reaction.rate_constants[1]
                        break
                    if (
                        set(reaction.products) == {self.e_free, s}
                        and reaction.reversible
                    ):
                        # E_S -- S + E
                        assert (
                            denom is None
                        ), "Found multiple E_S -- S + E micro-reactions"
                        denom = reaction.rate_constants[1]
                        assert len(reaction.substrates) == 1
                        assert reaction.substrates[0] in self.enzyme_states
                        num = reaction.rate_constants[0]

                if num is None or denom is None:
                    if self.reversible:
                        raise ValueError(
                            f"Cannot find micro-reaction for E+{s} -> E:{s}"
                        )
                    continue

                kis[f"Ki_{s}"] = num / denom

        return kis

    def get_kinetic_parameters(self):
        """Get the kinetic parameters of the reaction.

        Get the maximum reaction rates and the half-saturation (K_m) constants.

        Half-saturation constants are the substrate (product) concentration at
        which the reaction rate is half of the maximum forward (backward) rate.
        """
        if self.kinetic_parameters is not None:
            return self.kinetic_parameters

        pars = self._compute_kms()
        pars["V_mf"] = self.vmax_f
        pars["V_mr"] = self.vmax_r
        pars["flux"] = self.net_flux
        pars["keq_micro"] = self._compute_keq()
        pars |= self._compute_kis()
        return pars

    def _compute_keq(self) -> sp.Expr:
        """Compute the equilibrium constant of the reaction.

        Compute the equilibrium constant of net reaction in terms of
        microscopic rate constants.
        """
        if not self.reversible:
            return sp.oo

        # keq = \prod products_ss / \prod substrates_ss
        # solve flux = 0 for first product, then multiply by all reactants,
        # exclude E0 = 0 solution
        flux_tmp = self.net_flux.subs(self.e0, sp.Symbol(self.e0.name, positive=True))
        ss_prod_0 = sp.solve(flux_tmp, self.products[0])
        assert len(ss_prod_0) == 1
        ss_prod_0 = ss_prod_0[0]
        # cancel out remaining reactants
        keq = ss_prod_0 * sp.Mul(*self.products[1:]) / sp.Mul(*self.substrates)
        return keq.simplify()

    @property
    def reversible(self):
        return not self.vmax_r.is_zero

    def simplify_flux(self):
        """Substitute the kinetic parameters for the microscopic rate
        constants.

        NOTE: This seems rather fragile and may well fail.
        """
        kp = self.get_kinetic_parameters()
        vmax_f = kp["V_mf"]
        vmax_r = kp["V_mr"]
        sym_vmax_f = sp.Symbol("V_mf")
        sym_vmax_r = sp.Symbol("V_mr")
        kms = {sp.Symbol(k): v for k, v in kp.items() if k.startswith("Km_")}
        kis = {sp.Symbol(k): v for k, v in kp.items() if k.startswith("Ki_")}
        keq_micro = kp["keq_micro"]
        sym_keq = sp.Symbol("Keq")
        micro_rate_constants = set(
            chain.from_iterable(rxn.rate_constants for rxn in self.reactions)
        )

        # see https://doi.org/10.1016/0926-6569(63)90211-6.
        n, d = (self.net_flux / self.e0).cancel().as_numer_denom()
        if self.reversible:
            assert isinstance(n, sp.Add)
            assert len(n.args) == 2
            n1 = n.args[0] / sp.Mul(*self.substrates)
            n2 = -n.args[1] / sp.Mul(*self.products)
            assert n1 / n2 == keq_micro
        else:
            assert isinstance(n, sp.Mul)
            n1 = n / sp.Mul(*self.substrates)
            n2 = sp.Integer(1)

        reactants = set(self.substrates + self.products)

        # group by common reactant factors
        # reactants => coefficients
        coeffs = {}
        for expr in d.args:
            cur_reactants = expr.free_symbols & reactants
            assert (expr / sp.Mul(*cur_reactants)).free_symbols.isdisjoint(reactants)
            key = tuple(sorted(cur_reactants, key=str))
            coeffs[key] = coeffs.get(key, sp.Float(0)) + expr / sp.Mul(*cur_reactants)

        # find coefficients the product of substrates and products in the
        # denominator
        coeff_subs = coeffs[tuple(sorted(self.substrates, key=str))]
        assert (n1 / coeff_subs - vmax_f / self.e0).simplify().is_zero
        if self.reversible:
            coeff_prods = coeffs[tuple(sorted(self.products, key=str))]
            assert (n2 / coeff_prods - vmax_r / self.e0).simplify().is_zero
        else:
            coeff_prods = sp.Integer(1)
        # assemble new numerator
        new_n = (
            sym_vmax_f
            * (sym_vmax_r if self.reversible else sp.Integer(1))
            * (
                sp.Mul(*self.substrates)
                - (sp.Mul(*self.products) / sym_keq if self.reversible else sp.Float(0))
            )
        )

        # now replace the denominator terms
        d_summands = []

        for cur_reactants, coeff in coeffs.items():
            expr = None
            # try once with vf, once with vr substitution,
            #  see which one gets rid of all rate constants
            # is there any rule for the mixed product/substrate case?
            #  there has to be ...
            for trial in [1, 2]:
                expr = coeff * n2 / coeff_prods / coeff_subs
                expr = expr.simplify()

                # Each term will have either Vmax_f or Vmax_r in the numerator
                if not self.reversible:
                    pass
                elif set(cur_reactants).isdisjoint(self.products):
                    # if there is no product term, substitute vmax_r
                    expr = (expr / (vmax_r / self.e0) * sym_vmax_r).simplify()
                elif set(cur_reactants).isdisjoint(self.substrates):
                    expr = (
                        expr / (vmax_f / self.e0) * sym_vmax_f / sym_keq * keq_micro
                    ).simplify()
                else:
                    # mixed products and substrates
                    if trial == 1:
                        expr = (expr / (vmax_r / self.e0) * sym_vmax_r).simplify()
                    elif trial == 2:
                        expr = (
                            expr / (vmax_f / self.e0) * sym_vmax_f / sym_keq * keq_micro
                        ).simplify()

                # - try any km of a reactant that is not included or any Ki
                # - Km always occurs in the numerator, Ki can occur everywhere
                # - if the reactants only include substrates, only substrate
                #   parameters can be present, vice versa for products
                #   (mixed -> both)
                # - there is never a Ki and Km of the same reactant in the
                #   same term

                # if count_ops is reduced by a substitution, we accept it
                for km_sym, km_expr in kms.items():
                    if sp.Symbol(km_sym.name.removeprefix("Km_")) in cur_reactants:
                        continue

                    trial_expr = (expr / km_expr * km_sym).simplify()
                    if trial_expr.count_ops() < expr.count_ops():
                        expr = trial_expr
                        # there is maximally one K_m in each coefficient
                        break

                for ki_sym, ki_expr in kis.items():
                    trial_expr = (expr / ki_expr * ki_sym).simplify()
                    if trial_expr.count_ops() < expr.count_ops():
                        # Ki is in the numerator
                        expr = trial_expr
                    elif sp.Symbol(ki_sym.name.removeprefix("Ki_")) in cur_reactants:
                        trial_expr = (expr * ki_expr / ki_sym).simplify()
                        if trial_expr.count_ops() < expr.count_ops():
                            # Ki is in the denominator
                            # occurs there only if the respective reactant is included
                            # in the current term
                            expr = trial_expr

                if expr.free_symbols.isdisjoint(micro_rate_constants):
                    d_summands.append(expr * sp.Mul(*cur_reactants))
                    break

                if set(cur_reactants).isdisjoint(self.products):
                    # vmax{f,r} choice is clear, no need to retry
                    break
            else:
                raise ValueError(
                    "Could not simplify denominator term: " f"{cur_reactants}: {expr}"
                )

        new_d = sp.Add(*d_summands)
        return new_n / new_d
