from .micro_rxn import Rxn
from itertools import chain, product
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
        # FIXME: this does not work for random mechanisms
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
        assert len(res), "No steady-state solution found"
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

    @property
    def micro_rate_constants(self) -> set[sp.Symbol]:
        """Get the microscopic rate constants of the reaction."""
        return set(chain.from_iterable(rxn.rate_constants for rxn in self.reactions))

    def simplify_flux(self, **kwargs):
        """Substitute the kinetic parameters for the microscopic rate
        constants.
        """
        cf = self.coeff_form()
        return cf.simplify(**kwargs)

    def coeff_form(self):
        """Get the rate equation in coefficient form."""
        return CoefficientFormRateEquation(
            self.net_flux, self.substrates, self.products, self.reversible
        )


class CoefficientFormRateEquation:
    """Steady-state rate equation in coefficient form.

    See Cleland1963a p.112ff. https://doi.org/10.1016/0926-6569(63)90211-6.
    """

    # TODO doesn't work for random mechanisms, right?

    def __init__(
        self,
        rate_eq: sp.Expr,
        substrates: list[sp.Symbol],
        products: list[sp.Symbol],
        reversible: bool = True,
    ):
        """
        Initialize.

        :param rate_eq:
            The rate equation of the reaction without the total enzyme
            coefficient.
        :param substrates:
            The symbols for the substrates of the reaction.
        :param products:
            The symbols for the products of the reaction.
        :param reversible:
            Whether the reaction is reversible.
        """
        self.rate_eq = rate_eq
        self.reversible = reversible
        self.substrates = substrates
        self.products = products
        assert set(substrates).isdisjoint(products)
        self._dissect()

    def _dissect(self):
        """Dissect the rate equation."""
        self.n, self.d = self.rate_eq.cancel().as_numer_denom()
        n, d = self.n, self.d

        if self.reversible:
            assert isinstance(n, sp.Add)
            assert len(n.args) == 2
            # arg-order may vary
            if n.args[1].args[0] == -1:
                n1 = n.args[0] / sp.Mul(*self.substrates)
                n2 = -n.args[1] / sp.Mul(*self.products)
            else:
                assert n.args[0].args[0] == -1
                n1 = n.args[1] / sp.Mul(*self.substrates)
                n2 = -n.args[0] / sp.Mul(*self.products)
            # TODO assert n1 / n2 == keq_micro
        else:
            assert isinstance(n, sp.Mul)
            n1 = n / sp.Mul(*self.substrates)
            n2 = sp.Integer(1)

        # product coefficient in the numerator
        self.n1 = n1
        # substrate coefficient in the numerator
        self.n2 = n2

        reactants = set(self.substrates + self.products)

        # group by common reactant factors
        # reactants => coefficients
        coeffs: dict[tuple[sp.Symbol], sp.Expr] = {}
        for expr in d.args:
            cur_reactants = expr.free_symbols & reactants
            assert (expr / sp.Mul(*cur_reactants)).free_symbols.isdisjoint(reactants)
            key = tuple(sorted(cur_reactants, key=str))
            coeffs[key] = coeffs.get(key, sp.Integer(0)) + expr / sp.Mul(*cur_reactants)

        self.coefficients = coeffs

    def coefficient_of(self, reactants: list[sp.Symbol]) -> sp.Expr:
        return self.coefficients[tuple(sorted(reactants, key=str))]

    def _collect_candidates(self) -> dict[str, list[sp.Expr]]:
        """Collect candidate expressions to define the different kinetic
        parameters.

        Find all ratios of coefficients that are valid to define the
        kinetic parameters (Km, Ki).
        """
        candidates = {}
        for s in [*self.substrates, *self.products]:
            # look for K_m_{s}, K_i_{s}
            for reactants_n, coeff_n in self.coefficients.items():
                # `s` must not be in the numerator
                if s in reactants_n:
                    continue

                # find a valid coefficient for the denominator
                #  we need {*reactants_n, s}
                for reactants_d, coeff_d in self.coefficients.items():
                    if set(reactants_d) != set(reactants_n) | {s}:
                        continue
                    # if the denominator is the product of all
                    #  substrates or products, then the ratio will define
                    #  a K_m parameter, otherwise a K_i parameter
                    if set(reactants_d) != set(self.substrates) and set(
                        reactants_d
                    ) != set(self.products):
                        if not (coeff_n / coeff_d).cancel().as_numer_denom()[1].is_Atom:
                            continue
                        sym = f"Ki_{s}"
                    else:
                        sym = f"Km_{s}"
                    candidates[sym] = candidates.get(sym, set()) | {
                        (coeff_n / coeff_d).cancel()
                    }
        # TODO: sort for deterministic output
        candidates = {k: list(v) for k, v in candidates.items()}

        return candidates

    def simplify(self, exhaustive: bool = False) -> tuple[sp.Expr, dict[str, sp.Expr]]:
        """Replace micro-rate constants by macroscopic kinetic parameters."""
        sym_vmax_f = sp.Symbol("V_mf")
        sym_vmax_r = sp.Symbol("V_mr") if self.reversible else sp.Integer(1)
        sym_keq = sp.Symbol("Keq")

        # compute admissible values for K_m_{} and K_i_{}
        candidates = self._collect_candidates()

        debug = False
        if debug:
            from pprint import pprint
        else:

            def pprint(*args, **kwargs):
                pass

            print = pprint

        pprint(
            candidates,
            sort_dicts=False,
        )

        coeff_all_substrates = self.coefficient_of(self.substrates)
        coeff_all_products = (
            self.coefficient_of(self.products) if self.reversible else sp.Integer(1)
        )

        magic = self.n2 / (coeff_all_substrates * coeff_all_products)
        # magic == V2 / coeff_all_substrates =

        # V1 = n1 / coeff_all_substrates
        # V2 = n2 / coeff_all_products
        # V1 * V2 = n1 * n2 / (coeff_all_substrates * coeff_all_products)
        # keq  = n1 / n2 = V1 * coeff_all_substrates / (V2 * coeff_all_products) = V1 / V2 * coeff_all_substrates / coeff_all_products
        # V1 / Keq = V2 * coeff_all_products / coeff_all_substrates
        V2 = self.n2 / coeff_all_products
        V1 = self.n1 / coeff_all_substrates
        keq = self.n1 / self.n2
        for k, v in self.coefficients.items():
            # tmp2 = v / magic
            # tmp2 = tmp2.simplify()
            print(k, v)

        # The simplified coefficients look like this:
        #  prod_k / prod_ki * v_term
        #   v_term is "V2"       if only substrates
        #             "V1 / Keq" if only products
        #             either of those otherwise
        #  prod_k is the product of various Km and Ki terms
        #    if V1 is used, Km values or Ki values to complement the products
        #    if V2 is used, Km values or Ki values to complement the substrates
        # prod_ki is 1 for only products / only substrates, otherwise:
        #    if V1 is used, Ki values to cancel out the substrates
        #    if V2 is used, Ki values to cancel out the products
        print()

        solutions = []

        for cur_candidates in product(*candidates.values()):
            cur_candidates = dict(zip(candidates.keys(), cur_candidates))
            print("cur_candidates", cur_candidates)

            simplified = {}
            for involved_reactants, coeff in self.coefficients.items():
                print("  ", involved_reactants, coeff)
                found_match = False
                v_terms = []
                if set(involved_reactants) - set(self.substrates) == set():
                    v_terms.append(V2)
                elif set(involved_reactants) - set(self.products) == set():
                    v_terms.append(V1 / keq)
                else:
                    v_terms.append(V1 / keq)
                    v_terms.append(V2)
                for v_term in v_terms:
                    # try all V terms
                    sym_v = sym_vmax_r if v_term == V2 else sym_vmax_f / sym_keq
                    if v_term == V1 / keq:
                        k_choices = [
                            [
                                (sp.Symbol(s), cur_candidates[s])
                                for s in [f"Km_{r}", f"Ki_{r}"]
                                if s in cur_candidates
                            ]
                            for r in self.products
                            if r not in involved_reactants
                        ]
                        if len(v_terms) > 1:
                            # mixed substrates and products, add 1/Ki terms
                            tmp = [
                                [
                                    (
                                        1 / sp.Symbol(f"Ki_{k}"),
                                        1 / cur_candidates[f"Ki_{k}"],
                                    )
                                ]
                                for k in involved_reactants
                                if k in self.substrates and f"Ki_{k}" in cur_candidates
                            ]
                            k_choices += tmp
                    else:
                        k_choices = [
                            [
                                (sp.Symbol(s), cur_candidates[s])
                                for s in [f"Km_{r}", f"Ki_{r}"]
                                if s in cur_candidates
                            ]
                            for r in self.substrates
                            if r not in involved_reactants
                        ]
                        if len(v_terms) > 1:
                            # mixed substrates and products, add 1/Ki terms
                            tmp = [
                                [
                                    (
                                        1 / sp.Symbol(f"Ki_{k}"),
                                        1 / cur_candidates[f"Ki_{k}"],
                                    )
                                ]
                                for k in involved_reactants
                                if k in self.products and f"Ki_{k}" in cur_candidates
                            ]
                            k_choices += tmp

                    for cur_choice in product(*k_choices):
                        cur_choice = dict(cur_choice)
                        sym_trial = sym_v * sp.Mul(*cur_choice.keys())
                        trial = v_term * sp.Mul(*cur_choice.values())
                        tmp = trial / magic - coeff
                        tmp2 = tmp.cancel()
                        print(sym_trial, tmp2)
                        if tmp2.is_zero:
                            found_match = True
                            simplified[involved_reactants] = sym_trial
                if not found_match:
                    # no need to try the other coefficients
                    break
            else:
                # all coefficients match
                print("MATCH", cur_candidates)
                # TODO: add option for exhaustive search
                simplified_rate_eq = self._assemble_simplified(
                    simplified,
                    cur_candidates,
                    sym_vmax_f,
                    sym_vmax_r,
                    sym_keq,
                    keq,
                    V1,
                    V2,
                )
                if not exhaustive:
                    return simplified_rate_eq, cur_candidates
                solutions.append((simplified_rate_eq, cur_candidates))
        else:
            if exhaustive and solutions:
                return solutions
            raise AssertionError("No match found")

    def _assemble_simplified(
        self,
        simplified: dict,
        cur_candidates: dict[str, sp.Expr],
        sym_vmax_f: sp.Symbol,
        sym_vmax_r: sp.Symbol,
        sym_keq: sp.Symbol,
        keq: sp.Expr,
        V1: sp.Expr,
        V2: sp.Expr,
    ) -> tuple[sp.Expr, dict[str, sp.Expr]]:
        """Assemble the simplified rate equation."""
        # compute V_mr from Haldane relationship
        # There is always at least a K_eq = V1/V2 * \prod K_i_or_m_product / \prod K_i_or_m_substrate
        # We just need to find out which ones
        # V_mr = V_mf / Keq * \prod K_i_or_m_substrate / \prod K_i_or_m_product
        if self.reversible:
            cands = [
                [
                    (sp.Symbol(s), cur_candidates[s])
                    for s in [f"Km_{r}", f"Ki_{r}"]
                    if s in cur_candidates
                ]
                for r in self.products
            ] + [
                [
                    (1 / sp.Symbol(s), 1 / cur_candidates[s])
                    for s in [f"Km_{r}", f"Ki_{r}"]
                    if s in cur_candidates
                ]
                for r in self.substrates
            ]
            for cur_choice in product(*cands):
                cur_choice = dict(cur_choice)
                trial = keq - V1 / V2 * sp.Mul(*cur_choice.values())
                sym_trial = sym_vmax_f / sym_keq * sp.Mul(*cur_choice.keys())
                trial = trial.cancel()
                print(sym_trial, trial.cancel())
                if trial.is_zero:
                    cur_candidates["V_mr"] = sym_trial
                    break
            else:
                raise AssertionError("No match found")

        simplified = (
            sym_vmax_f
            * sym_vmax_r
            * (
                sp.Mul(*self.substrates)
                - (
                    sp.Mul(*self.products) / sym_keq
                    if self.reversible
                    else sp.Integer(0)
                )
            )
            / sp.Add(*[sp.Mul(*k) * v for k, v in simplified.items()])
        )

        return simplified

    @property
    def e0(self) -> sp.Symbol:
        return sp.Symbol("E0")

    @property
    def fwd_rate_constants(self) -> set[sp.Symbol]:
        """Get the forward rate constants.

        I.e. substrate association and product dissociation rate constants.
        """
        return (self.n1 / self.e0).free_symbols

    @property
    def bwd_rate_constants(self) -> set[sp.Symbol]:
        """Get the backward rate constants.

        I.e. substrate dissociation and product association rate constants.
        """
        return (self.n2 / self.e0).free_symbols
