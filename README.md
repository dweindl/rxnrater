# rxnrater

This is a package to derive rate expressions from enzyme mechanisms based on
the steady-state assumption.

This project is in the early stages of development and is not necessarily
ready for production use. Double-check the results.

## Installation

```bash
# Clone the repository
# then
pip install .
```

## Usage

Example:

```python
from pprint import pprint
from rxnrater import EnzymeReaction
# Define the enzyme mechanism (E + S ⇌ E:S ⇌ E + P)
mechanism = """
E + S -- E:S , k1 , k2
E:S -- E:P , k3 , k4
E:P -- E + P , k5 , k6
"""
er = EnzymeReaction(mechanism)
pprint(er.get_kinetic_parameters())
pprint(er.simplify_flux())
```
```
{'Km_P': (k2*k4 + k2*k5 + k3*k5)/(k6*(k2 + k3 + k4)),
 'Km_S': (k2*k4 + k2*k5 + k3*k5)/(k1*(k3 + k4 + k5)),
 'V_mf': E0*k3*k5/(k3 + k4 + k5),
 'V_mr': E0*k2*k4/(k2 + k3 + k4),
 'flux': E0*(-P*k2*k4*k6 + S*k1*k3*k5)/(P*k2*k6 + P*k3*k6 + P*k4*k6 + S*k1*k3 + S*k1*k4 + S*k1*k5 + k2*k4 + k2*k5 + k3*k5),
 'keq_micro': k1*k3*k5/(k2*k4*k6)}
(V_mf*V_mr*(S - P/Keq)/(Km_S*V_mr + S*V_mr + P*V_mf/Keq),
 {'Km_P': (k2*k4 + k2*k5 + k3*k5)/(k2*k6 + k3*k6 + k4*k6),
  'Km_S': (k2*k4 + k2*k5 + k3*k5)/(k1*k3 + k1*k4 + k1*k5),
  'V_mr': Km_P*V_mf/(Keq*Km_S)})
```
See `tests/test_enzyme_rxn.py` for more examples.

### Reaction syntax

The syntax is still a bit awkward:

* Each line contains one micro-reaction:
* The syntax is `reactants -- products , k_forward , k_reverse`
  for reversible reactions, or
  `reactants -> products , k_forward` for irreversible reactions.
  the rate constants are separated by ` , ` (spaces are required).
* Individual reactions are assumed to follow mass action kinetics.
* Enzyme-species are denoted by `E` or anything starting with `E:`
* Reactants are separated by ` + ` (spaces are required).
* Species names must consist of `[A-Za-z0-9_]` and start with a letter.


## Availability

The source code is available at [https://github.com/dweindl/rxnrater](https://github.com/dweindl/rxnrater).
