# Scientific rationale for material workflows

This note records why the unified workflow separates constraints, fidelities,
and model-form checks. It is a design reference, not a claim that one optimizer
or one potential form is universally best.

## Parameter fitting and evidence

Force matching established that transferable interatomic potentials benefit
from energies, forces, and stresses over diverse configurations, rather than
only a few equilibrium scalar properties
([Ercolessi and Adams, 1994](https://doi.org/10.1209/0295-5075/26/8/005)).
The Nb EAM work of Fellinger *et al.* is a metal-specific example that fits DFT
force, energy, and stress data and validates structures, surfaces, defects, and
thermal properties separately
([Fellinger *et al.*, 2010](https://doi.org/10.1103/PhysRevB.81.144119)).
ForceBalance likewise treats a force field as a collection of parameterized
targets with explicit scales and regularization rather than an unstructured
single score
([Wang *et al.*, 2014](https://doi.org/10.1021/jz500737m)).

FFOpt therefore retains every target value, residual, tolerance, uncertainty,
and provenance. A scalar acquisition may guide sampling, but it cannot replace
the property-level record or the independent validation set.

## Constraints and feasible-region learning

Constrained Bayesian optimization models the objective and expensive
constraints separately, so an optimizer can seek improvement without treating
an infeasible low objective as a valid solution
([Gardner *et al.*, 2014](https://proceedings.mlr.press/v32/gardner14.html)).
Level-set estimation instead asks which parts of a domain lie above or below a
threshold and focuses evaluation near unresolved boundaries
([Gotovos *et al.*, 2013](https://www.ijcai.org/Proceedings/13/Papers/202.pdf)).

The Fe workflow combines these two ideas: it first records and covers the
structural feasible set, then minimizes an elastic objective only inside exact
observed structural constraints. Each constraint remains a continuous margin;
the pass/fail column is only a reporting threshold.

Trust-region BO is useful when dimensionality or nonstationarity makes one
global surrogate unreliable. TuRBO maintains and adapts multiple local regions
([Eriksson *et al.*, 2019](https://papers.nips.cc/paper_files/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html));
SCBO extends trust-region ideas to difficult constrained problems
([Eriksson and Poloczek, 2021](https://proceedings.mlr.press/v130/eriksson21a.html)).
The current BCC search has only three free coordinates, so a globally audited
constrained GP plus explicit coverage is the default. Multiple trust regions
become justified if the feasible set is shown to be fragmented or the parameter
dimension grows.

## Static and finite-temperature elasticity

The LAMMPS elasticity guidance distinguishes straightforward 0 K finite
deformation from finite-temperature elastic constants, which require time
averaging. It describes both finite deformation and the Born-matrix plus stress
fluctuation and kinetic contributions
([LAMMPS elasticity documentation](https://docs.lammps.org/Howto_elastic.html)).
Consequently, 0 K and 300 K results are correlated evidence sources, not values
that should be connected by one fixed universal scale factor.

Multi-fidelity modelling explicitly represents a correlation and discrepancy
between cheap and expensive information sources
([Kennedy and O'Hagan, 2000](https://doi.org/10.1093/biomet/87.1.1)).
BOCA further makes the value of a lower fidelity depend on its information and
cost relative to the target fidelity
([Kandasamy *et al.*, 2017](https://proceedings.mlr.press/v70/kandasamy17a.html)).
The workflow therefore uses exact 0 K elasticity for broad screening and active
learning, while a diverse subset receives replicated 300 K calculations. The
paired observations must be used to measure rank retention, bias, noise, and
the false-negative rate of the static screen.

For alpha-Fe, the low-temperature elastic constants reported by Rayne and
Chandrasekhar imply approximately `B = 173.1 GPa`, `Cprime = 52.5 GPa`, and
`C44 = 121.9 GPa`
([Rayne and Chandrasekhar, 1961](https://doi.org/10.1103/PhysRev.122.1714)).
At 300 K, resonant-ultrasound measurements give
`B = 166.2 +/- 0.9 GPa`, `Cprime = 48.15 +/- 0.9 GPa`, and
`C44 = 115.87 +/- 0.17 GPa`
([Adams *et al.*, 2006](https://doi.org/10.1063/1.2365714)).
These three independent cubic quantities are the primary finite-temperature
targets. Adams is deliberately an experiment independent of the Rayne 0 K
extrapolation; the pair must therefore not be used to infer a universal thermal
scale factor.  The ultrasonic constants are adiabatic, whereas a thermostatted
finite-temperature stress--strain calculation is closer to an isothermal
response.  That thermodynamic-definition difference is retained as a reported
systematic limitation rather than hidden inside the experimental uncertainty.
Isotropic `G`, `E`, and Poisson's ratio are derived diagnostics.

## Model-form limitation of elemental LJ

Central pair models obey restrictive elastic relations in their static,
zero-pressure setting. Metallic bonding motivated many-body forms such as
Finnis--Sinclair
([Finnis and Sinclair, 1984](https://doi.org/10.1080/01418618408244210))
and embedded-atom models
([Daw and Baskes, 1984](https://doi.org/10.1103/PhysRevB.29.6443)).
The ordinary Lennard-Jones phase diagram also does not make zero-pressure BCC
the generally stable bulk phase; BCC can instead be a limited or metastable
region of the model
([Sousa *et al.*, 2022](https://doi.org/10.1021/acs.jpcc.2c01255)).

A two-type corner/body representation introduces extra degrees of freedom but
also attaches permanent labels to translationally equivalent Fe sites.
Permutation invariance among identical atoms is a fundamental requirement for
an elemental potential representation
([Bartok *et al.*, 2013](https://doi.org/10.1103/PhysRevB.87.184115)).
Accordingly, FFOpt may optimize and validate the two-sublattice LJ surrogate,
but its report must distinguish computational composability from physical
transferability and must run label/termination/defect diagnostics before making
a broad materials claim.

## Stopping active learning

Expected-improvement stopping criteria quantify whether useful optimization
gain remains
([Nguyen *et al.*, 2017](https://proceedings.mlr.press/v77/nguyen17a.html)).
Uncertainty-based atomistic active-learning systems also distinguish in-domain,
candidate, and failed/extrapolative configurations; examples include D-optimal
selection
([Podryabinkin and Shapeev, 2017](https://doi.org/10.1016/j.commatsci.2017.09.031))
and FLARE
([Vandermause *et al.*, 2020](https://doi.org/10.1038/s41524-020-0283-z)).

FFOpt's scientific stop therefore requires more than a fixed round count:
constraint-boundary uncertainty, constrained improvement, high-fidelity
candidate stability, independent-seed confirmation, and applicability-domain
checks must all be reported. A budget limit is recorded as `budget_exhausted`,
not silently relabelled as convergence.
