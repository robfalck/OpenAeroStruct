"""
B-018: after `run_optimizer()`, a component whose own output isn't part of any declared
design_vars/constraints/objectives can be left with a stale, un-synced promoted/connected input,
even though its upstream source is already fresh.

Here `aero_point_0.surface_perf.coeffs` computes both `CL1` (-> `CL`, read by the declared
constraint) and `CDi` (-> `CD`, read by nothing declared). `coeffs.CDi` itself is fresh after
`run_optimizer()`, but `CD`'s own connected input of the same name stays at its pre-optimization
value, so `CD`'s output is wrong until an extra `run_model()` call forces full propagation.

Three minimal (non-OAS) attempts to reproduce this with plain `ExecComp`s -- a shared value
feeding both a "needed" and "unneeded" downstream consumer, with and without a nesting `Group`
boundary -- did not reproduce it, so whatever triggers this needs more of the structure below
(a multi-output component sharing inputs with an implicit `aero_states` solve upstream, a
multi-level `VLMFunctionals`-style chain, or something else not yet isolated).
"""

import numpy as np

import om4.api as om
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective

from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.utils.testing import get_single_section_surface

surface = get_single_section_surface()

model = om.Group(
    subsystems={
        "surface": Geometry(surface=surface),
        "aero_point_0": AeroPoint(
            surfaces=[surface],
            promotes_inputs=["v", "Mach_number", "re", "rho", "cg", "alpha"],
        ),
    },
    connections=[
        om.Connection(src="surface.mesh", tgt="aero_point_0.surface.def_mesh"),
        om.Connection(src="surface.mesh", tgt="aero_point_0.aero_states.surface_def_mesh"),
    ],
    design_vars={"alpha": DesignVar(lower=0.0, upper=10.0)},
    constraints={"aero_point_0.CL": Constraint(equals=0.3)},
    objectives={"alpha": Objective()},
    optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-6, maxiter=1000),
)
prob = om.Problem(model=model)

prob.set_val("v", 1.0, units="m/s")
prob.set_val("alpha", 10.0, units="deg")
prob.set_val("Mach_number", 0.3)
prob.set_val("re", 1.0e5, units="1/m")
prob.set_val("rho", 0.38, units="kg/m**3")
prob.set_val("cg", np.zeros(3), units="m")

prob.run_optimizer()

cd_before = prob.get_val("aero_point_0.surface_perf.CD")[0]
cdi_source = prob.get_val("aero_point_0.surface_perf.coeffs.CDi")[0]
cdi_input = prob.get_val("aero_point_0.surface_perf.CD.CDi")[0]
print(f"CD before extra run_model = {cd_before}")
print(f"coeffs.CDi (source, fresh)     = {cdi_source}")
print(f"CD.CDi (TotalDrag's own input, stale) = {cdi_input}")

prob.run_model()
print(f"CD after extra run_model = {prob.get_val('aero_point_0.surface_perf.CD')[0]} (matches coeffs.CDi-derived value)")
