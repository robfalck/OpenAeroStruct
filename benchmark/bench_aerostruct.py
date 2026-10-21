"""Time OM3 (from main) against the om4 port on the test_aerostruct_analysis model.

Run from the repo root: pixi run python benchmark/bench_aerostruct.py
"""

import argparse
import contextlib
import io
import sys
import time

import numpy as np
import openmdao.api as om3

import om4.api as om

sys.path.insert(0, "ai/tools")
from om3_crosscheck import om3_oas  # noqa: E402

om3_oas()
from oas_om3.integration.aerostruct_groups import AerostructGeometry as G3, AerostructPoint as P3  # noqa: E402
from openaerostruct.integration.aerostruct_groups import AerostructGeometry as G4, AerostructPoint as P4  # noqa: E402
from openaerostruct.meshing.mesh_generator import generate_mesh  # noqa: E402
from openaerostruct.utils.constants import grav_constant  # noqa: E402

FLOW = {
    "v": (248.136, "m/s"),
    "alpha": (5.0, "deg"),
    "Mach_number": (0.84, None),
    "re": (1e6, "1/m"),
    "rho": (0.38, "kg/m**3"),
    "CT": (grav_constant * 17e-6, "1/s"),
    "R": (11.165e6, "m"),
    "W0": (0.4 * 3e5, "kg"),
    "speed_of_sound": (295.4, "m/s"),
    "load_factor": (1.0, None),
    "empty_cg": (np.zeros(3), "m"),
}


def surf(nx, ny):
    mesh, tw = generate_mesh({"num_y": ny, "num_x": nx, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5})
    return {
        "name": "wing",
        "symmetry": True,
        "S_ref_type": "wetted",
        "fem_model_type": "tube",
        "thickness_cp": np.array([0.1, 0.2, 0.3]),
        "twist_cp": tw,
        "mesh": mesh,
        "CL0": 0.0,
        "CD0": 0.015,
        "k_lam": 0.05,
        "t_over_c_cp": np.array([0.15]),
        "c_max_t": 0.303,
        "with_viscous": True,
        "with_wave": False,
        "E": 70e9,
        "G": 30e9,
        "yield": 500e6,
        "safety_factor": 2.5,
        "mrho": 3e3,
        "fem_origin": 0.35,
        "wing_weight_ratio": 2.0,
        "struct_weight_relief": False,
        "distributed_fuel_weight": False,
        "exact_failure_constraint": False,
    }


CONN = [
    ("local_stiff_transformed", "coupled.wing.local_stiff_transformed"),
    ("nodes", "coupled.wing.nodes"),
    ("mesh", "coupled.wing.mesh"),
    ("radius", "wing_perf.radius"),
    ("thickness", "wing_perf.thickness"),
    ("nodes", "wing_perf.nodes"),
    ("cg_location", "total_perf.wing_cg_location"),
    ("structural_mass", "total_perf.wing_structural_mass"),
    ("t_over_c", "wing_perf.t_over_c"),
]


def build3(S):
    p = om3.Problem(reports=False)
    iv = om3.IndepVarComp()
    for k, (v, u) in FLOW.items():
        iv.add_output(k, val=v, units=u)
    p.model.add_subsystem("prob_vars", iv, promotes=["*"])
    p.model.add_subsystem("wing", G3(surface=dict(S)))
    p.model.add_subsystem("AS_point_0", P3(surfaces=[dict(S)]))
    for k in FLOW:
        p.model.connect(k, "AS_point_0." + k)
    for a, b in CONN:
        p.model.connect("wing." + a, "AS_point_0." + b)
    p.setup()
    return p


COLLAPSE_BY_GROUP = False
FEN = False


def build4(S):
    conns = [om.Connection(src="wing." + a, tgt="AS_point_0." + b) for a, b in CONN]
    pt = P4(surfaces=[dict(S)], promotes_inputs=list(FLOW), fen=FEN)
    overrides = {"*": {"collapse_by": "group"}} if COLLAPSE_BY_GROUP else {}
    p = om.Problem(
        model=om.Group(subsystems={"wing": G4(surface=dict(S)), "AS_point_0": pt}, connections=conns),
        model_overrides=overrides,
    )
    for k, (v, u) in FLOW.items():
        p.set_val(k, v, units=u)
    return p


# Captured before any monkeypatching below, so LINEAR_SOLVERS["direct"] always builds the
# real scipy-backed solver even after use_linear_solver() has pointed om.DirectSolver
# elsewhere for a previous solver in the same process (needed to run multiple
# --linear-solver values back to back).
_ScipyDirectSolver = om.DirectSolver

LINEAR_SOLVERS = {
    "direct": lambda: _ScipyDirectSolver(assembled_format="sparse"),
    "gs": lambda: om.LinearBlockGS(maxiter=100, atol=1e-10, rtol=1e-10),
    "gmres": lambda: om.ScipyGMRES(maxiter=200, atol=1e-10, rtol=1e-10),
    # PETSc backends need petsc4py (included in the default pixi environment). "petsc" (PETSc's
    # own native LU) and "superlu_dist" are left out: the former errors on this model's Jacobian
    # ("Matrix is missing diagonal entries"), the latter has a documented serial-transpose bug
    # (see PETScLU.solve's docstring) and compute_totals needs transpose solves in rev mode.
    "petsc-superlu": lambda: om.PETScDirectSolver(assembled_format="sparse", sparse_solver_name="superlu"),
    "petsc-klu": lambda: om.PETScDirectSolver(assembled_format="sparse", sparse_solver_name="klu"),
    "petsc-umfpack": lambda: om.PETScDirectSolver(assembled_format="sparse", sparse_solver_name="umfpack"),
    "petsc-mumps": lambda: om.PETScDirectSolver(assembled_format="sparse", sparse_solver_name="mumps"),
}


def use_linear_solver(name):
    """Make `AerostructPoint` build its coupled group with the named om4 linear solver (om3 is unaffected)."""
    # AerostructPoint hardcodes om.DirectSolver for the coupled group; swap it for this run.
    om.DirectSolver = lambda **kwargs: LINEAR_SOLVERS[name]()


def t(f):
    t0 = time.perf_counter()
    r = f()
    return time.perf_counter() - t0, r


def bench(build, S):
    ts, p = t(lambda: build(S))
    t1, _ = t(p.run_model)

    def rerun(a):
        p.set_val("alpha", a, units="deg")
        return t(p.run_model)[0]

    t2 = min(rerun(4.0), rerun(6.0))
    fb = float(np.ravel(p.get_val("AS_point_0.fuelburn"))[0])
    tt, J = t(lambda: p.compute_totals(of=["AS_point_0.fuelburn"], wrt=["wing.thickness_cp"]))
    dfb = float(np.ravel(np.asarray(J["AS_point_0.fuelburn", "wing.thickness_cp"]))[0])
    return ts, t1, t2, tt, fb, dfb


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--linear-solver",
        choices=sorted(LINEAR_SOLVERS),
        default="direct",
        help=(
            "om4 linear solver for the coupled group (default: direct, as in OM3). om3 always uses DirectSolver."
        ),
    )
    ap.add_argument("--no-om3", action="store_true", help="skip the OM3 reference runs")
    ap.add_argument(
        "--collapse-by-group",
        action="store_true",
        help='om4 only: set collapse_by="group" on every system via model_overrides (default: "cycle")',
    )
    ap.add_argument(
        "--fen",
        action="store_true",
        help="om4 only: wrap AS_point_0's coupled group as a FunctionalExecutionNode, hiding its "
        "internal states from the outer linear system.",
    )
    args = ap.parse_args()
    COLLAPSE_BY_GROUP = args.collapse_by_group
    FEN = args.fen
    use_linear_solver(args.linear_solver)

    builds = (("om4", build4),) if args.no_om3 else (("om3", build3), ("om4", build4))
    print(
        f"om4 coupled linear solver: {args.linear_solver}, collapse_by: {'group' if COLLAPSE_BY_GROUP else 'cycle'}, "
        f"fen: {FEN}"
    )
    for nx, ny in ((2, 5), (3, 21), (4, 41)):
        S = surf(nx, ny)
        res = {}
        for label, build in builds:
            with contextlib.redirect_stdout(io.StringIO()):
                res[label] = bench(build, S)
        print(f"nx={nx} ny={ny}")
        for k, v in res.items():
            print(
                f"  {k}: build/setup {v[0]:.2f}s  first run {v[1]:.2f}s  rerun {v[2]:.2f}s  totals {v[3]:.2f}s"
                f"  fuelburn {v[4]:.4f}  dfb/dt[0] {v[5]:.6f}"
            )
