"""Parallel version of the multipoint aerostructural optimization example (Q400).

Only `run_model` + `check_totals` are exercised here (as in the original): the
design_vars/constraints/objective declarations below exist so `parallel_deriv_color`
can group them for parallel reverse-mode linear solves, not to actually drive an
optimizer.
"""

import unittest

import numpy as np

import om4.api as om
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
from om4.utils.mpi import MPI, MPI_and_PETSc
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta

# Provide coordinates for a portion of an airfoil for the wingbox cross-section (chord scaled to
# 1). We use the 10% to 60% portion of the NASA SC2-0612 airfoil, from airfoiltools.com.
# The first and last x-coordinates of the upper and lower skins must be the same.

# fmt: off
upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])
lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6])
upper_y = np.array([ 0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538, 0.0545,  0.0551,  0.0557, 0.0563,  0.0568, 0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,  0.0599,  0.06,    0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,    0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,   0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541])  # noqa: E201, E241
lower_y = np.array([-0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532, -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583, -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -0.0597, -0.0598, -0.0598, -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582, -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519, -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444])
# fmt: on


@unittest.skipUnless(MPI_and_PETSc, "MPI and PETSc required")
class Test(unittest.TestCase):
    N_PROCS = 2

    @unittest.skip(
        "Blocked on OM4_NEEDS B-020: under real 2-rank MPI, Problem construction raises a "
        "false 'already exists in OrderedVars' RuntimeError for this surface's two "
        "holder+spline control-point siblings (twist_bsp, t_over_c_bsp)."
    )
    def test_multipoint_MPI(self):
        span = 28.42  # wing span in m
        root_chord = 3.34  # root chord in m

        nx = 3  # number of chordwise nodal points (should be odd)
        ny = 11  # number of spanwise nodal points for the half-span

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 1] = np.linspace(-span / 2, 0, ny)
        mesh[0, :, 0] = 0.34 * root_chord * np.linspace(1.0, 0.0, ny)
        mesh[2, :, 0] = root_chord * (np.linspace(0.4, 1.0, ny) + 0.34 * np.linspace(1.0, 0.0, ny))
        mesh[1, :, 0] = (mesh[2, :, 0] + mesh[0, :, 0]) / 2

        Wf_reserve = 500.0  # [kg] reserve fuel mass

        surf_dict = {
            "name": "wing",
            "symmetry": True,
            "S_ref_type": "wetted",
            "mesh": mesh,
            "twist_cp": np.array([6.0, 7.0, 7.0, 7.0]),
            "fem_model_type": "wingbox",
            "data_x_upper": upper_x,
            "data_x_lower": lower_x,
            "data_y_upper": upper_y,
            "data_y_lower": lower_y,
            "spar_thickness_cp": np.array([0.004, 0.004, 0.004, 0.004]),  # [m]
            "skin_thickness_cp": np.array([0.003, 0.006, 0.010, 0.012]),  # [m]
            "original_wingbox_airfoil_t_over_c": 0.12,
            "CL0": 0.0,
            "CD0": 0.0142,
            "with_viscous": True,
            "with_wave": True,
            "k_lam": 0.05,
            "c_max_t": 0.38,
            "t_over_c_cp": np.array([0.1, 0.1, 0.15, 0.15]),
            "E": 73.1e9,  # [Pa] Young's modulus
            "G": (73.1e9 / 2 / 1.33),  # [Pa] shear modulus
            "yield": 420.0e6,  # [Pa] yield stress
            "safety_factor": 1.5,
            "mrho": 2.78e3,  # [kg/m^3] material density
            "strength_factor_for_upper_skin": 1.0,
            "wing_weight_ratio": 1.25,
            "exact_failure_constraint": False,
            "struct_weight_relief": True,
            "distributed_fuel_weight": True,
            "fuel_density": 803.0,  # [kg/m^3]
            "Wf_reserve": Wf_reserve,
        }

        surfaces = [surf_dict]
        name = surf_dict["name"]

        subsystems = {name: AerostructGeometry(surface=surf_dict)}
        connections = []

        # om4 has no IndepVarComp; this identity component stands in for one, giving
        # "fuel_mass" a real output other subsystems can connect to while "fuel_mass_dv"
        # (its input) is the free value used as the design variable.
        fuel_mass_comp = om.ExecComp(
            exprs=["fuel_mass = fuel_mass_dv"],
            inputs={"fuel_mass_dv": {"val": 3000.0, "units": "kg"}},
            outputs={"fuel_mass": {"val": 3000.0, "units": "kg"}},
            promotes_inputs=["fuel_mass_dv"],
            promotes_outputs=["fuel_mass"],
        )
        subsystems["fuel_mass_comp"] = fuel_mass_comp

        # Both AS points share the same reserve fuel condition; everything else (v, Mach_number,
        # re, rho, speed_of_sound, load_factor, alpha) differs per point and is set directly below.
        shared_point_promotes = ["CT", "R", "W0", "empty_cg"]

        AS_points = {}
        for i in range(2):
            point_name = f"AS_point_{i}"
            AS_points[point_name] = AerostructPoint(
                surfaces=surfaces, internally_connect_fuelburn=False, promotes_inputs=shared_point_promotes
            )

        # The two AS points are the only things that need to run in parallel; everything else
        # (geometry, fuel bookkeeping) is identical/cheap work duplicated on every rank.
        # promotes_inputs/promotes_outputs=["*"] here plays the same role as OM3's
        # add_subsystem("parallel", ParallelGroup(), promotes=["*"], ...): it makes this
        # grouping transparent to naming, so "AS_point_0.<var>" resolves the same whether or
        # not it happens to sit inside a parallel split.
        subsystems["parallel"] = om.Group(
            parallel=True,
            subsystems=AS_points,
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        for i in range(2):
            point_name = f"AS_point_{i}"
            com_name = point_name + "." + name + "_perf."

            connections += [
                om.Connection(src="fuel_mass", tgt=point_name + ".total_perf.L_equals_W.fuelburn"),
                om.Connection(src="fuel_mass", tgt=point_name + ".total_perf.CG.fuelburn"),
                om.Connection(src="wing.struct_setup.fuel_vols", tgt=point_name + ".coupled.wing.struct_states.fuel_vols"),
                om.Connection(src="fuel_mass", tgt=point_name + ".coupled.wing.struct_states.fuel_mass"),
                om.Connection(
                    src=name + ".local_stiff_transformed",
                    tgt=point_name + ".coupled." + name + ".local_stiff_transformed",
                ),
                om.Connection(src=name + ".nodes", tgt=point_name + ".coupled." + name + ".nodes"),
                om.Connection(src=name + ".mesh", tgt=point_name + ".coupled." + name + ".mesh"),
                om.Connection(src=name + ".element_mass", tgt=point_name + ".coupled." + name + ".element_mass"),
                om.Connection(src=name + ".nodes", tgt=com_name + "nodes"),
                om.Connection(src=name + ".cg_location", tgt=point_name + "." + "total_perf." + name + "_cg_location"),
                om.Connection(
                    src=name + ".structural_mass", tgt=point_name + "." + "total_perf." + name + "_structural_mass"
                ),
                om.Connection(src=name + ".Qz", tgt=com_name + "Qz"),
                om.Connection(src=name + ".J", tgt=com_name + "J"),
                om.Connection(src=name + ".A_enc", tgt=com_name + "A_enc"),
                om.Connection(src=name + ".htop", tgt=com_name + "htop"),
                om.Connection(src=name + ".hbottom", tgt=com_name + "hbottom"),
                om.Connection(src=name + ".hfront", tgt=com_name + "hfront"),
                om.Connection(src=name + ".hrear", tgt=com_name + "hrear"),
                om.Connection(src=name + ".spar_thickness", tgt=com_name + "spar_thickness"),
                om.Connection(src=name + ".t_over_c", tgt=com_name + "t_over_c"),
            ]

        fuel_vol_delta = WingboxFuelVolDelta(surface=surf_dict)
        subsystems["fuel_vol_delta"] = fuel_vol_delta
        connections += [
            om.Connection(src="wing.struct_setup.fuel_vols", tgt="fuel_vol_delta.fuel_vols"),
            om.Connection(src="AS_point_0.fuelburn", tgt="fuel_vol_delta.fuelburn"),
        ]

        fuel_diff = om.ExecComp(
            exprs=["fuel_diff = (fuel_mass - fuelburn) / fuelburn"],
            inputs={"fuel_mass": {"val": 1.0, "units": "kg"}, "fuelburn": {"val": 1.0, "units": "kg"}},
            promotes_inputs=["fuel_mass"],
            promotes_outputs=["fuel_diff"],
        )
        subsystems["fuel_diff"] = fuel_diff
        connections.append(om.Connection(src="AS_point_0.fuelburn", tgt="fuel_diff.fuelburn"))

        # Dummy constraint on the sum of fuel burns from AS_point_0 and AS_point_1 (doesn't make
        # physical sense; it exists to show that linear solves for point_0 and point_1 can be
        # parallelized even for a response that depends on both).
        fuel_sum_comp = om.ExecComp(
            exprs=["fuel_sum = fuelburn1 + fuelburn2"],
            inputs={"fuelburn1": {"val": 1.0, "units": "kg"}, "fuelburn2": {"val": 1.0, "units": "kg"}},
            outputs={"fuel_sum": {"val": 1.0, "units": "kg"}},
            promotes_outputs=["fuel_sum"],
        )
        subsystems["fuel_sum"] = fuel_sum_comp
        connections += [
            om.Connection(src="AS_point_0.fuelburn", tgt="fuel_sum.fuelburn1"),
            om.Connection(src="AS_point_1.fuelburn", tgt="fuel_sum.fuelburn2"),
        ]

        # We have 6 responses (objective + constraints), which would require 6 linear solves in
        # series for reverse-mode derivatives. 4 depend only on AS_point_0, and 2 only on
        # AS_point_1, so they form 2 pairs that can be solved in parallel by sharing a
        # parallel_deriv_color. color1 pairs AS_point_0.fuelburn with AS_point_1.L_equals_W;
        # color2 pairs AS_point_0.CL with AS_point_1.wing_perf.failure.
        if MPI.COMM_WORLD.size == 1:
            color1 = None
            color2 = None
        else:
            color1 = "parcon1"
            color2 = "parcon2"

        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={
                    "wing.twist_cp": DesignVar(
                        lower=-15.0, upper=15.0, transform=AffineTransform.from_scaler_adder(scaler=0.1)
                    ),
                    "wing.spar_thickness_cp": DesignVar(
                        lower=0.003, upper=0.1, transform=AffineTransform.from_scaler_adder(scaler=1e2)
                    ),
                    "wing.skin_thickness_cp": DesignVar(
                        lower=0.003, upper=0.1, transform=AffineTransform.from_scaler_adder(scaler=1e2)
                    ),
                    "wing.geometry.t_over_c_cp": DesignVar(
                        lower=0.07, upper=0.2, transform=AffineTransform.from_scaler_adder(scaler=10.0)
                    ),
                    "fuel_mass_dv": DesignVar(
                        lower=0.0, upper=2e5, transform=AffineTransform.from_scaler_adder(scaler=1e-5)
                    ),
                    "AS_point_1.alpha": DesignVar(lower=-15.0, upper=15.0),
                },
                constraints={
                    "AS_point_0.CL": Constraint(equals=0.6, parallel_deriv_color=color2),
                    "AS_point_1.L_equals_W": Constraint(equals=0.0, parallel_deriv_color=color1),
                    "AS_point_1.wing_perf.failure": Constraint(upper=0.0, parallel_deriv_color=color2),
                    "fuel_vol_delta.fuel_vol_delta": Constraint(lower=0.0),
                    "fuel_diff": Constraint(equals=0.0),
                    "fuel_sum": Constraint(
                        lower=0.0, upper=1e10, transform=AffineTransform.from_scaler_adder(scaler=1e-5)
                    ),
                },
                objectives={
                    "AS_point_0.fuelburn": Objective(
                        transform=AffineTransform.from_scaler_adder(scaler=1e-5), parallel_deriv_color=color1
                    )
                },
            )
        )

        prob.set_val("fuel_mass_dv", 3000.0, units="kg")

        # v, Mach_number, re, rho, speed_of_sound, load_factor differ between the two points, so
        # each is set directly rather than through a shared source.
        prob.set_val("AS_point_0.v", 0.5 * 310.95, units="m/s")
        prob.set_val("AS_point_1.v", 0.3 * 340.294, units="m/s")
        prob.set_val("AS_point_0.Mach_number", 0.5)
        prob.set_val("AS_point_1.Mach_number", 0.3)
        prob.set_val("AS_point_0.re", 0.569 * 310.95 * 0.5 * 1.0 / (1.56 * 1e-5), units="1/m")
        prob.set_val("AS_point_1.re", 1.225 * 340.294 * 0.3 * 1.0 / (1.81206 * 1e-5), units="1/m")
        prob.set_val("AS_point_0.rho", 0.569, units="kg/m**3")
        prob.set_val("AS_point_1.rho", 1.225, units="kg/m**3")
        prob.set_val("AS_point_0.speed_of_sound", 310.95, units="m/s")
        prob.set_val("AS_point_1.speed_of_sound", 340.294, units="m/s")
        prob.set_val("AS_point_0.load_factor", 1.0)
        prob.set_val("AS_point_1.load_factor", 2.5)
        # struct_weight_relief/distributed_fuel_weight loads inside the coupled group use their
        # own load_factor input, which isn't promoted past the coupled group's own boundary
        # (TD-107, ai/TECH_DEBT.md).
        prob.set_val("AS_point_0.coupled.load_factor", 1.0)
        prob.set_val("AS_point_1.coupled.load_factor", 2.5)

        prob.set_val("AS_point_0.alpha", 0.0, units="deg")
        # AS_point_1.alpha is a design variable above (OM3 called this DV "alpha_maneuver" and
        # connected it in; om4 can drive the AerostructPoint's own promoted input directly).

        prob.set_val("CT", 0.43 / 3600, units="1/s")
        prob.set_val("R", 2e6, units="m")
        prob.set_val("W0", 25400 + Wf_reserve, units="kg")
        prob.set_val("empty_cg", np.zeros(3), units="m")

        # coupled.linear_solver keeps AerostructPoint's own default (DirectSolver): om4 builds
        # Group trees declaratively and identically on every rank (no post-setup, per-rank
        # solver assignment as in OM3), and Problem(model_overrides=...) can't reach a system
        # nested inside AerostructPoint's auto-generated subsystems (OM4_NEEDS B-019).
        prob.run_model()

        totals = prob.check_totals(
            of=["fuel_sum"],
            wrt=["wing.spar_thickness_cp"],
            only_incorrect=True,
            atol=1.0,
            rtol=1e-4,
        )

        assert_near_equal(MPI.COMM_WORLD.size, 2, 1e-8)
        assert_near_equal(prob.get_val("fuel_sum", units="kg")[0], 5646.50827504, 1e-5)
        assert not totals, totals


if __name__ == "__main__":
    unittest.main()
