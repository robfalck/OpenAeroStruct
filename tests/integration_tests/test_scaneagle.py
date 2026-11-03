import unittest

import numpy as np

from om4.utils.assert_utils import assert_near_equal
from openaerostruct.utils.testing import assert_opt_successful

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant

import om4.api as om
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective
from om4.transforms.transform import AffineTransform
from om4.derivatives.finite_difference import FiniteDifference


def setup_problem():
    """Build an om4 problem for the ScanEagle aerostructural optimization."""
    # Total number of nodes to use in the spanwise (num_y) and
    # chordwise (num_x) directions. Vary these to change the level of fidelity.
    num_y = 21
    num_x = 3

    # Create a mesh dictionary to feed to generate_mesh to actually create
    # the mesh array.
    mesh_dict = {
        "num_y": num_y,
        "num_x": num_x,
        "wing_type": "rect",
        "symmetry": True,
        "span_cos_spacing": 0.5,
        "span": 3.11,
        "root_chord": 0.3,
    }

    mesh = generate_mesh(mesh_dict)

    # Apply camber to the mesh
    camber = 1 - np.linspace(-1, 1, num_x) ** 2
    camber *= 0.3 * 0.05

    for ind_x in range(num_x):
        mesh[ind_x, :, 2] = camber[ind_x]

    # Introduce geometry manipulation variables to define the ScanEagle shape
    zshear_cp = np.zeros(10)
    zshear_cp[0] = 0.3

    xshear_cp = np.zeros(10)
    xshear_cp[0] = 0.15

    chord_cp = np.ones(10)
    chord_cp[0] = 0.5
    chord_cp[-1] = 1.5
    chord_cp[-2] = 1.3

    radius_cp = 0.01 * np.ones(10)

    # Define wing parameters
    surface = {
        # Wing definition
        "name": "wing",  # name of the surface
        "symmetry": True,  # if true, model one half of wing
        # reflected across the plane y = 0
        "S_ref_type": "wetted",  # how we compute the wing area,
        # can be 'wetted' or 'projected'
        "fem_model_type": "tube",
        "taper": 0.8,
        "zshear_cp": zshear_cp,
        "xshear_cp": xshear_cp,
        "chord_cp": chord_cp,
        "sweep": 20.0,
        "twist_cp": np.array([2.5, 2.5, 5.0]),  # np.zeros((3)),
        "thickness_cp": np.ones((3)) * 0.008,
        # Give OAS the radius and mesh from before
        "radius_cp": radius_cp,
        "mesh": mesh,
        # Aerodynamic performance of the lifting surface at
        # an angle of attack of 0 (alpha=0).
        # These CL0 and CD0 values are added to the CL and CD
        # obtained from aerodynamic analysis of the surface to get
        # the total CL and CD.
        # These CL0 and CD0 values do not vary wrt alpha.
        "CL0": 0.0,  # CL of the surface at alpha=0
        "CD0": 0.015,  # CD of the surface at alpha=0
        # Airfoil properties for viscous drag calculation
        "k_lam": 0.05,  # percentage of chord with laminar
        # flow, used for viscous drag
        "t_over_c_cp": np.array([0.12]),  # thickness over chord ratio
        "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
        # thickness
        "with_viscous": True,
        "with_wave": True,  # if true, compute wave drag
        # Material properties taken from http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
        "E": 85.0e9,
        "G": 25.0e9,
        "yield": 350.0e6,
        "safety_factor": 1,
        "mrho": 1.6e3,
        "fem_origin": 0.35,  # normalized chordwise location of the spar
        "wing_weight_ratio": 1.0,  # multiplicative factor on the computed structural weight
        "struct_weight_relief": True,  # True to add the weight of the structure to the loads on the structure
        "distributed_fuel_weight": False,
        # Constraints
        "exact_failure_constraint": False,  # if false, use KS function
    }

    # Add the AerostructGeometry group, which computes all the intermediary
    # parameters for the aero and structural analyses, like the structural
    # stiffness matrix and some aerodynamic geometry arrays
    aerostruct_group = AerostructGeometry(surface=surface)

    name = "wing"
    point_name = "AS_point_0"

    # Create the aerostruct point group. This contains all the actual
    # aerostructural analyses.
    AS_point = AerostructPoint(
        surfaces=[surface],
        promotes_inputs=[
            "v",
            "alpha",
            "Mach_number",
            "re",
            "rho",
            "CT",
            "R",
            "W0",
            "speed_of_sound",
            "empty_cg",
            "load_factor",
        ],
    )

    # Issue quite a few connections within the model to make sure all of the
    # parameters are connected correctly.
    com_name = point_name + "." + name + "_perf"
    connections = [
        om.Connection(
            src=name + ".local_stiff_transformed", tgt=point_name + ".coupled." + name + ".local_stiff_transformed"
        ),
        om.Connection(src=name + ".nodes", tgt=point_name + ".coupled." + name + ".nodes"),
        # Connect aerodynamic mesh to coupled group mesh
        om.Connection(src=name + ".mesh", tgt=point_name + ".coupled." + name + ".mesh"),
        # Connect performance calculation variables
        om.Connection(src=name + ".radius", tgt=com_name + ".radius"),
        om.Connection(src=name + ".thickness", tgt=com_name + ".thickness"),
        om.Connection(src=name + ".nodes", tgt=com_name + ".nodes"),
        om.Connection(src=name + ".cg_location", tgt=point_name + "." + "total_perf." + name + "_cg_location"),
        om.Connection(
            src=name + ".structural_mass", tgt=point_name + "." + "total_perf." + name + "_structural_mass"
        ),
        om.Connection(src=name + ".t_over_c", tgt=com_name + ".t_over_c"),
    ]

    prob = om.Problem(
        model=om.Group(
            subsystems={name: aerostruct_group, point_name: AS_point},
            connections=connections,
            design_vars={
                # Lock the root of the wing at a 5 deg twist
                "wing.twist_cp": DesignVar(
                    lower=np.array([-5.0, -5.0, 5.0]), upper=np.array([15.0, 15.0, 5.0])
                ),
                "wing.thickness_cp": DesignVar(
                    lower=0.001, upper=0.01, transform=AffineTransform.from_scaler_adder(scaler=1e3)
                ),
                "wing.sweep": DesignVar(lower=10.0, upper=30.0),
                "alpha": DesignVar(lower=-10.0, upper=10.0),
            },
            constraints={
                # Make sure the spar doesn't fail, we meet the lift needs, and the aircraft
                # is trimmed through CM=0.
                "AS_point_0.wing_perf.failure": Constraint(upper=0.0),
                "AS_point_0.wing_perf.thickness_intersects": Constraint(upper=0.0),
                "AS_point_0.L_equals_W": Constraint(equals=0.0),
                # Instead of using an equality constraint here, we have to give it a little
                # wiggle room to make SLSQP work correctly.
                "AS_point_0.CM": Constraint(lower=-0.001, upper=0.001),
            },
            objectives={
                # We're trying to minimize fuel burn
                "AS_point_0.fuelburn": Objective(transform=AffineTransform.from_scaler_adder(scaler=0.1)),
            },
            optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-7),
        )
    )

    prob.set_val("v", 22.876, units="m/s")
    prob.set_val("alpha", 5.0, units="deg")
    prob.set_val("Mach_number", 0.071)
    prob.set_val("re", 1.0e6, units="1/m")
    prob.set_val("rho", 0.770816, units="kg/m**3")
    prob.set_val("CT", grav_constant * 8.6e-6, units="1/s")
    prob.set_val("R", 1800e3, units="m")
    prob.set_val("W0", 10.0, units="kg")
    prob.set_val("speed_of_sound", 322.2, units="m/s")
    prob.set_val("load_factor", 1.0)
    prob.set_val("empty_cg", np.array([0.35, 0.0, 0.0]), units="m")

    return prob


class Test(unittest.TestCase):
    def test_opt(self):
        prob = setup_problem()

        optResult = prob.run_optimizer()
        assert_opt_successful(self, optResult)

        assert_near_equal(prob.get_val("AS_point_0.fuelburn")[0], 4.518756027353296, 1e-5)
        assert_near_equal(prob.get_val("wing.twist_cp"), np.array([2.74343152, 12.2239185, 5.0]), 1e-5)
        assert_near_equal(prob.get_val("wing.sweep")[0], 18.908583449616266, 1e-5)
        assert_near_equal(prob.get_val("alpha")[0], 1.475153325082263, 1e-5)

    def test_totals(self):
        prob = setup_problem()
        prob.run_model()

        totals = prob.check_totals(
            of=[
                "AS_point_0.fuelburn",
                "AS_point_0.wing_perf.failure",
                "AS_point_0.wing_perf.thickness_intersects",
                "AS_point_0.L_equals_W",
                "AS_point_0.CM",
            ],
            wrt=["wing.twist_cp", "wing.thickness_cp", "wing.sweep", "alpha"],
            method=FiniteDifference.central(step=1e-3, step_calc="rel"),
            only_incorrect=True,
            atol=1e-5,
            rtol=1e-5,
        )
        assert not totals, totals


if __name__ == "__main__":
    unittest.main()
