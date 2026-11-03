import om4.api as om
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
import unittest
import numpy as np

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.utils.testing import assert_opt_successful

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta

# Provide coordinates for a portion of an airfoil for the wingbox cross-section as an nparray with dtype=complex (to work with the complex-step approximation for derivatives).
# These should be for an airfoil with the chord scaled to 1.
# We use the 10% to 60% portion of the NASA SC2-0612 airfoil for this case
# We use the coordinates available from airfoiltools.com. Using such a large number of coordinates is not necessary.
# The first and last x-coordinates of the upper and lower skins must be the same

upper_x = np.array(
    [
        0.1,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.19,
        0.2,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.26,
        0.27,
        0.28,
        0.29,
        0.3,
        0.31,
        0.32,
        0.33,
        0.34,
        0.35,
        0.36,
        0.37,
        0.38,
        0.39,
        0.4,
        0.41,
        0.42,
        0.43,
        0.44,
        0.45,
        0.46,
        0.47,
        0.48,
        0.49,
        0.5,
        0.51,
        0.52,
        0.53,
        0.54,
        0.55,
        0.56,
        0.57,
        0.58,
        0.59,
        0.6,
    ],
)
lower_x = np.array(
    [
        0.1,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.19,
        0.2,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.26,
        0.27,
        0.28,
        0.29,
        0.3,
        0.31,
        0.32,
        0.33,
        0.34,
        0.35,
        0.36,
        0.37,
        0.38,
        0.39,
        0.4,
        0.41,
        0.42,
        0.43,
        0.44,
        0.45,
        0.46,
        0.47,
        0.48,
        0.49,
        0.5,
        0.51,
        0.52,
        0.53,
        0.54,
        0.55,
        0.56,
        0.57,
        0.58,
        0.59,
        0.6,
    ],
)
upper_y = np.array(
    [
        0.0447,
        0.046,
        0.0472,
        0.0484,
        0.0495,
        0.0505,
        0.0514,
        0.0523,
        0.0531,
        0.0538,
        0.0545,
        0.0551,
        0.0557,
        0.0563,
        0.0568,
        0.0573,
        0.0577,
        0.0581,
        0.0585,
        0.0588,
        0.0591,
        0.0593,
        0.0595,
        0.0597,
        0.0599,
        0.06,
        0.0601,
        0.0602,
        0.0602,
        0.0602,
        0.0602,
        0.0602,
        0.0601,
        0.06,
        0.0599,
        0.0598,
        0.0596,
        0.0594,
        0.0592,
        0.0589,
        0.0586,
        0.0583,
        0.058,
        0.0576,
        0.0572,
        0.0568,
        0.0563,
        0.0558,
        0.0553,
        0.0547,
        0.0541,
    ],
)
lower_y = np.array(
    [
        -0.0447,
        -0.046,
        -0.0473,
        -0.0485,
        -0.0496,
        -0.0506,
        -0.0515,
        -0.0524,
        -0.0532,
        -0.054,
        -0.0547,
        -0.0554,
        -0.056,
        -0.0565,
        -0.057,
        -0.0575,
        -0.0579,
        -0.0583,
        -0.0586,
        -0.0589,
        -0.0592,
        -0.0594,
        -0.0595,
        -0.0596,
        -0.0597,
        -0.0598,
        -0.0598,
        -0.0598,
        -0.0598,
        -0.0597,
        -0.0596,
        -0.0594,
        -0.0592,
        -0.0589,
        -0.0586,
        -0.0582,
        -0.0578,
        -0.0573,
        -0.0567,
        -0.0561,
        -0.0554,
        -0.0546,
        -0.0538,
        -0.0529,
        -0.0519,
        -0.0509,
        -0.0497,
        -0.0485,
        -0.0472,
        -0.0458,
        -0.0444,
    ],
)


class Test(unittest.TestCase):
    def test(self):
        """
        This is an opt problem that tests the fuel volume constraint with the wingbox model
        """

        # Create a dictionary to store options about the surface
        mesh_dict = {
            "num_y": 7,
            "num_x": 2,
            "wing_type": "CRM",
            "symmetry": True,
            "num_twist_cp": 6,
            "chord_cos_spacing": 0,
            "span_cos_spacing": 0,
        }

        mesh, twist_cp = generate_mesh(mesh_dict)

        surf_dict = {
            # Wing definition
            "name": "wing",  # name of the surface
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "S_ref_type": "wetted",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            "fem_model_type": "wingbox",
            "spar_thickness_cp": np.array([0.004, 0.005, 0.005, 0.008, 0.008, 0.01]),  # [m]
            "skin_thickness_cp": np.array([0.005, 0.01, 0.015, 0.020, 0.025, 0.026]),
            "twist_cp": np.array([4.0, 5.0, 8.0, 8.0, 8.0, 9.0]),
            "mesh": mesh,
            "data_x_upper": upper_x,
            "data_x_lower": lower_x,
            "data_y_upper": upper_y,
            "data_y_lower": lower_y,
            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            "CL0": 0.0,  # CL of the surface at alpha=0
            "CD0": 0.0078,  # CD of the surface at alpha=0
            # Airfoil properties for viscous drag calculation
            "k_lam": 0.05,  # percentage of chord with laminar
            # flow, used for viscous drag
            "t_over_c_cp": np.array([0.08, 0.08, 0.08, 0.10, 0.10, 0.08]),
            "original_wingbox_airfoil_t_over_c": 0.12,
            "c_max_t": 0.38,  # chordwise location of maximum thickness
            "with_viscous": True,
            "with_wave": False,  # if true, compute wave drag
            # Structural values are based on aluminum 7075
            "E": 73.1e9,  # [Pa] Young's modulus
            "G": (73.1e9 / 2 / 1.33),  # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
            "yield": 420.0e6,  # [Pa] yield stress
            "safety_factor": 1.5,  # safety factor
            "mrho": 2.78e3,  # [kg/m^3] material density
            "strength_factor_for_upper_skin": 1.0,  # the yield stress is multiplied by this factor for the upper skin
            # 'fem_origin' : 0.35,    # normalized chordwise location of the spar
            "wing_weight_ratio": 1.25,
            "struct_weight_relief": True,
            "distributed_fuel_weight": False,
            # Constraints
            "exact_failure_constraint": False,  # if false, use KS function
            "fuel_density": 803.0,  # [kg/m^3] fuel density (only needed if the fuel-in-wing volume constraint is used)
            "Wf_reserve": 15000.0,  # [kg] reserve fuel mass
        }

        surfaces = [surf_dict]

        # Create the problem and assign the model group
        subsystems = {}
        connections = []

        # Loop over each surface in the surfaces list
        for surface in surfaces:
            # Get the surface name and create a group to contain components
            # only for this surface
            name = surface["name"]

            aerostruct_group = AerostructGeometry(surface=surface)

            # Add tmp_group to the problem with the name of the surface.
            subsystems[name] = aerostruct_group

        # Loop through and add a certain number of aero points
        for i in range(1):
            point_name = "AS_point_{}".format(i)
            # Connect the parameters within the model for each aero point

            # Create the aero point group and add it to the model
            AS_point = AerostructPoint(
                surfaces=surfaces,
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

            subsystems[point_name] = AS_point

            for _surface in surfaces:
                com_name = point_name + "." + name + "_perf."
                connections.append(
                    om.Connection(
                        src=name + ".local_stiff_transformed",
                        tgt=point_name + ".coupled." + name + ".local_stiff_transformed",
                    )
                )
                connections.append(om.Connection(src=name + ".nodes", tgt=point_name + ".coupled." + name + ".nodes"))

                # Connect aerodyamic mesh to coupled group mesh
                connections.append(om.Connection(src=name + ".mesh", tgt=point_name + ".coupled." + name + ".mesh"))
                connections.append(
                    om.Connection(src=name + ".element_mass", tgt=point_name + ".coupled." + name + ".element_mass")
                )

                # Connect performance calculation variables
                connections.append(om.Connection(src=name + ".nodes", tgt=com_name + "nodes"))
                connections.append(
                    om.Connection(
                        src=name + ".cg_location", tgt=point_name + "." + "total_perf." + name + "_cg_location"
                    )
                )
                connections.append(
                    om.Connection(
                        src=name + ".structural_mass", tgt=point_name + "." + "total_perf." + name + "_structural_mass"
                    )
                )

                # Connect wingbox properties to von Mises stress calcs
                connections.append(om.Connection(src=name + ".Qz", tgt=com_name + "Qz"))
                connections.append(om.Connection(src=name + ".J", tgt=com_name + "J"))
                connections.append(om.Connection(src=name + ".A_enc", tgt=com_name + "A_enc"))
                connections.append(om.Connection(src=name + ".htop", tgt=com_name + "htop"))
                connections.append(om.Connection(src=name + ".hbottom", tgt=com_name + "hbottom"))
                connections.append(om.Connection(src=name + ".hfront", tgt=com_name + "hfront"))
                connections.append(om.Connection(src=name + ".hrear", tgt=com_name + "hrear"))

                connections.append(om.Connection(src=name + ".spar_thickness", tgt=com_name + "spar_thickness"))
                connections.append(om.Connection(src=name + ".t_over_c", tgt=com_name + "t_over_c"))

            # =======================================================================================
            # Here we add the fuel volume constraint component to the model
            # =======================================================================================
            subsystems["fuel_vol_delta"] = WingboxFuelVolDelta(surface=surface)
            connections.append(om.Connection(src="AS_point_0.fuelburn", tgt="fuel_vol_delta.fuelburn"))
            connections.append(om.Connection(src="wing.struct_setup.fuel_vols", tgt="fuel_vol_delta.fuel_vols"))
            # =======================================================================================
            # =======================================================================================

        # Set up the problem and add design variables, constraint, and objective
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
                },
                constraints={
                    "AS_point_0.CL": Constraint(equals=0.5),
                    "AS_point_0.wing_perf.failure": Constraint(upper=0.0),
                    # =======================================================================================
                    # Here we add the fuel volume constraint
                    # =======================================================================================
                    "fuel_vol_delta.fuel_vol_delta": Constraint(lower=0.0),
                    # =======================================================================================
                    # =======================================================================================
                },
                objectives={
                    "AS_point_0.fuelburn": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e-5)),
                },
                optimizer=om.ScipyMinimizeOptimizer(tol=1e-9, optimizer="SLSQP"),
            )
        )
        prob.set_val("v", 0.85 * 295.07, units="m/s")
        prob.set_val("alpha", 0.0, units="deg")
        prob.set_val("Mach_number", 0.85)
        prob.set_val("re", 0.348 * 295.07 * 0.85 * 1.0 / (1.43 * 1e-5), units="1/m")
        prob.set_val("rho", 0.348, units="kg/m**3")
        prob.set_val("CT", 0.53 / 3600, units="1/s")
        prob.set_val("R", 14.307e6, units="m")
        prob.set_val("W0", 148000 + surf_dict["Wf_reserve"], units="kg")
        prob.set_val("speed_of_sound", 295.07, units="m/s")
        prob.set_val("load_factor", 1.0)
        prob.set_val("empty_cg", np.zeros((3)), units="m")

        optResult = prob.run_optimizer()

        assert_opt_successful(self, optResult)
        assert_near_equal(prob.get_val("AS_point_0.fuelburn")[0], 76869.3858256513, 1e-4)
        assert_near_equal(prob.get_val("wing.structural_mass")[0], 14520.819967195273, 1e-4)
        assert_near_equal(prob.get_val("fuel_vol_delta.fuel_vol_delta")[0], 43.02081603846062, 1e-4)
        assert_near_equal(prob.get_val("AS_point_0.CL")[0], 0.5, 1e-5)


if __name__ == "__main__":
    unittest.main()
