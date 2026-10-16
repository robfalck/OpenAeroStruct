import om4.api as om
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
import unittest
import numpy as np

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.structures.struct_groups import SpatialBeamAlone


class Test(unittest.TestCase):
    def test(self):
        # Create a dictionary to store options about the surface
        mesh_dict = {
            "num_y": 5,
            "num_x": 3,
            "wing_type": "rect",
            "symmetry": True,
            "span_cos_spacing": 1.0,
            "span": 10,
            "chord": 1,
        }

        mesh = generate_mesh(mesh_dict)

        surf_dict = {
            # Wing definition
            "name": "wing",  # name of the surface
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "fem_model_type": "tube",
            "mesh": mesh,
            # Structural values are based on aluminum 7075
            "E": 70.0e9,  # [Pa] Young's modulus of the spar
            "G": 30.0e9,  # [Pa] shear modulus of the spar
            "yield": 500.0e6,
            "safety_factor": 2.5,  # [Pa] yield stress divided by 2.5 for limiting case
            "mrho": 3.0e3,  # [kg/m^3] material density
            "fem_origin": 0.35,  # normalized chordwise location of the spar
            "t_over_c_cp": np.array([0.15]),  # maximum airfoil thickness
            "thickness_cp": np.ones((3)) * 0.0075,
            "wing_weight_ratio": 1.0,
            "struct_weight_relief": False,  # True to add the weight of the structure to the loads on the structure
            "distributed_fuel_weight": False,
            "exact_failure_constraint": False,
        }

        # Create the problem and assign the model group
        subsystems = {}
        connections = []

        ny = surf_dict["mesh"].shape[1]

        loads = np.zeros((ny, 6))
        loads[0, 2] = 1e4

        struct_group = SpatialBeamAlone(surface=surf_dict)

        subsystems[surf_dict["name"]] = struct_group

        # Setup problem and add design variables, constraint, and objective

        # Add design variables, constraisnt, and objective on the problem

        # Set up the problem
        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={
                    "wing.thickness_cp": DesignVar(
                        lower=0.001, upper=0.25, transform=AffineTransform.from_scaler_adder(scaler=1e2)
                    )
                },
                constraints={"wing.failure": Constraint(upper=0.0), "wing.thickness_intersects": Constraint(upper=0.0)},
                objectives={
                    "wing.structural_mass": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e-4))
                },
                optimizer=om.ScipyMinimizeOptimizer(tol=1e-9, optimizer="SLSQP"),
            )
        )
        prob.set_val(surf_dict["name"] + ".loads", loads, units="N")

        prob.run_optimizer()

        assert_near_equal(prob.get_val("wing.structural_mass")[0], 116.702381059, 1e-4)


if __name__ == "__main__":
    unittest.main()
