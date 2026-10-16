import sys, numpy as np

sys.path.insert(0, "ai/tools")
import openmdao.api as om3, om4.api as om
from om3_crosscheck import om3_oas

om3_oas()
from oas_om3.functionals.total_performance import TotalPerformance as T3
from openaerostruct.functionals.total_performance import TotalPerformance as T4
from openaerostruct.meshing.mesh_generator import generate_mesh


def surfaces(sym):
    m1, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    m2 = generate_mesh({"num_y": 5, "num_x": 2, "wing_type": "rect", "symmetry": sym}) + np.array([30.0, 0, 2.0])
    base = dict(
        symmetry=sym,
        CL0=0.05,
        CD0=0.015,
        k_lam=0.05,
        c_max_t=0.303,
        t_over_c_cp=np.array([0.15]),
        fem_model_type="tube",
        S_ref_type="wetted",
        with_viscous=True,
        with_wave=False,
    )
    return [dict(base, name="wing", mesh=m1), dict(base, name="tail", mesh=m2)]


fails = 0
for sym in (True, False):
    for usr in (False, True):
        for fb in (True, False):
            S = surfaces(sym)
            p0 = om3.Problem(reports=False)
            p0.model.add_subsystem(
                "tp",
                T3(surfaces=[dict(s) for s in S], user_specified_Sref=usr, internally_connect_fuelburn=fb),
                promotes=["*"],
            )
            p0.setup()
            md = p0.model.get_io_metadata(iotypes="input", metadata_keys=["units", "shape"], return_rel_names=False)
            outs = set(p0.model.get_io_metadata(iotypes="output", return_rel_names=False))
            outs_prom = {
                m["prom_name"]
                for a, m in p0.model.get_io_metadata(iotypes="output", metadata_keys=[], return_rel_names=False).items()
                if not a.startswith("_auto_ivc")
            }
            ins = {}
            for a, m in md.items():
                pn = m["prom_name"]
                if pn in outs_prom or pn in ins:
                    continue
                ins[pn] = (m["shape"], m["units"])
            rng = np.random.default_rng(7)
            vals = {k: (0.5 + rng.random(sh), u) for k, (sh, u) in ins.items()}
            for k in ("S_ref_total",):
                if k in vals:
                    vals[k] = (np.array([40.0]), vals[k][1])
            # OM3
            p3 = om3.Problem(reports=False)
            ivc = om3.IndepVarComp()
            for k, (v, u) in vals.items():
                if "." not in k:
                    ivc.add_output(k, val=v, units=u)
            p3.model.add_subsystem("ivc", ivc, promotes=["*"])
            p3.model.add_subsystem(
                "tp",
                T3(surfaces=[dict(s) for s in S], user_specified_Sref=usr, internally_connect_fuelburn=fb),
                promotes=["*"],
            )
            p3.setup(force_alloc_complex=True)
            for k, (v, u) in vals.items():
                if "." in k:
                    p3.set_val(k, v, units=u)
            p3.run_model()
            p4 = om.Problem(
                model=om.Group(
                    subsystems={
                        "tp": T4(
                            surfaces=[dict(s) for s in S],
                            user_specified_Sref=usr,
                            internally_connect_fuelburn=fb,
                            promotes_inputs=["*"],
                            promotes_outputs=["*"],
                        )
                    }
                )
            )
            for k, (v, u) in vals.items():
                p4.set_val(k, v, units=u)
            p4.run_model()
            of = ["CL", "CD", "L", "D", "CM", "fuelburn", "total_weight", "L_equals_W", "cg"]
            vd = max(
                np.max(abs(np.ravel(p3.get_val(o)) - np.ravel(p4.get_val(o)))) / max(1e-12, np.max(abs(p3.get_val(o))))
                for o in of
            )
            wrt = ["wing_structural_mass", "v", "rho", "W0", "wing_CL"]
            J3 = p3.compute_totals(of=of, wrt=wrt, return_format="array")
            out = []
            for mode in ("fwd", "rev"):
                J4 = p4.compute_totals(of=of, wrt=wrt, mode=mode).todense()
                out.append(f"{mode} {np.max(abs(J3 - J4)) / np.max(abs(J3)):.1e}")
            bad = vd > 1e-9
            fails += bad
            print(f"sym={sym!s:5} usrS={usr!s:5} fb={fb!s:5} values rel diff {vd:.1e} totals {', '.join(out)}")
print("FAILURES", fails)
