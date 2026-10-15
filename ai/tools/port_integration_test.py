"""Mechanically port an OM3 OAS analysis-style integration test to om4.

Usage: pixi run python ai/tools/port_integration_test.py tests/integration_tests/test_x.py [...]

Handles the standard template used by the aero/aerostruct analysis tests:
  * ``om.IndepVarComp`` flow variables become ``prob.set_val`` calls,
  * ``prob.model.add_subsystem`` / ``prob.model.connect`` become a ``subsystems`` dict and
    ``connections`` list; a flow variable connected to ``point_name + ".<same name>"`` is promoted
    on the point group instead,
  * ``prob.setup()`` is replaced by the om4 ``Problem`` construction,
  * ``prob.run_driver()`` becomes ``prob.run_model()`` (or ``run_optimizer()`` with a ScipyOptimizeDriver), ``prob["x"]`` becomes ``prob.get_val("x")``.

Files that use a driver, design variables or other OM3 APIs are reported and left untouched
(they need a hand port). Always review the diff and run the test.
"""

import ast
import sys


def _src(lines_cum, text, node):
    a = lines_cum[node.lineno - 1] + node.col_offset
    b = lines_cum[node.end_lineno - 1] + node.end_col_offset
    return text[a:b], a, b


def port(path):
    text = open(path).read()
    # A driver and design variables that are declared but never run (no run_driver) are dropped.
    # With run_driver, ScipyOptimizeDriver is ported to ScipyMinimizeOptimizer on the root Group.
    use_opt = "run_driver" in text
    if use_opt and any(k in text for k in ("pyoptsparse", "pyOptSparse", "recording_options", "add_recorder")):
        return "needs hand port (driver/optimization)"

    tree = ast.parse(text)
    lines = text.split("\n")
    lines_cum = [0]
    for ln in lines:
        lines_cum.append(lines_cum[-1] + len(ln.encode()) + 1)
    # ast col offsets are utf-8 bytes; work on bytes to keep offsets exact
    btext = text.encode()

    def seg(node):
        a = lines_cum[node.lineno - 1] + node.col_offset
        b = lines_cum[node.end_lineno - 1] + node.end_col_offset
        return btext[a:b].decode(), a, b

    stmts = [n for n in ast.walk(tree) if isinstance(n, (ast.Expr, ast.Assign))]
    stmts.sort(key=lambda n: (n.lineno, n.col_offset))

    def is_model_call(node, meth):
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == meth
            and ast.unparse(node.value.func.value) == "prob.model"
        )

    # pass 1: indep vars, point-name -> var, flow promotes
    indep = []  # (name, val_src, units_src)
    indep_names = set()
    var_of_name = {}  # source of subsystem name expr -> var name
    for st in stmts:
        if (
            isinstance(st, ast.Expr)
            and isinstance(st.value, ast.Call)
            and isinstance(st.value.func, ast.Attribute)
            and st.value.func.attr == "add_output"
            and ast.unparse(st.value.func.value) == "indep_var_comp"
        ):
            call = st.value
            name = call.args[0].value
            kw = {k.arg: seg(k.value)[0] for k in call.keywords}
            indep.append((name, kw["val"], kw.get("units")))
            indep_names.add(name)
        if (
            is_model_call(st, "add_subsystem")
            and len(st.value.args) >= 2
            and not isinstance(st.value.args[1], ast.Call)
        ):
            nm = seg(st.value.args[0])[0]
            if isinstance(st.value.args[1], ast.Name):
                var_of_name[nm] = st.value.args[1].id
    opt_opts = {}  # ScipyOptimizeDriver option -> source
    opt_blocks = {"design_vars": [], "constraints": [], "objectives": []}
    tot_args = {}
    for st in stmts:
        if (
            isinstance(st, ast.Assign)
            and isinstance(st.targets[0], ast.Subscript)
            and ast.unparse(st.targets[0].value) == "prob.driver.options"
        ):
            opt_opts[st.targets[0].slice.value] = seg(st.value)[0]
        if isinstance(st, ast.Assign) and ast.unparse(st.targets[0]) == "prob.driver":
            if not ast.unparse(st.value).startswith("om.ScipyOptimizeDriver("):
                return "needs hand port (driver is not ScipyOptimizeDriver)"
            for k in st.value.keywords:
                opt_opts[k.arg] = seg(k.value)[0]
        for meth, blk, cls in (
            ("add_design_var", "design_vars", "DesignVar"),
            ("add_constraint", "constraints", "Constraint"),
            ("add_objective", "objectives", "Objective"),
        ):
            if is_model_call(st, meth) and use_opt:
                c = st.value
                kw = {k.arg: seg(k.value)[0] for k in c.keywords}
                args = []
                tf = {}
                for k, v in kw.items():
                    if k in ("scaler", "adder", "ref", "ref0"):
                        tf[k] = v
                    elif k in ("lower", "upper", "equals", "units", "indices"):
                        args.append(f"{k}={v}")
                    else:
                        return f"needs hand port ({meth} kwarg {k})"
                if tf:
                    if {"ref", "ref0"} & tf.keys() and {"scaler", "adder"} & tf.keys():
                        return "needs hand port (ref and scaler)"
                    a = ", ".join(f"{k}={v}" for k, v in tf.items())
                    if {"ref", "ref0"} & tf.keys():
                        args.append(f"transform=AffineTransform.from_ref_ref0({a})")
                    else:
                        args.append(f"transform=AffineTransform.from_scaler_adder({a})")
                opt_blocks[blk].append(f"{seg(c.args[0])[0]}: {cls}({', '.join(args)})")
        if (
            isinstance(st, ast.Expr)
            and isinstance(st.value, ast.Call)
            and ast.unparse(st.value.func) == "assert_check_totals"
        ):
            tot_args = {k.arg: seg(k.value)[0] for k in st.value.keywords}
    unknown = set(opt_opts) - {"optimizer", "tol", "maxiter", "disp", "debug_print"}
    if unknown:
        return f"needs hand port (driver options {sorted(unknown)})"
    promote_of_var = {}
    drop = set()
    for st in stmts:
        if is_model_call(st, "connect"):
            a, b = st.value.args[:2]
            if (
                isinstance(a, ast.Constant)
                and a.value in indep_names
                and isinstance(b, ast.BinOp)
                and isinstance(b.right, ast.Constant)
                and b.right.value == "." + a.value
            ):
                var = var_of_name.get(seg(b.left)[0])
                if var:
                    promote_of_var.setdefault(var, []).append(a.value)
                    drop.add(id(st))
    # promotes given on add_subsystem
    for st in stmts:
        if is_model_call(st, "add_subsystem"):
            args = st.value.args
            if len(args) >= 2 and isinstance(args[1], ast.Name):
                for k in st.value.keywords:
                    if k.arg == "promotes_inputs":
                        promote_of_var.setdefault(args[1].id, [])
                        for e in k.value.elts:
                            if e.value not in promote_of_var[args[1].id]:
                                promote_of_var[args[1].id].append(e.value)

    edits = []
    has_setup = False
    for st in stmts:
        s, a, b = seg(st)
        indent = " " * st.col_offset
        # fixed-width ' ' only valid when indentation is spaces
        if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call):
            tgt = st.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in promote_of_var and ast.unparse(st.value.func) in ("AeroPoint",):
                # append promotes_inputs kwarg
                call_src, ca, cb = seg(st.value)
                new = call_src[:-1] + ", promotes_inputs=" + repr(promote_of_var[tgt.id]).replace("'", '"') + ")"
                edits.append((ca, cb, new))
                continue
            if isinstance(tgt, ast.Name) and tgt.id == "prob" and ast.unparse(st.value) == "om.Problem()":
                edits.append((a, b, "subsystems = {}\n" + indent + "connections = []"))
                continue
            if (
                isinstance(tgt, ast.Name)
                and ast.unparse(st.value) == "om.IndepVarComp()"
                and tgt.id == "indep_var_comp"
            ):
                edits.append((a, b, "pass  # DELETE"))
                continue
        if ast.unparse(st).startswith(("prob.driver", "prob.model.add_design_var", "prob.model.add_constraint")) or (
            ast.unparse(st).startswith("prob.model.add_objective")
        ):
            edits.append((a, b, "pass  # DELETE"))
            continue
        if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Subscript):
            sub = st.targets[0]
            if isinstance(sub.value, ast.Name) and sub.value.id == "prob":
                edits.append((a, b, f"prob.set_val({seg(sub.slice)[0]}, {seg(st.value)[0]})"))
                continue
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            c = st.value
            f = ast.unparse(c.func)
            if f == "indep_var_comp.add_output":
                edits.append((a, b, "pass  # DELETE"))
            elif is_model_call(st, "add_subsystem"):
                if len(c.args) >= 2 and isinstance(c.args[0], ast.Constant) and c.args[0].value == "prob_vars":
                    edits.append((a, b, "pass  # DELETE"))
                else:
                    kw = {k.arg for k in c.keywords}
                    if kw - {"promotes_inputs", "promotes_outputs"}:
                        return f"needs hand port (add_subsystem kwargs {kw})"
                    if c.args and len(c.args) == 2 and isinstance(c.args[1], ast.Name):
                        edits.append((a, b, f"subsystems[{seg(c.args[0])[0]}] = {c.args[1].id}"))
                    else:
                        return "needs hand port (add_subsystem form)"
            elif is_model_call(st, "connect"):
                if id(st) in drop:
                    edits.append((a, b, "pass  # DELETE"))
                else:
                    x, y = seg(c.args[0])[0], seg(c.args[1])[0]
                    edits.append((a, b, f"connections.append(om.Connection(src={x}, tgt={y}))"))
            elif f == "prob.setup":
                has_setup = True
                gkw = ""
                if use_opt:
                    ok = {k: v for k, v in opt_opts.items() if k in ("optimizer", "tol", "maxiter")}
                    ok.setdefault("optimizer", '"SLSQP"')
                    o = ", ".join(f"{k}={v}" for k, v in ok.items())
                    parts = [f"{blk}={{{', '.join(items)}}}" for blk, items in opt_blocks.items() if items]
                    parts.append(f"optimizer=om.ScipyMinimizeOptimizer({o})")
                    gkw = ", " + ", ".join(parts)
                lines_ = [
                    f"prob = om.Problem(model=om.Group(subsystems=subsystems, connections=connections{gkw}))",
                ]
                for name, val, units in indep:
                    u = f", units={units}" if units else ""
                    lines_.append(f'prob.set_val("{name}", {val}{u})')
                edits.append((a, b, ("\n" + indent).join(lines_)))
            elif f == "prob.run_driver" and not use_opt:
                edits.append((a, b, "prob.run_model()"))
    if not has_setup:
        return "needs hand port (no prob.setup())"
    # run_driver / check_totals / assert_check_totals
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = ast.unparse(n.func)
            if f == "prob.run_driver" and use_opt:
                _, a, b = seg(n)
                edits.append((a, b, "prob.run_optimizer()"))
            elif f.endswith(".check_totals"):
                _, a, b = seg(n)
                kws = [f"{k.arg}={seg(k.value)[0]}" for k in n.keywords if k.arg in ("of", "wrt")]
                kws += [
                    "only_incorrect=True",
                    f"atol={tot_args.get('atol', '1e-6')}",
                    f"rtol={tot_args.get('rtol', '1e-6')}",
                ]
                edits.append((a, b, f"{seg(n.func)[0]}({', '.join(kws)})"))
    for st in stmts:
        if (
            isinstance(st, ast.Expr)
            and isinstance(st.value, ast.Call)
            and ast.unparse(st.value.func) == "assert_check_totals"
        ):
            _, a, b = seg(st)
            edits.append((a, b, "assert not totals, totals"))
    # subscript loads
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Subscript)
            and isinstance(n.ctx, ast.Load)
            and isinstance(n.value, ast.Name)
            and n.value.id == "prob"
        ):
            _, a, b = seg(n)
            edits.append((a, b, f"prob.get_val({seg(n.slice)[0]})"))

    out = bytearray(btext)
    for a, b, new in sorted(edits, key=lambda e: -e[0]):
        out[a:b] = new.encode()
    res = out.decode()
    # remove DELETE placeholders (whole line)
    res = "\n".join(ln for ln in res.split("\n") if ln.strip() != "pass  # DELETE")
    res = res.replace("import openmdao.api as om\n", "")
    extra = ""
    if use_opt:
        extra = (
            "from om4.variables.design_var import DesignVar\n"
            "from om4.variables.response import Constraint, Objective\n"
            "from om4.transforms.transform import AffineTransform\n"
        )
    res = res.replace(
        "from openmdao.utils.assert_utils import assert_near_equal, assert_check_totals",
        "from openmdao.utils.assert_utils import assert_near_equal",
    )
    res = res.replace(
        "from openmdao.utils.assert_utils import assert_near_equal",
        "import om4.api as om\n" + extra + "from om4.utils.assert_utils import assert_near_equal",
    )
    open(path, "w").write(res)
    return "ported"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: {port(p)}")
