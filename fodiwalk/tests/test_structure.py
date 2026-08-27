"""test_structure.py -- the SHAPE of the split package, turned into gates.

`fodiwalk/fodiwalk.py` was a 698-line god class with four jobs (D1 of
`dev-docs/REFACTOR.md` section 2). Three agents split it into stage
packages. A shape that only a document describes goes back to being a
slop after three commits, thus every clause of the target tree
(`REFACTOR.md` section 3) becomes ONE test here: gates G4 to G8.

Style: one test, one contract, and the name says the contract, as
`test_contracts.py` does. A contract about the SOURCE uses `ast`, never
`import` plus `inspect` -- an import that happens inside a function body
is exactly the leak that runtime inspection misses (D6).

Every `ast` test takes a `root` path, defaulted to the live package, so a
COPY of the tree, broken on purpose, can be pointed at the same test as a
negative control. `agentic-log/04.structure-agent/` records the controls
run against this file; they are not part of this suite.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent

KNOWN_STAGE_PKGS = {"core", "augment_graph", "embed", "make_graph", "misc"}

# The ROOT packages a module of `fodiwalk` MAY import. `forcedirected` holds
# the engine `ForceDirected`, the SELL-C-sigma kernel, the CSR helpers and
# the update rules, for this package and for `fodined` alike; `fodiwalk`
# imports them from it DIRECTLY since 2026-08-27 (four forwarder modules of
# `core` and `misc` carried the names from 2026-08-26 and were deleted). It
# is ALLOWED, and it is allowed for one reason: `forcedirected` imports
# numpy, scipy, jax and its own modules and NOTHING of this repository, thus
# every dependency points AT it and a cycle is impossible.
# `_external_targets` therefore judges `fodiwalk.*` targets only, and
# `test_fodiwalk_imports_no_root_package_but_the_engine` below is what makes
# this set a RULE and not a note: without it the name would document an
# intent that nothing enforces, and `fodiwalk` could import `evaluator` or
# `fodined` with every gate still green.
ALLOWED_ROOT_PKGS = {"forcedirected"}


# ===========================================================================
# helpers -- shared by more than one contract below
# ===========================================================================
def _lines(path: pathlib.Path) -> int:
    return len(path.read_text().splitlines())


def _external_targets(path: pathlib.Path, own_pkg: str) -> set[str]:
    """The `fodiwalk` STAGE packages that the module at `path` reaches,
    module level AND function level, minus `own_pkg` itself (an import of
    a sibling module inside the same package is not a cross-package
    reach). Catches `import fodiwalk.x` and `from ..x import y` alike.

    A ROOT package of `ALLOWED_ROOT_PKGS` is not a stage of `fodiwalk` and
    never enters this set: it cannot import back, thus it cannot make a
    cycle, and the one-way rule these gates keep is about the stages."""
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):                  # not `tree.body`: a
        if isinstance(node, ast.Import):          # function-local import
            for alias in node.names:              # is the leak D6 planted
                parts = alias.name.split(".")
                if parts[0] == "fodiwalk" and len(parts) > 1:
                    if parts[1] != own_pkg:
                        out.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            level, mod = node.level, node.module or ""
            if level == 0:
                if mod == "fodiwalk" or mod.startswith("fodiwalk."):
                    parts = mod.split(".")
                    if len(parts) > 1:
                        if parts[1] != own_pkg:
                            out.add(parts[1])
                    else:
                        for alias in node.names:
                            if alias.name in KNOWN_STAGE_PKGS and (
                                    alias.name != own_pkg):
                                out.add(alias.name)
            elif level == 1:
                continue                          # own_pkg itself: a sibling
            else:                                 # level >= 2: fodiwalk/...
                if mod:
                    tgt = mod.split(".")[0]
                    if tgt != own_pkg:
                        out.add(tgt)
                else:
                    for alias in node.names:
                        if alias.name in KNOWN_STAGE_PKGS and (
                                alias.name != own_pkg):
                            out.add(alias.name)
    return out


def _assert_package_imports_only(root: pathlib.Path, pkg: str,
                                  allowed: set[str]) -> None:
    for path in sorted((root / pkg).glob("*.py")):
        bad = _external_targets(path, pkg) - allowed
        assert not bad, (
            f"{path.relative_to(root)} imports {sorted(bad)}, outside "
            f"{{{pkg!r}}} | {sorted(allowed)}")


# ===========================================================================
# 1. the dependency runs one way (G7, plus REFACTOR.md section 3)
# ===========================================================================
# `core -> core` only is `test_core_imports_only_core` of
# `test_contracts.py`; it is not duplicated here.
def test_core_imports_no_embed(root: pathlib.Path = PKG):
    """Extends `test_core_imports_only_core` (`test_contracts.py`), written
    before the `embed` package existed and thus silent on it. `embed` reads
    `core`; `core` must never read `embed`, module level or function level,
    or the one-way dependency of REFACTOR.md section 3 becomes a cycle.

    `core` reaching `forcedirected` is NOT such a leak, and this gate stays
    silent on it by design (`ALLOWED_ROOT_PKGS`): that root package holds
    the engine and reads nothing of this repository."""
    for path in sorted((root / "core").glob("*.py")):
        bad = _external_targets(path, "core") & {"embed"}
        assert not bad, f"{path.relative_to(root)} imports embed"


def test_augment_graph_imports_no_embed_no_model(root: pathlib.Path = PKG):
    """Stage 2 may read `core` (package docstring, since the stage boundary
    moved 2026-08-21): `planes.py` needs `core.forces.planes_of`/`fuse` and
    `core.plan_contract.PLANE_CHECKS` to prepare a law's data. It must
    never reach `embed` or the model, or the seam would run backward."""
    _assert_package_imports_only(root, "augment_graph", allowed={"core"})


def test_embed_imports_only_core_no_augment_graph_no_model(
        root: pathlib.Path = PKG):
    """Stage 3 is CONSUMPTION ONLY (`dev-docs/fodiwalk-module.md`): it reads
    the engine of `core` to build the plan and the jitted steps, and it
    must never reach back into stage 2's data preparation or the model, or
    `core` would end up two hops from `Config` through the back door
    REFACTOR.md section 3 forbids."""
    _assert_package_imports_only(root, "embed", allowed={"core"})


def test_make_graph_imports_only_make_graph(root: pathlib.Path = PKG):
    """Stage 1 is a reader of edge lists (package docstring); it has no
    reason to import any other stage."""
    _assert_package_imports_only(root, "make_graph", allowed=set())


def test_misc_imports_only_augment_graph(root: pathlib.Path = PKG):
    """`evaluation.py` reads `far_pairs` for the hop sample; that is
    existing and allowed (package docstring). Any other package would be a
    new, undocumented coupling."""
    _assert_package_imports_only(root, "misc", allowed={"augment_graph"})


# ===========================================================================
# 2. no cross-module import of a private name (D6)
# ===========================================================================
def test_no_module_imports_a_private_name_of_another_module(
        root: pathlib.Path = PKG):
    """D6 was `from .core.sell_c_sigma import make_plan, _step`: the god
    class reached a private helper of another module. The split renamed it
    to `step` and kept `_step` only as an alias that a package's own
    `__init__.py` may re-export as part of its declared surface (that is
    `core/__init__.py` today, and is why `__init__.py` is excluded below)
    -- a PLAIN module reaching into another module's private name is the
    defect, and it stays forbidden everywhere else."""
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                assert not alias.name.startswith("_"), (
                    f"{path.relative_to(root)} imports private name "
                    f"{alias.name!r} from {('.' * node.level) + (node.module or '')!r}")


# ===========================================================================
# 3. no `if` chain on a policy or a law name in the model (D2)
# ===========================================================================
def test_model_holds_no_compare_against_a_policy_or_law_literal(
        root: pathlib.Path = PKG):
    """D2. `_build_D`, `graph_walk` and `_plane` picked the physics with an
    `if` chain on a string; a missing branch silently ran another law's
    planes and a run went to NaN with no error (see `core/forces.py`).
    `POLICIES`, `PLANE_BUILDERS`, `FORCE_PLANES`, `weights.RULES` and
    `optim.RULES` replace the chain with a registry, thus `model.py` must
    hold no `Compare` against one of these literals. A docstring or a
    comment that names a policy does not fail this test: only a `Compare`
    node does."""
    literals = {"walk", "walk_edges", "nbr_walk", "buckets", "cap",
                "fdlinear", "fdlinear_fused"}
    path = root / "model.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for value in (node.left, *node.comparators):
            if isinstance(value, ast.Constant) and value.value in literals:
                raise AssertionError(
                    f"{path.relative_to(root)}:{node.lineno} compares "
                    f"against {value.value!r}, a policy/law name")


# ===========================================================================
# 4. the module size (D1)
# ===========================================================================
# The three exceptions were already over the cap BEFORE the split, each is
# ONE cohesive concern, and REFACTOR.md section 7 puts all three out of
# scope. The number PINS the file at its measured size on the day of the
# split (2026-08-20): a file that needs more needs a split, not a raised
# cap. `tests/*.py` is a test file, not a module of the package, and is
# excluded outright rather than exempted by name.
MAX_LINES = 300
MODEL_MAX_LINES = 200
# `core/sell_c_sigma.py` (558) and `core/force_directed.py` (421) were the
# other two entries. The kernel moved to `forcedirected/` 2026-08-25..26 and
# the engine with it; both files stayed as forwarders and 2026-08-27 deleted
# them. A cap on a file that no longer exists says nothing, thus the two
# entries GO with the files. `forcedirected/` is not judged here: this gate
# reads `fodiwalk` only.
SIZE_EXCEPTIONS = {
    "augment_graph/walks.py": 422,   # the walks
}


def test_no_module_exceeds_300_lines_except_the_named_exceptions(
        root: pathlib.Path = PKG):
    """D1: a 698-line, 4-job file was the defect. A module past 300 lines
    is that defect returning; the three pinned exceptions above are the
    only files allowed to stay over the line, and only up to their
    measured size."""
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        cap = SIZE_EXCEPTIONS.get(rel, MAX_LINES)
        n = _lines(path)
        assert n <= cap, f"{rel} is {n} lines, over its cap of {cap}"


def test_model_py_is_at_most_200_lines(root: pathlib.Path = PKG):
    """`model.py` is the wiring only (its own docstring): the config, the
    three stage seams and the epoch-loop hook, and nothing an algorithm
    could grow inside. 200 lines is the budget REFACTOR.md section 3 sets
    for that."""
    n = _lines(root / "model.py")
    assert n <= MODEL_MAX_LINES, f"model.py is {n} lines, over {MODEL_MAX_LINES}"


# ===========================================================================
# 5. every policy builder is a module-level function (D1, G8)
# ===========================================================================
def test_every_policy_builder_is_a_module_function_not_a_method():
    """D1: `_build_D_walk`, `_build_D_nbr_walk` and `_build_D_buckets` were
    bound methods of the god class and reached its state through `self`.
    Stage 2 is pure: a policy builder takes `(A, n, spec, rng)` and gives
    an `Augmentation`, with no bound state, thus its first parameter is
    never `self`."""
    from fodiwalk.augment_graph.policies import POLICIES

    assert POLICIES, "the policy registry is empty"
    for name, fn in POLICIES.items():
        assert inspect.isfunction(fn), (
            f"POLICIES[{name!r}] is {fn!r}, not a plain function")
        params = list(inspect.signature(fn).parameters)
        assert params, f"POLICIES[{name!r}] takes no parameter"
        assert params[0] != "self", (
            f"POLICIES[{name!r}]'s first parameter is `self`: it is a "
            f"bound method, not a module-level function")


# ===========================================================================
# 6. the registries are complete (D2, G8)
# ===========================================================================
# `FORCE_PLANES` vs `FORCE_FN` is `test_b3_registry_covers_every_law` of
# `test_contracts.py`; it is not duplicated here.
def _config_pairs_choices(root: pathlib.Path) -> set[str]:
    """The policy names `Config.pairs` documents, read from its own inline
    comment in `config.py` -- the same text a person edits when a policy is
    added or removed, thus the two cannot drift apart unnoticed."""
    text = (root / "config.py").read_text()
    for line in text.splitlines():
        if line.strip().startswith("pairs:"):
            _, _, comment = line.partition("#")
            assert comment, "config.py's `pairs` field has no `#` comment"
            return {tok.strip() for tok in comment.split("|")}
    raise AssertionError("config.py has no `pairs:` field")


def test_policies_registry_covers_every_pairs_choice_of_config(
        root: pathlib.Path = PKG):
    """A policy name that `Config.pairs` documents and `POLICIES` does not
    hold would raise `ValueError` only at call time, with no test catching
    it first -- the registry is only as good as its coverage."""
    from fodiwalk.augment_graph.policies import POLICIES

    choices = _config_pairs_choices(root)
    assert set(POLICIES) == choices, (
        f"POLICIES {sorted(POLICIES)} != config.py's documented "
        f"pairs choices {sorted(choices)}")


def test_plane_builders_registry_key_set_equals_plane_checks():
    """`augment_graph/planes.py` builds a plane; `core/plan_contract.py`
    asserts what the name promises (I1, I2, I4). A name held by one table
    and not the other is a plane built and not asserted, or asserted and
    never built -- `augment_graph/planes.py` already guards this with a
    module-level `assert`; this test gives it a name and a failure message
    in the suite, and it runs even if that module-level assert is ever
    loosened."""
    from fodiwalk.augment_graph.planes import PLANE_BUILDERS
    from fodiwalk.core.plan_contract import PLANE_CHECKS

    assert set(PLANE_BUILDERS) == set(PLANE_CHECKS), (
        f"PLANE_BUILDERS {sorted(PLANE_BUILDERS)} != "
        f"PLANE_CHECKS {sorted(PLANE_CHECKS)}")


# ===========================================================================
# 7. the god class and the empty models/ directory stay deleted (D1, D10)
# ===========================================================================
def test_the_god_class_module_and_the_empty_models_dir_stay_deleted(
        root: pathlib.Path = PKG):
    """D1 and D10. `fodiwalk/fodiwalk.py` was the 698-line, 4-job class;
    `fodiwalk/models/` was an empty directory the split leaves behind
    (REFACTOR.md section 3). A later change might reintroduce either by
    habit -- a new god method growing back in a familiar filename, or an
    editor recreating the directory. This fails the moment it does."""
    assert not (root / "fodiwalk.py").exists(), (
        f"{root / 'fodiwalk.py'} exists: the god class came back")
    assert not (root / "models").exists(), (
        f"{root / 'models'} exists: the empty directory of D10 came back")


def _repo_root_pkgs(root: pathlib.Path) -> set:
    """The packages that sit beside `fodiwalk` in the repository.

    Read from the FILESYSTEM and not from a list, thus a package added
    tomorrow is judged too.

    `or any(d.glob("*.py"))` is NOT tidy-able down to the `__init__.py`
    test, and this comment exists to stop the next reader from doing it.
    Python 3 imports a directory of `.py` files with NO `__init__.py` as a
    NAMESPACE package, and it imports cleanly. Three root directories are
    in that state today -- `fdwalk`, `experiments`, `archive` -- thus the
    `__init__.py` test alone made this gate blind to them, and
    `import fdwalk.walks` inside a stage module PASSED while working at
    runtime. That is the exact leak class this test exists to stop.
    """
    return {d.name for d in root.parent.iterdir()
            if d.is_dir() and ((d / "__init__.py").exists()
                               or any(d.glob("*.py")))}


def _root_targets(path: pathlib.Path, repo_pkgs: set) -> set:
    """The repository packages OUTSIDE `fodiwalk` that `path` imports.

    Module level and function level alike, as `_external_targets` does:
    a leak hidden inside a function body is the defect D6 planted.
    """
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            head = name.split(".")[0]
            if head in repo_pkgs and head != "fodiwalk":
                out.add(head)
    return out


def test_fodiwalk_imports_no_root_package_but_the_engine(
        root: pathlib.Path = PKG):
    """`fodiwalk` may reach ONE package beside it: `forcedirected`, the
    engine and the kernel that `fodined` shares.

    Every other root package is a defect. `forcedirected` reads nothing of
    this repository, thus every dependency points AT it and no cycle is
    possible; `evaluator`, `fodined` and whatever lands next carry their own
    dependencies, and an import of one from a stage module makes `fodiwalk`
    depend on a consumer of itself.

    This gate is what gives `ALLOWED_ROOT_PKGS` force. Before it, the set
    named an intent that no assertion consulted.
    """
    repo_pkgs = _repo_root_pkgs(root)
    assert "forcedirected" in repo_pkgs, (
        "the engine package is missing from the repository root")
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path) or path.parent.name == "tests":
            continue
        bad = _root_targets(path, repo_pkgs) - ALLOWED_ROOT_PKGS
        assert not bad, (
            f"{path.relative_to(root)} imports the root package(s) "
            f"{sorted(bad)}, which are not in ALLOWED_ROOT_PKGS "
            f"{sorted(ALLOWED_ROOT_PKGS)}")
