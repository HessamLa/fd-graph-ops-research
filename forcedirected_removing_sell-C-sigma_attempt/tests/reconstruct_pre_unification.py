"""reconstruct_pre_unification.py -- the pre-unification original, re-derived.

`fodiwalk/core/sell_c_sigma.py` held one of the two copies of the algorithm.
That working-tree file was NEVER COMMITTED: git HEAD holds an older version
(sha256 `c9d3af95...`, before the `_step` -> `step` rename), no stash holds
it, and `../fdmap_backup` holds `fodined` and `archive` only. The 2026-08-25
unification overwrote it with the forwarder, thus `PARITY.md`'s central
claim -- that the copy into the shared package was byte identical -- names a
file that no longer exists.

IT IS STILL RE-DERIVABLE, and this script proves it. Every edit made to
`forcedirected/sell_c_sigma.py` after the copy is enumerated in `PARITY.md`
and each one is invertible. Reverse all three -- the docstring correction,
the provenance header, and the `to_csr` rename with its alias -- and the
result hashes to `cfdbe266...318d`, the sha256 recorded at the copy.

The match is evidence and not circularity: an edit left out of the
enumeration would break the hash.

IT IS NOT A TRIPWIRE. The reconstruction reverses the edits of ONE day, thus
it applies to ONE version of the source. The first legitimate change to
`forcedirected/sell_c_sigma.py` -- a fix in the layout, a different `k_max`,
anything -- makes the reversal inapplicable, and that is normal and not a
defect. This script therefore pins BOTH hashes: it checks the SOURCE first,
and when the source has moved on it says so and exits 0. It fails only in
the one case that means something: the source is still the 2026-08-25 file
and the reconstruction no longer reproduces the original. That case says an
edit went unrecorded or the enumeration in `PARITY.md` is wrong; fix the
record, never the expected hash.

Once the tree is committed, a moved-on source is reconstructed from the
commit that holds the 2026-08-25 version instead, and this script is then
the METHOD rather than the check.

The OTHER original, `fodined/embedding/sell_c_sigma.py`, IS in git at
`89abaf2` (sha256 `b4a500a2...`) and needs no reconstruction.

2026-08-26: the `sellcsigma` package was ABSORBED into `forcedirected`, the
root engine package, and the implementation moved with sha256 UNCHANGED.
Only `SOURCE` above moved with it; every reversal below is untouched, and
that is why this script still reaches MATCH. The kernel file still names
`sellcsigma/PARITY.md` in one comment: it stays, because the file's sha256
is the anchor of this reconstruction and an editorial fix would break it.

Run: .venv/bin/python forcedirected/tests/reconstruct_pre_unification.py
     (from the repository root)
"""
import hashlib, re, sys

SOURCE = "forcedirected/sell_c_sigma.py"

# The implementation as it stood on 2026-08-25, the only version these
# reversals apply to.
SOURCE_SHA_2026_08_25 = (
    "0406d2ecfd74572878f6c0cf7e22036d4a89b58fa3a67f65f1707f77fd40bebe")
# The pre-unification `fodiwalk/core/sell_c_sigma.py`, the file being
# recovered. Measured at the copy, and confirmed independently by
# fdmap-f4 [d170fb] on the live file before the forwarder replaced it.
ORIGINAL_SHA = (
    "cfdbe266898a2ad3503bd3dbe14cbe5c17b29e9b6c71414c4e2e958c4152318d")

s = open(SOURCE).read()
source_sha = hashlib.sha256(s.encode()).hexdigest()
if source_sha != SOURCE_SHA_2026_08_25:
    print(f"{SOURCE} has moved on since 2026-08-25.")
    print(f"  now:        {source_sha}")
    print(f"  2026-08-25: {SOURCE_SHA_2026_08_25}")
    print("These reversals apply to the 2026-08-25 version only, thus the")
    print("check is SKIPPED and this is not a failure. To recover the")
    print("pre-unification original, run this script against that version:")
    print("  git show <commit>:forcedirected/sell_c_sigma.py")
    print(f"The target stays {ORIGINAL_SHA}.")
    sys.exit(0)


# 3. undo the docstring correction (rewrite + the dated note block)
new_claim = """plain, dense ``sum(axis=1)`` -- no scatter over the k axis, one predictable
shape per batch. The per-ROW write at the end of ``step`` is still a
scatter (``dZ.at[rows].add(..., mode="drop")``), and it is NOT
contention-free: a row wider than ``k_max`` becomes several virtual rows
that carry the SAME owner id, thus several values land on one address.
What the layout removes is the SCALE of that contention. ``segment_sum``
contends once per stored pair; this layout contends once per hub SPLIT,
which is ``ceil(width / k_max)`` and is a small number even on a
power-law graph (measured: 0 to 196 splits over n = 1k to 1M, BA). Only
*how the same sum gets computed* differs (see"""
old_claim = """plain, dense ``sum(axis=1)`` -- no scatter, no atomics, one predictable
shape per batch. Only *how the same sum gets computed* differs (see"""
assert s.count(new_claim) == 1
s = s.replace(new_claim, old_claim, 1)
s = re.sub(r"# CORRECTION 2026-08-25.*?# ADOPTED AT: .*?\n#\n", "", s, count=1, flags=re.S)

# 2. undo the provenance header
s = re.sub(r"# Provenance: moved VERBATIM from `fodiwalk/core/sell_c_sigma\.py` on\n"
           r"# 2026-08-25.*?# `to_csr` name above\. Parity record: `sellcsigma/PARITY\.md`\.\n#\n",
           "", s, count=1, flags=re.S)

# 1. undo the to_csr rename, the alias and its comment block
s = s.replace("def to_csr(D) -> sp.csr_matrix:", "def _to_csr(D) -> sp.csr_matrix:", 1)
s = s.replace("    D = to_csr(D)", "    D = _to_csr(D)")
s = s.replace("        D = to_csr(D)", "        D = _to_csr(D)")
s = re.sub(r"\n\n# `to_csr` is the public name\..*?\n_to_csr = to_csr\n", "", s, count=1, flags=re.S)

got = hashlib.sha256(s.encode()).hexdigest()
print("reconstructed sha256:  ", got)
print("pre-unification sha256:", ORIGINAL_SHA)
if got == ORIGINAL_SHA:
    print("MATCH -- the pre-unification original is recovered byte-exact.")
    print("Pass a path as argv[1] to write it out.")
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(s)
    sys.exit(0)
print("NO MATCH. The source is still the 2026-08-25 version, thus an edit")
print("went unrecorded or the enumeration in PARITY.md is wrong. Fix the")
print("record. Do NOT adjust the expected hash.")
sys.exit(1)
