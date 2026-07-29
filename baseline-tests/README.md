# Official transformer baseline (seed=1, independent PRNG)

This directory’s older Spike/Verilator READMEs recorded runs under the **legacy
shared PRNG**. Those numbers remain historical.

**Current official baseline** for before/after hardware work:

- Generator: independent per-tensor streams from `PRNG_SEED=1`
- Workflow: Spike float gold → export expected int8 → Verilator `SKIP_GOLD` + snapshot check
- L sweep results: [`../sweeps/l_sweep/README.md`](../sweeps/l_sweep/README.md)
- Correctness docs: [`../correctness/README.md`](../correctness/README.md)

Do not compare new seed-1 Verilator cycles directly to the old L=16 `145744` /
L=32 `493148` figures; the input tensors differ.
