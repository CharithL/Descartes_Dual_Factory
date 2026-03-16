# Phase 5: Final Integration

**Status: Planned (Task 25)**

Phase 5 wires everything together: updates `__init__.py` files for clean imports, runs a full integration smoke test on synthetic data, and verifies the complete pipeline from probe registry through verdict generation.

## What Will Be Included

### Package Init Updates

- `l5pc/probing/__init__.py` — re-export key functions (hardened_probe, mlp_delta_r2, train_sae, etc.)
- `l5pc/surrogates/__init__.py` — re-export surrogate registry
- `l5pc/factory/__init__.py` — re-export orchestrator, verdict generator, evaluator

### Integration Smoke Tests

Full-pipeline tests on synthetic data verifying:

1. **Registry** — All core probes available (15+ always True)
2. **Verdict generator** — Correct verdict for confirmed zombie evidence bundle
3. **Verdict generator** — Correct verdict for mandatory (causal ablation) evidence
4. **Surrogate genome** — Random composition, mutation, crossover all produce valid genomes
5. **Config independence** — Factory config and probing config coexist without collision

### Verification Criteria

- All `__init__.py` re-exports resolve without circular imports
- `from l5pc.factory.orchestrator import DualFactoryOrchestrator` works
- `ZombieVerdictGenerator_v3` produces correct verdicts for all 8 verdict types
- `SurrogateGenomeComposer` round-trips through compose/mutate/crossover
- No existing tests broken
