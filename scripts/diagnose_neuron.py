#!/usr/bin/env python3
"""Quick diagnostic to verify NEURON mechanisms are compiled and loadable.

Usage:
    python scripts/diagnose_neuron.py
"""
import os
import sys
import glob
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("NEURON Mechanism Diagnostic")
print("=" * 60)

# 1. Check mechanism directory
mech_dir = project_root / 'mechanisms'
print(f"\n[1] Mechanism directory: {mech_dir}")
print(f"    Exists: {mech_dir.exists()}")

if mech_dir.exists():
    mod_files = list(mech_dir.glob('*.mod'))
    print(f"    .mod files found: {len(mod_files)}")
    for f in sorted(mod_files):
        print(f"      - {f.name}")

    # Check for compiled output
    print(f"\n[2] Compiled mechanism search:")
    so_files = list(mech_dir.rglob('libnrnmech.so'))
    dll_files = list(mech_dir.rglob('nrnmech.dll'))
    all_compiled = so_files + dll_files
    if all_compiled:
        for f in all_compiled:
            print(f"    FOUND: {f}")
    else:
        print(f"    NOT FOUND: No compiled mechanism library!")
        print(f"    Contents of mechanisms/:")
        for item in sorted(mech_dir.iterdir()):
            print(f"      {item.name}{'/' if item.is_dir() else ''}")
        x86 = mech_dir / 'x86_64'
        if x86.exists():
            print(f"    Contents of mechanisms/x86_64/:")
            for root, dirs, files in os.walk(x86):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), mech_dir)
                    print(f"      {rel}")
        print(f"\n    FIX: Run from project root:")
        print(f"      cd {mech_dir} && nrnivmodl . && cd ..")

# 3. Try loading NEURON
print(f"\n[3] NEURON import test:")
try:
    from neuron import h
    print(f"    NEURON imported successfully")
    print(f"    NEURON version: {h.nrnversion()}")
except ImportError as e:
    print(f"    FAILED: {e}")
    sys.exit(1)

# 4. Try loading compiled mechanisms
print(f"\n[4] Loading compiled mechanisms:")
loaded = False
if mech_dir.exists():
    candidates = list(mech_dir.rglob('libnrnmech.so'))
    candidates += list(mech_dir.rglob('nrnmech.dll'))
    for dll_path in candidates:
        try:
            h.nrn_load_dll(str(dll_path))
            print(f"    LOADED: {dll_path}")
            loaded = True
            break
        except Exception as e:
            print(f"    Failed {dll_path}: {e}")

if not loaded:
    print(f"    NO compiled mechanisms could be loaded!")

# 5. Test mechanism availability
print(f"\n[5] Mechanism availability test:")
test_sec = h.Section(name='test')

mechanisms = ['NaTa_t', 'Ca_HVA', 'Ca_LVAst', 'Ih', 'Im',
              'K_Pst', 'K_Tst', 'Nap_Et2', 'SK_E2', 'SKv3_1',
              'CaDynamics_E2']

available = []
missing = []
for mech in mechanisms:
    try:
        test_sec.insert(mech)
        available.append(mech)
    except Exception:
        missing.append(mech)

print(f"    Available: {len(available)}/{len(mechanisms)}")
for m in available:
    print(f"      ✓ {m}")
for m in missing:
    print(f"      ✗ {m}")

h.delete_section(sec=test_sec)

# 6. Quick spike test if mechanisms available
if 'NaTa_t' in available and 'SKv3_1' in available:
    print(f"\n[6] Quick spike test:")
    soma = h.Section(name='test_soma')
    soma.L = 20
    soma.diam = 20
    soma.insert('pas')
    soma(0.5).pas.g = 3.38e-5
    soma(0.5).pas.e = -90

    soma.insert('NaTa_t')
    soma(0.5).gNaTa_tbar_NaTa_t = 2.04

    soma.insert('SKv3_1')
    soma(0.5).gSKv3_1bar_SKv3_1 = 0.693

    soma.insert('K_Pst')
    soma(0.5).gK_Pstbar_K_Pst = 0.05

    soma(0.5).ek = -85
    soma(0.5).ena = 50

    # Inject current
    stim = h.IClamp(soma(0.5))
    stim.delay = 100
    stim.dur = 500
    stim.amp = 0.5  # nA

    # Record voltage
    v = h.Vector()
    v.record(soma(0.5)._ref_v)

    h.load_file('stdrun.hoc')
    h.dt = 0.025
    h.celsius = 37
    h.v_init = -75
    h.tstop = 700
    h.finitialize(h.v_init)
    h.continuerun(h.tstop)

    import numpy as np
    v_arr = np.array(v.as_numpy())
    v_max = v_arr.max()
    n_spikes = np.sum(np.diff((v_arr > 0).astype(int)) > 0)

    print(f"    V_max = {v_max:.1f} mV")
    print(f"    Spikes detected: {n_spikes}")

    if n_spikes > 0:
        print(f"    ✓ Cell can spike! Mechanisms working correctly.")
    else:
        print(f"    ✗ No spikes detected. Check gbar values.")

    h.delete_section(sec=soma)
else:
    print(f"\n[6] Spike test SKIPPED (critical mechanisms missing)")

print(f"\n{'=' * 60}")
if missing:
    print("RESULT: MECHANISMS NOT AVAILABLE")
    print(f"  Run: cd mechanisms && nrnivmodl . && cd {project_root}")
    print(f"  Then re-run: python scripts/run_phase1.py --only-step 1 --force")
else:
    print("RESULT: ALL MECHANISMS AVAILABLE ✓")
print("=" * 60)
