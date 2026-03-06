"""
L5PC DESCARTES -- Bahl 2012 Reduced L5PC Model Wrapper

Wraps the Bahl et al. (2012) reduced compartmental model of a layer 5
pyramidal cell (ModelDB accession 146026).  The reduced model has 6
functional compartments:

    soma          -- Cell body.  NaTa_t, K_Pst, K_Tst, SKv3_1, SK_E2, Im.
    basal         -- Basal dendrite (single equivalent).  Passive + Ih.
    apical_trunk  -- Proximal apical dendrite.  NaTa_t, Ih.
    nexus         -- Apical nexus / calcium hot zone.  Ca_HVA, Ca_LVAst,
                     SK_E2, Ih, NaTa_t, SKv3_1.
    tuft          -- Distal apical tuft.  Ih, passive.

The code first attempts to load morphology and mechanism .mod files from
``data/models/bahl/``.  If those files are not found (e.g. on a machine
without the ModelDB download), it falls back to building a simplified
ball-and-stick model programmatically with standard Bahl 2012 parameters.

References
----------
Bahl, A., Stemmler, M. B., Herz, A. V. M., & Bhatt, D. H. (2012).
Automated optimization of a reduced layer 5 pyramidal cell model based
on experimental data.  Journal of Neuroscience Methods, 210(1), 22-34.
"""
import os
import warnings
import numpy as np

from l5pc.config import (
    BAHL_MODEL_DIR,
    BAHL_REGIONS,
    CHANNEL_SPECS,
    N_BASAL_SYN,
    N_APICAL_SYN,
    N_SOMA_SYN,
    NEURON_DT_MS,
)


class BahlCell:
    """Reduced L5PC following Bahl et al. 2012.

    Attributes
    ----------
    sections : dict
        Mapping of region name -> NEURON Section.
    gbar_values : dict
        Nested dict {region: {channel_name: gbar}} storing the maximal
        conductance inserted in each compartment.  Used by Level B
        effective-conductance computation.
    synapses : list
        All synapse / NetCon / NetStim objects (prevents garbage collection).
    """

    # Class-level flag: mechanisms only need to be loaded once per process
    _mechanisms_loaded = False

    def __init__(self):
        from neuron import h, nrn  # noqa: delayed import
        self.h = h

        self.sections = {}
        self.gbar_values = {r: {} for r in BAHL_REGIONS}
        self.synapses = []       # prevent GC of NEURON objects
        self._netcons = []
        self._netstims = []

        self._try_load_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _try_load_model(self):
        """Try to load the full Bahl model from ModelDB files; fall back."""
        h = self.h

        # First, try to load compiled mechanisms from mechanisms/ directory
        self._load_compiled_mechanisms()

        model_dir = str(BAHL_MODEL_DIR)
        loaded = False
        if os.path.isdir(model_dir):
            hoc_file = os.path.join(model_dir, 'init.hoc')
            mod_dir = os.path.join(model_dir, 'mechanisms')
            if os.path.isfile(hoc_file):
                try:
                    # Load compiled mechanisms from model directory
                    if os.path.isdir(mod_dir):
                        h.nrn_load_dll(
                            os.path.join(mod_dir, 'nrnmech.dll')
                            if os.name == 'nt' else
                            os.path.join(mod_dir, 'x86_64', '.libs',
                                         'libnrnmech.so')
                        )
                    h.load_file(hoc_file)
                    loaded = True
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load Bahl model from {model_dir}: {exc}. "
                        "Falling back to programmatic construction."
                    )

        if not loaded:
            self._build_programmatic_model()

        # Verify that critical mechanisms were inserted
        self._verify_mechanisms()

    def _load_compiled_mechanisms(self):
        """Load compiled .mod mechanisms from the project mechanisms/ dir.

        Searches multiple possible paths because nrnivmodl output location
        varies across NEURON versions:
          - x86_64/.libs/libnrnmech.so  (NEURON <= 8.0)
          - x86_64/libnrnmech.so        (NEURON 8.1+)
          - nrnmech.dll                  (Windows)

        Only loads once per process (tracked via class-level flag).
        """
        # Skip if already loaded in this NEURON process
        if BahlCell._mechanisms_loaded:
            return

        h = self.h
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        mech_dir = os.path.join(project_root, 'mechanisms')

        if not os.path.isdir(mech_dir):
            print(f"[BahlCell] mechanisms/ directory not found at {mech_dir}")
            return

        # Search multiple possible compiled library locations
        if os.name == 'nt':
            candidates = [
                os.path.join(mech_dir, 'nrnmech.dll'),
            ]
        else:
            candidates = [
                os.path.join(mech_dir, 'x86_64', '.libs', 'libnrnmech.so'),
                os.path.join(mech_dir, 'x86_64', 'libnrnmech.so'),
                os.path.join(mech_dir, 'arm64', '.libs', 'libnrnmech.so'),
                os.path.join(mech_dir, 'arm64', 'libnrnmech.so'),
            ]
            # Also search with glob for any .so file under mech_dir
            import glob
            candidates += glob.glob(
                os.path.join(mech_dir, '**', 'libnrnmech.so'),
                recursive=True
            )

        for dll_path in candidates:
            if os.path.isfile(dll_path):
                try:
                    h.nrn_load_dll(dll_path)
                    print(f"[BahlCell] Loaded mechanisms from: {dll_path}")
                    BahlCell._mechanisms_loaded = True
                    return
                except Exception as exc:
                    # "already exists" means mechanisms were loaded by a
                    # previous nrn_load_dll call (e.g. via nrnivmodl -loadflags)
                    if 'already exists' in str(exc):
                        BahlCell._mechanisms_loaded = True
                        return
                    print(f"[BahlCell] Failed to load {dll_path}: {exc}")

        # Nothing loaded — print diagnostic info
        print(f"[BahlCell] WARNING: No compiled mechanisms found!")
        print(f"[BahlCell]   Searched in: {mech_dir}")
        print(f"[BahlCell]   Run: cd mechanisms && nrnivmodl . && cd ..")

    def _build_programmatic_model(self):
        """Build a simplified ball-and-stick Bahl model from scratch.

        Geometry and channel densities follow published Bahl 2012 values.
        The model has 5 sections representing the 6 functional regions
        (tuft is a separate section off the nexus).
        """
        h = self.h

        # ---- Create sections ----
        soma = h.Section(name='soma')
        basal = h.Section(name='basal')
        apical_trunk = h.Section(name='apical_trunk')
        nexus = h.Section(name='nexus')
        tuft = h.Section(name='tuft')

        # ---- Geometry (Bahl 2012 reduced model) ----
        # soma
        soma.L = 20.0        # um
        soma.diam = 20.0
        soma.nseg = 1

        # basal (single equivalent cylinder)
        basal.L = 200.0
        basal.diam = 3.0
        basal.nseg = 1

        # apical trunk
        apical_trunk.L = 500.0
        apical_trunk.diam = 4.0
        apical_trunk.nseg = 5

        # nexus (calcium hot zone, ~685-885 um from soma)
        nexus.L = 200.0
        nexus.diam = 3.0
        nexus.nseg = 3

        # tuft
        tuft.L = 300.0
        tuft.diam = 2.0
        tuft.nseg = 3

        # ---- Topology ----
        basal.connect(soma(0.0))
        apical_trunk.connect(soma(1.0))
        nexus.connect(apical_trunk(1.0))
        tuft.connect(nexus(1.0))

        # ---- Passive properties (all sections) ----
        for sec in [soma, basal, apical_trunk, nexus, tuft]:
            sec.Ra = 100.0       # Ohm-cm
            sec.cm = 1.0         # uF/cm^2
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = 3.38e-5    # S/cm^2
                seg.pas.e = -90.0      # mV

        # ---- Active conductances ----
        # We insert mechanisms and set gbar values following the Bahl 2012
        # optimised parameter set.

        # ----- SOMA -----
        _insert_mech(soma, 'NaTa_t',  gbar=2.04)
        _insert_mech(soma, 'Nap_Et2', gbar=0.0015)
        _insert_mech(soma, 'K_Pst',   gbar=0.0513)
        _insert_mech(soma, 'K_Tst',   gbar=0.0812)
        _insert_mech(soma, 'SKv3_1',  gbar=0.693)
        _insert_mech(soma, 'SK_E2',   gbar=0.0441)
        _insert_mech(soma, 'Im',      gbar=0.00001)
        _insert_mech(soma, 'Ca_HVA',  gbar=0.000994)
        _insert_mech(soma, 'Ca_LVAst', gbar=0.00343)
        _insert_calcium_dynamics(soma)

        # ----- BASAL -----
        _insert_mech(basal, 'Ih', gbar=0.0002)

        # ----- APICAL TRUNK -----
        _insert_mech(apical_trunk, 'NaTa_t', gbar=0.0213)
        _insert_mech(apical_trunk, 'Nap_Et2', gbar=0.0003)
        _insert_mech(apical_trunk, 'Ih', gbar=0.0002)
        _insert_mech(apical_trunk, 'SKv3_1', gbar=0.003)
        _insert_mech(apical_trunk, 'Im', gbar=0.00001)

        # ----- NEXUS (calcium hot zone) -----
        _insert_mech(nexus, 'NaTa_t',   gbar=0.0213)
        _insert_mech(nexus, 'SKv3_1',   gbar=0.003)
        _insert_mech(nexus, 'SK_E2',    gbar=0.0012)
        _insert_mech(nexus, 'Ca_HVA',   gbar=0.000555)
        _insert_mech(nexus, 'Ca_LVAst', gbar=0.0187)
        _insert_mech(nexus, 'Ih',       gbar=0.0002)
        _insert_mech(nexus, 'Im',       gbar=0.00001)
        _insert_calcium_dynamics(nexus)

        # ----- TUFT -----
        _insert_mech(tuft, 'Ih',       gbar=0.0002)
        _insert_mech(tuft, 'NaTa_t',   gbar=0.008)
        _insert_mech(tuft, 'SKv3_1',   gbar=0.001)
        _insert_mech(tuft, 'Ca_LVAst', gbar=0.005)
        _insert_calcium_dynamics(tuft)

        # ---- Ionic concentrations ----
        for sec in [soma, basal, apical_trunk, nexus, tuft]:
            if hasattr(sec(0.5), 'ek'):
                sec(0.5).ek = -85.0
            if hasattr(sec(0.5), 'ena'):
                sec(0.5).ena = 50.0

        # ---- Store sections ----
        self.sections = {
            'soma': soma,
            'basal': basal,
            'apical_trunk': apical_trunk,
            'nexus': nexus,
            'tuft': tuft,
        }

        # ---- Collect gbar values ----
        self._collect_gbar_values()

    # Class-level: only print verification once
    _verified_once = False

    def _verify_mechanisms(self):
        """Check that critical active mechanisms are present in the cell."""
        if 'soma' not in self.sections:
            print("[BahlCell] WARNING: No soma section found!")
            return

        soma = self.sections['soma']
        seg = soma(0.5)

        critical = ['NaTa_t', 'SKv3_1', 'K_Pst']
        missing = [m for m in critical if not hasattr(seg, m)]

        if missing:
            present = [m for m in critical if hasattr(seg, m)]
            print(f"[BahlCell] CRITICAL: Missing mechanisms in soma: {missing}")
            print(f"[BahlCell]   Present: {present}")
            print(f"[BahlCell]   The cell will be PASSIVE ONLY (no spikes).")
            print(f"[BahlCell]   Fix: cd mechanisms && nrnivmodl . && cd ..")
        elif not BahlCell._verified_once:
            print(f"[BahlCell] OK: All {len(critical)} critical channels "
                  f"present in soma")
            BahlCell._verified_once = True

    def _collect_gbar_values(self):
        """Scan every section and build the gbar_values dict."""
        for region, sec in self.sections.items():
            seg = sec(0.5)
            for chan_name in CHANNEL_SPECS:
                if hasattr(seg, chan_name):
                    mech = getattr(seg, chan_name)
                    # Standard NEURON convention: gbar is named
                    # gNaTa_tbar_NaTa_t, but the mechanism attribute
                    # is exposed via the mechanism reference.
                    gbar_attr = f'gbar'
                    try:
                        gval = getattr(mech, gbar_attr)
                        self.gbar_values[region][chan_name] = float(gval)
                    except AttributeError:
                        # Try the long-form name NEURON sometimes uses
                        long_attr = f'g{chan_name}bar_{chan_name}'
                        try:
                            gval = getattr(seg, long_attr)
                            self.gbar_values[region][chan_name] = float(gval)
                        except AttributeError:
                            pass

    # ------------------------------------------------------------------
    # Synapse creation
    # ------------------------------------------------------------------

    def create_synapses(self, input_dict):
        """Attach synapses driven by pre-generated spike-time arrays.

        Basal and apical receive AMPA + NMDA excitatory synapses.
        Soma receives GABAa inhibitory synapses.

        Parameters
        ----------
        input_dict : dict
            Output of ``stimulation.generate_trial_inputs``.
            Keys: 'basal' (T, N_BASAL_SYN), 'apical' (T, N_APICAL_SYN),
                  'soma' (T, N_SOMA_SYN).

        Notes
        -----
        Spike times are extracted from the binary spike arrays and delivered
        via NetStim + NetCon pairs.  Each synapse gets its own NetStim so
        that spike times are independent.
        """
        h = self.h

        # Extract spike times from binary arrays
        dt = NEURON_DT_MS

        # --- Basal excitatory (AMPA + NMDA) ---
        basal_sec = self.sections['basal']
        basal_spikes = input_dict['basal']  # (T, N_BASAL_SYN)
        for syn_idx in range(basal_spikes.shape[1]):
            spike_bins = np.where(basal_spikes[:, syn_idx] > 0.5)[0]
            spike_times = spike_bins * dt
            self._attach_excitatory_synapse(basal_sec, 0.5, spike_times)

        # --- Apical excitatory (AMPA + NMDA) ---
        # Distribute across nexus and tuft for more realistic placement
        apical_spikes = input_dict['apical']  # (T, N_APICAL_SYN)
        n_apical = apical_spikes.shape[1]
        n_nexus_syn = n_apical // 2
        n_tuft_syn = n_apical - n_nexus_syn

        for syn_idx in range(n_nexus_syn):
            spike_bins = np.where(apical_spikes[:, syn_idx] > 0.5)[0]
            spike_times = spike_bins * dt
            pos = 0.3 + 0.4 * (syn_idx / max(n_nexus_syn - 1, 1))
            self._attach_excitatory_synapse(
                self.sections['nexus'], pos, spike_times
            )

        for syn_idx in range(n_tuft_syn):
            col = n_nexus_syn + syn_idx
            spike_bins = np.where(apical_spikes[:, col] > 0.5)[0]
            spike_times = spike_bins * dt
            pos = 0.2 + 0.6 * (syn_idx / max(n_tuft_syn - 1, 1))
            self._attach_excitatory_synapse(
                self.sections['tuft'], pos, spike_times
            )

        # --- Somatic inhibitory (GABAa) ---
        soma_sec = self.sections['soma']
        soma_spikes = input_dict['soma']  # (T, N_SOMA_SYN)
        for syn_idx in range(soma_spikes.shape[1]):
            spike_bins = np.where(soma_spikes[:, syn_idx] > 0.5)[0]
            spike_times = spike_bins * dt
            self._attach_inhibitory_synapse(soma_sec, 0.5, spike_times)

    def _attach_excitatory_synapse(self, sec, pos, spike_times):
        """Attach combined AMPA + NMDA synapse at sec(pos)."""
        h = self.h

        if len(spike_times) == 0:
            return

        # AMPA component (Exp2Syn)
        ampa = h.Exp2Syn(sec(pos))
        ampa.tau1 = 0.2    # ms, rise
        ampa.tau2 = 1.7    # ms, decay
        ampa.e = 0.0       # mV, reversal

        # NMDA component (Exp2Syn with slower kinetics; voltage-dependence
        # is approximate -- a full NMDA model would need a custom .mod file,
        # but Exp2Syn captures the temporal profile)
        nmda = h.Exp2Syn(sec(pos))
        nmda.tau1 = 2.0    # ms
        nmda.tau2 = 65.0   # ms
        nmda.e = 0.0       # mV

        # Create VecStim-like delivery via NetStim + event list
        # We use one NetStim per synapse with number=0, then play spike
        # times through a vector event mechanism.
        stim = h.VecStim() if hasattr(h, 'VecStim') else None

        if stim is not None:
            # Use VecStim if available (needs vecstim.mod)
            spike_vec = h.Vector(spike_times.tolist())
            stim.play(spike_vec)

            nc_ampa = h.NetCon(stim, ampa)
            nc_ampa.weight[0] = 0.0005   # uS -- AMPA weight
            nc_ampa.delay = 0.0

            nc_nmda = h.NetCon(stim, nmda)
            nc_nmda.weight[0] = 0.0002   # uS -- NMDA weight
            nc_nmda.delay = 0.0

            self.synapses.extend([ampa, nmda])
            self._netstims.extend([stim, spike_vec])
            self._netcons.extend([nc_ampa, nc_nmda])
        else:
            # Fallback: use one NetStim per spike (less efficient but
            # always works with vanilla NEURON).
            for t_spike in spike_times:
                ns = h.NetStim()
                ns.number = 1
                ns.start = float(t_spike)
                ns.noise = 0

                nc_a = h.NetCon(ns, ampa)
                nc_a.weight[0] = 0.0005
                nc_a.delay = 0.0

                nc_n = h.NetCon(ns, nmda)
                nc_n.weight[0] = 0.0002
                nc_n.delay = 0.0

                self._netstims.append(ns)
                self._netcons.extend([nc_a, nc_n])

            self.synapses.extend([ampa, nmda])

    def _attach_inhibitory_synapse(self, sec, pos, spike_times):
        """Attach GABAa inhibitory synapse at sec(pos)."""
        h = self.h

        if len(spike_times) == 0:
            return

        gaba = h.Exp2Syn(sec(pos))
        gaba.tau1 = 0.5    # ms
        gaba.tau2 = 5.0    # ms
        gaba.e = -80.0     # mV, GABAa reversal

        stim = h.VecStim() if hasattr(h, 'VecStim') else None

        if stim is not None:
            spike_vec = h.Vector(spike_times.tolist())
            stim.play(spike_vec)

            nc = h.NetCon(stim, gaba)
            nc.weight[0] = 0.001   # uS
            nc.delay = 0.0

            self.synapses.append(gaba)
            self._netstims.extend([stim, spike_vec])
            self._netcons.append(nc)
        else:
            for t_spike in spike_times:
                ns = h.NetStim()
                ns.number = 1
                ns.start = float(t_spike)
                ns.noise = 0

                nc = h.NetCon(ns, gaba)
                nc.weight[0] = 0.001
                nc.delay = 0.0

                self._netstims.append(ns)
                self._netcons.append(nc)

            self.synapses.append(gaba)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_sections(self):
        """Return dict mapping region names to NEURON Section objects.

        Returns
        -------
        dict : {str: nrn.Section}
            Keys are from config.BAHL_REGIONS.
        """
        return dict(self.sections)

    def get_gbar_values(self):
        """Return maximal conductances per channel per region.

        Returns
        -------
        dict : {region: {channel_name: gbar_value}}
            gbar in S/cm^2 (NEURON convention).
        """
        return dict(self.gbar_values)


# ======================================================================
# Module-level helpers
# ======================================================================

def _insert_mech(sec, mech_name, gbar=None):
    """Insert a mechanism into a NEURON section and optionally set gbar.

    Tries two common NEURON naming conventions for the conductance
    parameter.
    """
    try:
        sec.insert(mech_name)
    except Exception as exc:
        print(f"[WARN] Could not insert mechanism '{mech_name}' into "
              f"section '{sec.name()}': {exc}")
        return

    if gbar is not None:
        # Convention 1: gbar_<mech>  (e.g. gbar_NaTa_t)
        attr1 = f'gbar_{mech_name}'
        # Convention 2: g<mech>bar_<mech>  (e.g. gNaTa_tbar_NaTa_t)
        attr2 = f'g{mech_name}bar_{mech_name}'
        for seg in sec:
            set_ok = False
            for attr in [attr1, attr2]:
                try:
                    setattr(seg, attr, gbar)
                    set_ok = True
                    break
                except (AttributeError, NameError):
                    continue
            if not set_ok:
                # Last resort: try mechanism-level attribute
                try:
                    mech_obj = getattr(seg, mech_name)
                    mech_obj.gbar = gbar
                except Exception:
                    pass


def _insert_calcium_dynamics(sec):
    """Insert a simple calcium accumulation mechanism.

    Uses the built-in CaDynamics_E2 mechanism if available (from the
    Hay/Bahl model packages), otherwise inserts a minimal calcium
    diffusion mechanism.
    """
    try:
        sec.insert('CaDynamics_E2')
        for seg in sec:
            seg.CaDynamics_E2.decay = 80.0    # ms
            seg.CaDynamics_E2.gamma = 0.0005   # fraction
            seg.CaDynamics_E2.depth = 0.1      # um
            seg.CaDynamics_E2.minCai = 1e-4    # mM
    except Exception:
        # Fallback: ensure cai is at least initialised
        try:
            sec.insert('cadyn')
        except Exception:
            # If no calcium dynamics mechanism is available, we still
            # want cai to exist for recording purposes.
            try:
                sec.insert('cai')
            except Exception:
                pass
