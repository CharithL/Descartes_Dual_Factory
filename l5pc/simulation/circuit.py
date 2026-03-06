"""Phase 3: Cortical microcircuit builder using NetPyNE.

100-neuron circuit: L5PCs (replacement targets) + PV+ basket cells +
SOM+ Martinotti + L2/3 pyramidal + L4 spiny stellate.
"""
import numpy as np
from l5pc.config import CIRCUIT_CELL_COUNTS


def build_circuit_config(replacement_indices=None):
    """Build NetPyNE simulation configuration for the cortical microcircuit.

    Args:
        replacement_indices: Array of L5PC indices to replace with surrogates.
            None = all biological (baseline).

    Returns:
        NetPyNE simConfig and netParams objects
    """
    try:
        from netpyne import specs, sim
    except ImportError:
        raise ImportError("NetPyNE required for Phase 3. pip install netpyne")

    netParams = specs.NetParams()
    simConfig = specs.SimConfig()

    # --- Cell types ---
    # L5 thick-tufted pyramidal cell (biological)
    netParams.cellParams['L5PC_bio'] = {
        'conds': {'cellType': 'L5PC', 'cellModel': 'bio'},
        'secs': _l5pc_section_params(),
    }

    # L5PC surrogate (point process placeholder — replaced at runtime)
    netParams.cellParams['L5PC_surr'] = {
        'conds': {'cellType': 'L5PC', 'cellModel': 'surrogate'},
        'secs': {
            'soma': {
                'geom': {'diam': 20, 'L': 20, 'nseg': 1},
                'mechs': {'pas': {'g': 0.0001, 'e': -70}},
            }
        },
    }

    # PV+ basket cell (fast-spiking interneuron)
    netParams.cellParams['PV_basket'] = {
        'conds': {'cellType': 'PV'},
        'secs': _pv_section_params(),
    }

    # SOM+ Martinotti cell
    netParams.cellParams['SOM_martinotti'] = {
        'conds': {'cellType': 'SOM'},
        'secs': _som_section_params(),
    }

    # L2/3 pyramidal (simplified)
    netParams.cellParams['L23_pyr'] = {
        'conds': {'cellType': 'L23'},
        'secs': _l23_section_params(),
    }

    # L4 spiny stellate (simplified)
    netParams.cellParams['L4_stellate'] = {
        'conds': {'cellType': 'L4'},
        'secs': _l4_section_params(),
    }

    # --- Populations ---
    n_l5pc = CIRCUIT_CELL_COUNTS['L5PC']
    if replacement_indices is None:
        replacement_indices = []
    replacement_indices = set(replacement_indices)

    # Split L5PCs into bio and surrogate
    n_bio = n_l5pc - len(replacement_indices)
    n_surr = len(replacement_indices)

    netParams.popParams['L5PC_bio'] = {
        'cellType': 'L5PC', 'cellModel': 'bio',
        'numCells': n_bio,
    }
    if n_surr > 0:
        netParams.popParams['L5PC_surr'] = {
            'cellType': 'L5PC', 'cellModel': 'surrogate',
            'numCells': n_surr,
        }

    netParams.popParams['PV'] = {
        'cellType': 'PV', 'numCells': CIRCUIT_CELL_COUNTS['PV_basket'],
    }
    netParams.popParams['SOM'] = {
        'cellType': 'SOM', 'numCells': CIRCUIT_CELL_COUNTS['SOM_martinotti'],
    }
    netParams.popParams['L23'] = {
        'cellType': 'L23', 'numCells': CIRCUIT_CELL_COUNTS['L23_pyr'],
    }
    netParams.popParams['L4'] = {
        'cellType': 'L4', 'numCells': CIRCUIT_CELL_COUNTS['L4_stellate'],
    }

    # --- Connectivity ---
    # L4 → L5PC (feedforward)
    netParams.connParams['L4->L5PC'] = {
        'preConds': {'cellType': 'L4'},
        'postConds': {'cellType': 'L5PC'},
        'probability': 0.1,
        'weight': 0.005,
        'delay': 2.0,
        'synMech': 'AMPA',
        'sec': 'basal' if 'basal' in netParams.cellParams.get('L5PC_bio', {}).get('secs', {}) else 'soma',
    }

    # L2/3 → L5PC apical (top-down)
    netParams.connParams['L23->L5PC'] = {
        'preConds': {'cellType': 'L23'},
        'postConds': {'cellType': 'L5PC'},
        'probability': 0.08,
        'weight': 0.003,
        'delay': 3.0,
        'synMech': 'AMPA',
        'sec': 'apical_trunk',
    }

    # L5PC → L5PC recurrent
    netParams.connParams['L5PC->L5PC'] = {
        'preConds': {'cellType': 'L5PC'},
        'postConds': {'cellType': 'L5PC'},
        'probability': 0.05,
        'weight': 0.002,
        'delay': 1.5,
        'synMech': 'AMPA',
    }

    # PV → L5PC soma (perisomatic inhibition, drives gamma)
    netParams.connParams['PV->L5PC'] = {
        'preConds': {'cellType': 'PV'},
        'postConds': {'cellType': 'L5PC'},
        'probability': 0.4,
        'weight': 0.01,
        'delay': 1.0,
        'synMech': 'GABA',
        'sec': 'soma',
    }

    # SOM → L5PC dendrites (dendritic inhibition)
    netParams.connParams['SOM->L5PC'] = {
        'preConds': {'cellType': 'SOM'},
        'postConds': {'cellType': 'L5PC'},
        'probability': 0.2,
        'weight': 0.005,
        'delay': 2.0,
        'synMech': 'GABA',
        'sec': 'apical_trunk',
    }

    # L5PC → PV (excitatory drive for PING)
    netParams.connParams['L5PC->PV'] = {
        'preConds': {'cellType': 'L5PC'},
        'postConds': {'cellType': 'PV'},
        'probability': 0.3,
        'weight': 0.005,
        'delay': 1.0,
        'synMech': 'AMPA',
    }

    # L5PC → SOM
    netParams.connParams['L5PC->SOM'] = {
        'preConds': {'cellType': 'L5PC'},
        'postConds': {'cellType': 'SOM'},
        'probability': 0.15,
        'weight': 0.003,
        'delay': 1.5,
        'synMech': 'AMPA',
    }

    # --- Synapse mechanisms ---
    netParams.synMechParams['AMPA'] = {
        'mod': 'Exp2Syn', 'tau1': 0.5, 'tau2': 2.0, 'e': 0,
    }
    netParams.synMechParams['GABA'] = {
        'mod': 'Exp2Syn', 'tau1': 1.0, 'tau2': 5.0, 'e': -80,
    }

    # --- Background input ---
    netParams.stimSourceParams['background'] = {
        'type': 'NetStim', 'rate': 5, 'noise': 1.0,
    }
    netParams.stimTargetParams['bg->all'] = {
        'source': 'background',
        'conds': {},
        'weight': 0.001,
        'delay': 1.0,
        'synMech': 'AMPA',
    }

    # --- Simulation config ---
    simConfig.duration = 1000  # ms
    simConfig.dt = 0.025
    simConfig.recordStep = 0.5
    simConfig.recordCells = ['all']
    simConfig.recordTraces = {'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
    simConfig.analysis = {
        'plotRaster': {'saveFig': True},
        'plotTraces': {'include': [0, 1, 2], 'saveFig': True},
    }
    simConfig.saveJson = True
    simConfig.verbose = False

    return netParams, simConfig


def run_circuit(replacement_indices=None):
    """Build and simulate the circuit.

    Returns:
        dict with simulation results for validation
    """
    try:
        from netpyne import sim
    except ImportError:
        raise ImportError("NetPyNE required for Phase 3")

    netParams, simConfig = build_circuit_config(replacement_indices)
    sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)

    # Extract results
    spike_trains = _extract_spike_trains(sim)
    lfp = _estimate_lfp(sim)

    return {
        'spike_trains': spike_trains,
        'lfp': lfp,
        'spike_counts': spike_trains.sum(axis=1) if spike_trains is not None else None,
        'sim_data': sim.allSimData,
    }


def _extract_spike_trains(sim_obj, dt_ms=0.5, duration_ms=1000):
    """Extract binary spike trains from NetPyNE simulation."""
    T = int(duration_ms / dt_ms)
    n_cells = len(sim_obj.net.cells)
    trains = np.zeros((n_cells, T), dtype=np.float32)

    spk_times = sim_obj.allSimData.get('spkt', [])
    spk_ids = sim_obj.allSimData.get('spkid', [])

    for t, gid in zip(spk_times, spk_ids):
        gid = int(gid)
        t_idx = int(t / dt_ms)
        if 0 <= gid < n_cells and 0 <= t_idx < T:
            trains[gid, t_idx] = 1.0

    return trains


def _estimate_lfp(sim_obj):
    """Estimate LFP as sum of membrane potentials (crude approximation)."""
    traces = sim_obj.allSimData.get('V_soma', {})
    if not traces:
        return np.zeros(100)
    all_v = []
    for cell_key, v_trace in traces.items():
        all_v.append(np.array(v_trace))
    if all_v:
        return np.mean(all_v, axis=0)
    return np.zeros(100)


# --- Simplified section parameters ---
# These define minimal multicompartment neurons for the circuit.
# Full morphologies can be substituted from ModelDB.

def _l5pc_section_params():
    """Simplified L5PC sections for circuit context."""
    return {
        'soma': {
            'geom': {'diam': 20, 'L': 20, 'nseg': 1, 'Ra': 100, 'cm': 1},
            'mechs': {
                'pas': {'g': 0.00003, 'e': -70},
                'NaTa_t': {'gNaTa_tbar': 2.04},
                'SKv3_1': {'gSKv3_1bar': 0.693},
            },
        },
        'basal': {
            'geom': {'diam': 3, 'L': 200, 'nseg': 3, 'Ra': 100, 'cm': 2},
            'mechs': {'pas': {'g': 0.00003, 'e': -70}},
            'topol': {'parentSec': 'soma', 'parentX': 0, 'childX': 0},
        },
        'apical_trunk': {
            'geom': {'diam': 5, 'L': 400, 'nseg': 5, 'Ra': 100, 'cm': 2},
            'mechs': {
                'pas': {'g': 0.00003, 'e': -70},
                'Ih': {'gIhbar': 0.0002},
            },
            'topol': {'parentSec': 'soma', 'parentX': 1, 'childX': 0},
        },
    }


def _pv_section_params():
    return {
        'soma': {
            'geom': {'diam': 15, 'L': 15, 'nseg': 1, 'Ra': 100, 'cm': 1},
            'mechs': {'pas': {'g': 0.0001, 'e': -65}},
        },
    }


def _som_section_params():
    return {
        'soma': {
            'geom': {'diam': 15, 'L': 15, 'nseg': 1, 'Ra': 100, 'cm': 1},
            'mechs': {'pas': {'g': 0.00005, 'e': -65}},
        },
    }


def _l23_section_params():
    return {
        'soma': {
            'geom': {'diam': 18, 'L': 18, 'nseg': 1, 'Ra': 100, 'cm': 1},
            'mechs': {'pas': {'g': 0.00003, 'e': -70}},
        },
    }


def _l4_section_params():
    return {
        'soma': {
            'geom': {'diam': 15, 'L': 15, 'nseg': 1, 'Ra': 100, 'cm': 1},
            'mechs': {'pas': {'g': 0.00005, 'e': -70}},
        },
    }
