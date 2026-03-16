"""
L5PC DESCARTES -- Probe Availability Registry

Import-time dependency checking for optional packages.
One WARNING per session per missing package, then silent.
AVAILABLE_PROBES dict gates the orchestrator.
"""

import logging

logger = logging.getLogger(__name__)

AVAILABLE_PROBES = {}
_WARNED = set()


def _check_optional(probe_name, package_name, install_hint):
    """Check if an optional dependency is available. Warn once if not."""
    global AVAILABLE_PROBES
    try:
        __import__(package_name)
        AVAILABLE_PROBES[probe_name] = True
    except ImportError:
        AVAILABLE_PROBES[probe_name] = False
        if probe_name not in _WARNED:
            logger.warning(
                "WARNING: %s not installed — %s probes disabled (%s)",
                package_name, probe_name, install_hint,
            )
            _WARNED.add(probe_name)


def is_available(probe_name):
    """Check if a probe is available for scheduling."""
    return AVAILABLE_PROBES.get(probe_name, False)


# === Core probes (always available — PyTorch + sklearn + scipy) ===
AVAILABLE_PROBES['ridge'] = True
AVAILABLE_PROBES['mlp'] = True
AVAILABLE_PROBES['sae'] = True
AVAILABLE_PROBES['hardening'] = True
AVAILABLE_PROBES['resample_ablation'] = True
AVAILABLE_PROBES['cca'] = True      # sklearn.cross_decomposition
AVAILABLE_PROBES['rsa'] = True      # scipy.spatial.distance
AVAILABLE_PROBES['cka'] = True      # numpy only
AVAILABLE_PROBES['koopman'] = True  # scipy.linalg + numpy
AVAILABLE_PROBES['dsa'] = True      # numpy.linalg
AVAILABLE_PROBES['mine'] = True     # PyTorch
AVAILABLE_PROBES['mdl'] = True      # sklearn + numpy
AVAILABLE_PROBES['temporal'] = True  # sklearn + numpy
AVAILABLE_PROBES['gate_specific'] = True  # PyTorch
AVAILABLE_PROBES['transfer_entropy'] = True  # sklearn.metrics
AVAILABLE_PROBES['das'] = True      # sklearn + PyTorch
AVAILABLE_PROBES['frequency'] = True  # scipy.signal
AVAILABLE_PROBES['pi_vae'] = True   # PyTorch

# === Optional probes (require extra packages) ===
_check_optional('tda', 'ripser', 'pip install ripser persim')
_check_optional('sindy', 'pysindy', 'pip install pysindy')
_check_optional('cebra', 'cebra', 'pip install cebra')


def get_available_probe_names():
    """Return list of all available probe names."""
    return [name for name, available in AVAILABLE_PROBES.items() if available]


def get_unavailable_probe_names():
    """Return list of all unavailable probe names."""
    return [name for name, available in AVAILABLE_PROBES.items() if not available]
