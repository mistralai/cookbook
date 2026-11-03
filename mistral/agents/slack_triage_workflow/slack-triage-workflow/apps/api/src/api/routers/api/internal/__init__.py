"""Infrastructure routes: unversioned, and the only ones served without a caller.

These paths are wired into gateways, startup checks, probes and healthchecks, so they must not move
when the API surface gains a version. ``health.py`` serves all of them. ``/health`` always answers
200 for a human; the ``/health/*`` paths are the Kubernetes probes and stay off the gateway because
the kubelet reaches the pod directly. The probes stay out of the OpenAPI spec too: nothing calls
them through the generated client or a connector.
"""

from fastapi import params

# This package names no URL segment, so its modules hang directly off the API prefix: ``health.py``
# is ``/health``, not ``/internal/health``.
segment = ""

# The access rule, cascading to every module here. Empty is the declaration, not an omission: these
# paths are served without a caller. A subtree that omits it raises ``UndeclaredAccessRuleError``.
dependencies: tuple[params.Depends, ...] = ()
