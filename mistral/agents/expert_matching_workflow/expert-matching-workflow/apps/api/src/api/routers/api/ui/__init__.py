"""The web page. It holds no data, so it is served without a caller; the API calls it makes are not."""

from fastapi import params

# Serve the page at the API prefix itself: ``route.py`` is ``/api``.
segment = ""
dependencies: tuple[params.Depends, ...] = ()
