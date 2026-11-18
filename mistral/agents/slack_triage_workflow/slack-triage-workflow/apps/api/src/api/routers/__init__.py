"""Every router this app serves. The directory tree is the URL tree.

Each module serves (``router``) or declares (``segment``, ``dependencies``). A module lives in one
group, which is the access declaration: ``internal`` is unversioned and anonymous, and an
authenticated group such as ``v1`` declares ``dependencies`` that cascade to everything beneath
it. A new group must declare ``dependencies``, or the import fails, so a group cannot default to
public.
"""
