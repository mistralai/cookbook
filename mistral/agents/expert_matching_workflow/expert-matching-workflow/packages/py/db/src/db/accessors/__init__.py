"""Table accessors, one module per table, contributed by the capability that owns the table.

A model describes a row; an accessor is the only place that reads or writes one, so the hosts stay
free of ORM queries. The base package ships none: each owning capability drops its
``db/accessors/<table>.py`` here, and a caller imports it directly --
``from db.accessors.<table> import <fn>`` -- so nothing in the base has to name it.
"""
