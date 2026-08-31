"""Make TomaNet modules visible to ultralytics' YAML parser.

`ultralytics.nn.tasks.parse_model` resolves a YAML module name through the *globals of
that module*, so the classes have to be bound there. Two things are needed:

  1. Name binding      - handled here, at runtime, no source edit required.
  2. Channel handling  - `parse_model` only injects (c1, c2) and applies the width
                         multiplier for modules listed in its local `base_modules`
                         frozenset. That set is a local variable, so it cannot be
                         reached from outside; `scripts/patch_ultralytics.py` edits the
                         installed source to add our names.

Import this module before building a model from a TomaNet YAML.
"""

from __future__ import annotations

from tomanet.modules import CoordAtt, TomaBlock, TomaLayer

_EXPORTS = (CoordAtt, TomaBlock, TomaLayer)


def register() -> list[str]:
    """Bind TomaNet modules into ultralytics' namespaces. Returns the names bound."""
    import ultralytics.nn.modules as ul_modules
    import ultralytics.nn.tasks as ul_tasks

    for cls in _EXPORTS:
        setattr(ul_tasks, cls.__name__, cls)
        setattr(ul_modules, cls.__name__, cls)

    return [cls.__name__ for cls in _EXPORTS]


def is_patched() -> bool:
    """True if the installed ultralytics source carries the TomaNet channel patch."""
    from pathlib import Path

    import ultralytics.nn.tasks as ul_tasks

    return "TOMANET PATCH" in Path(ul_tasks.__file__).read_text(encoding="utf-8")
