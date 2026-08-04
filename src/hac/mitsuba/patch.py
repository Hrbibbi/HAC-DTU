from typing import Any, cast

import mitsuba as _mi

# This patches Mitsuba to work slightly better with static type checking tools
# Without this, Pylance (and others, I expect) paints everything red :(
mi = cast(Any, _mi)  # type: ignore
