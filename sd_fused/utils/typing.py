# type: ignore

try:
    from typing import Literal, Final, Protocol
except ImportError:
    from typing_extensions import Literal, Final, Protocol
