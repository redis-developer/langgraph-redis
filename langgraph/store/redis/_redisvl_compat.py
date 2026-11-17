"""Escaping for store keys in ``@key:{...}`` tag queries, until redisvl has it.

This copies the tag value handling of ``TokenEscaper.for_tags()`` from
https://github.com/redis/redis-vl-python/pull/837 (at f665bc4).
"""

import re

_QUERY_SYNTAX = ",.<>{}[]\\\"':;!@#$%^&()-+=~/ "
# Store keys are literal values, so wildcards are escaped too.
_WILDCARDS = "*?"
# The query parser reads these as a term break inside a tag value unless they
# are escaped.
_TAG_SEPARATORS = "`\t\n\r\x0b\x0c"
# Redis trims these from both ends of a tag value when it indexes it (on JSON
# too, because the store's key fields set SEPARATOR ","), so they stay
# unescaped at the ends of a key and the query drops them the same way.
_TAG_EDGE_WHITESPACE = "\t\n\r\x0b\x0c"


def _char_class(chars: str) -> str:
    return "[" + "".join(re.escape(char) for char in chars) + "]"


class _TagValueEscaper:
    """Escape a literal tag value the way ``TokenEscaper.for_tags()`` does."""

    _escaped_chars_re: re.Pattern[str] = re.compile(
        _char_class(_QUERY_SYNTAX + _WILDCARDS + "|" + _TAG_SEPARATORS)
    )

    def escape(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"Value must be a string object for token escaping, got type {type(value)}"
            )
        core = value.strip(_TAG_EDGE_WHITESPACE)
        if not core:
            return value
        start = len(value) - len(value.lstrip(_TAG_EDGE_WHITESPACE))
        end = start + len(core)
        escaped = self._escaped_chars_re.sub(r"\\\g<0>", core)
        return value[:start] + escaped + value[end:]


# TODO(https://github.com/redis/redis-vl-python/pull/837): once a redisvl
# release ships TokenEscaper.for_tags(), replace this with
# `_tag_escaper = TokenEscaper.for_tags()` and delete the rest of this module.
_tag_escaper = _TagValueEscaper()
