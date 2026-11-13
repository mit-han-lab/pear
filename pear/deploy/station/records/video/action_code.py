import json
from typing import Any


class ActionCode:
    """A model's action as lines of JSON: the braces alone, one top-level key on each line, and an object wider than
    the column limit wrapped."""

    INDENT = "  "  # one level of JSON indentation
    NO_ACTION = "No action"  # shown for a turn without an action

    def __init__(self, action: dict[str, Any] | None, max_columns: int | None = None) -> None:
        self.max_columns = max_columns  # the characters a line may take; None: as many as it needs

        if action is None:
            lines = [[(self.NO_ACTION, "note")]]
        else:
            lines = [[("{", "p")], *self._members(action), [("}", "p")]]

        # the syntax roles: p punctuation, k a top-level key, key an inner key, s another value, num a zero, lit a
        # number that is not zero, note a remark in place of the action
        self.spans: list[tuple[int, int, str, str]] = []  # (line, column, text, syntax role)
        for line_index, line in enumerate(lines):
            column = 0
            for text, syntax_role in line:
                if syntax_role:
                    self.spans.append((line_index, column, text, syntax_role))
                column += len(text)

        self.line_count = len(lines)
        self.column_count = max(sum(len(text) for text, _ in line) for line in lines)

    def _members(self, action: dict[str, Any]) -> list[list[tuple[str, str | None]]]:
        """Returns the lines of the action's members, each a list of (text, syntax role) pieces."""
        lines = []
        for member_index, (key, value) in enumerate(action.items()):
            comma = [(",", "p")] if member_index < len(action) - 1 else []
            line = [(self.INDENT, None), *self._member(key, value, "k", is_expanded=True), *comma]
            if (
                self.max_columns is None
                or self._length(line) <= self.max_columns
                or not isinstance(value, dict)
                or not value
            ):
                lines.append(line)
                continue

            lines.append([(self.INDENT, None), ('"', "p"), (key, "k"), ('": {', "p")])
            row = [(self.INDENT * 2, None)]
            for item_index, (inner_key, item) in enumerate(value.items()):
                piece = self._member(inner_key, item, "key", is_expanded=False)
                piece += [(",", "p")] if item_index < len(value) - 1 else []
                if len(row) > 1 and self._length(row) + 1 + self._length(piece) > self.max_columns:
                    lines.append(row)
                    row = [(self.INDENT * 2, None)]
                elif len(row) > 1:
                    row.append((" ", None))
                row += piece

            lines += [row, [(self.INDENT, None), ("}", "p"), *comma]]

        return lines

    def _member(self, key: str, value: Any, syntax_role: str, is_expanded: bool) -> list[tuple[str, str | None]]:
        """Returns the pieces of one member: its key, in ``syntax_role``, and its value."""
        return [('"', "p"), (key, syntax_role), ('": ', "p"), *self._value(value, is_expanded)]

    def _value(self, value: Any, is_expanded: bool) -> list[tuple[str, str | None]]:
        """Returns the pieces of one value; an object is written out on the line only when it is expanded."""
        if isinstance(value, dict) and is_expanded:
            pieces = [("{", "p")]
            for item_index, (key, item) in enumerate(value.items()):
                pieces += [(", ", "p")] if item_index else []
                pieces += self._member(key, item, "key", is_expanded=False)
            return pieces + [("}", "p")]

        if isinstance(value, str):
            return [('"', "p"), (value, "s"), ('"', "p")]

        text = json.dumps(value)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return [(text, "lit" if value else "num")]

        return [(text, "s")]

    @staticmethod
    def _length(pieces: list[tuple[str, str | None]]) -> int:
        """Returns the characters the pieces take."""
        return sum(len(text) for text, _ in pieces)
