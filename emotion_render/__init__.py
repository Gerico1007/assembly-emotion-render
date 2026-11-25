"""Acted speech for the G.Music Assembly: a tagged script in, a voiced wav out.

Two rules hold everywhere in this package:

I1  A tag is consumed or refused. It is never spoken and never dropped silently.
I2  No cloned voice without the consent of the person whose voice it is.
"""

__version__ = "0.1.0"


class Refused(ValueError):
    """The render cannot happen as asked. The message says why and what is accepted.

    `code` is stable, so a program can act on it (the MCP server returns it as is).
    `detail` carries what the caller can use to fix the request, such as the tags an
    engine accepts. The message is what a person reads, and the CLI prints it unchanged.
    """

    def __init__(self, message: str, code: str = "refused", **detail):
        super().__init__(message)
        self.code = code
        self.detail = detail

    @property
    def message(self) -> str:
        return str(self)

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, **self.detail}
