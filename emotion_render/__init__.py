"""Acted speech for the G.Music Assembly: a tagged script in, a voiced wav out.

Two rules hold everywhere in this package:

I1  A tag is consumed or refused. It is never spoken and never dropped silently.
I2  No cloned voice without the consent of the person whose voice it is.
"""

__version__ = "0.1.0"


class Refused(ValueError):
    """The render cannot happen as asked. The message says why and what is accepted."""
