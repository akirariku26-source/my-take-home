"""
Sentence accumulator for streaming TTS.

Voice agent use-case: an LLM streams text token-by-token.  Rather than waiting
for the full response, we detect sentence boundaries and submit each sentence
to TTS as soon as it's complete.  This cuts first-audio latency from
"full LLM generation time" to "first sentence generation time".
"""

import re

# Matches the end of a sentence: .!? optionally followed by whitespace or EOL.
# Also splits on newlines so paragraph breaks trigger a flush.
_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"     # .!? followed by whitespace
    r"|(?<=[.!?])$"       # .!? at end of string
    r"|(?<=\n)\s*"         # newline
)

_MIN_LEN = 3  # Don't dispatch fragments shorter than this


class SentenceAccumulator:
    """
    Buffer incoming text chunks and yield complete sentences.

    Usage::

        acc = SentenceAccumulator()

        for token in llm_stream:
            for sentence in acc.push(token):
                audio = await tts.synthesize(sentence, ...)

        for sentence in acc.flush():
            audio = await tts.synthesize(sentence, ...)
    """

    def __init__(self) -> None:
        self._buf = ""

    def push(self, text: str) -> list[str]:
        """
        Append *text* to the internal buffer.
        Returns a (possibly empty) list of complete sentences ready for TTS.
        """
        self._buf += text
        parts = _BOUNDARY.split(self._buf)

        if len(parts) <= 1:
            return []

        # All parts except the last are complete.
        complete = [p.strip() for p in parts[:-1] if p.strip() and len(p.strip()) >= _MIN_LEN]
        self._buf = parts[-1]
        return complete

    def flush(self) -> list[str]:
        """Return any remaining buffered text as a final sentence."""
        remaining = self._buf.strip()
        self._buf = ""
        return [remaining] if remaining and len(remaining) >= _MIN_LEN else []

    def reset(self) -> None:
        """Discard the buffer (e.g. on client disconnect)."""
        self._buf = ""
