"""
Sentence accumulator for streaming TTS.

Voice agent use-case: an LLM streams text token-by-token.  Rather than waiting
for the full response, we detect sentence boundaries and submit each sentence
to TTS as soon as it's complete.  This cuts first-audio latency from
"full LLM generation time" to "first sentence generation time".
"""

import re

# Matches dispatch boundaries in order of strength:
#   Strong — sentence-ending punctuation (.!?) or newline: dispatch at any length ≥ _MIN_LEN
#   Soft   — clause boundaries (,;:—) followed by whitespace: dispatch when fragment is long
#            enough that the TTS model gets meaningful prosodic context
_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"     # strong: sentence-ending punctuation + whitespace
    r"|(?<=[.!?])$"       # strong: sentence-ending at end of string
    r"|(?<=\n)\s*"        # strong: newline / paragraph break
    r"|(?<=[,;:\u2014])\s+"  # soft: comma / semicolon / colon / em-dash + whitespace
)

# Minimum fragment length before dispatching to TTS.  Short fragments (e.g.
# "Hi,") carry forward and merge with the next part rather than being dropped
# or sent as tiny isolated synthesis calls.
_MIN_LEN = 10


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
        Returns a (possibly empty) list of dispatch-ready fragments.

        Fragments shorter than _MIN_LEN are not dropped — they are carried
        forward and merged into the next fragment so no text is ever lost.
        """
        self._buf += text
        parts = _BOUNDARY.split(self._buf)

        if len(parts) <= 1:
            return []

        complete = []
        carry = ""
        for part in parts[:-1]:
            candidate = f"{carry} {part}".strip() if carry else part.strip()
            if candidate and len(candidate) >= _MIN_LEN:
                complete.append(candidate)
                carry = ""
            else:
                carry = candidate  # too short — merge into next fragment

        last = parts[-1]
        self._buf = f"{carry} {last}".strip() if carry else last
        return complete

    def flush(self) -> list[str]:
        """Return any remaining buffered text as a final fragment.

        Dispatches regardless of length — a flush is an explicit end-of-turn
        signal so even a short remainder must be synthesized.
        """
        remaining = self._buf.strip()
        self._buf = ""
        return [remaining] if remaining else []

    def reset(self) -> None:
        """Discard the buffer (e.g. on client disconnect)."""
        self._buf = ""
