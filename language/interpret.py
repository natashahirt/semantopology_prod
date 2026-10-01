"""Turn a sentence into the image motive CLIP should see.

The building is not extracted. ``run.py`` already received it as ``--problem``.
Paper figures pass ``--clip`` and never call this module: a language model
will not reproduce a published prompt string.
"""

from __future__ import annotations

from typing import Callable, Optional

_SYSTEM = (
    'Extract only the image motive from the user message: the visual subject '
    'a structure should resemble. Reply with that phrase alone. Do not name '
    'a building type, and do not add a label.'
)


def interpret_motive(sentence: str, *, complete: Optional[Callable[[str], str]] = None) -> str:
    """Return the CLIP text for ``sentence``.

    ``complete`` receives the sentence and returns the model text. The default
    calls GPT-4 through the ``openai`` package and ``OPENAI_API_KEY``.
    """
    text = sentence.strip()
    if not text:
        raise ValueError('sentence must be non-empty')
    if complete is None:
        complete = _openai_complete
    motive = complete(text).strip().strip('"').strip()
    lowered = motive.lower()
    for prefix in ('image motive:', 'motive:'):
        if lowered.startswith(prefix):
            motive = motive.split(':', 1)[1].strip()
            break
    if not motive:
        raise RuntimeError('language step returned an empty image motive')
    return motive


def _openai_complete(sentence: str) -> str:
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            'The language step needs the openai package and OPENAI_API_KEY. '
            'Paper figures use --clip and do not call this.') from exc
    response = openai.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': _SYSTEM},
            {'role': 'user', 'content': sentence},
        ],
    )
    return response.choices[0].message.content or ''
