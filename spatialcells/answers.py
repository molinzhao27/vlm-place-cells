"""The answer a free-text API reply states, for the two-choice test.

Models that reason in the visible reply name both options along the way, so the first option mentioned is not the
answer (claude_test.py and api_test.py scored it that way until 2026-11-01). The rule, in order:
  1. the last line is only an option number (bare, bold, fenced, or "Frame N");
  2. an <answer> tag;
  3. the first line opens with an option number, bare or bold (the answer, then a caveat or the reasoning);
  4. the last line says "closer to", "nearer to" or "close to" something: the first option named after it;
  5. the options in bold: the only one, or the last one if both are;
  6. the last line, unless a bullet, names exactly one option;
  7. the reply names exactly one option.
Anything else is unparsed (-1) and counts as wrong, like every unparsed answer. two_choice.py's own parser is unchanged:
open models answer with a number, and no reply there names both options outside the thinking runs.
"""
import re

ONLY = re.compile(r"[\*\s_`#>]*(?:(?:final\s+)?(?:answer|frame)\s*:?\s*)?\**\s*0*(\d+)[\*\s_`\.]*", re.I)
TAG = re.compile(r"<answer>\s*(?:frame\s*)?0*(\d+)\s*</answer>", re.I)
OPENS = re.compile(r"[\*\s_`#>]*(?:frame\s*)?0*(\d+)\b\**(?!['’])", re.I)
BOLD = re.compile(r"\*\*\s*(?:Frame\s*)?0*(\d+)\s*\.?\s*\*\*", re.I)
CLOSER = re.compile(r"(?:closer|nearer|close)\s+to\b", re.I)


def stated_answer(text, a, b, tail_only=False):
    """The option (a or b) this reply states, else -1. Options must be ints; zero-padded numbers in the text match.
    tail_only: the text is the end of a longer reply (api_test.py stores 600 characters), so it has no first line."""
    text = (text or "").split("</think>")[-1].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip(" `")]      # code-fence lines are not content
    if not lines:
        return -1
    opts = (a, b)
    in_opts = lambda s: [int(x) for x in re.findall(r"\d+", s) if int(x) in opts]
    m = ONLY.fullmatch(lines[-1])
    if m and int(m.group(1)) in opts:
        return int(m.group(1))
    tags = [int(x) for x in TAG.findall(text) if int(x) in opts]
    if tags:
        return tags[-1]
    m = None if tail_only else OPENS.match(lines[0])
    if m and int(m.group(1)) in opts and (len(lines) > 1 or ONLY.fullmatch(lines[0]) or lines[0].startswith("*")):
        return int(m.group(1))
    close = list(CLOSER.finditer(lines[-1]))
    if close:
        after = in_opts(lines[-1][close[-1].end():])
        if after:
            return after[0]
    bold = [int(x) for x in BOLD.findall(text) if int(x) in opts]
    if bold:
        return bold[-1]
    last = set(in_opts(lines[-1]))
    if len(last) == 1 and not lines[-1].startswith(("-", "*", "•")):   # a bullet describes an option, it doesn't choose one
        return last.pop()
    named = set(in_opts(text))
    return named.pop() if len(named) == 1 else -1
