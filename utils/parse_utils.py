
import re
from typing import List, Dict

TAG_LIST = ["logic", "commonsense", "focus", "spatial", "external", "conclude"]     

def parse_think_answer(raw: str) -> Dict[str, object]:
    think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL | re.IGNORECASE)

    if not (think_match and answer_match):
        raise ValueError("Input lacks required tags.")

    think_block = think_match.group(1).strip()
    answer = answer_match.group(1).strip()

    reason_steps: List[str] = []
    tag_traces: List[str] = []

    tag_regex = re.compile(rf"<({'|'.join(TAG_LIST)})>\s*(.*)", flags=re.IGNORECASE)

    for line in think_block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)   # remove "1. ", "2. " etc.
        m = tag_regex.match(line)
        if m:
            tag = m.group(1).lower()
            step_txt = m.group(2).strip()
            # remove possible closing tag at the end
            step_txt = re.sub(r"</[^>]+>\s*$", "", step_txt).strip()
            tag_traces.append(f"<{tag}>")
            reason_steps.append(step_txt)

    return {
        "answer": answer,
        "plan": reason_steps,
        "tag_trace": tag_traces,
    }

def extract_last_boxed_text(raw: str) -> str:
    """
    Return the LaTeX code inside the *last* \\boxed{...} in `raw`.
    If no \\boxed{...} exists or braces never balance, return "".
    """
    if raw is None:
        return ""
    key = r'\boxed{'
    start = raw.rfind(key)          # find *last* occurrence
    if start == -1:
        return ""

    i = start + len(key)            # index of first char after the opening brace
    depth = 1                       # we are already inside one '{'
    out = []

    while i < len(raw) and depth:
        ch = raw[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:          # matched the initial '{' → finished
                break
        out.append(ch)
        i += 1

    return ''.join(out).strip()
    