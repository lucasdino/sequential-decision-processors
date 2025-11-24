import re
from typing import List, Tuple

ACTION_RE = re.compile(r"<action>(.*?)</action>", flags=re.DOTALL)
THINK_RE  = re.compile(r"<think>(.*?)</think>",   flags=re.DOTALL)

def general_projection(prompt, actions: List[str]) -> Tuple[List[str], List[int], List[str]]:
    """
    - Take the LAST <action>...</action> (case-sensitive tags).
    - Valid iff an <action> block exists AND both <think> and </think> exist in the original.
    - If no <action>, use the last 30 chars of the lowercased input and mark invalid.
    """
    n = len(actions)
    parsed_actions  = ["Action not provided correctly. Ensure your action is in <action>...</action> tags and you don't think too long."] * n
    valids          = [0]  * n
    parsed_thinking = ["Invalid thinking trace: remember to enclose thinking traces in <think>...</think> tags."] * n

    for i, generation in enumerate(actions):
        # last <action>…</action>
        am = list(ACTION_RE.finditer(generation))
        if am:
            parsed_actions[i] = am[-1].group(1).strip()  # keep original casing/content
            valids[i] = 1
        # else:
        #     parsed_actions[i] = generation.lower()[-125:]

        # last <think>…</think> must exist (case-sensitive)
        tm = list(THINK_RE.finditer(generation))
        if tm:
            parsed_thinking[i] = tm[-1].group(1).strip()
        else:
            valids[i] = 0

    return parsed_actions, valids, parsed_thinking