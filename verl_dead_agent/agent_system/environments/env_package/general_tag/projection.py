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
    parsed_actions  = ["[ERROR] Action unable to be parsed."] * n
    valids          = [0]  * n
    parsed_thinking = ["[ERROR] Thinking trace unable to be parsed."] * n

    for i, generation in enumerate(actions):
        # last <action>…</action>
        am = list(ACTION_RE.finditer(generation))
        if am:
            if len(am[-1].group(1)) > 150:
                parsed_actions[i] = f"[ERROR] Parsed action was too long (>150 chars). Final 100 chars of action: {am[-1].group(1)[-100].strip()}"
            else:
                parsed_actions[i] = am[-1].group(1).strip()  # keep original casing/content
                valids[i] = 1
        else:
            parsed_actions[i] += f" Final 125 chars of your last generation: {generation[-125:].strip()}"
        # else:
        #     parsed_actions[i] = generation.lower()[-125:]

        # last <think>…</think> must exist (case-sensitive)
        tm = list(THINK_RE.finditer(generation))
        if tm:
            parsed_thinking[i] = tm[-1].group(1).strip()
        # else:
        #     valids[i] = 0

    return parsed_actions, valids, parsed_thinking