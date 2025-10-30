from typing import List
import re

def general_projection(prompt, actions: List[str]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)
    thinking_traces = ["Invalid thinking trace: remember to enclose thinking traces in <think>...</think> tags."] * len(actions)
    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

        # check for <summary>...</summary>
        if 'scq' in prompt or 'sctq' in prompt:
            summary_start_idx = original_str.find("<summary>")
            summary_end_idx = original_str.find("</summary>")
            if summary_start_idx == -1 and summary_end_idx == -1:
                valids[i] = 0

        # # Enforce Ordering
        # if 'scq' in prompt or 'sctq' in prompt:
        #     if summary_start_idx > think_start_idx or summary_end_idx < think_end_idx:
        #         valids[i] = 0

        # Try to extract the thinking trace:
        # Get the first '<think'> tag:
        think_start_idx = original_str.find("<think>")
        # Get the last '</think>' tag:
        think_end_idx = original_str.rfind("</think>")
        # Check if the stripped original string is larger than 100 characters: if so, make the placeholder only the first 100 characters. 
        # Otherwise, keep the original string.
        # thinking_traces[i] = original_str.strip()[:100] if len(original_str.strip()) > 100 else original_str.strip()
        thinking_traces[i] = "Invalid thinking trace: remember to enclose thinking traces in <think>...</think> tags."
        if think_start_idx != -1 and think_end_idx != -1:
            # Split the string to get the thinking trace
            thinking_traces[i] = original_str[think_start_idx + len("<think>"):think_end_idx].strip()

    return actions, valids, thinking_traces
