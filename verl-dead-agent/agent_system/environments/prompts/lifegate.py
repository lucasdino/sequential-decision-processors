LIFEGATE_TEMPLATE_BASECASE = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""

# New propmt 7/15
LIFEGATE_TEMPLATE_SCQ = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first critique your performance based on your previous past and observations. Summarize the most important lession from these experiences in a single sentence. This summary should be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""

LIFEGATE_TEMPLATE_SCQ_LONGER = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first perform a retrospective on your previous actions and observations. Summarize the most important information to keep in mind going forward. This summary should be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""

LIFEGATE_TEMPLATE_SCQ_LONGER_WARNING = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
There is a mysterious force that can interfere with your actions and may cause you to fail. Be cautious and make sure to think carefully about your actions.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first perform a retrospective on your previous actions and observations. Summarize the most important information to keep in mind going forward. This summary should be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""

LIFEGATE_TEMPLATE_SCQ_LONGER_NO_WARNING = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first perform a retrospective on your previous actions and observations. Summarize the most important information to keep in mind going forward. This summary should be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""

LIFEGATE_TEMPLATE_THINK_THEN_SCQ_LONGER = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Then, given this reasoning process, perform a retrospective on your previous actions and observations. Summarize the most important information to keep in mind going forward. This summary should be enclosed within <summary> </summary> tags.
Once you've finished your summarization, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""

LIFEGATE_TEMPLATE_SCQ_LONGER_NO_WARNING = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first perform a retrospective on your previous actions and observations. Summarize the most important information to keep in mind going forward. This summary should be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""

LIFEGATE_TEMPLATE_SCQ_LONGER_INSTRUCTIONS_FIRST = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
When you are ready to take an action, you should first perform a retrospective on your previous actions and observations. 
Summarize the most important information to keep in mind going forward. This summary MUST be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
"""

LIFEGATE_TEMPLATE_SCQ_LONGER_INSTRUCTIONS_FIRST_MORE_INFO = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the lifegate room somewhere on the other side of the map.
When you are ready to take an action, you should first perform a retrospective on your previous actions and observations. 
Summarize the most important information to keep in mind going forward. This summary MUST be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
"""

LIFEGATE_TEMPLATE_SCQ_LONGER_INSTRUCTIONS_FIRST_MORE_INFO_WITH_THINK = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the lifegate room somewhere on the other side of the map.
When you are ready to take an action, you should first perform a retrospective on your previous actions and observations. 
Summarize the most important information to keep in mind going forward. This summary MUST be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations, thinking traces, and actions you took: {action_history_with_think}
You are now at step {current_step} and your current observation is: {current_observation}.
"""

LIFEGATE_TEMPLATE_INSTRUCTIONS_FIRST_MORE_INFO_WITH_THINK = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the lifegate room somewhere on the other side of the map.
When you are ready to take an action, you should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations, thinking traces, and actions you took: {action_history_with_think}
You are now at step {current_step} and your current observation is: {current_observation}.
"""

LIFEGATE_TEMPLATE_INSTRUCTIONS_FIRST_NO_SCQRT = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the lifegate room somewhere on the other side of the map.
When you are ready to take an action, you should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations, thinking traces, and actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
"""

LIFEGATE_TEMPLATE_NO_WARNING = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""


# New propmt 7/15
LIFEGATE_TEMPLATE_TRAIN = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You are currently training, so you should prioritize exploring the environment and learning best practices.
You should first critique your performance based on your previous past and observations. Summarize the most important lession from these experiences in a single sentence. This summary should be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""

# New propmt 7/15
LIFEGATE_TEMPLATE_TEST = """
You are an expert agent operating in the Lifegate Environment. Your task is to reach the room with the lifegate on the other side of the map.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You are currently testing, so you do your best to complete the task with minimal errors.
You should first critique your performance based on your previous past and observations. Summarize the most important lession from these experiences in a single sentence. This summary should be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
"""