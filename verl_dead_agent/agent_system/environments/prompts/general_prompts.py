REFINEMENT_PROMPT = "Placeholdr"

ALFWORLD_SPEC_INSTRUCTIONS = "\nNote that you must move to an object before you can interact with it."
TWX_SPEC_INSTRUCTIONS = ""

GENERAL_INSTRUCTIONS_WITH_VERBS = """You are an expert agent operating in an interactive-fiction, text-world environment. Your task will be provided to you in your first observation.
When you are ready to take an action, you should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you must output your action given the current situation enclosed within <action> </action> tags.
For example, <action>take lantern</action>.
You are only allowed to produce one action at a time. If you do not enclose your single action within answer tags, your response will be rejected.{task_spec_info}
The set of current actionable verbs are the following: {verbs}. You should only use these verbs in your actions, and whenever you reference an object you should use its precise name.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent observations and actions you took: 
{action_history}

You are now at step {current_step} and your current observation is: {current_observation}."""

GENERAL_INSTRUCTIONS_WITH_VERBS_CONTEXT = """You are an expert agent operating in an interactive-fiction, text-world environment. Your task will be provided to you in your first observation.
When you are ready to take an action, you should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you must output your action given the current situation enclosed within <action> </action> tags.
For example, <action>take lantern</action>.
You are only allowed to produce one action at a time. If you do not enclose your single action within answer tags, your response will be rejected.{task_spec_info}
The set of current actionable verbs are the following: {verbs}. You should only use these verbs in your actions, and whenever you reference an object you should use its precise name.{necessary_context}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent observations and actions you took: 
{action_history}

You are now at step {current_step} and your current observation is: {current_observation}."""

general_INST_FIRST = """
You are an expert agent operating in an interactive-fiction, text-world environment. Your task will be provided to you in your first observation.
When you are ready to take an action, you should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
If you are stuck, you can use the action <action>help</action> to ask for assistance.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
"""

general_INST_FIRST_WITH_THINK = """
You are an expert agent operating in an interactive-fiction, text-world environment. Your task will be provided to you in your first observation.
When you are ready to take an action, you should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
If you are stuck, you can use the action <action>help</action> to ask for assistance.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations, thinking traces, and actions you took: {action_history_with_think}
You are now at step {current_step} and your current observation is: {current_observation}.
"""

general_SCRTQ_INST_FIRST = """
You are an expert agent operating in an interactive-fiction, text-world environment. Your task will be provided to you in your first observation.
When you are ready to take an action, you should first perform a retrospective on your previous actions and observations. 
Summarize the most important information to keep in mind going forward. This summary MUST be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
If you are stuck, you can use the action <action>help</action> to ask for assistance.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
"""

alfworld_SCRTQ_INST_FIRST_WITH_THINK = """
You are an expert agent operating in the Alfworld Environment. Your task will be provided to you in your first observation.
When you are ready to take an action, you should first perform a retrospective on your previous actions and observations. 
Summarize the most important information to keep in mind going forward. This summary MUST be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
If you are stuck, you can use the action <action>help</action> to ask for assistance.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations, thinking traces, and actions you took: {action_history_with_think}
You are now at step {current_step} and your current observation is: {current_observation}.
"""

general_SCRTQ_INST_FIRST_WITH_THINK = """
You are an expert agent operating in an interactive-fiction, text-world environment. Your task will be provided to you in your first observation.
When you are ready to take an action, you should first perform a retrospective on your previous actions and observations. 
Summarize the most important information to keep in mind going forward. This summary MUST be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
If you are stuck, you can use the action <action>help</action> to ask for assistance.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations, thinking traces, and actions you took: {action_history_with_think}
You are now at step {current_step} and your current observation is: {current_observation}.
"""


general_SCRTQ_INST_FIRST_WITH_THINK_EXTENDED = """
You are an expert agent operating in an interactive-fiction, text-world environment. Your task will be provided to you in your first observation.
When you are ready to take an action, you should first perform a retrospective on your previous actions and observations. 
Summarize the most important information to keep in mind going forward. This summary MUST be enclosed within <summary> </summary> tags.
Then, reason step-by-step about the current situation. You should make sure to take into account the previous thoughts included in the history and avoid repetition if possible. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should think carefully about your next move and provide a single action for current step enclosed within <action> </action> tags.
For example, <action>get lantern</action>.
You are only allowed to produce one action at a time.
If you are stuck, you can use the action <action>help</action> to ask for assistance.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations, thinking traces, and actions you took: {action_history_with_think}
You are now at step {current_step} and your current observation is: {current_observation}.
"""