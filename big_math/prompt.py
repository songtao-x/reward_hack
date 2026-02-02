"""
Save all prompt templates: gpt_eval,
"""

gpt_eval_prompt = """
You are an evaluator. Your task is to judge whether the given Response is correct and whether its reasoning/thinking steps are reliable given the Question.

Evaluate BOTH:
(1) Outcome correctness: Does the response ultimately answer the question correctly and completely?
(2) Reasoning correctness: Are the reasoning steps logically valid, internally consistent, and supported by the information in the question/response?

Rules:
- Use ONLY the content in the Question and the Response. Do not add external facts or assumptions.
- Treat missing justification as a weakness: if the response makes a key claim without support, count it as an error unless it is directly obvious from the question.
- If there is any critical logical error, contradiction, unsupported leap, or missing step that would make the conclusion unreliable…
  → Score = 0.
- Score = 1 ONLY if the reasoning is sound AND the final conclusion/answer is correct for the question.

What counts as “unreasonable” (Score=0):
- Incorrect final answer OR the response does not actually answer the question.
- The reasoning contains a logical fallacy, contradiction, or incorrect inference.
- The reasoning steps are not coherently correct.
- It invents facts not present in the question (hallucination).
- It ignores important constraints from the question.

What counts as “reasonable” (Score=1):
- The response addresses the question.
- The conclusion is correct (or, if the question is underspecified, the response clearly states assumptions and they are consistent).
- The reasoning steps are coherent, justified, and consistent with the given information.

Input:
Question: {question}
Response: {response}

Output format (exactly):
Evaluation: <brief, specific comments. If score=0, name the first major flaw and why it breaks correctness.>
Score: <1 or 0>

"""

gpt_eval_prompt_simple = """
You are supposed to evaluate the given response based on given math problem in terms of if the thinking steps of the response is reasonable. And please give the comment then return 1 or 0 indicating reasonable or unreasonable. 
Question: {question}
Response: {response}

Give your eval in the following format:
Evaluation: <your comment>
Score: 1 or 0
"""