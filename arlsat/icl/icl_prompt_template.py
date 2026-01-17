"""
Here is the prompt template for ARLSAT tasks. ICL prompt templates
"""

import json



ICL_PROMPT = """
You will be given some examples of logic problems question answering and then you will be given a new input, please follow the examples to answer the question in the given input.
Here are exampels:
{example0}
{example1}
{example2}
Instruction: You should follow the above examples to complete the following task.

"""

EXAMPLE_PROMPT = """
Input: 
{input}

Response: 
{response}
"""


prompt_with_hint = {
"system": """A conversation between user and assistant. The user asks questions and the assistant always tries to help the user. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer>answer here</answer> Within the answer tags, provide only the final answer.""",
"user": """{label} You will be given a analytical logic problem including context, options, question, please provide your thinking steps and return ONLY the index of the correct option.
Context: {context}
Options:{options}
Question: {question}
Show your step by step work in <think> </think> tags. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags, for example <answer> 0 </answer>
"""
}

with open('positive_icl_prompt.txt', 'r') as f:
    positive_icl_template = f.read()


ICL_POSOITIVE_PROMPT = {
"system": """You are an assistant and you will always help users. You will be given some examples of logic problems question answering and then you will be given a new input, please follow the examples to answer the question in the given input.""",
"user": """
Here are exampels:

Input: 
You will need to solve the logic problem, provide thinking steps and return the index of the correct option as your final answer.
            The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>, <answer>answer here</answer>. Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.

Here is the context:
A disc jockey will play a sequence consisting of exactly seven different songs: three ballads—F, G, and H—and four dance tunes—R, S, V, and X. The following conditions must be met: No dance tune can be played immediately after another dance tune. H must be played earlier in the sequence than V. V and S must be separated from each other by exactly one song. S must be played immediately before or immediately after F. F must be played immediately after R, unless G is played earlier than R.
Options (0-indexed):
0: G
1: R
2: S
3: V
4: X

Question:
Which one of the following could be the fourth song in the sequence?
Show your step by step work in <think> </think> tags. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags, for example <answer> 0 </answer>
    

Response: 
<think>Summary of constraints: dances (R,S,V,X) cannot immediately follow another dance; H must come before V; V and S are separated by exactly one song; S is adjacent to F; F must immediately follow R unless G is earlier than R.

Construct a sequence with G as the 4th song that satisfies all constraints:
1: X (dance)
2: H (ballad)
3: V (dance)
4: G (ballad)
5: S (dance)
6: F (ballad)
7: R (dance)

Check constraints:
- No dance immediately followed by a dance: every dance (positions 1,3,5,7) is followed either by a ballad or by the end.
- H (pos2) is before V (pos3).
- V (pos3) and S (pos5) are separated by exactly one song (pos4).
- S (pos5) is adjacent to F (pos6).
- F (pos6) does not follow R (pos7), but G (pos4) is earlier than R (pos7), so the exception applies.

Thus G can be the fourth song.</think>
<answer> 0 </answer>


Input: 
You will need to solve the logic problem, provide thinking steps and return the index of the correct option as your final answer.
            The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>, <answer>answer here</answer>. Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.

Here is the context:
There are exactly three recycling centers in Rivertown: Center 1, Center 2, and Center 3. Exactly five kinds of material are recycled at these recycling centers: glass, newsprint, plastic, tin, and wood. Each recycling center recycles at least two but no more than three of these kinds of material. The following conditions must hold: Any recycling center that recycles wood also recycles newsprint. Every kind of material that Center 2 recycles is also recycled at Center 1. Only one of the recycling centers recycles plastic, and that recycling center does not recycle glass.
Options (0-indexed):
0: glass
1: newsprint
2: plastic
3: tin
4: wood

Question:
If Center 2 recycles three kinds of material, then which one of the following kinds of material must Center 3 recycle?
Show your step by step work in <think> </think> tags. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags, for example <answer> 0 </answer>
    

Response: 
<think>Label centers C1, C2, C3. Materials: G (glass), N (newsprint), P (plastic), T (tin), W (wood).
Given: C2 recycles 3 kinds, and every material C2 recycles is also recycled at C1, so C2 ⊆ C1. But C1 recycles at most 3, so C1 must recycle exactly the same 3 as C2 (C1 = C2).
"Only one center recycles plastic (P)". If P were in C1 (and thus in C2), that would be at least two centers recycling P, contradicting uniqueness. Therefore P is not in C1 or C2.
Since all five materials are recycled somewhere, P must be recycled by C3. Also the center that recycles P does not recycle G, so C3 does not recycle G (hence G is among the three recycled by C1/C2). Thus C3 must recycle plastic (P).</think>

<answer> 2 </answer>


Input: 
You will need to solve the logic problem, provide thinking steps and return the index of the correct option as your final answer.
            The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>, <answer>answer here</answer>. Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.

Here is the context:
Exactly five cars—Frank's, Marquitta's, Orlando's, Taishah's, and Vinquetta's—are washed, each exactly once. The cars are washed one at a time, with each receiving exactly one kind of wash: regular, super, or premium. The following conditions must apply: The first car washed does not receive a super wash, though at least one car does. Exactly one car receives a premium wash. The second and third cars washed receive the same kind of wash as each other. Neither Orlando's nor Taishah's is washed before Vinquetta's. Marquitta's is washed before Frank's, but after Orlando's. Marquitta's and the car washed immediately before Marquitta's receive regular washes.
Options (0-indexed):
0: Vinquetta's car receives a premium wash.
1: Exactly two cars receive a super wash.
2: The fifth car washed receives a super wash.
3: The fourth car washed receives a super wash.
4: The second car washed receives a regular wash.

Question:
Which one of the following must be true?
Show your step by step work in <think> </think> tags. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags, for example <answer> 0 </answer>
    

Response: 
<think>
We have five cars V, O, M, F, T in the wash order 1–5. Given constraints:
- First car is not S; at least one S exists.
- Exactly one P overall.
- Positions 2 and 3 have the same wash.
- Neither O nor T is washed before V → V is before O and before T.
- M is washed after O and before F → O < M < F.
- M and the car immediately before M are both regular (R).

From "V before O and T" and "O before M before F", every car except V must come after V. Therefore V must be first (position 1). The remaining order (positions 2–5) must place O, M, F in that order, with T inserted anywhere among them. So the four possible orders are:
A: V, T, O, M, F
B: V, O, T, M, F
C: V, O, M, T, F
D: V, O, M, F, T

Use the M constraint: the car immediately before M and M itself are R. In each order this fixes:
- If M is in position 3 or 4, then positions (2,3) or (3,4) are both R respectively.
Also positions 2 and 3 must be the same wash.

Analyze each order:
A (V,T,O,M,F): M is pos4, so pos3 and pos4 are R. With pos2 = pos3, pos2 is R as well. So positions 2–4 are R. Position1 (V) cannot be S; to have at least one S, pos5 must be S. The unique valid assignment becomes V = P (to have exactly one P), pos2–4 = R, pos5 = S. In particular pos2 = R.

B (V,O,T,M,F): M is pos4, so pos3 and pos4 are R. With pos2 = pos3, pos2 is R. Same conclusion: pos2 = R.

C (V,O,M,T,F): M is pos3, so pos2 and pos3 are R. Thus pos2 = R. The remaining positions (1,4,5) can be assigned R/S/P subject to "first not S", "exactly one P", and "at least one S". Several assignments are possible (P could be at pos1, pos4, or pos5), but pos2 remains R in every allowed assignment.

D (V,O,M,F,T): M is pos3, so pos2 and pos3 are R. Thus pos2 = R. Again multiple wash assignments possible for the remaining positions, but pos2 is always R.

Across all possible orderings and allowed wash assignments, position 2 is always regular. Therefore "The second car washed receives a regular wash" must be true.
</think>
<answer> 4 </answer>

Instruction: You should follow the above examples to complete the following task.

Input:
Context: {context}
Options:{options}
Question: {question}
Show your step by step work in <think> </think> tags at first. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags.

Response:

"""
}


ICL_NEG_PROMPT = {
"system": """You are an assistant and you will always help users. You will be given some examples of logic problems question answering and then you will be given a new input, please follow the examples to answer the question in the given input.""",
"user": """
Here are exampels:

Input: 
You will need to solve the logic problem, think step by step and ONLY return the index of the correct option as your final answer.
                The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>, <answer>answer here</answer>. Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.

Here is the context:
Of the eight students—George, Helen, Irving, Kyle, Lenore, Nina, Olivia, and Robert—in a seminar, exactly six will give individual oral reports during three consecutive days—Monday, Tuesday, and Wednesday. Exactly two reports will be given each day—one in the morning and one in the afternoon—according to the following conditions: Tuesday is the only day on which George can give a report. Neither Olivia nor Robert can give an afternoon report. If Nina gives a report, then on the next day Helen and Irving must both give reports, unless Nina's report is given on Wednesday.
Options (0-indexed):
0: Mon. morning: Helen; Mon. afternoon: Robert Tues. morning: Olivia; Tues. afternoon: Irving Wed. morning: Lenore; Wed. afternoon: Kyle
1: Mon. morning: Irving; Mon. afternoon: Olivia Tues. morning: Helen; Tues. afternoon: Kyle Wed. morning: Nina; Wed. afternoon: Lenore
2: Mon. morning: Lenore; Mon. afternoon: Helen Tues. morning: George; Tues. afternoon: Kyle Wed. morning: Robert; Wed. afternoon: Irving
3: Mon. morning: Nina; Mon. afternoon: Helen Tues. morning: Robert; Tues. afternoon: Irving Wed. morning: Olivia; Wed. afternoon: Lenore
4: Mon. morning: Olivia; Mon. afternoon: Nina Tues. morning: Irving; Tues. afternoon: Helen Wed. morning: Kyle; Wed. afternoon: George

Question:
Which one of the following could be the schedule of the students' reports?
Show your step by step work in <think> </think> tags. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags,     for example <answer> 0 </answer>
        

Response: 
<think>
We have students G,H,I,K,L,N,O,R. Mon-Tue-Wed each have one morning and one afternoon. Constraints:
- George can give only on Tuesday.
- Olivia and Robert cannot give an afternoon report.
- If Nina gives a report and it's not on Wednesday, then the following day Helen and Irving must both give reports.

Check each option:

Option 0:
Tue afternoon: Irving (allowed). But Olivia is Tue morning (morning slot, allowed). Robert is Mon afternoon — Robert cannot give an afternoon report. Violation. Reject.

Option 1:
Tue morning: Helen, Tue afternoon: Kyle. But George is not scheduled at all. Required six reports; missing George and possibly others. Reject.

Option 2:
Mon: Lenore morning, Helen afternoon (Olivia not used, Robert not used). Tue: George morning (allowed for George), Kyle afternoon. Wed: Robert morning (allowed, not afternoon), Irving afternoon. All eight students accounted for. Check the Nina condition: Nina is not giving a report, so no issue. All constraints satisfied. Accept.

Option 3:
Mon afternoon: Helen, Tue morning: Robert, Tue afternoon: Irving, Wed morning: Olivia, Wed afternoon: Lenore. Missing George and Kyle, and Robert is Tue morning (allowed). But George absent violates the requirement that exactly six give reports (and G must give on Tuesday). Reject.

Option 4:
Mon afternoon: Nina. Because Nina is not on Wednesday, the rule triggers: next day (Tue) must have both Helen and Irving giving reports. But Tue here has Irving and Helen are Wed, so violation. Reject.

Only option 2 satisfies all constraints.
</think>
<answer>2</answer>


Input: 
You will need to solve the logic problem, think step by step and ONLY return the index of the correct option as your final answer.
                The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>, <answer>answer here</answer>. Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.

Here is the context:
Of the eight students—George, Helen, Irving, Kyle, Lenore, Nina, Olivia, and Robert—in a seminar, exactly six will give individual oral reports during three consecutive days—Monday, Tuesday, and Wednesday. Exactly two reports will be given each day—one in the morning and one in the afternoon—according to the following conditions: Tuesday is the only day on which George can give a report. Neither Olivia nor Robert can give an afternoon report. If Nina gives a report, then on the next day Helen and Irving must both give reports, unless Nina's report is given on Wednesday.
Options (0-indexed):
0: Helen, George, and Nina
1: Irving, Robert, and Helen
2: Nina, Helen, and Olivia
3: Olivia, Robert, and Irving
4: Robert, George, and Helen

Question:
If Kyle and Lenore do not give reports, then the morning reports on Monday, Tuesday, and Wednesday, respectively, could be given by
Show your step by step work in <think> </think> tags. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags,     for example <answer> 0 </answer>
        

Response: 
</think>We have G, H, I, K, L, N, O, R. K and L do NOT give reports, so the 6 reporters are G, H, I, N, O, R assigned to Mon, Tue, Wed (two each day: AM and PM).

Constraints:
- Tue is the only day George can give a report → G must be Tue (AM or PM).
- Olivia and Robert cannot give an afternoon report → O and R must be morning reports.
- If N gives a report and it's not Wednesday, then the next day both H and I must give reports.

Because O and R must be mornings, they cannot occupy Tue (only one afternoon slot exists, so Tue must have one AM and one PM; O and R cannot both be Tue AM/PM). Therefore O and R must be Mon AM and Wed AM (in some order). Thus Tue AM is taken by one of G, H, I, N (but G can only be on Tue). The other two days (Mon and Wed) each have an additional AM slot occupied by O or R and a PM slot to be filled.

If N is not on Wednesday, then N’s report would force the next day (the day after N) to have both H and I giving reports. But since Tue is the only day G can give and G must be Tue, N cannot be Tue (G must be Tue). If N were Mon (not Wed), then the next day Tue would have to include both H and I, forcing Tue to be H and I with no room for G — impossible. Thus N cannot be Mon or Tue, so N must be Wed. If N is Wed, the conditional about H and I is vacuously satisfied, so no extra restriction.

We now test the five listed triples to see which could be the three morning reporters (the AM slots Mon, Tue, Wed). The only triples that can be mornings are those that include G (must be Tue) and two of (O,R,H,I,N) with O and R both included (since they must be mornings). Check each option:

0: Helen, George, and Nina — includes G and two of (O,R,H,I,N) but not both O and R → cannot (O and R both must be mornings) → reject.

1: Irving, Robert, and Helen — includes I and H but not both O and R → O missing → cannot (R is included but O missing) → reject.

2: Nina, Helen, and Olivia — includes N, H, O but not R → R missing → cannot (both O and R must be mornings) → reject.

3: Olivia, Robert, and Helen — includes O and R (good) and H, but not N or I — possible with Tue = G (not H) and other placements, so feasible → accept.

4: Robert, George, and Helen — like option 1, includes R and G and H but not both O and R → O missing → cannot (R present but O missing) → reject.

Thus only option 3 can be the three morning reports.</think>
<answer> 3 </answer>


Instruction: You should follow the above examples to complete the following task.

Input:
Context: {context}
Options: {options}
Question: {question}
Show your step by step work in <think> </think> tags at first. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags.

Response:

"""
}

ICL_POS_Q3_PROMPT = {
"system": """You are an assistant and you will always help users. You will be given some examples of logic problems question answering and then you will be given a new input, please follow the examples to answer the question in the given input.""",
"user": """
Here are exampels:

Input: 
You will need to solve the logic problem, think step by step and ONLY return the index of the correct option as your final answer.
                The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>, <answer>answer here</answer>. Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.

Here is the context:
Of the eight students—George, Helen, Irving, Kyle, Lenore, Nina, Olivia, and Robert—in a seminar, exactly six will give individual oral reports during three consecutive days—Monday, Tuesday, and Wednesday. Exactly two reports will be given each day—one in the morning and one in the afternoon—according to the following conditions: Tuesday is the only day on which George can give a report. Neither Olivia nor Robert can give an afternoon report. If Nina gives a report, then on the next day Helen and Irving must both give reports, unless Nina's report is given on Wednesday.
Options (0-indexed):
0: Mon. morning: Helen; Mon. afternoon: Robert Tues. morning: Olivia; Tues. afternoon: Irving Wed. morning: Lenore; Wed. afternoon: Kyle
1: Mon. morning: Irving; Mon. afternoon: Olivia Tues. morning: Helen; Tues. afternoon: Kyle Wed. morning: Nina; Wed. afternoon: Lenore
2: Mon. morning: Lenore; Mon. afternoon: Helen Tues. morning: George; Tues. afternoon: Kyle Wed. morning: Robert; Wed. afternoon: Irving
3: Mon. morning: Nina; Mon. afternoon: Helen Tues. morning: Robert; Tues. afternoon: Irving Wed. morning: Olivia; Wed. afternoon: Lenore
4: Mon. morning: Olivia; Mon. afternoon: Nina Tues. morning: Irving; Tues. afternoon: Helen Wed. morning: Kyle; Wed. afternoon: George

Question:
Which one of the following could be the schedule of the students' reports?
Show your step by step work in <think> </think> tags. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags,     for example <answer> 0 </answer>
        

Response: 
We check each option against constraints:
Constraints recap: reports Mon–Wed (2 each day: AM/PM). Tuesday is the only day George can give a report. Olivia and Robert cannot give afternoon reports. If Nina gives a report and it's not on Wednesday, then on the next day both Helen and Irving must give reports.

Option 0:
Mon AM Helen, Mon PM Robert (allowed: R not PM). Tue AM Olivia, Tue PM Irving. Wed AM Lenore, Wed PM Kyle.
- George absent — allowed (he must be on Tue only).
- Olivia is AM on Tue — allowed (not PM).
- Robert is PM Mon — NOT allowed (Robert cannot give an afternoon report).
→ Option 0 violates Robert constraint. Reject.

Option 1:
Mon AM Irving, Mon PM Olivia. Tue AM Helen, Tue PM Kyle. Wed AM Nina, Wed PM Lenore.
- George absent — allowed.
- Olivia PM Mon — NOT allowed (Olivia cannot give an afternoon report).
→ Option 1 violates Olivia constraint. Reject.

Option 2:
Mon AM Lenore, Mon PM Helen. Tue AM George, Tue PM Kyle. Wed AM Robert, Wed PM Irving.
- George on Tue AM — allowed.
- Robert Wed AM (not PM) — allowed for Robert.
- Olivia not scheduled at all — allowed (not required).
- Nina not scheduled at all — the "if Nina..." condition is irrelevant.
→ Option 2 satisfies all constraints. Accept.

Option 3:
Mon AM Nina, Mon PM Helen. Tue AM Robert, Tue PM Irving. Wed AM Olivia, Wed PM Lenore.
- George absent — allowed.
- Robert Tue AM (not PM) — allowed.
- Olivia Wed AM (not PM) — allowed.
- Nina Mon AM: since it's not Wednesday, the rule requires that on the next day (Tue) both Helen and Irving give reports. Tue reports are Robert and Irving — Helen is missing. → Option 3 violates the Nina condition. Reject.

Option 4:
Mon AM Olivia, Mon PM Nina. Tue AM Irving, Tue PM Helen. Wed AM Kyle, Wed PM George.
- George on Wed PM — NOT allowed (George must give a report only on Tuesday).
→ Option 4 violates George constraint. Reject.

Only Option 2 is consistent with all conditions.</think>
<answer>2</answer>


Input: 
You will need to solve the logic problem, think step by step and ONLY return the index of the correct option as your final answer.
                The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>, <answer>answer here</answer>. Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.

Here is the context:
Of the eight students—George, Helen, Irving, Kyle, Lenore, Nina, Olivia, and Robert—in a seminar, exactly six will give individual oral reports during three consecutive days—Monday, Tuesday, and Wednesday. Exactly two reports will be given each day—one in the morning and one in the afternoon—according to the following conditions: Tuesday is the only day on which George can give a report. Neither Olivia nor Robert can give an afternoon report. If Nina gives a report, then on the next day Helen and Irving must both give reports, unless Nina's report is given on Wednesday.
Options (0-indexed):
0: Helen, George, and Nina
1: Irving, Robert, and Helen
2: Nina, Helen, and Olivia
3: Olivia, Robert, and Irving
4: Robert, George, and Helen

Question:
If Kyle and Lenore do not give reports, then the morning reports on Monday, Tuesday, and Wednesday, respectively, could be given by
Show your step by step work in <think> </think> tags. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags,     for example <answer> 0 </answer>
        

Response: 
 or <answer> 1 </answer> etc. (Use no other text.)

We have 8 students; two don't give reports (Kyle and Lenore), so the 6 reporters are G, H, I, N, O, R. Reports are Mon AM, Mon PM, Tue AM, Tue PM, Wed AM, Wed PM; Tue is the only day G can give. Olivia and Robert cannot give afternoon reports, so O and R must be mornings. Therefore Tue AM and Tue PM cannot both be afternoons; at least one of Tue AM or Tue PM must be a morning. Since O and R must be mornings, among the three mornings (Mon AM, Tue AM, Wed AM) exactly two are O and R and the third is one of G, H, I, N. So the set of three morning reporters must include both O and R and exactly one from (G,H,I,N). Check options:
0: H, G, N — missing O and R → impossible.
1: I, R, H — missing O → impossible.
2: N, H, O — missing R → impossible.
3: O, R, I — includes O and R and I (one of H,I,N) → possible.
4: R, G, H — missing O and N → impossible (must include exactly one from (G,H,I,N), but here none).
Thus only option 3 can be the three mornings.
</think>
<answer> 3 </answer>


Instruction: You should follow the above examples to complete the following task.

Input:
Context: {context}
Options: {options}
Question: {question}
Show your step by step work in <think> </think> tags at first. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags.

Response:

"""
}


INSTRUCTION_PROMPT ={
"system": """You are an assistant and you will always help users. You will need to solve the logic problem, think step by step and ONLY return the index of the correct option as your final answer.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>, <answer>answer here</answer>. Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.""",
"user": """
Here is the context:
{context}
Options: 
{options}

Question:
{question}

Show your step by step work in <think> </think> tags. And return exactly one character: 0,1,2,3, or 4 as final answer in <answer> </answer> tags.

"""
}




def main():
    # with open('/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/gpt/gpt5/data/gpt5_response_sft_context.json', 'r') as f:
    #     data = json.load(f)
    # examples_prompts = []
    # for entry in data[:3]:
    #     examples_prompts.append(EXAMPLE_PROMPT.format(input=entry[0], response=entry[1]))
    
    # final_prompt = ICL_PROMPT.format(example0=examples_prompts[0], example1=examples_prompts[1], example2=examples_prompts[2])

    # with open('pos_icl_prompt.txt', 'w') as f:
    #     f.write(final_prompt)
    
    with open('/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b/reasonable_inference_.json', 'r') as f:
        neg_data = json.load(f)

    examples_prompts = []
    for entry in neg_data[:2]:
        examples_prompts.append(EXAMPLE_PROMPT.format(input=entry['prompt'], response=entry['gen'][0]))
    final_prompt = ICL_PROMPT.format(example0=examples_prompts[0], example1=examples_prompts[1], example2='')

    print(final_prompt)

    with open('pos_icl_q3_prompt.txt', 'w') as f:
        f.write(final_prompt)

if __name__ == "__main__":
    main()





