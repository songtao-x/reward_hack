from openai import OpenAI

KEY = ""
OpenAI.api_key = KEY
client = OpenAI(api_key=OpenAI.api_key)

def gpt_completion(prompt, model="gpt-5-mini"):
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "medium"},
    )
    return response.output_text



text = "hello"
response = gpt_completion(text)
print(response)

