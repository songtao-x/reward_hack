import json
import random

random.seed(224)


prompt_template = """
User: {prompt}
Assistant: {response}
"""

response_template = """
Response: {response}
"""

def load_data():
    with open("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/gpt/gpt5/data/gpt5_response_sft_context.json", 'r') as f:
        gpt5_data = json.load(f)
        gpt5_data = [prompt_template.format(prompt=item[0], response=item[1]) for item in gpt5_data[:max(250, len(gpt5_data))]]

    with open("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b/correct_inference.json", 'r') as f:
        qwen_data = json.load(f)
    qwen = []
    for i, item in enumerate(qwen_data[:250]):
        try:
            qwen_data = prompt_template.format(prompt=item['prompt'], response=item['gen'][0])
            qwen.append(qwen_data)
        except:
            print(i)
            continue
    print(f'Number of GPT5 data: {len(gpt5_data)}')
    print(f'Number of Qwen data: {len(qwen)}')

    random.shuffle(gpt5_data)
    random.shuffle(qwen)

    return gpt5_data, qwen


def load_data_response():
    with open("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/gpt/gpt5/data/gpt5_response_sft_context.json", 'r') as f:
        gpt5_data = json.load(f)
        gpt5_data = [response_template.format(prompt=item[0], response=item[1]) for item in gpt5_data[:max(250, len(gpt5_data))]]

    with open("/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/sft/data/qwen3_4b/correct_inference.json", 'r') as f:
        qwen_data = json.load(f)
    qwen = []
    for i, item in enumerate(qwen_data[:250]):
        try:
            qwen_data = response_template.format(prompt=item['prompt'], response=item['gen'][0])
            qwen.append(qwen_data)
        except:
            print(i)
            continue
    print(f'Number of GPT5 data: {len(gpt5_data)}')
    print(f'Number of Qwen data: {len(qwen)}')

    random.shuffle(gpt5_data)
    random.shuffle(qwen)
    return gpt5_data, qwen



def get_trainset(K=50):
    gpt5_data, qwen = load_data()
    dataset = gpt5_data[:K] + qwen[:K]
    label = [1] * K + [0] * K

    ds = [{'data': d, 'label': l} for d, l in zip(dataset, label)]

    with open(f"data/train/trainset_{K}.json", 'w') as f:
        json.dump(ds, f, indent=4)

def get_testset(K=100):
    gpt5_data, qwen = load_data()
    dataset = gpt5_data[50:K+50] + qwen[50:K+50]
    label = [1] * K + [0] * K
    ds = [{'data': d, 'label': l} for d, l in zip(dataset, label)]

    with open(f"data/test/testset_{K}.json", 'w') as f:
        json.dump(ds, f, indent=4)


def get_trainset_response(K=50):
    gpt5_data, qwen = load_data_response()
    dataset = gpt5_data[:K] + qwen[:K]
    label = [1] * K + [0] * K

    ds = [{'data': d, 'label': l} for d, l in zip(dataset, label)]

    with open(f"data/train/trainset_res_{K}.json", 'w') as f:
        json.dump(ds, f, indent=4)

def get_testset_response(K=100):
    gpt5_data, qwen = load_data_response()
    dataset = gpt5_data[:K] + qwen[:K]
    label = [1] * K + [0] * K
    ds = [{'data': d, 'label': l} for d, l in zip(dataset, label)]

    with open(f"data/test/testset_res_{K}.json", 'w') as f:
        json.dump(ds, f, indent=4)

def main():
    get_trainset(K=50)
    get_testset(K=100)

    # get_trainset_response(K=50)
    # get_testset_response(K=100)


if __name__ == '__main__':
    main()







