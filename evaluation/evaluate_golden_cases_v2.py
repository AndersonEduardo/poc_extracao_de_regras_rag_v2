import time
import requests
import json
import pandas as pd
from dotenv import load_dotenv; load_dotenv()
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


URL = 'http://localhost:8000/rag/answer'
HEADER = {'Content-Type': 'application/json'}
REFERENCE_DATASET = 'data/golden_dataset/golden_cases.csv'
RESULTS_OUTPUT_PATH = 'evaluation/results/evaluation_results.csv'
LLM_JUDGE = 'gpt-5.1'
JUDGE_SYSTEM_PROMPT = """
You are a careful clinical guideline evaluation judge.
Your job is to compare a system output against a reference answer for a medical decision-support golden case.

Return only JSON with:
{{
  "verdict": "correct" | "partial" | "incorrect",
  "score": 1.0 | 0.5 | 0.0,
  "rationale": "short explanation"
}}

Judging rules:
- Mark correct when the system output contains all mandatory components from the reference and does not contradict restrictions.
- When the reference allows alternatives with "ou" / "or", any valid alternative is sufficient.
- Mark partial when the output is clinically related but incomplete or only partially overlaping with reference data.
- Mark incorrect when it omits an essential recommendation, recommends a prohibited item, contradicts the reference, or is unrelated.
- Judge only against the supplied reference answer, not against outside clinical knowledge.
""".strip()
JUDGE_QUERY_PROMPT = '''
# EXPERIMENT OUTPUT:
{experiment_output}

# REFERENCE_DATASET:
{reference_output}
'''

class JudgeDataModel(BaseModel):
    verdict:str
    score:str
    rationale:str


def solve(query:dict):

    if not isinstance(query, dict):
    
        raise TypeError(f'`query` must be a python dict (found {type(query)})')

    user_query = json.dumps(query)

    # query = {'query': "Recomendação de medicamentos para paciente com estratificação de risco muito alta."}
    query = {'query': str(user_query).replace('{', '{{').replace('}', '}}')}
    # query = {'query': f"{str(query)}"}

    # print()
    # print('--> DENTRO')
    # print('- type(query) =', type(query))
    # print()
    # print('- query =', query)
    # print()

    # r = requests.post(url=URL, headers=HEADER, data=json.dumps(query))
    # r = requests.post(url=URL, data=json.dumps(query))
    r = requests.post(url=URL, json=query)

    if r.status_code != 200:

        raise Exception(f'API status code failure: status code = {r.status_code}')
    
    r_json = r.json()

    output = r_json['answer']['results']

    return output


def load_reference_dateset():

    return pd.read_csv(REFERENCE_DATASET)


def llm_judge(reference_output:str, experiment_output:str):

    llm = ChatOpenAI(model=LLM_JUDGE, temperature=0)

    llm_structured = llm.with_structured_output(JudgeDataModel)

    prompt = ChatPromptTemplate(
        [
            ('system', JUDGE_SYSTEM_PROMPT), 
            ('user', JUDGE_QUERY_PROMPT.format(experiment_output=experiment_output, reference_output=reference_output))
        ]
    )

    chain_judge = prompt | llm_structured
    # chain_judge = prompt | llm
    
    return chain_judge.invoke({'reference_output':reference_output, 'experiment_output':experiment_output})


def get_table(output:list):

    return pd.DataFrame([{'verdict':x.verdict, 'core':x.score, 'rationale':x.rationale} for x in output])


def main():

    reference_dataset = load_reference_dateset()

    output = list()

    for i, row in reference_dataset.iterrows():

        print(f'[STATUS] Evaluating reference data instance:\n\t-Index: {i}\n\t-Contents: {row}\n')

        reference_input = json.loads(row['parametros_clinicos'])
        reference_output = row['resposta_correta']

        # print()
        # print('##'*10)
        # print(f'---> reference_input (type = {type(reference_input)}):\n', reference_input)
        # print('##'*10)
        # print(f'---> reference_output (type = {type(reference_output)}):\n', reference_output)
        # print('##'*10)
        # print()

        experiment_output_raw = solve(query=reference_input)
        experiment_output = json.dumps(experiment_output_raw)

        reference_output = reference_output.replace('{', '{{').replace('}', '}}')
        experiment_output = experiment_output.replace('{', '{{').replace('}', '}}')


        # print()
        # print('##'*10)
        # print(f'---> reference_input (type = {type(reference_input)}):\n', reference_input)
        # print('##'*10)
        # print(f'---> reference_output (type = {type(experiment_output)}):\n', experiment_output)
        # print('##'*10)
        # print()


        judge_output = llm_judge(reference_output=reference_output, experiment_output=experiment_output)

        output_i = {
            'id': int(i),
            'query': row['parametros_clinicos'],
            'reference_output': reference_output,
            'experiment_output': experiment_output,
            'verdict': judge_output.verdict, 
            'score': float(judge_output.score),
            'rationale':judge_output.rationale
        }

        # output.append(judge_output)
        output.append(output_i)

        print(f'[STATUS] Reference data instance {i} completed.')

        # break

        time.sleep(0.3)

    return output


if __name__ == '__main__':

    print()
    print('##### Runing Evaluation #####')
    print()

    print('[STATUS] Running')
    results = main()
    print('[STATUS] Evaluation completed.')

    print('-- OUTPUT --')
    # results = pd.DataFrame(results)
    # results_table = get_table(results)
    results_table = pd.DataFrame(results)
    print(results_table)
    print('\nOutput summary:\n', results_table['verdict'].value_counts(), '\n')
    print('\nMean score:\n', results_table['score'].mean(), '\n')
    results_table.to_csv(RESULTS_OUTPUT_PATH, index=False)
    # print(results)
    print('-------------')
    print()
    print('##### Evaluation Finished #####')
    print()
