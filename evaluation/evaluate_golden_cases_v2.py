import json
import asyncio
import os

import httpx
import pandas as pd

from dotenv import load_dotenv; load_dotenv()
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ragas import EvaluationDataset, aevaluate
from ragas.metrics import faithfulness, answer_correctness, context_recall
from ragas.llms import LangchainLLMWrapper


URL = 'http://localhost:8000/rag/answer'
HEADER = {'Content-Type': 'application/json'}
REFERENCE_DATASET = 'data/golden_dataset/golden_cases_ragas.csv'
RESULTS_OUTPUT_PATH = 'evaluation/results/evaluation_results.xlsx'
RAGAS_METRICS = [answer_correctness, context_recall, faithfulness]
# Observações:
# answer_correctness: avalia a exatidão da response em comparação com a reference dos especialistas.
# context_recall: analisa se o retrieved_contexts (experimental) contém as informações necessárias para chegar à resposta descrita na reference. No ambiente médico, ela ajuda a identificar se o buscador de vetores omitiu alguma contraindicação ou sintoma vital que os especialistas incluíram no gabarito.
# faithfulness: checa se cada afirmação médica presente na response pode ser diretamente inferida a partir do retrieved_contexts. Se a IA gerar uma conduta correta com base em conhecimento prévio interno dela, mas que não estava nos documentos que o sistema recuperou, a nota cai. Isso impede o perigoso fenômeno de "alucinação factual não rastreável".
LLM_JUDGE = 'gpt-5.1'
API_TIMEOUT_SECONDS = 120
API_CONCURRENCY = int(os.getenv("API_CONCURRENCY", "1"))
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


def load_reference_dateset():

    return pd.read_csv(REFERENCE_DATASET)


# def llm_judge(reference_output:str, experiment_output:str):

#     llm = ChatOpenAI(model=LLM_JUDGE, temperature=0)

#     llm_structured = llm.with_structured_output(JudgeDataModel)

#     prompt = ChatPromptTemplate(
#         [
#             ('system', JUDGE_SYSTEM_PROMPT), 
#             ('user', JUDGE_QUERY_PROMPT.format(experiment_output=experiment_output, reference_output=reference_output))
#         ]
#     )

#     chain_judge = prompt | llm_structured
#     # chain_judge = prompt | llm
    
#     return chain_judge.invoke({'reference_output':reference_output, 'experiment_output':experiment_output})


def get_table(output:list):

    return pd.DataFrame([{'verdict':x.verdict, 'core':x.score, 'rationale':x.rationale} for x in output])


def format_response_for_ragas(results: list[dict]) -> str:

    if not results:

        return ""

    parts = []

    for item in results:

        suggested_items = item.get("suggested-items", [])
        rationale = item.get("clinical-rationale", "")

        if suggested_items:

            parts.append("Itens sugeridos: " + "; ".join(suggested_items))

        if rationale:

            parts.append("Justificativa clínica: " + rationale)

    return "\n".join(parts)


async def solve(query:dict, client:httpx.AsyncClient):

    if not isinstance(query, dict):
    
        raise TypeError(f'`query` must be a python dict (found {type(query)})')

    user_query = json.dumps(query)

    query = {'query': str(user_query).replace('{', '{{').replace('}', '}}')}

    r = await client.post(url=URL, json=query)
    r.raise_for_status()
    
    r_json = r.json()

    output = {
        'response': format_response_for_ragas(r_json['answer']['results']),
        'retrieved_contexts': [x['content_preview'] for x in r_json['sources']]
    }

    return output


async def solve_dataframe_row(
    index:int,
    row:pd.Series,
    client:httpx.AsyncClient,
    semaphore:asyncio.Semaphore
) -> dict:

    async with semaphore:

        print(f'[STATUS] Evaluating reference data instance:\n\t-Index: {index}\n\t-Contents: {row}\n')

        reference_input = json.loads(row['user_input'])
        reference_output = row['reference']

        try:

            experiment_output = await solve(query=reference_input, client=client)

        except httpx.HTTPStatusError as exc:

            response_text = exc.response.text[:1000]
            raise RuntimeError(
                f"API failure while evaluating reference data instance {index}: "
                f"status_code={exc.response.status_code}, response={response_text}"
            ) from exc

        output_i = {
            'id': int(index),
            'user_input': row['user_input'],
            'ratianale': row['rationale'],
            'reference': reference_output,
            'response': experiment_output['response'],
            'retrieved_contexts': experiment_output['retrieved_contexts']
        }

        print(f'[STATUS] Reference data instance {index} completed.')

        return output_i


async def solve_dataframe(df:pd.DataFrame):

    if not isinstance(df, pd.DataFrame):

        raise TypeError('`df` must be a pandas dataframe.')

    semaphore = asyncio.Semaphore(API_CONCURRENCY)

    async with httpx.AsyncClient(timeout=API_TIMEOUT_SECONDS) as client:

        tasks = [
            solve_dataframe_row(
                index=i,
                row=row,
                client=client,
                semaphore=semaphore
            )
            for i, row in df.iterrows()
        ]

        output = await asyncio.gather(*tasks)

    return pd.DataFrame(output)


async def main():

    llm = ChatOpenAI(model=LLM_JUDGE, temperature=0)

    llm_judge = LangchainLLMWrapper(llm)
    
    reference_dataset = load_reference_dateset()

    experiment_output = await solve_dataframe(reference_dataset)

    reference_dataset_dict = experiment_output.to_dict(orient="records")
    
    reference_dataset_ragas = EvaluationDataset.from_list(reference_dataset_dict)

    output_ragas = await aevaluate(
        dataset = reference_dataset_ragas,
        metrics = RAGAS_METRICS,
        llm = llm_judge
    )

    return {'summary':output_ragas, 'dataframe':experiment_output}


if __name__ == '__main__':

    print()
    print('##### Runing Evaluation #####')
    print()

    print('[STATUS] Running')
    # results = main()
    results = asyncio.run(main())
    print('[STATUS] Evaluation completed.')

    print('-- OUTPUT --')

    print('- Ragas Summary:')
    print(results['summary'])
    print('\n- Output table:')
    print(results['dataframe'])

    results['dataframe'].to_excel(RESULTS_OUTPUT_PATH, index=False)

    print('-------------')
    print()
    print('##### Evaluation Finished #####')
    print()
