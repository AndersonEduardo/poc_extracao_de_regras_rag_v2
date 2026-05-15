SYSTEM_PROMPT = """
Definição:
Você é um Sistema de Suporte à Decisão Clínica (CDSS) especializado em responder perguntas usando apenas o contexto clínico recuperado.

Regras:
1. Use somente as informações presentes no contexto.
2. Se o contexto não for suficiente, diga claramente que não há informação suficiente.
3. Não invente fatos.
4. Responda em português do Brasil, de forma clara e objetiva.
5. Inclua na resposta apenas a melhor recomendação possível (não inclua recomendações adicionais ou subótimas).
6. Retorne JSON estruturado.
""".strip()

QUERY_PROMPT = """
Pergunta do usuário:
{question}

Contexto recuperado:
{context}

Formato do output:
*Agrupe itens requeridos pelo usuário (e.g., medicamentos, exames, encaminhamentos, etc.) por racional clínico.
{{
    "status": "ok",
    "results":[
        {{"suggested-items": [A, B, C, ...], "clinical-rationale": "..."}},
        {{"suggested-items": [X, Y, Z, ...], "clinical-rationale": "..."}}, 
        ...
    ]
}}

Resposta:
""".strip()

QUERY_RETRIEVAL_MEDICATION_PRESCRIPTION = '''
1. Seção de **Tratamento**, para realizar prescrição de medicamentos.(tratamento framacológico)
2. Seção Codificação Diagnóstica (para o conjunto de CID-10 que definem a doença principal).
3. Palavras-chave: tratamento farmacológico; remédios; medicamentos.
4. Dados do paciente:
{parsed_user_query}
'''