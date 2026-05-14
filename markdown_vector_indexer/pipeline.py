from __future__ import annotations

import ast
import json
import logging
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# from .langchain_einstein import ChatEinstein

LOGGER = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """
# SISTEMA

Você e um especialista sênior em extração de regras clínicas e fluxos de decisão.
Sua tarefa e analisar uma consulta do usuário junto com trechos recuperados de documentos internos e produzir regras decisórias claras, completas, objetivas, auditáveis e não ambíguas.
As regras que você produz sempre são fortemente parsimoniosas, seguindo sempre o fluxo decisório mais simples, direto e elegante possível, sem jamais deixar lacunas de cobertura do documento clínico.

# TAREFA

Criar fluxos de decisão (e.g., árvores de decisão).

# REGRAS OBRIGATÓRIAS PARA EXTRAÇÃO DE REGRAS DECISÓRIAS

1. **TODAS** as regras decisórias existentes no documento clínico devem ser extraídas.
    1.1. Dica: no Documento Principal onde ocorrer nomes de doenças, códigos CID-10, nome de medicamentos ou exames, etc., com certeza é um trecho de regra decisória.
2. Sempre que possível, use a estrutura hirárquica do próprio documento para as regras de decisão.
3. Use somente as informações sustentadas pelos trechos fornecidos.
4. Não invente critérios ausentes no contexto.
5. Gere mais do que um fluxo, se necessário.
6. Gere fluxos de decisão somente quando houver base suficiente.
7. Os passos decisórios **NÃO** devem ligar documentos clínicos diferentes (i.e., devem começar e terminar dentro de um mesmo clínico Documento Principal).
8. Cada fluxo deve ser escrito em linguagem operacional.
9. Qualquer fluxo de decisão deve, preferencialmente, ser inciado pelo conjunto de decisões relacionadas às codificações de CID-10 (ou, também, outros códigos, como CIAP, etc.), quando disponíveis.
    9.1. Encontrar e usar *todos* os códigos CID-10 da doença focal do Documento Principal (e.g., seções ou subseções intituladas como: "Diagnóstico", "Codificação Diagnóstica", etc.).
    9.2. Sempre agrupe decisões sobre CID-10 (ou outros códigos) no mesmo nó, quando possível.
    9.3. Nunca fragmente nós de decisão de CID-10 (ou qualquer outro codificação clínica) em múltiplos nós, a menos que haja uma justificativa clínica clara para isso.
10. Cada regra deve seguir o padrão: "SE ... ENTÃO ... SENÃO ...".
    10.1. Exemplo 1: "Se diagnóstico CID-10 == 'E10' OU 'T89' ENTÃO ir para N2 SENÃO ir para N3".
    10.2. Exemplo 2: "SE idade > 65 e idade <= 80 ENTÃO ir para N8 SENÃO N7".
    10.3. Exemplo 3: "SE sexo == 'feminino' ENTÃO N10 SENÃO N11".
11. Os nós de decisão devem ser de um dos dois tipos:
    11.1. nó de decisão (`decision`);
    11.2. nó terminal (`terminal`).
    11.2.1. cada sequência de decisões (ramos de decisão) dentro do fluxo decisório deve ter seu próprio nó terminal.
    11.2.2. não usar um mesmo nó terminal para múltiplos ramos de decisão, mesmo nos casos que o output clínico seja exatamente o mesmo.
12. Quaisquer outras doenças adicionais (e.g., comorbidade) à doença/problema de saúde focal, mencionadas ao longo do Documento Principal, devem ser manejados por meio de códigos CID-10 da seguinte maneira:

    SE "<CID-10 da comorbidade>" em "historico_de_saude" ENTÃO ... SENÃO ...

    12.1. Os códigos CID-10 das doenças/problemas de saúde mencionados devem prioritariamente ser aqueles que estão explicitamente disponíveis no Documento Principal.
    12.2. Nos casos em que não houver CID-10 explícito, os principais códigos CID-10 devem ser inferidos e utilizados, sempre que possível.
    12.3. Exemplos ilustrativos (assumir como exemplo o cenário de Documento Principal focado em "diabetes"):
        12.3.1. "paciente com hipertensão": "paciente com hipertensão" -> "hipertensão = sim"
        12.3.2. "histórico de depressão": "histórico de depressão" -> "historico_de_depressao = sim"
        12.3.3. "cirurgia bariátrica": "cirurgia bariátrica = sim" -> "cirurgia_bariatrica = sim"
        *Importante:* "historico_de_saude" é uma lista de códigos CID-10 a ser enviada no contexto da query do usuário.
        *Importante:* A lista "historico_de_saude" contém apenas códigos CID-10 de doenças ou problemas de saúde adicionais, ou seja, não inclui o CID-10 da doença focal do Documento Principal, que deve ser tratado separadamente como 'cid10'.
    12.4. Toda e qualquer inferência realizada deverá ser anotada em um campo específico chamado "Observações" (`observations`), no nó de decisão correspondente.
13. Quando possível, crie intervalos, limiares, frequências e condições de exclusão. Se criar, explicite isso em "Observações" (`observations`).
14. Se a informação disponível for insuficiente, explicite isso em "Observações" (`observations`).
15. Responda apenas em JSON válido.
16. Sempre valide a consistência interna do fluxo (e.g., se um nó N3 é citado como destino, ele deve existir no fluxo).

# COMO PROCEDER

1. Leia o contexto documental atentamente e passo a passo.
2. Identifique cada um dos parâmetros clínicos relevantes.
3. Normnalize os parâmetros usando as keywords fornecidas, sempre que possível.
    3.1. Ao utilizar as keywords, assuma as regras de padronização gerais: snake_case, sem acentos, cedilha, etc.
    3.2. Exemplo: "circunferência abdominal" -> "circunferencia_abdominal"; "síndrome dos ovários policísticos" -> "sindrome_dos_ovarios_policisticos".
4. Identifique valores limiares, frequências, códigos, condições de exclusão e contraindicações.
5. Identifique as relações entre os parâmetros clínicos relevantes.
6. Estruture os fluxos de decisão seguindo as regras acima (vide seção "REGRAS OBRIGATÓRIAS PARA EXTRAÇÃO DE REGRAS DECISÓRIAS").
7. Se múltiplos fluxos possíveis/necessários, separe-os claramente usando flow-ID.
    7.1. Cada fluxo (flow-ID) tem seus próprios nós de decisão (node-ID), que devem ser únicos dentro do fluxo, mas podem se repetir entre fluxos diferentes.
8. Sempre revise a consistência interna do fluxo antes de gerar o output final.
9. Sempre revise se o fluxo **cobre todas as regras relevantes** do documento clínico.
    9.1. Se o Documento Principal cita uma doença/sintoma qualquer, ela/ele *DEVE* entrar em alguma regra decisória.
10. Gere o output.

*Regra da Elegibiliade*: Qualquer fluxo decisório deve ser iniciado pela checagem dos códigos CID-10 da doença/problema de saúde focal do Documento Principal. Caso o paciente não tenha ao menos 1 (um) dos códigos CID-10 focais, o fluxo deve ser direcionado para um nó terminal, seguindo os devidos padrões do sistema.

# FORMATO DE OUTPUT

{
  "documento_principal": <"string">,
  "fluxos": [
    {
      "flow-ID": <"string" seguindo o padrão "F1">,
      "node-ID": <"string" seguindo o padrão "N1">,
      "type": "decision",
      "regra": "SE ... ENTÃO ... SENÃO ...",
      "action_if_true": <"string" com o node-ID>,
      "action_if_false": <"string" com o node-ID>,
      "observations": <"string"; use "" quando não houver inferência, conflito, ambiguidade ou informação não explícita nas fontes>
    },
    ...
    {
      "flow-ID": <"string" seguindo o padrão "F1">,
      "node-ID": <"string" seguindo o padrão "N1">,
      "type": "terminal",
      "result": <"string" com o output clínico ou de recomendação final>, # "fim" quando não houver recomendação.
      "observations": <"string"; use "" quando não houver inferência, conflito, ambiguidade ou informação não explícita nas fontes>
    }
  ]
}

# EXEMPLOS

1. Exemplo 1:
1.1. Trecho do documento clínico:
```
- Nefrologista:
    - Taxa de filtração glomerular < 30
- Cardiologista:
    - PAS > 140 ou PAD > 90 por mais de 6 meses
- Cirurgia vascular:
    - L97 Úlcera dos membros inferiores não classificada em outra parte
    - M86 Osteomielite
    - M87.3 Outras osteonecroses secundárias
    - M87.8 Outras osteonecroses
    - M87.9 Osteonecrose não especificada
    - R02 Gangrena não classificada em outra parte
```

1.2. Regra textual extraída:
```
FLOW-ID: F1
NODE-ID: N1
TYPE: decision
REGRA: SE taxa_de_filtracao_glomerular < 30 ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N2
TYPE: terminal
RESULT: nefrologista
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N3
TYPE: terminal
RESULT: fim  # sempre "fim" quando não houver recomendação.
OBSERVATIONS: 


FLOW-ID: F2
NODE-ID: N1
TYPE: decision
REGRA: SE pas_6_meses > 140 OU pad_6_meses > 90 ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS:

FLOW-ID: F2
NODE-ID: N2
TYPE: terminal
RESULT: cardiologista
OBSERVATIONS: 

FLOW-ID: F2
NODE-ID: N3
TYPE: terminal
RESULT: fim  # sempre "fim" quando não houver recomendação.
OBSERVATIONS: 


FLOW-ID: F3
NODE-ID: N1
TYPE: decision
REGRA: SE CID-10 em {L97, M86, M87.3, M87.8, M87.9, R02} ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS: 

FLOW-ID: F3
NODE-ID: N2
TYPE: terminal
RESULT: cirurgia_vascular
OBSERVATIONS: 

FLOW-ID: F3
NODE-ID: N3
TYPE: terminal
RESULT: fim  # sempre "fim" quando não houver recomendação.
OBSERVATIONS: 
```

2. Exemplo 2:
2.1. Trecho do documento clínico:
```
## 8. Exames complementares de acompanhamento após inserção de CID-10 diabetes

- a. trimestral:
    - Hemoglobina glicada:
        - Se Diabetes mellitus descompensado:
            - HbA1c >= 7%

- b. semestral:
    - Hemoglobina glicada:
        - Se Diabetes mellitus compensado:
            - HbA1c < 7%
```

2.2. Regra textual extraída:
```
FLOW-ID: F1
NODE-ID: N1
TYPE: decision
REGRA: SE HbA1c >= 7% ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS:

FLOW-ID: F1
NODE-ID: N2
TYPE: terminal
RESULT: acompanhamento_trimestral
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N3
TYPE: terminal
RESULT: acompanhamento_semestral
OBSERVATIONS: 
```
""".strip()


DECISION_FLOW_REVIEW_PROMPT = '''
# TAREFAS:

1. Veja o fluxo decisório abaixo e corrija os trechos de comorbidades (nomes de doenças, agudas ou crônicas) seguindo a seguinte regra: 

"SE <nome da doença> == 'sim'" -> "SE <nome da doença> em historico_de_saude"

2. Normalize nomes e valores de parâmetros clínicos para ficarem corretos seguindo as regras:

"... <nome do parâmetro clínico> ..." -> "... <nome_do_parametro_clinico> ..." # retirar acentuação e usar snake_case.
"... <exemplo: estratificacao_risco> ..." -> "... <estratificacao_de_risco> ..." # corrigir a escrita, para ficar gramaticalmente correto.

# OUTPUT:

Me retorne o mesmo fluxo, só que atualizado para a correção das tarefas.

# FLUXO DECISÓRIO:

{decision_flow}
'''.strip()


DEFAULT_USER_TEMPLATE = """
1. Consulta do usuário:

{query}

2. Documento Principal:

{document_name}

3. Dicionário de keywords do Documento Principal:

Ao definir os parâmetros clínicos, use as keywords abaixo para garantir estabilidade e consistência terminológica no sistema como um todo.

{document_keywords}

4. Trechos recuperados do Documento Principal:

Os seguintes trechos foram recuperados do Documento Principal e são potencialmente relevantes para responder a consulta do usuário. 
Use os trechos que forem pertinentes e relevantes como base para extrair as regras decisórias, mas não se limite a eles.

{retrieved_context}
""".strip()


RULESET_GENERATION_SYSTEM_PROMPT = '''
# SISTEMA

Você é um especialista em engenharia de regras clínicas e em transformar fluxos decisórios em linguagem natural em rulesets executáveis em Python.

# TAREFA

Sua tarefa é converter o fluxo recebido do usuário em um ruleset Python estruturado como um dicionário, pronto para uso por um motor de decisão baseado em nós de decisão.

# OBJETIVO

Transformar blocos textuais contendo FLOW-ID, NODE-ID, TYPE, REGRA, ACTION_IF_TRUE, ACTION_IF_FALSE, RESULT e OBSERVATIONS em um dicionário Python com estrutura consistente, legível e executável.

# INSTRUÇÕES GERAIS

1. Gere somente código Python válido.
2. Não escreva explicações antes nem depois do código.
3. O nome da variável de saída deve ser exatamente: `ruleset`.
4. O formato do output deve ser exatamente um dicionário Python com esta estrutura geral:

ruleset = {
    "flow_id": "...",
    "start_node": "...",
    "nodes": {
        "N1": {
            "type": <"decision">,
            "condition": "...",
            "action_if_true": "...",
            "action_if_false": "...",
            "description": "...",
            "observations": "..."  # use "" quando não houver inferência real
        },
        ...
        "NX": {
            "type": <"terminal">,
            "result": "...", # output clínico ou de recomendação final, ou mensagem de encerramento do fluxo ("fim")
            "description": "..."
        },
        ...
    }
}

5. Cada nó deve ser representado por uma chave dentro de `nodes`.
6. O nó inicial (`start_node`) deve ser o primeiro NODE-ID encontrado no fluxo.
7. Se um nó for citado como destino (`N3`, `N5`, etc.) mas não tiver definição no texto, crie esse nó como terminal placeholder, no formato:

"NX": {
    "type": "terminal",
    "result": "Fluxo encerrado em NX",
    "description": "Nó citado no fluxo, mas sem detalhamento adicional."
}

8. Para todos os outos casos, o nó terminal deve ser o último nó de um fluxo de decisões.
    8.1. O `type` do nó terminal deve ser `"terminal"`.
    8.2. O output clínico ou de recomendação final deve ser colocado diretamente no campo `result` do nó terminal.
    8.3. Quando o input trouxer `TYPE: terminal` e `RESULT: ...`, preserve esse nó como terminal e copie `RESULT` para o campo `"result"`.
    8.4. Cada sequência de decisões (ramos de decisão) dentro do fluxo decisório deve ter seu próprio nó terminal.
    8.5. Não usar um mesmo nó terminal para múltiplos ramos de decisão, mesmo nos casos que o output clínico seja exatamente o mesmo.
9. Usar snake_case para todo o conteúdo textual relacionado aos nomes de variáveis, parâmetros e seus valores correspondentes.
    9.1. Exemplos: "circunferência abdominal" -> "circunferencia_abdominal"; "estratificação de risco" -> "estratificacao_de_risco"; "muito alta" -> "muito_alta"; etc.
    9.2. *No caso específico do CID-10:* "CID-10" -> "cid10"; "CID10" -> "cid10"; "CID 10" -> "cid10"; etc.

# REGRAS DE INTERPRETAÇÃO

1. O campo `condition` deve ser uma expressão booleana Python em formato string, usando `context.get(...)`.
2. Padronize todas as variáveis em snake_case, sem acentos, sem cedilha, etc.
    2.1. Exemplos: "circunferência abdominal" -> "circunferencia_abdominal"; "CID 10" -> "cid10".
3. Use nomes de parâmetros e literais textuais de comparação sempre em letras minúsculas.
    3.1. Exemplo: `context.get("cid10") in {"e10", "e11", "e13", "e14"}`.
4. Converta operadores textuais para operadores Python:
   - E -> `and`
   - OU -> `or`
   - NÃO -> `not`
   - ≥ -> `>=`
   - ≤ -> `<=`
   - = ou “for” em contexto categórico -> `==` ou `in {...}`
5. Ao converter listas categóricas, prefira:
    5.1. `context.get("cid10") in {"e10", "e11", "e13", "e14"}`
6. Ao converter condições numéricas, siga as seguintes regras obrigatórias:
    6.1. *Regra do Maior*: sempre que a regra for com `>` ou `>=` use 0 (zero) como valor default para o `.get(..., <valor default>)`.
        6.1.1. Exemplo: context.get("glicemia", 0) >= 126
    6.2. *Regra do Menor*: sempre que a regra for com `<` ou `<=` use 1000000 (um milhão) como valor default para o `.get(..., <valor default>)`. 
        6.2.1. Exemplo: context.get('hdl_c', 1000000) < 35)
7. Para flags clínicas/binárias, prefira:
    7.1. `context.get("sedentarismo", False)`
8. Quando houver condições compostas entre parênteses no texto clínico, preserve a precedência lógica com parênteses explícitos na expressão Python.
9. Quando a regra textual indicar doença ou problema de saúde adicional (e.g., comorbidade) ao documento clínico principal como critério de decisão, converta para busca do CID-10 em `context.get("historico_de_saude", [])`.
    9.1. Exemplo: `SE "z86.4" em "historico_de_saude" ENTÃO ... SENÃO ...` -> `"z86.4" in context.get("historico_de_saude", [])`;
    9.2. Exemplo: `SE "i10" em "historico_de_saude" ENTÃO ... SENÃO ...` -> `"i10" in context.get("historico_de_saude", [])`;
    9.3. Exemplo: `SE "n18 ou n18.9 ou n03 ou z86.6" em "historico_de_saude" ENTÃO ... SENÃO ...` -> `"n18" in context.get("historico_de_saude", []) or "n18.9" in context.get("historico_de_saude", []) or "n03" in context.get("historico_de_saude", []) or "z86.6" in context.get("historico_de_saude", [])`;    
    9.4. Preserve em `"observations"` qualquer inferência de CID-10 recebida no fluxo textual ou em `OBSERVATIONS`.
    *Importante:* A lista "historico_de_saude" contém apenas códigos CID-10 de doenças ou problemas de saúde adicionais, ou seja, não inclui o CID-10 principal do documento clínico, que deve ser tratado separadamente como 'cid10'.
10. Nunca use `eval`, funções, lambdas ou código executável no campo `condition`; apenas string booleana.
11. Nunca invente variáveis clínicas fora do que puder ser inferido diretamente do texto. Se precisar nomear uma variável implícita, faça isso de forma conservadora, clara e coerente.

# REGRAS DE ENCADEAMENTO E OUTPUT

1. Use somente strings em `action_if_true` e `action_if_false`.
2. Cada `action_if_*` deve conter o nome do próximo nó, como `"N2"`.
3. Não use `if_true` nem `if_false`.
4. Não use objetos estruturados de prescrição.
5. Se o ramo levar a outro nó, o respectivo NODE-ID deve ser colocado no `action_if_true` ou `action_if_false` correspondente.
6. Se o ramo encerrar com um output clínico, o respectivo nó decisório deve ser do tipo `"terminal"` e o referido output textual deve ser adicionado diretamente em `result`.
7. Se houver conflito entre a REGRA e os campos `ACTION_IF_TRUE` / `ACTION_IF_FALSE`, priorize a interpretação semântica da REGRA e registre o problema em `observations`.
8. Se a regra disser “ENTÃO ir para NX”, use `"action_if_true": "NX"`.
9. Se a regra disser “SENÃO ir para NX”, use `"action_if_false": "NX"`.

# TIPOS DE NÓ

1. Use `"type": "decision"` para nós com condição (regra decisória).
2. Use `"type": "terminal"` para nós finais/placeholders.
    2.1. A conclusão de um ramo decisório deve ser um nó terminal.
    2.2. O output de um nó terminal sempre será o conteúdo de `result`.

# CAMPO DESCRIPTION

1. Sempre inclua um campo `description` curto e técnico para cada nó.
2. A descrição deve resumir a finalidade clínica ou lógica do nó.

# CAMPO OBSERVATIONS

1. Replique no campo `"observations"` o conteúdo recebido em `OBSERVATIONS` do texto da regra do fluxo decisório.
2. Adicione o racional à este campo quando houver inferência real ou observação relevante na conversão do texto para o formato estruturado do ruleset.
3. Nunca repita nem parafraseie o conteúdo de `"description"` em `"observations"`.
4. Caso não haja inferência real ou observação relevante, deixe o campo `"observations"` como string vazia (`""`).

# RESTRIÇÕES IMPORTANTES

1. Não omita nenhum nó presente no texto.
2. Não resuma o fluxo.
3. Não faça comentários em Python.
4. Não use markdown.
5. Não use aspas triplas.
6. Não retorne JSON; retorne dicionário Python válido.
7. Não adicione texto fora do código.

# EXEMPLOS

1. Exemplo 1:
1.1. Regra textual:
```
...
FLOW-ID: F1
NODE-ID: N1
TYPE: decision
REGRA: SE taxa_de_filtracao_glomerular < 30 ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N2
TYPE: terminal
RESULT: nefrologista
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N3
TYPE: terminal
RESULT: fim   # sempre "fim" quando não houver recomendação.
OBSERVATIONS: 
...
```

1.2. Ruleset Python:
```
...
"N1": {
    "type": "decision",
    "condition": "context.get('taxa_de_filtracao_glomerular', 1000000) < 30",
    "action_if_true": "N2",
    "action_if_false": "N3",
    "description": "Decisão baseada na taxa de filtração glomerular.",
    "observations": ""
},
"N2": {
    "type": "terminal",
    "result": "nefrologista",
    "description": "Fim do fluxo decisório com sugestão de encaminhamento para Nefrologista."
},
"N3": {
    "type": "terminal",
    "result": "fim", # sempre "fim" quando não houver recomendação.
    "description": "Fim do fluxo decisório, sem sugestões."
}
...
```

2. Exemplo 2:
2.1. Regra textual:
```
...
FLOW-ID: F1
NODE-ID: N1
TYPE: decision
REGRA: SE hemoglobina_glicada >= 7% ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS:

FLOW-ID: F1
NODE-ID: N2
TYPE: terminal
RESULT: acompanhamento_trimestral
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N3
TYPE: terminal
RESULT: acompanhamento_semestral
OBSERVATIONS: 
...
```

2.2. Ruleset Python:
```
...
"N1": {
    "type": "decision",
    "condition": "context.get('hemoglobina_glicada', 0) >= 7",
    "action_if_true": "N2",
    "action_if_false": "N3",
    "description": "Decisão clínica baseada no resultado de exame clínico para hemoglobina glicada.",
    "observations": ""
},
"N2": {
    "type": "terminal",
    "result": "acompanhamento_trimestral",
    "description": "Fim do fluxo decisório com sugestão de recomendação de acompanhamento trimestral do paciente."
},
"N3": {
    "type": "terminal",
    "result": "acompanhamento_semestral",
    "description": "Fim do fluxo decisório com sugestão de recomendação de acompanhamento semestral do paciente."
}
...
```

3. Exemplo 3: caso de comorbidades 
(IMPORTANTE: nesse exemplo, o contexto de documento/protocolo principal **não** é focado em **hipertensão**, mas sim em alguma outra doença ou problema de saúde)
3.1. Regra textual:
```
...
FLOW-ID: F1
NODE-ID: N1
TYPE: decision
REGRA: SE hipertensão = sim ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS:

FLOW-ID: F1
NODE-ID: N2
TYPE: terminal
RESULT: solicitar_ecocardiograma
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N3
TYPE: terminal
RESULT: fim  # sempre "fim" quando não houver recomendação.
OBSERVATIONS: 
...
```

3.2. Ruleset Python:
```
...
"N1": {
    "type": "decision",
    "condition": "'i10' in context.get('historico_de_saude', [])",
    "action_if_true": "N2",
    "action_if_false": "N3",
    "description": "Decisão baseada na presença de hipertensão no histórico de saúde do paciente, inferida por meio de código CID-10.",
    "observations": "O código CID-10 'I10' foi inferido por meio de I.A. generativa para a doença/problema de saúde 'Hipertensão', mencionado na regra decisória."
},
"N2": {
    "type": "terminal",
    "result": "solicitar ecocardiograma",
    "description": "Fim do fluxo decisório com sugestão de recomendação de solicitação de ecocardiograma."
},
"N3": {
    "type": "terminal",
    "result": "fim",  # sempre "fim" quando não houver recomendação.
    "description": "Fim do fluxo decisório, sem sugestões."
}
...
```
'''.strip()


RULESET_CORRECTION_SYSTEM_PROMPT = '''
# SISTEMA

Você é um especialista em engenharia de regras, fluxos decisórios clínicos e geração de rulesets executáveis em Python.
Sua tarefa é ler um fluxo decisório textual e convertê-lo em um `ruleset` Python no formato exato especificado abaixo.
O input textual pode conter nós decisórios (`TYPE: decision`) e nós terminais (`TYPE: terminal`) já explícitos.

# OBJETIVO

Transformar o fluxo decisório em um dicionário Python chamado `ruleset`, seguindo os seguintes critérios:

1. Cada NODE-ID representa um nó decisório.
2. Todo nó decisório deve ter:
   - uma condição booleana em string Python;
   - dois ramos explícitos:
     - `action_if_true`
     - `action_if_false`
3. Cada `action_if_*` deve conter apenas uma string com o próximo nó. Exemplo: `"N2"`.
4. O formato de saída deve ser estritamente compatível com a definição de output fornecida.
5. Nós com `TYPE: terminal` e `RESULT: ...` devem ser preservados como nós `"type": "terminal"` com o valor de `RESULT` no campo `"result"`.

# REGRAS DE CONVERSÃO

1. Retorne somente código Python válido.
2. Não use markdown.
3. Não escreva explicações, comentários ou texto fora do código.
4. O nome da variável final deve ser exatamente: `ruleset`

# ESTRUTURA DE OUTPUT OBRIGATÓRIA

A saída deve seguir exatamente este formato estrutural:

ruleset = {
    "flow_id": "...",
    "start_node": "...",
    "nodes": {
        "N1": {
            "type": <"decision">,
            "condition": "...",
            "action_if_true": "...",
            "action_if_false": "...",
            "description": "...",
            "observations": "..."  # use "" quando não houver inferência real
        },
        ...
        "NX": {
            "type": <"terminal">,
            "result": "...",  # sempre "fim" quando não houver recomendação.
            "description": "..."
        },
        ...
    }
}

# REGRAS DE INTERPRETAÇÃO DOS NÓS

1. Todo nó com `TYPE: decision` deve ser convertido em:
    1.1. "type": "decision"
    1.2. "condition": "..."
    1.3. "action_if_true": "..." # ID do próximo nó, como "N2"
    1.4. "action_if_false": "..." # ID do próximo nó, como "N2"
    1.5. "description": "..."
    1.6. "observations": "..."  # use "" quando não houver inferência real

1.7. Todo nó com `TYPE: terminal` deve ser convertido diretamente em:
    1.7.1. "type": "terminal"
    1.7.2. "result": <conteúdo de RESULT> # output clínico ou de recomendação final, ou mensagem de encerramento do fluxo
    1.7.3. "description": "..."

2. Se a REGRA disser “ENTÃO ir para NX”, isso significa:
    2.1. action_if_true = "NX"

3. Se a REGRA disser “SENÃO ir para NX”, então:
    3.1. action_if_false = "NX"

4. Sempre respeite `ACTION_IF_TRUE` como payload do ramo verdadeiro e `ACTION_IF_FALSE` como payload do ramo falso, desde que isso seja compatível com a semântica da REGRA.

5. Se houver aparente ambiguidade textual como:
    5.1. SE diagnóstico CID-10 == 'E10' OU 'T89':
    5.1.1. Isso deve ser interpretado como: context.get('cid10') in {'e10', 't89'}

6. Se houver texto como:
    6.1. SE estratificação de risco == 'alta' OU 'muito alta':
    6.1.1. Interpretar como: context.get('estratificacao_de_risco') in {'alta', 'muito_alta'}

# REGRAS PARA O CAMPO `condition`

1. O campo `condition` deve ser uma string contendo uma expressão booleana Python.
2. A expressão deve usar exclusivamente `context.get(...)`.
3. Usar snake_case para todo o conteúdo textual relacionado aos nomes de variáveis, parâmetros e seus valores correspondentes.
    3.1. Exemplos: "circunferência abdominal" -> "circunferencia_abdominal"; "estratificação de risco" -> "estratificacao_de_risco"; "muito alta" -> "muito_alta"; etc.
    3.2. *No caso específico do CID-10:* "CID-10" -> "cid10"; "CID10" -> "cid10"; "CID 10" -> "cid10"; etc.
4. Use nomes de parâmetros e literais textuais de comparação sempre em letras minúsculas.
    4.1. Exemplo: `context.get('cid10') in {'e10', 'e11', 't89'}`.
5. Converta expressões do fluxo para Python da seguinte forma:
    - `E` -> `and`
    - `OU` -> `or`
    - `==` mantém `==`
    - `≥` -> `>=`
    - `≤` -> `<=`
    - listas categóricas -> `in {...}` quando apropriado
6. Ao converter condições numéricas, siga as seguintes regras obrigatórias:
    6.1. *Regra do maior*: sempre que a regra for com `>` ou `>=` use 0 (zero) como placeholder: Exemplo: context.get("glicemia", 0) >= 126
    6.2. *Regra do menor*: sempre que a regra for com `<` ou `<=` use 1000000 (um milhão) como placeholder: Exemplo: context.get('hdl_c', 1000000) < 35)
7. Exemplos ilustrativos de conversão de expressões clínicas para o campo `condition`:
    7.1. `SE diagnóstico CID-10 == 'E10' OU 'T89' ...` -> `context.get('cid10') in {'e10', 't89'}`
    7.2. `SE prediabetes == 'sim' E idade < 60 ...` -> `context.get('cid10') in {"r73.0"} and context.get('idade', 1000000) < 60`
8. Quando a regra textual indicar doença ou problema de saúde adicional ao documento clínico principal como critério de decisão, converta para busca do CID-10 em `context.get("historico_de_saude", [])`.
    8.1. Exemplo: `SE "z86.4" em "historico_de_saude" ENTÃO ... SENÃO ...` -> `"z86.4" in context.get("historico_de_saude", [])`;
    8.2. Exemplo: `SE "i10" em "historico_de_saude" ENTÃO ... SENÃO ...` -> `"i10" in context.get("historico_de_saude", [])`;
    8.3. Exemplo: `SE "n18 ou n18.9 ou n03 ou z86.6" em "historico_de_saude" ENTÃO ... SENÃO ...` -> `"n18" in context.get("historico_de_saude", []) or "n18.9" in context.get("historico_de_saude", []) or "n03" in context.get("historico_de_saude", []) or "z86.6" in context.get("historico_de_saude", [])`;    
    8.4. Preserve em `"observations"` qualquer inferência de CID-10 recebida no fluxo textual ou em `OBSERVATIONS`.
    *Importante:* A lista "historico_de_saude" contém apenas códigos CID-10 de doenças ou problemas de saúde adicionais, ou seja, não inclui o CID-10 principal do documento clínico, que deve ser tratado separadamente como 'cid10'.
9. Nos casos de múltiplas condições adicionais agregadas, preserve a lógica clínica da forma mais fiel e consistente possível.
    9.1. Ou seja: evite fragmentar a lógica em múltiplos nós quando a regra original expressa uma condição composta que é clinicamente indivisível.

# REGRAS PARA OUTPUTS DOS RAMOS

1. `action_if_true` e `action_if_false` devem ser sempre strings.
2. Não use objetos, listas ou dicionários em `action_if_true` ou `action_if_false`.
3. Preserve o texto clínico com o máximo de fidelidade possível.
    3.1. Entretanto, sempre normalize variáveis clínicas usando as keywords fornecidas, quando possível.
    3.2. Ao utilizar as keywords, assuma e aplique as regras de padronização gerais do sistema: snake_case, sem acentos, sem cedilha, etc.
    3.3. Exemplo: "circunferência abdominal" -> "circunferencia_abdominal"; "ovários policísticos" -> "ovarios_policisticos"; etc.

# REGRAS PARA DESCRIÇÕES

1. Todo nó deve ter um campo `"description"`.
2. A descrição deve ser curta, técnica e objetiva.
3. Ela deve resumir a finalidade do nó.

# REGRAS PARA OBSERVATIONS

1. Todo nó deve ter um campo `"observations"` como string, podendo ser vazia.
2. Preserve nesse campo somente inferências da LLM, conflitos, ambiguidades ou informações não explícitas nas fontes.
3. Quando não houver inferência real ou observação relevante, preencha exatamente: `""`.
4. Nunca repita nem parafraseie o conteúdo de `"description"` em `"observations"`.
5. Nós terminais e placeholders não precisam conter `"observations"`.

# REGRAS IMPORTANTES DE QUALIDADE

1. Não omita nenhum nó do fluxo.
2. Não invente nós extras além dos necessários.
3. Não altere o FLOW-ID.
4. O `start_node` deve ser o primeiro NODE-ID do fluxo.
5. Preserve a natureza binária de cada decisão:
    5.1. um ramo verdadeiro
    5.2. um ramo falso
6. Não converta a saída para JSON.
7. Retorne um dicionário Python válido.
8. Não use funções, classes ou texto adicional.
9. Não use `eval`.
10. Não use placeholders vagos.
11. O resultado precisa estar pronto para ser consumido por um motor de decisão que leia:
   - `flow_id`
   - `start_node`
   - `nodes`
   - `condition`
   - `action_if_true`
   - `action_if_false`

# EXEMPLOS

1. Exemplo 1:
1.1. Regra textual:
```
...
FLOW-ID: F1
NODE-ID: N1
TYPE: decision
REGRA: SE taxa_de_filtracao_glomerular < 30 ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N2
TYPE: terminal
RESULT: nefrologista
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N3
TYPE: terminal
RESULT: fim  # sempre "fim" quando não houver recomendação.
OBSERVATIONS: 
...
```

1.2. Ruleset Python:
```
...
"N1": {
    "type": "decision",
    "condition": "context.get('taxa_de_filtracao_glomerular', 1000000) < 30",
    "action_if_true": "N2",
    "action_if_false": "N3",
    "description": "Decisão baseada na taxa de filtração glomerular.",
    "observations": ""
},
"N2": {
    "type": "terminal",
    "result": "nefrologista",
    "description": "Fim do fluxo decisório com sugestão de encaminhamento para Nefrologista."
},
"N3": {
    "type": "terminal",
    "result": "fim",  # sempre "fim" quando não houver recomendação.
    "description": "Fim do fluxo decisório, sem sugestões."
}
...
```

2. Exemplo 2:
2.1. Regra textual:
```
...
FLOW-ID: F1
NODE-ID: N1
TYPE: decision
REGRA: SE hemoglobina_glicada >= 7% ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS:

FLOW-ID: F1
NODE-ID: N2
TYPE: terminal
RESULT: acompanhamento_trimestral
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N3
TYPE: terminal
RESULT: acompanhamento_semestral
OBSERVATIONS: 
...
```

2.2. Ruleset Python:
```
...
"N1": {
    "type": "decision",
    "condition": "context.get('hemoglobina_glicada', 0) >= 7",
    "action_if_true": "N2",
    "action_if_false": "N3",
    "description": "Decisão clínica baseada no resultado de exame clínico para hemoglobina glicada.",
    "observations": ""
},
"N2": {
    "type": "terminal",
    "result": "acompanhamento_trimestral",
    "description": "Fim do fluxo decisório com sugestão de recomendação de acompanhamento trimestral do paciente."
},
"N3": {
    "type": "terminal",
    "result": "acompanhamento_semestral",
    "description": "Fim do fluxo decisório com sugestão de recomendação de acompanhamento semestral do paciente."
}
...
```

3. Exemplo 3: (IMPORTANTE: nesse exemplo, o contexto de documento/protocolo principal **não** é focado em **hipertensão**, mas sim em alguma outra doença ou problema de saúde)
3.1. Regra textual:
```
...
FLOW-ID: F1
NODE-ID: N1
TYPE: decision
REGRA: SE hipertensão = sim ENTÃO N2 SENÃO N3
ACTION_IF_TRUE: N2
ACTION_IF_FALSE: N3
OBSERVATIONS:

FLOW-ID: F1
NODE-ID: N2
TYPE: terminal
RESULT: solicitar_ecocardiograma
OBSERVATIONS: 

FLOW-ID: F1
NODE-ID: N3
TYPE: terminal
RESULT: fim  # sempre "fim" quando não houver recomendação.
OBSERVATIONS: 
...
```

3.2. Ruleset Python:
```
...
"N1": {
    "type": "decision",
    "condition": "'i10' in context.get('historico_de_saude', [])",
    "action_if_true": "N2",
    "action_if_false": "N3",
    "description": "Decisão baseada na presença de hipertensão no histórico de saúde do paciente, inferida por meio de código CID-10.",
    "observations": "O código CID-10 'I10' foi inferido por meio de I.A. generativa para a doença/problema de saúde 'Hipertensão', mencionado na regra decisória."
},
"N2": {
    "type": "terminal",
    "result": "solicitar ecocardiograma",
    "description": "Fim do fluxo decisório com sugestão de recomendação de solicitação de ecocardiograma."
},
"N3": {
    "type": "terminal",
    "result": "fim",  # sempre "fim" quando não houver recomendação.
    "description": "Fim do fluxo decisório, sem sugestões."
}
...
```
'''.strip()


@dataclass(slots=True)
class RetrievalConfig:
    chroma_dir: Path = Path("data/chromadb")
    keywords_dir: Path = Path("data/keywords")
    collection_name: str = "markdown_documents"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 15


@dataclass(slots=True)
class GenerationConfig:
    provider: str = "einstein"
    response_model: str = "gpt-5.1" #"gpt-4o"
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_TEMPLATE


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None


@dataclass(slots=True)
class DecisionFlow:
    flowID: str
    nodeID: str
    regra: str
    action_if_true: str
    action_if_false: str
    observations: str
    type: str = ""
    result: str = ""


@dataclass(slots=True)
class PipelineQueryResult:
    query: str
    document_name: str
    retrieved_chunks: list[RetrievedChunk]
    raw_response_text: str
    parsed_response: dict[str, Any]
    flows: list[DecisionFlow]


@dataclass(slots=True)
class RulesetGenerationResult:
    flow_id: str
    rendered_flow: str
    ruleset: dict[str, Any]


class DecisionFlowPipeline:
    """Pipeline RAG para transformar uma consulta em fluxos de decisao."""

    def __init__(
        self,
        retrieval: RetrievalConfig | None = None,
        generation: GenerationConfig | None = None,
    ) -> None:

        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):

            raise RuntimeError(
                "OPENAI_API_KEY nao encontrada. Defina a chave no arquivo .env do projeto."
            )

        self.retrieval = retrieval or RetrievalConfig()
        self.generation = generation or GenerationConfig()

        env_provider = os.getenv("LLM_PROVIDER")

        if env_provider and (generation is None or self.generation.provider == "einstein"):

            self.generation.provider = env_provider

        self.openai_client = OpenAI()
        self.chat_provider = self._create_chat_provider()
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.retrieval.chroma_dir.expanduser().resolve())
        )
        self.collection = self.chroma_client.get_collection(self.retrieval.collection_name)


    def _create_chat_provider(self) -> Any:

        provider = self.generation.provider.strip().lower()

        if provider == "openai":

            return self.openai_client


        if provider == "einstein":

            # return ChatEinstein(temperature=self.generation.temperature)
            raise NotImplementedError("O suporte ao ChatEinstein só está disponível para o ambiente de VDI institucional.")

        raise ValueError(
            f"Provedor de LLM nao suportado: {self.generation.provider!r}. "
            "Use 'einstein' ou 'openai'."
        )


    def query(
        self,
        query: str | list[str],
        *,
        top_k: int | None = None,
        source_name: str | None = None,
    ) -> PipelineQueryResult:

        print('[STATUS POC] (query) Rodando _normalize_query')
        normalized_query = self._normalize_query(query)
        print('[STATUS POC] (query) Done')

        print('[STATUS POC] (query) Rodando _retrieve_chunks')
        retrieved_chunks = self._retrieve_chunks(
            normalized_query,
            top_k=top_k,
            source_name=source_name,
        )
        print('[STATUS POC] (query) Done')

        print('[STATUS POC] (query) Rodando _resolve_primary_document_name')
        document_name = self._resolve_primary_document_name(retrieved_chunks)
        print('[STATUS POC] (query) Done')

        print('[STATUS POC] (query) Rodando _load_document_keywords')
        document_keywords = self._load_document_keywords(retrieved_chunks)
        print('[STATUS POC] (query) Done')

        print('[STATUS POC] (query) Rodando _build_user_prompt')
        prompt = self._build_user_prompt(
            normalized_query,
            document_name,
            retrieved_chunks,
            self._render_document_keywords(document_keywords),
        )
        print('[STATUS POC] (query) Done')

        print()
        print('##'*20)
        print('prompt:\n', prompt)
        print('##'*20)
        print()

        print('[STATUS POC] (query) Rodando _generate_response')
        raw_response_text = self._generate_response(prompt) # gera o decision flow
        print('[STATUS POC] (query) Done')

        print('[STATUS POC] (query) Rodando _parse_json_response')
        parsed_response = self._parse_json_response(raw_response_text)
        print('[STATUS POC] (query) Done')

        print('[STATUS POC] (query) Rodando _parse_flows')
        flows = self._parse_flows(parsed_response)
        print('[STATUS POC] (query) Done')

        print('[STATUS POC] (query) Retornando PipelineQueryResult')

        return PipelineQueryResult(
            query=normalized_query,
            document_name=document_name,
            retrieved_chunks=retrieved_chunks,
            raw_response_text=raw_response_text,
            parsed_response=parsed_response,
            flows=flows,
        )


    def _retrieve_chunks(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source_name: str | None = None,
    ) -> list[RetrievedChunk]:

        print('[STATUS POC] (_retrieve_chunks) Obtendo "limit"')
        limit = top_k or self.retrieval.top_k
        print('[STATUS POC] (_retrieve_chunks) Done')
    
        print()
        print('#!#!#!'*20)
        print('query:\n', query)
        print('#!#!#!'*20)
        print()

        print('[STATUS POC] (_retrieve_chunks) Rodando _embed_query')
        query_embedding = self._embed_query(query)
        print('[STATUS POC] (_retrieve_chunks) Done')

        print('[STATUS POC] (_retrieve_chunks) Construindo query_kwargs')
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
        }
        print('[STATUS POC] (_retrieve_chunks) Done')

        if source_name:

            source_names = list({
                source_name,
                unicodedata.normalize("NFC", source_name),
                unicodedata.normalize("NFD", source_name),
            })
            query_kwargs["where"] = {"source_name": {"$in": source_names}}

        print('[STATUS POC] (_retrieve_chunks) Rodando collection.query')
        results = self.collection.query(**query_kwargs)
        print('[STATUS POC] (_retrieve_chunks) Done')

        print('[STATUS POC] (_retrieve_chunks) Processando resultados da query')
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        print('[STATUS POC] (_retrieve_chunks) Done')

        print('[STATUS POC] (_retrieve_chunks) Construindo lista de RetrievedChunk')
        chunks = [
            RetrievedChunk(
                chunk_id=ids[index],
                text=documents[index],
                metadata=metadatas[index] or {},
                distance=distances[index] if index < len(distances) else None,
            )
            for index in range(len(ids))
        ]
        print('[STATUS POC] (_retrieve_chunks) Done')

        if not chunks and source_name:

            raise ValueError(
                f"Nenhum chunk foi recuperado do ChromaDB para o documento selecionado: {source_name}."
            )

        if not chunks:

            raise ValueError("Nenhum chunk foi recuperado do ChromaDB para a consulta.")


        print('[STATUS POC] (_retrieve_chunks) Filtrando chunks para manter apenas os do mesmo documento principal')
        primary_path = str(chunks[0].metadata.get("relative_path", ""))
        print('[STATUS POC] (_retrieve_chunks) Documento principal identificado:', primary_path)

        print('[STATUS POC] (_retrieve_chunks) Rodando filtragem de chunks para documento principal')
        filtered_chunks = [
            chunk
            for chunk in chunks
            if str(chunk.metadata.get("relative_path", "")) == primary_path
        ]
        print('[STATUS POC] (_retrieve_chunks) Done')

        print('[STATUS POC] (_retrieve_chunks) Retornando output')

        return sorted(filtered_chunks or chunks, key=self._chunk_index_sort_key)

    @staticmethod
    def _chunk_index_sort_key(chunk: RetrievedChunk) -> tuple[int, int]:
        raw_index = chunk.metadata.get("chunk_index")
        try:
            return (0, int(raw_index))
        except (TypeError, ValueError):
            return (1, 0)


    def _embed_query(self, query: str) -> list[float]:

        response = self.openai_client.embeddings.create(
            model = self.retrieval.embedding_model,
            input = query,
            encoding_format = "float",
        )

        return response.data[0].embedding


    def _build_user_prompt(
        self,
        query: str,
        document_name: str,
        chunks: list[RetrievedChunk],
        document_keywords: str,
    ) -> str:
        rendered_chunks = []
        for index, chunk in enumerate(chunks):
            distance_text = (
                f"{chunk.distance:.6f}" if isinstance(chunk.distance, float) else "n/a"
            )
            rendered_chunks.append(
                "\n".join(
                    [
                        f"[Trecho {index}]",
                        f"origem={chunk.metadata.get('relative_path', 'desconhecida')}",
                        f"chunk_index={chunk.metadata.get('chunk_index', 'n/a')}",
                        f"section_title={chunk.metadata.get('section_title') or 'n/a'}",
                        f"section_summary={chunk.metadata.get('section_summary') or 'n/a'}",
                        f"chunk_summary={chunk.metadata.get('chunk_summary') or 'n/a'}",
                        f"distance={distance_text}",
                        chunk.text,
                    ]
                )
            )

        return self.generation.user_prompt_template.format(
            query=query,
            document_name=document_name,
            document_keywords=document_keywords,
            retrieved_context="\n\n".join(rendered_chunks),
        )

    def _load_document_keywords(self, chunks: list[RetrievedChunk]) -> dict[str, str]:
        if not chunks:
            return {}

        relative_path = str(chunks[0].metadata.get("relative_path", "")).strip()
        if not relative_path:
            return {}

        keywords_path = (
            self.retrieval.keywords_dir.expanduser().resolve() / Path(relative_path)
        ).with_suffix(".json")
        if not keywords_path.exists():
            return {}

        try:
            payload = json.loads(keywords_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"O arquivo de keywords do documento principal nao contem um JSON valido: {keywords_path}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"O arquivo de keywords do documento principal deve conter um dicionario JSON: {keywords_path}"
            )

        normalized_keywords: dict[str, str] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    f"O arquivo de keywords do documento principal deve conter apenas pares string->string: {keywords_path}"
                )
            normalized_keywords[key] = value

        # return normalized_keywords # TODO: decisir depois como manter aqui.
        return list(normalized_keywords.keys())

    @staticmethod
    def _render_document_keywords(keywords: dict[str, str]) -> str:
        if not keywords:
            return "{}"
        return json.dumps(keywords, ensure_ascii=False, indent=2)


    def _generate_response(self, prompt: str) -> str:

        return self._generate_chat_text(
            self.generation.system_prompt,
            prompt,
            json_mode=True,
        )


    def review_decision_flow(self, result: PipelineQueryResult) -> PipelineQueryResult:

        rendered_decision_flow = json.dumps(
            result.parsed_response,
            ensure_ascii=False,
            indent=2,
        )
        prompt = DECISION_FLOW_REVIEW_PROMPT.format(
            decision_flow=rendered_decision_flow
        )

        raw_response_text = self._generate_chat_text(
            self.generation.system_prompt,
            prompt,
            json_mode=True,
        )
        parsed_response = self._parse_json_response(raw_response_text)
        flows = self._parse_flows(parsed_response)

        return PipelineQueryResult(
            query=result.query,
            document_name=result.document_name,
            retrieved_chunks=result.retrieved_chunks,
            raw_response_text=raw_response_text,
            parsed_response=parsed_response,
            flows=flows,
        )


    def _generate_chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
    ) -> str:
                
        provider = self.generation.provider.strip().lower()

        if provider == "openai":
        
            create_kwargs: dict[str, Any] = {
                "model": self.generation.response_model,
                "temperature": self.generation.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            if json_mode:

                create_kwargs["response_format"] = {"type": "json_object"}


            completion = self.chat_provider.chat.completions.create(**create_kwargs)
            
            # usage=CompletionUsage(completion_tokens=2141, prompt_tokens=10690, total_tokens=12831 
            # # TODO verificar como ajustar esse Log aqui, depois da POC.
            usage = getattr(completion, "usage", None)
            if usage is not None:
                print('[STATUS POC] (_generate_chat_text): Token usage summary:')
                print('\tcompletion.usage.prompt_tokens =', usage.prompt_tokens)
                print('\tcompletion.usage.completion_tokens =', usage.completion_tokens)
                print('\tcompletion.usage.total_tokens =', usage.total_tokens)

            content = completion.choices[0].message.content

        
        elif provider == "einstein":
        
            from langchain_core.messages import HumanMessage, SystemMessage

            message = self.chat_provider.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )

            content = message.content

        else:
        
            raise ValueError(
                f"Provedor de LLM nao suportado: {self.generation.provider!r}. "
                "Use 'einstein' ou 'openai'."
            )

        if not isinstance(content, str) or not content.strip():
        
            raise ValueError("A LLM retornou uma resposta vazia.")
        

        return content.strip()


    @staticmethod
    def _normalize_query(query: str | list[str]) -> str:
        if isinstance(query, str):
            normalized = query.strip()
        else:
            normalized = "; ".join(part.strip() for part in query if part.strip())

        if not normalized:
            raise ValueError("A query nao pode ser vazia.")
        return normalized

    @staticmethod
    def _resolve_primary_document_name(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "desconhecido"
        metadata = chunks[0].metadata
        return str(
            metadata.get("source_name")
            or metadata.get("document_name")
            or metadata.get("relative_path")
            or "desconhecido"
        )

    @staticmethod
    def _parse_json_response(raw_response_text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_response_text)
        except json.JSONDecodeError as exc:
            raise ValueError("A resposta da LLM nao veio em JSON valido.") from exc

        if not isinstance(parsed, dict):
            raise ValueError("A resposta JSON da LLM deve ser um objeto.")
        return parsed

    @staticmethod
    def _parse_flows(payload: dict[str, Any]) -> list[DecisionFlow]:

        raw_flows = payload.get("fluxos", [])
        if not isinstance(raw_flows, list):
            return []

        flows: list[DecisionFlow] = []
        for item in raw_flows:
            if not isinstance(item, dict):
                continue
            flows.append(
                DecisionFlow(
                    flowID=str(item.get("flow-ID", "")).strip(),
                    nodeID=str(item.get("node-ID", "")).strip(),
                    regra=str(item.get("regra", "")).strip(),
                    action_if_true=str(item.get("action_if_true", "")).strip(),
                    action_if_false=str(item.get("action_if_false", "")).strip(),
                    observations=str(item.get("observations", "")).strip(),
                    type=str(item.get("type", "")).strip().lower(),
                    result=str(item.get("result", "")).strip(),
                )
            )
        return flows


class RulesetPipeline(DecisionFlowPipeline):
    """Pipeline especializada para gerar regras clínicas a partir de consultas e documentos."""

    def __init__(
        self,
        retrieval: RetrievalConfig | None = None,
        generation: GenerationConfig | None = None,
    ) -> None:
        
        super().__init__(retrieval, generation)


    def prepare_ruleset_input(self, flows: list[DecisionFlow]) -> tuple[str, str]:

        return self.prepare_ruleset_inputs(flows)[0]


    def prepare_ruleset_inputs(self, flows: list[DecisionFlow]) -> list[tuple[str, str]]:

        grouped_flows = self._group_flows_by_id(flows)

        return [
            (flow_id, self._render_ruleset_generation_input(group))
            for flow_id, group in grouped_flows.items()
        ]


    def build_ruleset(self, flows: list[DecisionFlow]) -> dict[str, Any]:

        selected_flow_id, rendered_flow = self.prepare_ruleset_input(flows)

        raw_ruleset = self._generate_ruleset_response(rendered_flow)
        cleaned_ruleset = self._clean_ruleset_response(raw_ruleset)
        ruleset = self._parse_ruleset_python(cleaned_ruleset)

        self._validate_ruleset(ruleset, expected_flow_id=selected_flow_id)

        return ruleset


    def build_rulesets(self, flows: list[DecisionFlow]) -> list[RulesetGenerationResult]:

        results: list[RulesetGenerationResult] = []

        for selected_flow_id, rendered_flow in self.prepare_ruleset_inputs(flows):

            raw_ruleset = self._generate_ruleset_response(rendered_flow)

            cleaned_ruleset = self._clean_ruleset_response(raw_ruleset)

            ruleset = self._parse_ruleset_python(cleaned_ruleset)

            self._validate_ruleset(ruleset, expected_flow_id=selected_flow_id)

            results.append(
                RulesetGenerationResult(
                    flow_id=selected_flow_id,
                    rendered_flow=rendered_flow,
                    ruleset=ruleset,
                )
            )

        return results


    @staticmethod
    def _group_flows_by_id(flows: list[DecisionFlow]) -> dict[str, list[DecisionFlow]]:

        if not flows:
            raise ValueError("A lista de fluxos nao pode ser vazia.")

        grouped_flows: dict[str, list[DecisionFlow]] = {}

        for flow in flows:

            flow_id = flow.flowID.strip()

            if not flow_id:

                continue

            grouped_flows.setdefault(flow_id, []).append(flow)

        if not grouped_flows:

            raise ValueError("Nao foi encontrado um flowID valido nos fluxos informados.")

        return grouped_flows


    def correct_ruleset(self, rendered_flow: str, *, expected_flow_id: str) -> dict[str, Any]:

        if not isinstance(rendered_flow, str) or not rendered_flow.strip():
            raise ValueError("O texto do fluxo para correcao nao pode ser vazio.")

        if not isinstance(expected_flow_id, str) or not expected_flow_id.strip():
            raise ValueError("O flow_id esperado para correcao nao pode ser vazio.")

        raw_ruleset = self._generate_corrected_ruleset_response(rendered_flow)
        cleaned_ruleset = self._clean_ruleset_response(raw_ruleset)

        print()
        print('- raw_ruleset:\n', cleaned_ruleset)
        print()

        ruleset = self._parse_ruleset_python(cleaned_ruleset)

        print()
        print('- ruleset:\n', ruleset)
        print()

        self._validate_ruleset(ruleset, expected_flow_id=expected_flow_id.strip())

        return ruleset


    @staticmethod
    def _render_ruleset_generation_input(flows: list[DecisionFlow]) -> str:

        rendered_flows = []

        for flow in flows:

            flow_type = flow.type.strip().lower()
            rendered_fields = [
                f"FLOW-ID: {flow.flowID}",
                f"NODE-ID: {flow.nodeID}",
            ]

            if flow_type:

                rendered_fields.append(f"TYPE: {flow_type}")

            if flow_type == "terminal":

                rendered_fields.extend(
                    [
                        f"RESULT: {flow.result}",
                        f"OBSERVATIONS: {flow.observations}",
                    ]
                )

            else:

                rendered_fields.extend(
                    [
                        f"REGRA: {flow.regra}",
                        f"ACTION_IF_TRUE: {flow.action_if_true}",
                        f"ACTION_IF_FALSE: {flow.action_if_false}",
                        f"OBSERVATIONS: {flow.observations}",
                    ]
                )

            rendered_flows.append(
                "\n".join(rendered_fields)
            )

        return "\n\n".join(rendered_flows)


    def _generate_ruleset_response(self, rendered_flow: str) -> str:

        return self._generate_chat_text(
            RULESET_GENERATION_SYSTEM_PROMPT,
            rendered_flow,
        )


    def _generate_corrected_ruleset_response(self, rendered_flow: str) -> str:

        return self._generate_chat_text(
            RULESET_CORRECTION_SYSTEM_PROMPT,
            rendered_flow,
        )


    @staticmethod
    def _clean_ruleset_response(raw_ruleset: str) -> str:
        return raw_ruleset.strip("```python").strip("```").strip()


    @staticmethod
    def _parse_ruleset_python(raw_ruleset: str) -> dict[str, Any]:

        try:

            tree = ast.parse(raw_ruleset, mode="exec")

        except SyntaxError as exc:

            raise ValueError("A resposta da LLM nao contem Python valido.") from exc

        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):

            raise ValueError("A resposta da LLM deve conter apenas uma atribuicao a `ruleset`.")

        assign_node = tree.body[0]

        if len(assign_node.targets) != 1 or not isinstance(assign_node.targets[0], ast.Name):

            raise ValueError("A atribuicao retornada pela LLM e invalida.")

        target_name = assign_node.targets[0].id

        if target_name != "ruleset":

            raise ValueError("A resposta da LLM deve atribuir o dicionario a variavel `ruleset`.")

        try:
            parsed_ruleset = ast.literal_eval(assign_node.value)

        except (ValueError, SyntaxError) as exc:

            raise ValueError("O valor atribuido a `ruleset` deve ser um dicionario literal valido.") from exc

        if not isinstance(parsed_ruleset, dict):

            raise ValueError("O valor atribuido a `ruleset` deve ser um dicionario.")


        return parsed_ruleset


    @staticmethod
    def _canonical_flow_id(value: Any) -> str:

        if not isinstance(value, str) or not value.strip():

            raise ValueError("O flow_id deve ser uma string nao vazia.")

        return value.strip().casefold()


    @staticmethod
    def _validate_ruleset(ruleset: dict[str, Any], *, expected_flow_id: str) -> None:

        flow_id = ruleset.get("flow_id")
        start_node = ruleset.get("start_node")
        nodes = ruleset.get("nodes")

        try:
            canonical_flow_id = RulesetPipeline._canonical_flow_id(flow_id)
        except ValueError as exc:

            raise ValueError("O ruleset deve conter `flow_id` como string nao vazia.") from exc

        try:
            canonical_expected_flow_id = RulesetPipeline._canonical_flow_id(expected_flow_id)
        except ValueError as exc:

            raise ValueError("O flow_id esperado deve ser uma string nao vazia.") from exc

        expected_flow_id = expected_flow_id.strip()

        if canonical_flow_id != canonical_expected_flow_id:

            raise ValueError(
                f"O `flow_id` retornado ({flow_id!r}) difere do fluxo esperado ({expected_flow_id!r})."
            )

        ruleset["flow_id"] = expected_flow_id
        
        if not isinstance(start_node, str) or not start_node.strip():

            raise ValueError("O ruleset deve conter `start_node` como string nao vazia.")
        
        if not isinstance(nodes, dict) or not nodes:

            raise ValueError("O ruleset deve conter `nodes` como dicionario nao vazio.")
        
        if start_node not in nodes:

            raise ValueError("O `start_node` informado nao existe em `nodes`.")

        for node_id, node in nodes.items():

            if not isinstance(node_id, str) or not node_id.strip():

                raise ValueError("Todos os identificadores de no devem ser strings nao vazias.")
            
            if not isinstance(node, dict):

                raise ValueError(f"O no `{node_id}` deve ser representado como dicionario.")

            node_type = node.get("type")

            if node_type == "decision":

                RulesetPipeline._validate_decision_node(node_id, node, nodes)

                continue

            if node_type == "terminal":

                RulesetPipeline._validate_terminal_node(node_id, node)

                continue

            raise ValueError(
                f"No `{node_id}` possui `type` invalido: {node_type!r}. "
                "Esperado: 'decision' ou 'terminal'."
            )


    @staticmethod
    def _validate_decision_node(
        node_id: str,
        node: dict[str, Any],
        nodes: dict[str, Any],
    ) -> None:
        
        condition = node.get("condition")
        action_if_true = node.get("action_if_true")
        action_if_false = node.get("action_if_false")
        observations = node.get("observations")

        if not isinstance(condition, str) or not condition.strip():

            raise ValueError(f"No `{node_id}` deve conter `condition` como string nao vazia.")
        
        if not isinstance(action_if_true, str) or not action_if_true.strip():

            raise ValueError(f"No `{node_id}` deve conter `action_if_true` como string nao vazia.")
        
        if not isinstance(action_if_false, str) or not action_if_false.strip():

            raise ValueError(f"No `{node_id}` deve conter `action_if_false` como string nao vazia.")

        if "observations" not in node or not isinstance(observations, str):

            raise ValueError(f"No `{node_id}` deve conter `observations` como string.")


    @staticmethod
    def _validate_terminal_node(node_id: str, node: dict[str, Any]) -> None:

        result = node.get("result")

        if not isinstance(result, str) or not result.strip():

            raise ValueError(f"No terminal `{node_id}` deve conter `result` como string nao vazia.")


    @staticmethod
    def _normalize_context(context: dict[str, Any]) -> dict[str, Any]:
        return {
            key.lower() if isinstance(key, str) else key: (
                value.lower() if isinstance(value, str) else value
            )
            for key, value in context.items()
        }


    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Avalia uma expressão booleana Python armazenada como string.

        A expressão deve usar apenas `context.get(...)` para acessar variáveis.
        Exemplo:
            'context.get("glicemia", 0) >= 126'

        Parâmetros
        ----------
        condition : str
            Expressão booleana em formato string.
        context : dict
            Dicionário com os dados de entrada do paciente/contexto.

        Retorno
        -------
        bool
            Resultado da avaliação da condição.

        Observação
        ----------
        Esta implementação usa eval com builtins desabilitados para manter
        a interface simples e compatível com o formato de ruleset adotado.
        Em produção, o ideal é substituir por um avaliador seguro via AST/parser.
        """

        if not isinstance(condition, str) or not condition.strip():

            raise ValueError("A condição deve ser uma string não vazia.")

        safe_globals = {"__builtins__": {}}
        safe_locals = {"context": context}

        # print('---'*30)
        # print('-> condition:', condition, '\n')
        # print('-> context:', context)
        # print('---'*30)

        result = eval(condition, safe_globals, safe_locals)

        if not isinstance(result, bool):

            raise ValueError(f"A condição não retornou booleano: {condition}")

        return result


    def solve(self, ruleset:dict, query:dict, max_steps:int = 100):
        """
        Executa um ruleset baseado em nós de decisão.

        Parâmetros
        ----------
        ruleset : dict
            Dicionário contendo:
                - flow_id
                - start_node
                - nodes
        context : dict
            Dados de entrada usados para resolver as condições.
        max_steps : int
            Limite de segurança para evitar loops infinitos.

        Retorno
        -------
        dict
            Estrutura com:
                - flow_id
                - final_node
                - status
                - trace
                - actions
                - output
                - warnings

        Formato do retorno
        ------------------
        {
            "flow_id": "F1",
            "final_node": "FIM",
            "status": "completed",
            "trace": [...],
            "actions": [...],
            "output": "...",
            "warnings": [...]
        }
        """
        if not isinstance(ruleset, dict):
            raise ValueError("`ruleset` deve ser um dicionário.")

        flow_id = ruleset.get("flow_id")
        start_node = ruleset.get("start_node")
        nodes = ruleset.get("nodes")

        if not flow_id:
            raise ValueError("O ruleset deve conter `flow_id`.")
        if not start_node:
            raise ValueError("O ruleset deve conter `start_node`.")
        if not isinstance(nodes, dict) or not nodes:
            raise ValueError("O ruleset deve conter `nodes` como dicionário não vazio.")

        query = self._normalize_context(query)
        current_node_id = start_node
        trace: List[Dict[str, Any]] = []
        actions: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for step_index in range(max_steps):
            node = nodes.get(current_node_id)

            if node is None:
                raise KeyError(f"Nó `{current_node_id}` não encontrado no ruleset.")

            node_type = node.get("type")

            if node_type == "terminal":
                trace.append({
                    "step": step_index + 1,
                    "node_id": current_node_id,
                    "node_type": "terminal",
                    "result": node.get("result"),
                    "description": node.get("description")
                })

                return {
                    "flow_id": flow_id,
                    "final_node": current_node_id,
                    "status": "completed",
                    "trace": trace,
                    "actions": actions,
                    "output": node.get("result"),
                    "warnings": warnings
                }

            if node_type != "decision":
                raise ValueError(
                    f"Nó `{current_node_id}` possui type inválido: {node_type!r}. "
                    "Esperado: 'decision' ou 'terminal'."
                )

            condition = node.get("condition")
            action_if_true = node.get("action_if_true")
            action_if_false = node.get("action_if_false")
            warning = node.get("warning")

            if warning:
                warnings.append(f"{current_node_id}: {warning}")

            condition_result = self.evaluate_condition(condition=condition, context=query)

            if condition_result:
                selected_payload = action_if_true
                branch = "true"
            else:
                selected_payload = action_if_false
                branch = "false"

            actions.append({
                "node_id": current_node_id,
                "branch": branch,
                "action": selected_payload
            })

            next_node_id = selected_payload if isinstance(selected_payload, str) and selected_payload in nodes else None

            trace.append({
                "step": step_index + 1,
                "node_id": current_node_id,
                "node_type": "decision",
                "condition": condition,
                "condition_result": condition_result,
                "branch_taken": branch,
                "next_node": next_node_id,
                "selected_action": selected_payload,
                "description": node.get("description")
            })

            if not isinstance(selected_payload, str) or not selected_payload.strip():
                raise ValueError(
                    f"Nó `{current_node_id}` não definiu output para o ramo `{branch}`."
                )

            if next_node_id is None:
                return {
                    "flow_id": flow_id,
                    "final_node": None,
                    "status": "completed",
                    "trace": trace,
                    "actions": actions,
                    "output": selected_payload,
                    "warnings": warnings
                }

            current_node_id = next_node_id

        raise RuntimeError(
            f"Execução interrompida após {max_steps} passos. "
            "Possível ciclo no ruleset."
        )


    
