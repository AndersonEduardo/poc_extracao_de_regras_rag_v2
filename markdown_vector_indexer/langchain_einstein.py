from __future__ import annotations

import os

from typing import Any, Optional, List
from dotenv import load_dotenv


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from hiae_mlops.api_gateway import APIManager

load_dotenv()

REQUIRED_ENV_VARS = (
    "LLM_ENDPOINT",
    "PROVIDER_NAME",
    "REALM",
    "CLIENT_ID",
    "ENVIRONMENT",
    "MEULOGIN",
    "MINHASENHA",
)


class ChatEinstein(BaseChatModel):
    """
    Wrapper LangChain para seu provider institucional Einstein.
    Você adapta aqui o modo como o Einstein recebe prompt/mensagens e devolve texto.
    """

    api_manager: Any = None
    model_name: str = "einstein-default"
    temperature: float = float(os.getenv("TEMPERATURE", 0.0))
    timeout: Optional[float] = 180.0
    url: Optional[str] = None

    def __init__(self, **kwargs: Any) -> None:

        load_dotenv()

        missing = []

        if "url" not in kwargs and not os.getenv("LLM_ENDPOINT"):

            missing.append("LLM_ENDPOINT")

        if "api_manager" not in kwargs:

            missing.extend(
                name
                for name in REQUIRED_ENV_VARS
                if name != "LLM_ENDPOINT" and not os.getenv(name)
            )

        if missing:

            raise RuntimeError(
                "Variaveis de ambiente obrigatorias para a LLM institucional nao encontradas: "
                + ", ".join(missing)
            )

        kwargs.setdefault("url", os.getenv("LLM_ENDPOINT"))
        kwargs.setdefault(
            "api_manager",
            APIManager(
                provider_name=os.getenv("PROVIDER_NAME"),
                realm=os.getenv("REALM"),
                client_id=os.getenv("CLIENT_ID"),
                environment=os.getenv("ENVIRONMENT"),
                username=os.getenv("MEULOGIN"),
                password=os.getenv("MINHASENHA"),
            ),
        )

        super().__init__(**kwargs)


    @property
    def _llm_type(self) -> str:

        return "einstein-chat"


    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Converte a lista de mensagens (system/human/ai) em uma string.
        Isso é o jeito mais simples e funciona com a maioria dos endpoints
        que esperam um único 'prompt'.
        """

        chunks = []
        
        for m in messages:
            
            role = m.type  # 'system' | 'human' | 'ai' | etc
            content = m.content if isinstance(m.content, str) else str(m.content)
            chunks.append(f"{role.upper()}:\n{content}")
            
        return "\n\n".join(chunks)

    
    def _call_einstein(self, prompt: str, **kwargs: Any) -> str:
        """
        Aqui, chama o endpoint real usando api_manager.
        """

        response = self.api_manager.make_authenticated_request(
            endpoint = self.url,
            method = "POST",
            payload = {
                "input": {
                    "prompt": prompt, 
                    "temperature": self.temperature
                }
            }
        )

        print('[STATUS POC] (_call_einstein) Status da resposta da API de LLM institucional:', response)
        # print('[STATUS POC] (_call_einstein) response.json():', response.json())
        print('[STATUS POC] (_call_einstein) prompt_tokens:', response.json()['output']['additional_kwargs']['usage']['prompt_tokens'])
        print('[STATUS POC] (_call_einstein) completion_tokens:', response.json()['output']['additional_kwargs']['usage']['completion_tokens'])
        print('[STATUS POC] (_call_einstein) total_tokens:', response.json()['output']['additional_kwargs']['usage']['total_tokens'])
        print('[STATUS POC] (_call_einstein) stop_reason:', response.json()['output']['additional_kwargs']['stop_reason'])


        return response.json()['output']['content']

    
    def _generate(self, messages: List[BaseMessage], 
                  stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        
        prompt = self._messages_to_prompt(messages)

        text = self._call_einstein(prompt, **kwargs)

        
        # aplica stop tokens se necessário
        if stop:
            
            for s in stop:
                
                if s in text:
                    
                    text = text.split(s)[0]

        
        generation = ChatGeneration(message=AIMessage(content=text))

        
        return ChatResult(generations=[generation])
