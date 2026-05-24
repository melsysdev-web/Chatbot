from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# ==================
# MODELO
# ==================

modelo = ChatOllama(
    model="llama3.2",
    temperature=0.6
)

# ==================
# MEMORIA
# ==================

historial = InMemoryChatMessageHistory()

# ==================
# PROMPT
# ==================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres Mel, asistente virtual de tecnogigi Store.

Tu función:
- Recomendar productos tecnológicos.
- Responder preguntas frecuentes.
- Recordar información del cliente.

Sé amable y profesional.
"""
    ),

    MessagesPlaceholder("history"),

    ("human", "{input}")
])

# ==================
# CADENA
# ==================

chain = prompt | modelo

chat = RunnableWithMessageHistory(
    chain,
    lambda session_id: historial,
    input_messages_key="input",
    history_messages_key="history"
)

print("\n=== Chat activo ===")

while True:

    mensaje = input("\nCliente: ")

    if mensaje.lower() == "salir":
        break

    respuesta = chat.invoke(
        {"input": mensaje},
        config={
            "configurable": {
                "session_id": "mel"
            }
        }
    )

    print("\nMel:")
    print(respuesta.content)