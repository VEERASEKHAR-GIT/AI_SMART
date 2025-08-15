# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import (
#     PromptTemplate,
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate
# )
# from langchain.memory import ConversationBufferMemory

# def create_conversational_qa_chain(llm, retriever, system_prompt, temperature, response_format):
#     # Enable memory tracking
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

#     condense_question_template = (
#         "Given the following conversation and a follow up question, "
#         "rephrase the follow up question to be a standalone question. "
#         "If there is no chat history, just return the follow up question as is.\n\n"
#         "Chat History:\n"
#         "{chat_history}\n"
#         "Follow Up Input: {question}\n"
#         "Standalone question:"
#     )

#     condense_question_prompt = PromptTemplate.from_template(condense_question_template)

#     qa_template = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(
#             system_prompt + 
#             """
#             You are an expert assistant on the provided document. Your primary role is to answer questions based *solely* on the context given to you from the document.

#             If the user greets you (e.g., "Hi", "Hello", "Good morning"), greet back normally. Be polite and professional in your tone.

#             Maintain a natural flow of conversation. There is no need to begin every answer with "Hello!" except when the user greets you.

#             IMPORTANT:
#             1. If you do not find relevant information in the provided context to answer the question, or if the question is completely out of scope (e.g., asking about weather, current time, or general knowledge not related to the document), you MUST respond with exactly:
#                 "I'm sorry, unfortunately I don't have the answer to that question as it is outside the scope of the document I have been trained on."
#                 Do NOT attempt to answer if you are unsure or if the information is not in the provided context.
#             2. You must only use the provided context to answer the user queries. Do not use your own knowledge or access the internet for answering user queries, unless it's a general greeting.

#             Here is the relevant context from the document:
#             {context}
#             """ +
#             f"""Begin generating the response with a 2-3 short sentences about the question asked, followed by the main points generated in this format: {response_format}"""
#         ),
#         HumanMessagePromptTemplate.from_template(
#             "Chat History: {chat_history}\nUser question: {question}"
#         )
#     ])

#     combine_kwargs = {"prompt": qa_template}

#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         condense_question_prompt=condense_question_prompt,
#         combine_docs_chain_kwargs=combine_kwargs,
#         return_source_documents=True
#     )
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.memory import ConversationBufferMemory
import streamlit as st

def create_conversational_qa_chain(llm, retriever, system_prompt, temperature, response_format):
    # Debug: Check if LLM is properly initialized
    if llm is None:
        st.error("LLM model is not properly initialized")
        return None
        
    if retriever is None:
        st.error("Retriever is not properly initialized")
        return None

    # Enable memory tracking
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer'
    )

    # Simplified condense question template to avoid errors
    condense_question_template = (
        "Given the following conversation and a follow up question, "
        "rephrase the follow up question to be a standalone question. "
        "If there is no chat history, just return the follow up question as is.\n\n"
        "Chat History:\n"
        "{chat_history}\n"
        "Follow Up Input: {question}\n"
        "Standalone question:"
    )

    condense_question_prompt = PromptTemplate.from_template(condense_question_template)

    # Simplified QA template to reduce complexity
    qa_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            system_prompt + 
            """
            You are an expert assistant on the provided document. Your primary role is to answer questions based *solely* on the context given to you from the document.

            If the user greets you (e.g., "Hi", "Hello", "Good morning"), greet back normally. Be polite and professional in your tone.

            Maintain a natural flow of conversation. There is no need to begin every answer with "Hello!" except when the user greets you.

            IMPORTANT:
            1. Use both the provided context and the chat history to answer questions
            2. For follow-up questions, refer to previous parts of our conversation when relevant
            3. If you do not find relevant information in the provided context to answer the question, or if the question is completely out of scope (e.g., asking about weather, current time, or general knowledge not related to the document), you MUST respond with exactly:
                "I'm sorry, unfortunately I don't have the answer to that question as it is outside the scope of the document I have been trained on."
                Do NOT attempt to answer if you are unsure or if the information is not in the provided context.
            4. You must only use the provided context to answer the user queries. Do not use your own knowledge or access the internet for answering user queries, unless it's a general greeting.

            Here is the relevant context from the document:
            {context}
            """ +
            f"""Begin generating the response with a 2-3 short sentences about the question asked, followed by the main points generated in this format: {response_format}"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Chat History: {chat_history}\nUser question: {question}"
        )
    ])

    combine_kwargs = {"prompt": qa_template}

    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs=combine_kwargs,
            return_source_documents=True,
            verbose=False
        )
        return chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None