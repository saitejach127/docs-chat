from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

template = """
Question: {question}
Answer: Let's think step by step.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])


llm = LlamaCpp(model_path="./weights/gpt4all-lora-q-converted.bin")

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)