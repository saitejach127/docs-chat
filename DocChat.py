import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

class DocChat:

    def __init__(
        self,
        EMBEDDING_PATH,
        INTRODUCTION_QUESTION,
        SYSTEM_CONTEXT_MESSAGE,
        DOCUMENT_NAME,
        APIKEY,
        EMBEDDING_MODEL = "text-embedding-ada-002",
        GPT_MODEL = "gpt-3.5-turbo",
    ):
        self.EMBEDDING_PATH = EMBEDDING_PATH
        self.INTRODUCTION_QUESTION = INTRODUCTION_QUESTION
        self.SYSTEM_CONTEXT_MESSAGE = SYSTEM_CONTEXT_MESSAGE
        self.DOCUMENT_NAME = DOCUMENT_NAME
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
        self.GPT_MODEL = GPT_MODEL
        df = pd.read_csv(EMBEDDING_PATH)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        self.df = df
        openai.api_key = APIKEY

    # search function
    def strings_ranked_by_relatedness(
            self,
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) :
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = openai.Embedding.create(
            model=self.EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]


    def num_tokens(self, text: str, model: str) -> int:
        """Return the number of tokens in a string."""
        model = self.GPT_MODEL
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))


    def query_message(
        self,
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        strings, relatednesses = self.strings_ranked_by_relatedness(query, df)
        introduction = f'{self.INTRODUCTION_QUESTION}'
        question = f"\n\nQuestion: {query}"
        message = introduction
        for string in strings:
            next_article = f'\n\n {self.DOCUMENT_NAME}\n"""\n{string}\n"""'
            if (
                self.num_tokens(message + next_article + question, model=model)
                > token_budget
            ):
                break
            else:
                message += next_article
        return message + question


    def ask(
        self,
        query: str,
        previous_qas,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        model = self.GPT_MODEL
        df = self.df
        message = self.query_message(query, df, model=model, token_budget=token_budget)
        system_message = f"{self.SYSTEM_CONTEXT_MESSAGE}"
        for previous_qa in previous_qas:
            system_message += f"\nYFor this question {previous_qa.question} this is the answer provided {previous_qa.answer}"
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
        return response_message
