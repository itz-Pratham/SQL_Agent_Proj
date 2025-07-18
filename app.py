import streamlit as st
import pandas as pd
import duckdb
import os
import tempfile
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()
# Azure OpenAI credentials
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")

llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=1
)

st.title("🧠 Agentic AI: Ask Questions on Any Dataset")

uploaded_file = st.file_uploader("Upload your dataset (CSV only, max 100MB)", type=["csv"])

final_answer = ""

if uploaded_file:
    if uploaded_file.size > 100 * 1024 * 1024:
        st.error("❌ File too large. Please upload a CSV file under 100MB.")
    else:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', encoding_errors='ignore')

            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            df.fillna(value={col: 'unknown' if df[col].dtype == 'O' else 0 for col in df.columns}, inplace=True)

            st.subheader("📁 Preview of the Dataset")
            st.dataframe(df.head())

            # with tempfile.NamedTemporaryFile(suffix=".duckdb") as tmp_db:
            #     con = duckdb.connect(database=tmp_db.name)
            #     con.register("dataset", df)
            con = duckdb.connect(database=':memory:')
            con.register("dataset", df)


            user_question = st.text_input("Ask a question about your dataset")

            if user_question:
                schema_str = ", ".join([f"{col} ({str(dtype)})" for col, dtype in zip(df.columns, df.dtypes)])

                prompt_sql = PromptTemplate(
                    input_variables=["schema", "question"],
                    template=(
                        "Given the following table schema: {schema}\n"
                        "The table name is `dataset`"
                        "Write a SQL query to answer this question: {question}\n"
                        "Return only the SQL query."
                    )
                )

                sql_chain = LLMChain(llm=llm, prompt=prompt_sql)

                try:
                    generated_sql = sql_chain.run({"schema": schema_str, "question": user_question})

                    # if not generated_sql.strip().lower().startswith("select"):
                    #     raise ValueError("Ambiguous or invalid question. SQL generation failed.")

                    # st.code(generated_sql, language="sql")

                    # cleaned_sql=generated_sql.strip().replace("```sql","").replace("```","").replace('"""','').replace("'''","")

                    cleaned_sql = re.sub(r"```sql|```|'''|\"\"\"","",generated_sql,flags=re.IGNORECASE).strip()

                    # st.code(cleaned_sql, language="sql")

                    result = con.execute(cleaned_sql).fetchdf()

                    if result is None:
                        result_value = pd.DataFrame()
                    else:
                        result_value = result
                        

                    if result_value.empty:
                        final_answer = "I'm sorry, the query didn't return any meaningful result."
                    else:
                        prompt_nl = PromptTemplate(
                            input_variables=["question", "answer"],
                            template=(
                                "You are a helpful assistant. Given the question: \"{question}\" and the SQL result: \"{answer}\",\n"
                                "respond with a human-readable answer."
                            )
                        )

                        answer_chain = LLMChain(llm=llm, prompt=prompt_nl)
                        final_answer = answer_chain.run({"question": user_question, "answer": str(result_value)})

                except Exception as e:
                    final_answer = f"❌ Error while generating or executing query: {e}"

                st.subheader("📝 Final Answer")
                st.write(final_answer)

        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
else:
    st.info("Upload a CSV file to get started!")