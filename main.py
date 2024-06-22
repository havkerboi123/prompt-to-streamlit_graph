import pandas as pd
import openai
import re
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from classes import get_primer, format_question, run_request, format_response
import warnings

warnings.filterwarnings("ignore")

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key="")

def remove(text):
    patterns_to_remove = [
        'final_df = graph_df',
        'st.xlabel',
        'st.ylabel',
        'st.title',
        'st.suptitle("")',
        'st.show()',
        'st.pyplot()',
        '```python',
        '```',
        'suptitle'
    ]
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not any(pattern in line for pattern in patterns_to_remove)]
    return '\n'.join(filtered_lines)

def format_code(code: str) -> str:
    statements = re.split(r';|\n', code)
    statements = [stmt.strip() for stmt in statements if stmt.strip()]
    return '\n'.join(statements)

available_models = {"GPT4": "gpt-4", "GPT3.5": "gpt-3.5-turbo"}

if "datasets" not in st.session_state:
    datasets = {}
    datasets["test_data_set"] = pd.read_csv("/Users/mhmh/Desktop/p2/abd/t.csv")
    st.session_state["datasets"] = datasets
else:
    datasets = st.session_state["datasets"]


with st.sidebar:
    dataset_container = st.empty()
    index_no = 0
    chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:", datasets.keys(), index=index_no)
    st.write("Choose your model")
    use_model = {}
    for model_desc, model_name in available_models.items():
        label = f"{model_desc} ({model_name})"
        key = f"key_{model_desc}"
        use_model[model_desc] = st.checkbox(label, value=True, key=key)

question1 = "Bar chart showing the total number of males and females that died on titanic"
question2 = "bar chart showing count of every unique make/model"
go_btn = st.button("Visualize")



if go_btn:
    api_keys_entered = True
    if "GPT4" in use_model or "GPT3.5" in use_model:
        if not openai_key.startswith('sk-'):
            st.error("Please enter a valid OpenAI API key.")
            api_keys_entered = False
    if "Code Llama" in use_model:
        if not hf_key.startswith('hf_'):
            st.error("Please enter a valid HuggingFace API key.")
            api_keys_entered = False
    if api_keys_entered:
        plots = st.columns(len(use_model))
        primer1, primer2 = get_primer(datasets[chosen_dataset], 'datasets["' + chosen_dataset + '"]')
        for plot_num, model_type in enumerate(use_model):
            if use_model[model_type]:
                with plots[plot_num]:
                    st.subheader(model_type)
                    if True:
                        question_to_ask_1 = format_question(primer1, primer2, question1, model_type)
                        question_to_ask_2 = format_question(primer1, primer2, question2, model_type)
                        answer1 = run_request(question_to_ask_1, available_models[model_type], key=openai_key, alt_key=hf_key)
                        answer2 = run_request(question_to_ask_2, available_models[model_type], key=openai_key, alt_key=hf_key)
                        answer1 = primer2 + answer1
                        answer2 = primer2 + answer2
                        res1 = llm.invoke(f"Change this code so it can be plotted using streamlit. Remove this line if it exists: df = pd.read_csv('data_file.csv'). Do not add comments. Make sure st.bar_chart is used only.\nCode:{answer1}")
                        res2 = llm.invoke(f"Change this code so it can be plotted using streamlit. Remove this line if it exists: df = pd.read_csv('data_file.csv'). Do not add comments. Make sure st.bar_chart is used only. Do not use st.pyplot(). Ensure code is well formatted and will run\nCode:{answer2}")
                        res1.content=remove(res1.content)
                        res2.content=remove(res2.content)
                        
                    st.write("Prompt:Bar chart showing the total number of males and females that died on titanic")
                    exec(res1.content)
                    st.write("Prompt:Bar chart showing count of every unique make/model")
                    exec(res2.content)





file = "/Users/mhmh/Desktop/p2/abd/t.csv"
question = st.text_input("Ask me something about the data")
agent = create_csv_agent(llm, file)
if question:
    st.write(agent.run(question))
