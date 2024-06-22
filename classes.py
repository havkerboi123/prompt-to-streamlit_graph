

import openai
from langchain import HuggingFaceHub, LLMChain,PromptTemplate

def run_request(question_to_ask, model_type, key, alt_key):
    if model_type == "gpt-4" or model_type == "gpt-3.5-turbo" :
        # Run OpenAI ChatCompletion API
        task = "Generate Python Code Script."
        if model_type == "gpt-4":
            # Ensure GPT-4 does not include additional comments
            task = task + " The script should only include code, no comments."
        openai.api_key = ""
        response = openai.chat.completions.create(model='gpt-3.5-turbo-0125',
            messages=[{"role":"system","content":task},{"role":"user","content":question_to_ask}])
        llm_response = response.choices[0].message.content
  
    
    # rejig the response
    llm_response = format_response(llm_response)
    return llm_response

def format_response(res):
    df_code = []
    lines = res.split('\n')
    for line in lines:
        if line.strip().startswith('df') or line.strip().startswith('data') or line.strip().startswith('graph_df'):
            df_code.append(line)
        elif "import" not in line and 'st.' not in line:
            df_code.append(line)
    # Ensure the final dataframe is assigned to final_df
    df_code.append("final_df = graph_df")
    return '\n'.join(df_code)


def format_question(primer_desc,primer_code , question, model_type):
    # Fill in the model_specific_instructions variable
    instructions = ""
     
    primer_desc = primer_desc.format(instructions)  
    
    return  '"""\n' + primer_desc + question + '\n"""\n' + primer_code

def get_primer(df_dataset,df_name):
  
    primer_desc = "Use a dataframe called df from data_file.csv with columns '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'. "
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            primer_desc = primer_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. "   
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
    primer_desc = primer_desc + "\nAdd a title. Set the fig suptitle as empty."
    primer_desc = primer_desc + "{}" # Space for additional instructions if needed
    primer_desc = primer_desc + "\nUsing Python version 3.9.12, create a script using the dataframe df to graph df using st.bar_chart(df) "
    pimer_code = "import pandas as pd"
    
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    
    return primer_desc,pimer_code
