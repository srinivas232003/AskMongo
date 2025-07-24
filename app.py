# app.py
import streamlit as st
import pymongo
import json
import re
from bson.json_util import dumps
from dotenv import load_dotenv
import os
# from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
# ---- Setup ---- #
# llm = ChatOllama(
#     model = "qwen3:1.7b",
#     temperature = 0.8,
#     num_predict=4096,
#     # other params ...
# )
load_dotenv()
import bson
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import json
llm = ChatGroq(model="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])
# MongoDB setup
# client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
# db = client["agriculture"]
# collection = db["crops"]

# Schema Description (condensed here for brevity)
schema_description = """This table contains comprehensive information about a farmer's crop cycle.
- farmer: An object containing the farmer's personal details.
  - name: The full name of the farmer.
  - father_name: The name of the farmer's father.
- code: A unique integer code assigned to this specific crop record.
- location: An object with geographical details of the farmland.
  - village: The name of the village where the farm is located.
  - district: The name of the district where the farm is located.
- crop_info: An object detailing the crop's timeline and status.
  - sowing_date: The date when the crop seeds were sown.
  - expected_harvest_date: The originally planned or estimated date for harvesting the crop.
  - actual_harvest_date: The actual date on which the crop was harvested.
  - harvest_status: The current status of the harvest (e.g., "Harvested", "Pending").
- risk_status: A qualitative assessment of the risk associated with this crop (e.g., "Low risk", "Medium risk", "High risk").
- chemical: An object describing the chemical application on the crop.
  - name: The specific name of the pesticide or fertilizer used.
  - application_date: The date when the chemical was applied.
- acre: The total area of the farmland in acres.
- expected_quantity: The anticipated yield from the harvest in a standard unit (e.g., tonnes).
- actual_quantity: The final measured yield from the harvest in a standard unit (e.g., tonnes).
- assigned_to: The name of the field officer or agent responsible for overseeing this record.
- sample_collection: Indicates whether a crop sample was collected for analysis (e.g., "Collected", "Not collected").
- remarks: A field for any additional notes or comments regarding the crop cycle, such as reasons for yield differences.
"""

# The nested schema structure reflecting the JSON object.
table_schema = {
    "farmer": {
        "name": "string",
        "father_name": "string"
    },
    "code": "integer",
    "location": {
        "village": "string",
        "district": "string"
    },
    "crop_info": {
        "sowing_date": "date",
        "expected_harvest_date": "date",
        "actual_harvest_date": "date",
        "harvest_status": "string"
    },
    "risk_status": "string",
    "chemical": {
        "name": "string",
        "application_date": "date"
    },
    "acre": "integer",
    "expected_quantity": "integer",
    "actual_quantity": "integer",
    "assigned_to": "string",
    "sample_collection": "string",
    "remarks": "string"
}
json_ex_1 = {
  "pipeline_groups": [
    [
      { "$match": { "location.district": "Cuddalore", "actual_quantity": { "$gt": 10 } } },
    #   { "$project": { "farmer_name": "$farmer.name", "yield": "$actual_quantity", "_id": 0 } }
    ],
    [
    {
        "$match": {
          "location.district": "Cuddalore",
          "actual_quantity": { "$gt": 10 }
        }
    },
    {
        "$count": "matching_farmers"
    },
    ]
  ]
}
json_ex_string = json.dumps(json_ex_1)

# ---- Functions ---- #
def should_send_to_llm(result):
    if not result:
        return True
    if len(result) == 1 and isinstance(result[0], dict) and len(result[0]) <= 3:
        return True
    if len(result) <= 5:
        return True
    return False

def build_pipeline_prompt(user_question):
    # This string contains all the instructions and placeholders
    prompt_template= f"""You are an expert in crafting NoSQL queries for MongoDB with 10 years of experience, particularly in MongoDB.
I will provide you with the table_schema and schema_description in a specified format.
Your task is to read the user_question and create a NOSQL MongoDb pipeline accordingly.All pipeline groups must be written independent of each other,
there is no guarantee that each pipeline group will be executed one after another.

-- START CONTEXT ---
Table schema:
{table_schema}

Schema Description:
{schema_description}

Here is an example:
Input: How many farmers from Cuddalore district have an actual quantity greater than 10? List their names and yields.
Output: {json_ex_string}
--- END CONTEXT ---

Important Rules:
1. Always output a valid JSON object with a "pipeline_groups" key.
2. `"pipeline_groups"` is a **list of arrays**. Each array contains a separate aggregation pipeline for one logical objective.
3. **Each pipeline group must be fully independent** — no group depends on the output of another.
4. Each group in pipeline_groups must be a separate array containing related stages.
5. Use separate groups for different logical operations (e.g., getting data vs counting).
6. Do not include any explanations or extra text.
7. The output must be directly usable by json.loads().
8. Each pipeline stage must be a valid MongoDB aggregation stage.
9. If the user wants all matching records or full details, do not use $project — return full documents by omitting it.
10. Only use $project when the user asks for specific fields.
Your response must contain only the JSON object, nothing else."""

    # prompt = PromptTemplate(
    # partial_variables={
    #     "table_schema": table_schema,
    #     "schema_description": schema_description,
    #     "json_ex_string": json_ex_string
    # },
    # template=prompt_template)

    return [
        SystemMessage(content=prompt_template),
        HumanMessage(content=user_question)
    ]

def clean_and_parse_pipeline(raw_text):
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    print(fr"Raw pipeline text: {raw_text}")
    return json.loads(raw_text)["pipeline_groups"]

def summarize_with_llm(user_question, llm_outputs):
    combined_text = "\n\n".join(
        f"Result :\n{dumps(res, indent=2)}"
        for res in llm_outputs
    )
    summary_messages = [
    SystemMessage(
    content=(
        "You are a helpful data assistant. Your task is to summarize the given MongoDB query result "
        "based on the user's question.\n"
        "You are only seeing a part of the result, not the full dataset.\n"
        "Never mention or comment on what data is missing or unavailable.\n"
        "Just summarize what is given clearly and concisely, in a natural and informative tone.\n"
        "Do not use technical terms like 'field' or 'document'. Speak like you're explaining the result to a non-technical user.\n"
        "If the result is empty, say that no matching data was found.\n"
        "Do not assume anything beyond the given result."
    )
),

        HumanMessage(
    content=(
           f"User asked:\n{user_question}\n\n"
            f"Here are partial or summarized results from different pipeline groups in MongoDB:\n\n"
            f"{combined_text}\n\n"
        "Please summarize this result clearly and directly without mentioning missing data or fields."
    )
)


    ]
    return llm.invoke(summary_messages).content
import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Prerequisite:
# This code assumes your variables `llm` and `collection` are initialized,
# and your helper functions like `build_pipeline_prompt` are defined.
# ==============================================================================

## -----------------
## Sidebar
## -----------------
# Update the sidebar section with MongoDB URI input
with st.sidebar:
    st.title("🌾 MongoDB Q&A Assistant")
    st.markdown("This assistant translates your questions into MongoDB Aggregation Pipelines and fetches the results.")
    st.markdown("---")
    
    # Add MongoDB URI input section
    st.markdown("### MongoDB Connection")
    default_uri = ""
    mongodb_uri = st.text_input(
        "MongoDB URI",
        value=default_uri,
        type="password",  # Masks the URI for security
        help="Enter your MongoDB connection string"
    )
    
    st.markdown("---")
    st.markdown("### Controls")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.markdown("---")
    st.info("💡 Tip: Ask 'What is the average yield for wheat by region?'")

# Update MongoDB connection to use the user-provided URI
try:
    client = pymongo.MongoClient(mongodb_uri)
    # Test the connection
    client.admin.command('ping')
    st.sidebar.success("✅ Connected to MongoDB")
    db = client["agriculture"]
    collection = db["crops"]
except Exception as e:
    st.sidebar.error(f"❌ Failed to connect to MongoDB: {str(e)}")
    st.stop()  # Stop the app if connection fails

## -----------------
## App Initialization
## -----------------

# Initialize session state for messages if it doesn't exist
st.session_state.setdefault("messages", [])

st.header("Chat with your Data")

## -----------------
## Chat History Display
## -----------------

# Display the entire chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle complex assistant messages stored as dictionaries
        if isinstance(message["content"], dict):
            if "summary" in message["content"]:
                st.markdown(message["content"]["summary"])
            if "data" in message["content"]:
                st.dataframe(pd.DataFrame(message["content"]["data"]))
            # Display the pipeline in an expander, associated with its message
            if "pipeline" in message["content"]:
                with st.expander("View Generated Pipeline"):
                    st.code(message["content"]["pipeline"], language="json")
        # Handle simple user messages
        else:
            st.markdown(message["content"])

## -----------------
## User Input & Backend Processing
## -----------------
from datetime import datetime

def flatten_doc(doc):
    def format_date(date_field):
        if isinstance(date_field, dict) and '$date' in date_field:
            return date_field['$date'].split('T')[0]
        elif isinstance(date_field, datetime):
            return date_field.strftime('%Y-%m-%d')
        return date_field

    flat = {}

    # Safe nested get
    def safe_get(dct, path, default=None):
        for key in path.split("."):
            if isinstance(dct, dict) and key in dct:
                dct = dct[key]
            else:
                return default
        return dct

    fields_to_extract = {
        "Farmer Name": "farmer.name",
        "Father Name": "farmer.father_name",
        "Code": "code",
        "Village": "location.village",
        "District": "location.district",
        "Sowing Date": "crop_info.sowing_date",
        "Expected Harvest": "crop_info.expected_harvest_date",
        "Actual Harvest": "crop_info.actual_harvest_date",
        "Harvest Status": "crop_info.harvest_status",
        "Risk Status": "risk_status",
        "Chemical Name": "chemical.name",
        "Acres": "acre",
        "Expected Qty": "expected_quantity",
        "Actual Qty": "actual_quantity",
        "Assigned To": "assigned_to",
        "Sample Status": "sample_collection",
        "Remarks": "remarks"
    }

    for label, path in fields_to_extract.items():
        val = safe_get(doc, path)
        if "Date" in label or "Harvest" in label or "Sowing" in label:
            val = format_date(val)
        flat[label] = val

    return flat

    return processed_data
if user_question := st.chat_input("Ask your question here..."):
    # Add user's question to history immediately
    st.session_state.messages.append({"role": "user", "content": user_question})

    # This dictionary will hold all parts of the assistant's response
    assistant_response_content = {}
    
    # try:
    # Generate the pipeline
    response = llm.invoke(build_pipeline_prompt(user_question))
    pipeline_str = response.content
    print(fr"Generated Pipeline: {pipeline_str}")
    # Add the pipeline to our response dictionary
    assistant_response_content["pipeline"] = pipeline_str
    
    # Execute the pipeline
    pipeline_groups = clean_and_parse_pipeline(pipeline_str)
    print(fr"Parsed Pipeline Groups: {pipeline_groups}")
    llm_outputs, table_outputs = [], []
    for group in pipeline_groups:
        result = list(collection.aggregate(group))
        print(fr"Pipeline Group Result: {result}")
        if should_send_to_llm(result):
            llm_outputs.append(result)
        else:
            table_outputs.append(result)

    # Process and store results in the response dictionary
# Update the section where you process table outputs with this code:
    if table_outputs:
        all_data = []
        for data in table_outputs:
            # Process the MongoDB documents
            processed_docs = [flatten_doc(doc) for doc in data]
            all_data.extend(processed_docs)
        
        # Create DataFrame with processed data
        df = pd.DataFrame(all_data)
        
        # Store both raw and formatted data
        assistant_response_content["data"] = all_data
        assistant_response_content["formatted_data"] = df

    if llm_outputs:
        summary = summarize_with_llm(user_question, llm_outputs)
        assistant_response_content["summary"] = summary

    # except Exception as e:
    #     error_message = f"An error occurred: {e}"
    #     assistant_response_content["summary"] = error_message

    # Add the complete assistant response (summary, data, and pipeline) to history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})

    # Rerun the app to display the new messages
    st.rerun()

# # ==============================================================================
# # Prerequisite:
# # Ensure your variables `llm` and `collection` are initialized
# # and your helper functions `build_pipeline_prompt`, `clean_and_parse_pipeline`,
# # `should_send_to_llm`, and `summarize_with_llm` are defined before this point.
# # ==============================================================================

# # Set the title for your app
# st.title("🌾 MongoDB Q&A Assistant")

# # Initialize chat history in session state if it doesn't exist
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display prior chat messages on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         # Check if the content is a dictionary (for structured assistant messages)
#         if isinstance(message["content"], dict):
#             if "summary" in message["content"]:
#                 st.markdown(message["content"]["summary"])
#             if "pipeline" in message["content"]:
#                 with st.expander("View Generated Pipeline"):
#                     st.code(message["content"]["pipeline"], language="json")
#             if "data" in message["content"]:
#                 st.dataframe(pd.DataFrame(message["content"]["data"]))
#         else:
#             # For simple user messages
#             st.markdown(message["content"])

# # Get new user input from the chat input box at the bottom
# if user_question := st.chat_input("Ask a question about your crop data:"):
#     # Add user's question to history and display it
#     st.session_state.messages.append({"role": "user", "content": user_question})
#     with st.chat_message("user"):
#         st.markdown(user_question)

#     # Generate and display the assistant's response
#     with st.chat_message("assistant"):
#         # This dictionary will hold the structured response for session state
#         assistant_response_content = {}

#         try:
#             # 1. Generate MongoDB Pipeline
#             with st.spinner("### 🔧 Generating MongoDB Pipeline..."):
#                 response = llm.invoke(build_pipeline_prompt(user_question))
#                 pipeline_str = response.content
#                 assistant_response_content["pipeline"] = pipeline_str

#             with st.expander("View Generated Pipeline", expanded=True):
#                 st.code(pipeline_str, language="json")

#             # 2. Execute Pipeline and Process Results
#             with st.spinner("### ⚙️ Executing query..."):
#                 # Use your function to parse the pipeline string
#                 pipeline_groups = clean_and_parse_pipeline(pipeline_str)

#                 llm_outputs = []
#                 table_outputs = []

#                 # Note: The original code implies one pipeline can have multiple groups.
#                 # If your `clean_and_parse_pipeline` returns a single pipeline array,
#                 # you might wrap it in a list: `[clean_and_parse_pipeline(pipeline_str)]`
#                 for idx, group in enumerate(pipeline_groups):
#                     # Use your MongoDB collection object to run the aggregation
#                     result = list(collection.aggregate(group))
#                     # Use your logic to decide the result's destination
#                     if should_send_to_llm(result):
#                         llm_outputs.append((idx, result))
#                     else:
#                         table_outputs.append((idx, result))

#             # 3. Display Final Results (Summary and/or Tables)
#             # Display tabular data first
#             if table_outputs:
#                 st.markdown("### 📊 Tabular Results")
#                 all_data = []
#                 for idx, data in table_outputs:
#                     st.markdown(f"**Query Results:**")
#                     df = pd.DataFrame(data)
#                     st.dataframe(df)
#                     # Add data to the structured response for history
#                     all_data.extend(df.to_dict('records'))
#                 assistant_response_content["data"] = all_data

#             # Display LLM summary last
#             if llm_outputs:
#                 with st.spinner("### 🧠 Generating summary..."):
#                     # Use your summarization function
#                     summary = summarize_with_llm(user_question, llm_outputs)
#                     st.success(summary)
#                     assistant_response_content["summary"] = summary

#         except Exception as e:
#             error_message = f"An error occurred: {e}"
#             st.error(error_message)
#             assistant_response_content["summary"] = error_message

#     # Add the complete, structured assistant response to the chat history
#     st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})