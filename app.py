import streamlit as st
from datetime import datetime
from css import load_css_styles, load_footer
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card
import math
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from google.oauth2 import service_account
from gsheetsdb import connect
from utilities import usage_in_app, usage_number_in_app, df_usage_in_app, get_top_apps, weekly_growth_plot, get_driver, get_screenshot, add_corners, generate_app_image, show_apps, make_donut, make_donut_chart, df_tools_usage, calculate_tools_usage, calculate_use_sum, get_weekly_cumulative_app_count, get_weekly_cumulative_developer_count, add_cumulative_column, prepare_data_for_trends_plot, add_cumulative_column_proprietary_opensource, add_cumulative_column_usage_trends, prepare_gallery_data, load_weekly_chat_app, sort_LLM_tools, calculate_weekly_app_count, prepare_llm_data, sort_opensource_tools, redirect_button

# Configuration
st.set_page_config(layout="wide", page_title="State of LLM Apps 2023", page_icon="🎈",)
load_css_styles()
alt.themes.enable("dark")


############################################################
# Navigation


#if 'most_recent_week_start' not in st.session_state:
#    st.session_state['most_recent_week_start'] = ''

with st.sidebar:
    st.title(':gray[State of LLM Apps 2023]')
    #st.write(':gray[Explore the latest trends, tools, and use cases in LLM app development from apps hosted on Streamlit Community Cloud.]')
    
    #<h3>What's in the report?</h3>
    #st.header("What's in the report?")
    st.markdown('''
        <div class="sidebar">
          <a href="#key-takeaways" class="nav">💡 &nbsp; Key takeaways</a>
          <a href="#app-developer-growth" class="nav">📈 &nbsp; App and developer growth</a>
          <a href="#llms-adoption-at-a-glance" class="nav">📊 &nbsp; LLMs adoption at-a-glance</a>
          <a href="#top-models" class="nav">📊 &nbsp; Top models</a>
          <a href="#top-orchestration-tools" class="nav">📊 &nbsp; Top orchestration tools</a>
          <a href="#top-vector-retrieval-tools" class="nav">📊 &nbsp; Top vector retrieval tools</a>
          <a href="#are-chatbots-the-future" class="nav">💬 &nbsp; Are chatbots the future?</a>
          <a href="#gallery-of-llm-apps" class="nav">✨ &nbsp; Gallery of LLM apps</a>
          <a href="#concerns-building-with-llms" class="nav">🤔 &nbsp; Concerns building with LLMs</a>
          <a href="#llm-app-architecture" class="nav">🧱 &nbsp; LLM app architecture</a>
          <a href="#about-streamlit" class="nav">🎈 &nbsp; About Streamlit</a>
          <a href="#methodology" class="nav">🛠️ &nbsp; Methodology</a>
        </div>
    ''', unsafe_allow_html=True)
    
    add_vertical_space(1)
    
#    if st.session_state['most_recent_week_start']:
#        st.caption(f'Last updated {st.session_state["most_recent_week_start"].strftime("%B %d, %Y")}')


############################################################
# Load Google Sheets data
# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)
conn = connect(credentials=credentials)

# Perform SQL query on the Google Sheet.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_resource(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

@st.cache_data
def load_data(persist="disk"):
    sheet_url = st.secrets["private_gsheets_url"]
    rows = run_query(f'SELECT * FROM "{sheet_url}"')
    df = pd.DataFrame(rows)
    df.LLM_MODEL = [x.replace('"', '') for x in df.LLM_MODEL]
    most_recent_start_week = df['WEEK_START'].max()
    return df, most_recent_start_week

df = load_data()[0]
# st.session_state['most_recent_week_start'] = load_data()[1]
most_recent_week_start = load_data()[1]
df = df[df['WEEK_START'] < most_recent_week_start]

total_developer_number = df.OWNER.nunique()
total_app_number = df.SUBDOMAIN.nunique()

with st.sidebar:
    st.caption(f'Last updated {most_recent_week_start.strftime("%B %d, %Y")}')

############################################################
# Main page

#total_developer_number_round_down = math.floor(total_developer_number / 1000) * 1000
#total_app_number_round_down = math.floor(total_app_number / 1000) * 1000

@st.cache_data
def main_page():
    st.markdown("<div id='top'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <p align="center" style="margin-top:-40px">
        <img src="https://raw.githubusercontent.com/dataprofessor/thumbnail-image/master/streamlit-snowflake-logo.svg" height="55" class="img_hover">
    </p>
    <p style="margin-top:30px">
        <h1 style="text-align: center; font-size:110px; font-family: 'Open Sans', sans-serif; letter-spacing: -0.02em"; font-weight: 700;>
            <div class="row">
                <div style="display: inline-block;">
                    State of
                </div>
                <div style="display: inline-block;">
                    <span style="color:black; background-color: #29b5e8; margin-left:10px; margin-right:25px;">
                        &nbsp;LLM
                    </span>
                    <img src="https://raw.githubusercontent.com/dataprofessor/thumbnail-image/master/Sno-Blue-Arrow.svg" height="90" align="center" style="margin-right:-70px; margin-top:-20px">
                </div>
            </div>
            <div>
                &nbsp;Apps 2023
            </div>
        </h1>
    </p>
    <div class="frontpage">
    <p align="center" style="font-family: 'Open Sans', sans-serif; font-size:28px; line-height:1.25; margin-top:38px;">
        Explore emerging trends, tools, and use cases in LLM app development from 
        <span style="color:#FFBD45">
            <b>{total_app_number:,}</b>
        </span> 
        apps built by 
        <span style="color:#FFBD45">
            <b>{total_developer_number:,}</b>
        </span> developers hosted on Streamlit Community Cloud.
    </p>
    </div>
    """, unsafe_allow_html=True)

main_page()

load_footer()


# Page 2
st.markdown("""
<div class="container">
  <a data-scroll href="#first">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)



############################################################
# Data

#@st.cache_data
#def load_data_llm():
#    df_llm = df.copy()
#    df_llm = df_llm.groupby(['LLM_MODEL', 'WEEK_START']).sum().reset_index()
#    df_llm.drop(['WEEK_OVER_WEEK_USER_GROWTH'], axis=1)
#    df_llm['WEEK_OVER_WEEK_APP_GROWTH'] = (df_llm['WEEKLY_APP_COUNT'].pct_change()*100)
#    return df_llm
# df_llm = load_data_llm()

#@st.cache_data
#def load_data_top_apps():
#    top_apps_sheet_url = st.secrets["top_apps_gsheets_url"]
#    top_apps_rows = run_query(f'SELECT * FROM "{top_apps_sheet_url}"')
#    df_top_apps = pd.DataFrame(top_apps_rows)
#    return df_top_apps
# df_top_apps = load_data_top_apps()

@st.cache_data
def load_data_daily(persist="disk"):
    daily_sheet_url = st.secrets["daily_gsheets_url"]
    daily_rows = run_query(f'SELECT * FROM "{daily_sheet_url}"')
    df_daily = pd.DataFrame(daily_rows)
    return df_daily
df_daily = load_data_daily()

df_tool_popularity = df_usage_in_app(df)

# Load chat data
#@st.cache_data
#def load_chat_elements():
#    df2 = df[['LLM_MODEL', 'SUBDOMAIN', 'USES_CHAT_INPUT', 'USES_CHAT_MESSAGE', 'USES_STREAMLIT_CHAT', 'USES_TEXT_INPUT', 'USES_TEXT_AREA']]
#    df3 = df2.groupby('SUBDOMAIN').first().drop('LLM_MODEL', axis=1)
#    df4 = df3.fillna(value=0)
#    df_asint = df4.astype(int)
#    df_asint = df_asint.T.dot(df_asint)
#    df_asint.index = ['st.chat_input', 'st.chat_message', 'streamlit_chat', 'st.text_input', 'st.text_area']
#    df_asint.columns = ['st.chat_input', 'st.chat_message', 'streamlit_chat', 'st.text_input', 'st.text_area']
#    return df_asint

#df_chat_elements = load_chat_elements()

# st_chat_input_and_chat_message = df[ (df['USES_CHAT_INPUT']==1) & (df['USES_CHAT_MESSAGE']==1) ].SUBDOMAIN.nunique()

st_text_input = df[df['USES_TEXT_INPUT']==1].SUBDOMAIN.nunique()
st_text_input_pct = int((st_text_input/total_app_number)*100)


# Prepare gallery data

# Add chat elements to dataframe
columns_to_combine = ['USES_CHAT_INPUT', 'USES_CHAT_MESSAGE', 'USES_STREAMLIT_CHAT', 'USES_TEXT_INPUT', 'USES_TEXT_AREA']
st_chat_elements = ['st.chat_input', 'st.chat_message', 'streamlit_chat', 'st.text_input', 'st.text_area']
def combine_columns(row):
    combined_values = [element for element in st_chat_elements if row[columns_to_combine[st_chat_elements.index(element)]] == 1]
    return combined_values

# Function to classify rows
def classify_app_type(row):
    if 'st.chat_input' in row or 'st.chat_message' in row or 'streamlit_chat' in row:
        return 'chat'
    else:
        return 'single text input' 
        # non-chat

@st.cache_data
def prepare_top_apps_data(input_df, input_df_daily):
    top_apps = prepare_gallery_data(df, df_daily)
    top_apps['GH_URL'] = "https://github.com/" + top_apps['OWNER'] + "/" + top_apps['REPOSITORY']
    top_apps['APP_URL'] = [f'https://{x}.streamlit.app' for x in top_apps['SUBDOMAIN']]
    top_apps.index = np.arange(1, len(top_apps) + 1)
    
    # Apply the function to create the new combined column
    top_apps['ST_CHAT_ELEMENTS'] = top_apps.apply(combine_columns, axis=1)
    
    # Apply the function to create the "APP_TYPE" column
    top_apps['APP_TYPE'] = top_apps['ST_CHAT_ELEMENTS'].apply(classify_app_type)
    return top_apps

top_apps = prepare_top_apps_data(df, df_daily)

#####
# Prepare gallery data
#top_apps = prepare_gallery_data(df, df_daily)
#top_apps['GH_URL'] = "https://github.com/" + top_apps['OWNER'] + "/" + top_apps['REPOSITORY']
#top_apps['APP_URL'] = [f'https://{x}.streamlit.app' for x in top_apps['SUBDOMAIN']]
#top_apps.index = np.arange(1, len(top_apps) + 1)
    
# Apply the function to create the new combined column
#top_apps['ST_CHAT_ELEMENTS'] = top_apps.apply(combine_columns, axis=1)
    
# Apply the function to create the "APP_TYPE" column
#top_apps['APP_TYPE'] = top_apps['ST_CHAT_ELEMENTS'].apply(classify_app_type)
#####


single_text_input_count = len(top_apps[top_apps.APP_TYPE == 'single text input'])
st_chat_input_and_chat_message = top_apps[top_apps.APP_TYPE == 'chat'].SUBDOMAIN.nunique()
st_chat_input_and_chat_message_pct = int((st_chat_input_and_chat_message/total_app_number)*100)

# Weekly data of chat elements
@st.cache_data
def load_weekly_chat_data():
    df['ST_CHAT_ELEMENTS'] = df.apply(combine_columns, axis=1)
    df['APP_TYPE'] = df['ST_CHAT_ELEMENTS'].apply(classify_app_type)
    return df

df_weekly_chat_data = load_weekly_chat_data()
df_weekly_chat_app = load_weekly_chat_app(df_weekly_chat_data)


############################################################

st.markdown("<div id='first'></div>", unsafe_allow_html=True)

colored_header(
    label="Key takeaways",
    description="",
    color_name="light-blue-70",
)

openai_app_number = usage_number_in_app(df, 'openai')
openai_pct = int(round((openai_app_number/total_app_number)*100,0))

langchain_app_number = usage_number_in_app(df, 'langchain')
langchain_pct = round((langchain_app_number/total_app_number)*100,1)

llama_index_app_number = usage_number_in_app(df, 'llama_index')
llama_index_pct = round((llama_index_app_number/total_app_number)*100,1)

cohere_app_number = usage_number_in_app(df, 'cohere')
cohere_pct = round((cohere_app_number/total_app_number)*100,1)

anthropic_app_number = usage_number_in_app(df, 'anthropic')
anthropic_pct = round((anthropic_app_number/total_app_number)*100,1)

transformers_app_number = usage_number_in_app(df, 'transformers')
transformers_pct = round((transformers_app_number/total_app_number)*100,1)

huggingface_hub_app_number = usage_number_in_app(df, 'huggingface_hub')
huggingface_hub_pct = round((huggingface_hub_app_number/total_app_number)*100,1)

llama_cpp_app_number = usage_number_in_app(df, 'llama_cpp')
llama_cpp_pct = round((llama_cpp_app_number/total_app_number)*100,1)

pinecone_app_number = usage_number_in_app(df, 'pinecone')
pinecone_pct = round((pinecone_app_number/total_app_number)*100,1)

faiss_app_number = usage_number_in_app(df, 'faiss')
faiss_pct = round((faiss_app_number/total_app_number)*100,1)

chromadb_app_number = usage_number_in_app(df, 'chromadb')
chromadb_pct = round((chromadb_app_number/total_app_number)*100,1)

qdrant_client_app_number = usage_number_in_app(df, 'qdrant_client')
qdrant_client_pct = round((qdrant_client_app_number/total_app_number)*100,1)

weaviate_app_number = usage_number_in_app(df, 'weaviate')
weaviate_pct = round((weaviate_app_number/total_app_number)*100,1)

elasticsearch_app_number = usage_number_in_app(df, 'elasticsearch')
elasticsearch_pct = round((elasticsearch_app_number/total_app_number)*100,1)

pgvector_app_number = usage_number_in_app(df, 'pgvector')
pgvector_pct = round((pgvector_app_number/total_app_number)*100,1)

vector_opensource_pct = round(( (chromadb_app_number + weaviate_app_number + faiss_app_number + pgvector_app_number + qdrant_client_app_number) /total_app_number)*100,1)


st.write(f'As of :orange[**{most_recent_week_start.strftime("%B %d, %Y")}**], an analysis of :orange[**{total_app_number:,}**] LLM-powered Streamlit apps revealed the following insights:')

df_tools_usage = df_tools_usage(df_daily)
pct_llm_models, pct_vector_retrieval, pct_llm_orchestration = calculate_tools_usage(df_tools_usage)
pct_use_1, pct_use_2, pct_use_3 = calculate_use_sum(df_tools_usage)

takeaway_col = st.columns(4, gap="medium")

# Apps using 1 LLM category accounted for more than half of apps at :orange[**{pct_use_1}%**]
# while those using 2 and 3 LLM categories afforded percentage values of :orange[**{pct_use_2}%**] and :orange[**{pct_use_3}%**], respectively.

with takeaway_col[0]:
    st.markdown(f"""
        <h3>
          <b>
            <span style="color:#FFBD45">
                OpenAI
            </span> is
            <br> dominant
          </b>
        </h3>
        
        <h4>
          <b>
            <span style="color:#FFBD45">
                {openai_pct}%
            </span>
          </b> use GPT <br> models
        </h4>

        OpenAI has become the standard for LLM apps due to its pioneering GPT research, high-quality outputs, steerability, and accessible API. Their first-mover debut of ChatGPT and large transformer-based models sparked the imagination of developers, and the world, at large.
    """, unsafe_allow_html=True)

    redirect_button('#top-models', 'Go to Top models')
    
    #st.altair_chart(make_donut(pct_llm_models, 'LLM Models', 'blue'), use_container_width=True)

    

with takeaway_col[1]:
# langchain_app_number, langchain_pct
    st.markdown(f"""
        <h3>
          <b>
            The future is <br>
            <span style="color:#FFBD45">
                multi-agent
            </span>
          </b>
        </h3>
        
        <h4>
          <b>
            <span style="color:#FFBD45">
                {pct_llm_orchestration}%
            </span>
          </b> use <br> orchestration
        </h4>

        LangChain and LlamaIndex are orchestration frameworks with agents and tools designed to augment LLM capabilities. Agents can be combined to manage and optimize LLM functions, such as refining AI reasoning, addressing biases, and integrating external data sources.
    """, unsafe_allow_html=True)
    #redirect_button('Go to Top models', '#top-models')    
    #st.altair_chart(make_donut(pct_llm_orchestration, 'LLM Orchestration', 'green'), use_container_width=True)

    
with takeaway_col[2]:
    st.markdown(f"""
        <h3>
          <b> Most apps
            <span style="color:#FFBD45">bypass</span> vector magic
          </b>
        </h3>
        
        <h4>
          <b>
            Only
            <span style="color:#FFBD45">
                {pct_vector_retrieval}%
            </span> use
            <br> vector retrieval
          </b>
        </h4>

        Apps with **vector databases** and **vector search** are used to enable fast, contextual search by categorizing **large, unstructured datasets** (including text, images, video, or audio). :orange[**80%**] of apps rely on the LLM's built-in knowledge, suggesting this may suffice for the majority of use cases.
    """, unsafe_allow_html=True)
    

   #st.altair_chart(make_donut(pct_vector_retrieval, 'Vector Retrieval', 'orange'), use_container_width=True)


with takeaway_col[3]:
    st.markdown(f"""
        <h3>
          <b>
            Chatbots are <br>
            on the
            <span style="color:#FFBD45">
                rise
            </span>
          </b>
        </h3>
        
        <h4>
          <b>
            <span style="color:#FFBD45">
                {st_chat_input_and_chat_message_pct}%
            </span>
          </b> (and growing) 
          <br> are chatbots
        </h4>

        Chatbots let users iteratively refine answers, leaving room for fluid, human-like conversations with the LLM. Conversely, :orange[**{100-st_chat_input_and_chat_message_pct}**%] of LLM apps use text inputs with a single objective, generally not allowing for conversational refinement.
    """, unsafe_allow_html=True)
    

add_vertical_space(2)
 
st.markdown("""
<div class="container">
  <a data-scroll href="#second">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)


############################################################

st.markdown("<div id='second'></div>", unsafe_allow_html=True)

colored_header(
    label="App & developer growth",
    description="",
    color_name="light-blue-70",
)

line1, line2 = st.columns((2,1), gap="large")

# Load data
@st.cache_data
def load_weekly_growth_data(input_df):
    df_weekly_app_count = get_weekly_cumulative_app_count(input_df)
    df_weekly_developer_count = get_weekly_cumulative_developer_count(input_df)
    # df_weekly_growth['APP_DEV_RATIO'] = df_weekly_app_count['WEEKLY_COUNT']/df_weekly_developer_count['WEEKLY_COUNT']
    return pd.concat([df_weekly_app_count, df_weekly_developer_count], axis=0) 
df_weekly_growth = load_weekly_growth_data(df)

# Plotting
# Create a selection that chooses the nearest point & selects based on x-value
growth_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['WEEK_START'], empty='none')
    
# The basic line
growth_line = alt.Chart(df_weekly_growth).mark_line(interpolate='linear').encode(
                x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
                # y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
                y=alt.Y("WEEKLY_COUNT:Q", axis=alt.Axis(title="Weekly Count", titlePadding=15, titleFontSize=20, titleFontWeight=900)),
                color=alt.Color('COUNT_TYPE:N',
                                legend=alt.Legend(title=" ") ))
    
# Transparent selectors across the chart. This is what tells us the x-value of the cursor
growth_selectors = alt.Chart(df_weekly_growth).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(growth_nearest)
    
# Draw points on the line, and highlight based on selection
growth_points = growth_line.mark_point().encode(opacity=alt.condition(growth_nearest, alt.value(1), alt.value(0)))
    
# Draw text labels near the points, and highlight based on selection
growth_text = growth_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(growth_nearest, 'WEEKLY_COUNT:Q', alt.value(' ')))
    
# Draw a rule at the location of the selection
growth_rules = alt.Chart(df_weekly_growth).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(growth_nearest)
    
# Put the five layers into a chart and bind the data
growth_count = alt.layer(growth_line, growth_selectors, growth_points, growth_rules, growth_text
    ).properties(width=800, height=500
    ).configure(padding=20, background="#111111"
    ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
    ).configure_axis(labelFontSize=14)



with line1:
    st.altair_chart(growth_count, use_container_width=True)

            
with line2:
    # add_dev_ratio = round(df_weekly_growth.APP_DEV_RATIO.mean(), 1)
    add_dev_ratio = round((total_app_number/total_developer_number), 1)
    st.write(f'''
        ### :orange[**{total_developer_number:,}**] unique developers built :orange[**{total_app_number:,}**] total apps
        On average, one developer is creating :orange[**{add_dev_ratio}**] apps.

        LLMs power a wide variety of natural language processing tasks, allowing you to 'talk to' your data. Visit the app gallery below to see the thousands of use cases, including: 
        - Content generation
        - Language translation
        - Chatbots and virtual assistants
        - Data analysis and insights
        - Content summarization
        ''') 


st.markdown("""
<div class="container">
  <a data-scroll href="#third">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)

############################################################
# LLM adoption at a Glance
st.markdown("<div id='third'></div>", unsafe_allow_html=True)

colored_header(
    label="LLMs adoption at-a-glance",
    description="",
    color_name="light-blue-70",
)

llm_models = ['openai', 'anthropic', 'cohere', 'huggingface_hub', 'transformers', 'llama_cpp']   # 'pyllamacpp', 'diffusers'
vector_databases = ['chromadb', 'pinecone', 'weaviate', 'elasticsearch', 'faiss', 'pgvector', 'qdrant_client']
llm_orchestration = ['langchain', 'llama_index']

llm_tab = st.tabs(['% Overall adoption', 'Weekly adoption'])

with llm_tab[0]:

    line1, line2 = st.columns((2,1), gap="large")

    with line1:
        st.dataframe(
            df_tool_popularity,
            column_config={
                 "TOOL": st.column_config.TextColumn("LLM Tool", help="LLM Tool"),
        
                 "PCT_USAGE": st.column_config.ProgressColumn(
                    "Percent Usage in LLM Apps",
                    help="Percent Usage in LLM Apps",
                    format="%f%%",
                    min_value=0,
                    max_value=100,
                 ),
                
                 "LLM_CATEGORY": st.column_config.TextColumn(
                    "LLM Category", 
                    help="LLM Category",
                 ),
             },
             column_order=("TOOL", "PCT_USAGE", "LLM_CATEGORY"),
             height=380,
             use_container_width=True,
             hide_index=True,
        )
        
    with line2:
        # st.subheader(f':orange[**{openai_pct}%**] use OpenAI')
        st.subheader('Out of all LLM tech, OpenAI (GPT) is the [**most%**] used')
        st.write('''
        Generally speaking, there are 4 categories of an [LLM app’s architecture](#llm-app-architecture):
        - LLM model
        - Orchestration
        - Vector retrieval
        - Visual UI
        ''')

with llm_tab[1]:

    llm_widget_1, llm_widget_2, llm_widget_3 = st.columns((2,1,2))
    line1, line2 = st.columns((2,1), gap="large")

    with line2:
        st.subheader(':orange[**Compare**] trends of LLM tools')
        st.write('''
        Use the search fields above to compare growth rates of apps using various LLM technologies.
        ''')    
        
    with line1:
        #tools_type_options = st.selectbox('Tools type', ('All', 'LLM Models', 'Vector Databases'))
        
        #llm_models_list = ['openai', 'anthropic', 'cohere', 'huggingface_hub', 'transformers', 'llama_cpp', 'diffusers']
        #vector_databases_list = ['chromadb', 'pinecone', 'weaviate', 'elasticsearch', 'faiss', 'pgvector', 'qdrant_client']
        #llm_orchestration_list = ['langchain', 'llama_index']

        llm_category_options = llm_widget_1.multiselect('Select LLM models', llm_models, llm_models, key='key_llm_category_options')
        orchestration_category_options = llm_widget_2.multiselect('Select LLM orchestration', llm_orchestration, llm_orchestration, key='key_orchestration_category_options')
        vector_category_options = llm_widget_3.multiselect('Select vector retrieval', vector_databases, vector_databases, key='key_vector_category_options')
        
    # Load and prepare data
    #df_llm['LLM_CATEGORY'] = 'NA'
    #df_llm.loc[df_llm['LLM_MODEL'].isin(llm_category_options), 'LLM_CATEGORY'] = 'llm_models'
    #df_llm.loc[df_llm['LLM_MODEL'].isin(vector_category_options), 'LLM_CATEGORY'] = 'vector_databases'
    #df_llm.loc[df_llm['LLM_MODEL'].isin(orchestration_category_options), 'LLM_CATEGORY'] = 'llm_orchestration'
    #df_llm_category = df_llm[df_llm.LLM_CATEGORY != 'NA']

    #df_llm_category_agg = df_llm_category.groupby(['LLM_CATEGORY', 'WEEK_START']).sum()
    #df_llm_category_type = df_llm_category_agg.add_suffix('').reset_index()

    #cumulative_list = []
    #for x in df_llm_category_type['WEEK_START']:
    #  app_count = df_llm_category_type[df_llm_category_type.WEEK_START.isin(df_llm_category_type['WEEK_START'].unique()[:df_llm_category_type['WEEK_START'].unique().tolist().index(x) + 1])]['SUBDOMAIN'].nunique()
    #  cumulative_list.append(app_count)
    # df_llm_category_type['CUMULATIVE_WEEKLY_APP_COUNT'] = cumulative_list

    
    # Load data
    df_llm_category_type = add_cumulative_column(df, 'SUBDOMAIN', llm_category_options, vector_category_options, orchestration_category_options)

    
    # Plotting
    # Create a selection that chooses the nearest point & selects based on x-value
    llm_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                                fields=['WEEK_START'], empty='none')
        
    # The basic line
    llm_line = alt.Chart(df_llm_category_type).mark_line(interpolate='linear').encode(
                    x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
                    # y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
                    y=alt.Y("WEEKLY_APP_COUNT:Q", axis=alt.Axis(title="Weekly App Count", titlePadding=15, titleFontSize=20, titleFontWeight=900)),
                    color=alt.Color('LLM_CATEGORY:N',
                                     legend=alt.Legend(title=" ") ))
        
    # Transparent selectors across the chart. This is what tells us the x-value of the cursor
    llm_selectors = alt.Chart(df_llm_category_type).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(llm_nearest)
        
    # Draw points on the line, and highlight based on selection
    llm_points = llm_line.mark_point().encode(opacity=alt.condition(llm_nearest, alt.value(1), alt.value(0)))
        
    # Draw text labels near the points, and highlight based on selection
    #llm_text = llm_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(llm_nearest, f'{y_var}:Q', alt.value(' ')))
    llm_text = llm_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(llm_nearest, 'WEEKLY_APP_COUNT:Q', alt.value(' ')))
        
    # Draw a rule at the location of the selection
    llm_rules = alt.Chart(df_llm_category_type).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(llm_nearest)
        
    # Put the five layers into a chart and bind the data
    llm_app_count = alt.layer(llm_line, llm_selectors, llm_points, llm_rules, llm_text
        ).properties(width=800, height=500
        ).configure(padding=20, background="#111111"
        ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
        ).configure_axis(labelFontSize=14)

    with line1:
        st.altair_chart(llm_app_count, use_container_width=True)

   
st.markdown("""
<div class="container">
  <a data-scroll href="#fourth">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)



############################################################

# LLM models
st.markdown("<div id='fourth'></div>", unsafe_allow_html=True)

colored_header(
    label="Top models",
    description="",
    color_name="light-blue-70",
)

tab1, tab2 = st.tabs(['All models', 'Proprietary vs. Open source'])

with tab1:

    line_widget_1, line_widget_2 = st.columns((5,1))
    line1, line2 = st.columns((2,1), gap="large")
    
    # Load and prepare data
    # llm_models = ['openai', 'anthropic', 'huggingface_hub', 'cohere', 'llama_cpp', 'pyllamacpp', 'diffusers', 'transformers']
    # proprietary_models = ['openai', 'anthropic', 'cohere']
    # opensource_models = ['huggingface_hub', 'llama_cpp', 'pyllamacpp', 'diffusers', 'transformers']
    
    llm_models_options = line_widget_1.multiselect('Select LLM models', llm_models, llm_models)
    numbers_percent_options = line_widget_2.selectbox('Growth units', ('Count', '% Growth'))

    # df_llm_models = prepare_data_for_trends_plot(df_llm, llm_models_options)
    # df_llm_models = add_cumulative_column_usage_trends(df, 'SUBDOMAIN', llm_models_options)

    
    if numbers_percent_options=='Count':
       y_var = "WEEKLY_APP_COUNT"
       y_title = "Weekly App Count"
       
    if numbers_percent_options=='% Growth':
       y_var = "WEEKLY_PCT"
       y_title = "Relative Weekly App Growth (%)"


    df_llm = df[df.LLM_MODEL.isin(llm_models_options)]
    df_llm_models =  calculate_weekly_app_count(df_llm)
    
    df_llm_models_sort_list = sort_LLM_tools(df_llm_models, y_var)

    
    ## df_llm.LLM_MODEL = [x.replace('"', '') for x in df_llm.LLM_MODEL]
    ## df_llm_models2 = df_llm[df_llm.LLM_MODEL.isin(llm_models_options)]
    
    # df_llm_models = prepare_data_for_trends_plot(df_llm, llm_models_options)
    
        
    # Plotting
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['WEEK_START'], empty='none')
    
    # The basic line
    line = alt.Chart(df_llm_models).mark_line(interpolate='linear').encode(
        x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
        y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
        color=alt.Color('LLM_MODEL:N',
                         scale=alt.Scale(domain=df_llm_models_sort_list),
                         legend=alt.Legend(title=" ") ))
    
    # Transparent selectors across the chart. This is what tells us the x-value of the cursor
    selectors = alt.Chart(df_llm_models).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(nearest)
    
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(nearest, f'{y_var}:Q', alt.value(' ')))
    
    # Draw a rule at the location of the selection
    rules = alt.Chart(df_llm_models).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(nearest)
    
    # Put the five layers into a chart and bind the data
    app_count = alt.layer(line, selectors, points, rules, text
    ).properties(width=800, height=500
    ).configure(padding=20, background="#111111"
    ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
    ).configure_axis(labelFontSize=14)

    with line1:
        st.altair_chart(app_count, use_container_width=True)
        
    with line2:
        st.markdown(f'''
            ### :orange[**{openai_pct}%**] use GPT models  
            OpenAI has become the standard for LLM apps due to its pioneering GPT research, high-quality outputs, steerability, and accessible API. Their first-mover debut of ChatGPT and large transformer-based models sparked the imagination of developers, and the world, at large.
            ''')

with tab2:

    line_widget_1, line_widget_2, line_widget_3  = st.columns((2,4,1))
    line1, line2 = st.columns((2,1), gap="large")
    
    # Load and prepare data
    
    #llm_models_options = line_widget_1.multiselect('Select LLM models', llm_models, llm_models, key='tab2_llm_models_options')

    # Prepare proprietary vs open source data
    proprietary_models = ['openai', 'anthropic', 'cohere']
    opensource_models = ['huggingface_hub', 'llama_cpp', 'pyllamacpp', 'diffusers', 'transformers']
    
    proprietary_models_options = line_widget_1.multiselect('Select proprietary models', proprietary_models, proprietary_models)
    opensource_models_options = line_widget_2.multiselect('Select open source models', opensource_models, opensource_models)
    numbers_percent_options = line_widget_3.selectbox('Growth units', ('Count', '% Growth'), key='tab2_numbers_percent_options')

    all_llm_models = proprietary_models_options + opensource_models_options
    
    # Load data
    df_llm_model_type = add_cumulative_column_proprietary_opensource(df, 'SUBDOMAIN', proprietary_models_options, opensource_models_options, 'models')

    df_llm_model_type['TOTAL_WEEKLY_APP_COUNT'] = df_llm_model_type.groupby('WEEK_START').sum().reset_index()['WEEKLY_APP_COUNT']
    df_llm_model_type['WEEKLY_PCT'] = (df_llm_model_type['WEEKLY_APP_COUNT']/df_llm_model_type['TOTAL_WEEKLY_APP_COUNT']) * 100

    
    if numbers_percent_options=='Count':
       y_var = "WEEKLY_APP_COUNT"
       y_title = "Weekly App Count"
    if numbers_percent_options=='% Growth':
       y_var = "WEEKLY_PCT"
       y_title = "Relative Weekly App Growth (%)"


    df_llm_model_type_sort_list = sort_opensource_tools(df_llm_model_type, y_var)
    
       #df_llm_growth = df_llm_model_type.copy()
       #df_llm_growth = df_llm_growth.groupby(['MODEL_TYPE', 'WEEK_START']).sum().reset_index()
       ## df_llm_growth.drop(['WEEK_OVER_WEEK_USER_GROWTH'], axis=1)
       #df_llm_growth['WEEK_OVER_WEEK_APP_GROWTH'] = (df_llm_growth['WEEKLY_APP_COUNT'].pct_change()*100)
       #df_llm_model_type = df_llm_growth.copy()
    
    ## df_llm.LLM_MODEL = [x.replace('"', '') for x in df_llm.LLM_MODEL]
    #df_llm_models = df_llm[df_llm.LLM_MODEL.isin(all_llm_models)]

    ## df_llm_models['MODEL_TYPE'] = 'open source'
    #df_llm_models = df_llm_models.assign(MODEL_TYPE='open source')
    #df_llm_models.loc[df_llm_models['LLM_MODEL'].isin(proprietary_models_options), 'MODEL_TYPE'] = 'proprietary'
    
    #df_llm_models_agg = df_llm_models.groupby(['MODEL_TYPE', 'WEEK_START']).sum()
    #df_llm_model_type = df_llm_models_agg.add_suffix('').reset_index()


    # df_llm_model_type = add_cumulative_column_proprietary_opensource(df, 'SUBDOMAIN', proprietary_models, opensource_models)
    
    # Plotting
    
    # Create a selection that chooses the nearest point & selects based on x-value
    llm_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['WEEK_START'], empty='none')
    
    # The basic line
    llm_line = alt.Chart(df_llm_model_type).mark_line(interpolate='linear').encode(
        x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
        y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
        color=alt.Color('MODEL_TYPE:N',
                         scale=alt.Scale(domain=df_llm_model_type_sort_list),
                         legend=alt.Legend(title=" ") ))
    
    # Transparent selectors across the chart. This is what tells us the x-value of the cursor
    llm_selectors = alt.Chart(df_llm_model_type).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(llm_nearest)
    
    # Draw points on the line, and highlight based on selection
    llm_points = llm_line.mark_point().encode(opacity=alt.condition(llm_nearest, alt.value(1), alt.value(0)))
    
    # Draw text labels near the points, and highlight based on selection
    llm_text = llm_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(llm_nearest, f'{y_var}:Q', alt.value(' ')))
    
    # Draw a rule at the location of the selection
    llm_rules = alt.Chart(df_llm_model_type).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(llm_nearest)
    
    # Put the five layers into a chart and bind the data
    llm_app_count = alt.layer(llm_line, llm_selectors, llm_points, llm_rules, llm_text
    ).properties(width=800, height=500
    ).configure(padding=20, background="#111111"
    ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
    ).configure_axis(labelFontSize=14)

    
    with line1:
        st.altair_chart(llm_app_count, use_container_width=True)    

    with line2:
        st.markdown(f'''
            ### :orange[**{int(openai_pct + cohere_pct + anthropic_pct)}%**] use proprietary LLMs
            - **Proprietary LLMs** are developed and owned by an organization (_e.g._ OpenAI, Anthropic, Cohere), where source code, training data, weights, or other model details are not usually publicly disclosed.
            - **Open Source LLMs** require self-hosting or inferencing through a hosting provider (_e.g._ Llama models via HuggingFace), where source code, training data, weights, and other model details are more readily available.
            ''')


st.markdown("""
<div class="container">
  <a data-scroll href="#fifth">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)

############################################################

st.markdown("<div id='fifth'></div>", unsafe_allow_html=True)

colored_header(
    label="Top orchestration tools",
    description="",
    color_name="light-blue-70",
)

orchestration_widget_1, orchestration_widget_2 = st.columns((5,1))
line1, line2 = st.columns((2,1), gap="large")
    
# Load and prepare data
orchestration = ['langchain', 'llama_index']
    
orchestration_options = orchestration_widget_1.multiselect('Select LLM orchestration', orchestration, orchestration, key='key_orchestration_options')
orchestration_numbers_percent_options = orchestration_widget_2.selectbox('Growth units', ('Count', '% Growth'), key='key_orchestration_percent_options')

#df_orchestration = add_cumulative_column_usage_trends(df, 'SUBDOMAIN', orchestration_options)
#df_orchestration_sort_list = sort_LLM_tools(df_orchestration)


if orchestration_numbers_percent_options=='Count':
    y_var = "WEEKLY_APP_COUNT"
    y_title = "Weekly App Count"
if orchestration_numbers_percent_options=='% Growth':
    y_var = "WEEKLY_PCT"
    y_title = "Relative Weekly App Growth (%)"

df_orchestration = df[df.LLM_MODEL.isin(orchestration_options)]
df_orchestration_tools =  calculate_weekly_app_count(df_orchestration)
df_orchestration_tools_sort_list = sort_LLM_tools(df_orchestration_tools, y_var)


# df_llm.LLM_MODEL = [x.replace('"', '') for x in df_llm.LLM_MODEL]
# df_orchestration = df_llm[df_llm.LLM_MODEL.isin(orchestration_options)]

#df_orchestration = prepare_data_for_trends_plot(df_llm, orchestration_options)


# Plotting
# Create a selection that chooses the nearest point & selects based on x-value
orchestration_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['WEEK_START'], empty='none')
    
# The basic line
orchestration_line = alt.Chart(df_orchestration_tools).mark_line(interpolate='linear').encode(
                x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
                y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
                color=alt.Color('LLM_MODEL:N',
                                 scale=alt.Scale(domain=df_orchestration_tools_sort_list),
                                 legend=alt.Legend(title=" ") ))
    
# Transparent selectors across the chart. This is what tells us the x-value of the cursor
orchestration_selectors = alt.Chart(df_orchestration_tools).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(orchestration_nearest)
    
# Draw points on the line, and highlight based on selection
orchestration_points = orchestration_line.mark_point().encode(opacity=alt.condition(orchestration_nearest, alt.value(1), alt.value(0)))
    
# Draw text labels near the points, and highlight based on selection
orchestration_text = orchestration_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(orchestration_nearest, f'{y_var}:Q', alt.value(' ')))
    
# Draw a rule at the location of the selection
orchestration_rules = alt.Chart(df_orchestration_tools).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(orchestration_nearest)
    
# Put the five layers into a chart and bind the data
orchestration_app_count = alt.layer(orchestration_line, orchestration_selectors, orchestration_points, orchestration_rules, orchestration_text
    ).properties(width=800, height=500
    ).configure(padding=20, background="#111111"
    ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
    ).configure_axis(labelFontSize=14)

with line1:
    st.altair_chart(orchestration_app_count, use_container_width=True)

with line2:
    st.markdown(f'''
        ### :orange[**{pct_llm_orchestration}%**] apps use orchestration
        LangChain and LlamaIndex are orchestration frameworks with agents and tools designed to amplify LLM capabilities. 

        However, these :orange[**tools are not mutually exclusive**] and are often used together. These agents are tools to help manage and optimize LLM functions, such as refining AI reasoning, addressing biases, and integrating external data sources.

        In an effort to minimize hallucination and build trust in generated responses, LLM orchestration frameworks facilitate retrieval augmented generation, as well as the ability to test and evaluate LLM models.
        ''')


st.markdown("""
<div class="container">
  <a data-scroll href="#sixth">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)



############################################################

st.markdown("<div id='sixth'></div>", unsafe_allow_html=True)

colored_header(
    label="Top vector retrieval tools",
    description="",
    color_name="light-blue-70",
)

vector_tab1, vector_tab2, vector_tab3, vector_tab4 = st.tabs(['All Vector retrieval tools', 'Vector databases', 'Vector search', 'Proprietary vs. Open source'])

with vector_tab1:

    vector_widget_1, vector_widget_2 = st.columns((5,1))
    line1, line2 = st.columns((2,1), gap="large")
    
    # Load and prepare data
    all_vector_tools = ['pinecone', 'chromadb', 'weaviate', 'elasticsearch', 'faiss', 'pgvector', 'qdrant_client']
    
    vector_tool_options = vector_widget_1.multiselect('Select vector database', all_vector_tools, all_vector_tools, key='vector_tab0_vector_databases_options')
    vector_tool_numbers_percent_options = vector_widget_2.selectbox('Growth units', ('Count', '% Growth'), key='vector_tab0_vector_numbers_percent_options')

    #df_vector_tools = add_cumulative_column_usage_trends(df, 'SUBDOMAIN', vector_tool_options)


    
    if vector_tool_numbers_percent_options=='Count':
       y_var = "WEEKLY_APP_COUNT"
       y_title = "Weekly App Count"
    if vector_tool_numbers_percent_options=='% Growth':
       y_var = "WEEKLY_PCT"
       y_title = "Relative Weekly App Growth (%)"
        
    df_vector = df[df.LLM_MODEL.isin(vector_tool_options)]
    df_vector_tools =  calculate_weekly_app_count(df_vector)
    df_vector_tools_sort_list = sort_LLM_tools(df_vector_tools, y_var)
    
    ## df_llm.LLM_MODEL = [x.replace('"', '') for x in df_llm.LLM_MODEL]
    # df_vector_databases = df_llm[df_llm.LLM_MODEL.isin(vector_tool_options)]

    # df_vector_databases = prepare_data_for_trends_plot(df_llm, vector_tool_options)
    
    
    # Plotting
    # Create a selection that chooses the nearest point & selects based on x-value
    vector0_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['WEEK_START'], empty='none')
    
    # The basic line
    vector0_line = alt.Chart(df_vector_tools).mark_line(interpolate='linear').encode(
        x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
        y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
        color=alt.Color('LLM_MODEL:N',
                         scale=alt.Scale(domain=df_vector_tools_sort_list),
                         legend=alt.Legend(title=" ") ))
    
    # Transparent selectors across the chart. This is what tells us the x-value of the cursor
    vector0_selectors = alt.Chart(df_vector_tools).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(vector0_nearest)
    
    # Draw points on the line, and highlight based on selection
    vector0_points = vector0_line.mark_point().encode(opacity=alt.condition(vector0_nearest, alt.value(1), alt.value(0)))
    
    # Draw text labels near the points, and highlight based on selection
    vector0_text = vector0_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(vector0_nearest, f'{y_var}:Q', alt.value(' ')))
    
    # Draw a rule at the location of the selection
    vector0_rules = alt.Chart(df_vector_tools).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(vector0_nearest)
    
    # Put the five layers into a chart and bind the data
    vector0_app_count = alt.layer(vector0_line, vector0_selectors, vector0_points, vector0_rules, vector0_text
    ).properties(width=800, height=500
    ).configure(padding=20, background="#111111"
    ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
    ).configure_axis(labelFontSize=14)

    with line1:
        st.altair_chart(vector0_app_count, use_container_width=True)

    with line2:
        st.markdown(f"""
            ### :orange[**{pct_vector_retrieval}%**] of all apps use vector magic
            
            Apps with **vector databases** and **vector search** are used to enable fast, contextual search by categorizing large, unstructured datasets (including text, images, video, or audio). 
            
            While this feature isn't widespread across apps yet, it's incredibly effective for enhancing search functions and making stronger contextual, and faster recommendations.

            If an app requires additional information beyond its built-in capacity, it can fetch the necessary details from a large database using vectors. This process is called **Retrieval Augmented Generation** (RAG).
            """)


with vector_tab2:

    vector_widget_1, vector_widget_2 = st.columns((5,1))
    line1, line2 = st.columns((2,1), gap="large")
    
    # Load and prepare data
    # vector_databases = ['pinecone', 'chromadb', 'weaviate', 'elasticsearch', 'faiss', 'pgvector', 'qdrant_client']
    vector_databases = ['chromadb', 'pinecone', 'qdrant_client', 'weaviate']
    
    vector_databases_options = vector_widget_1.multiselect('Select vector database', vector_databases, vector_databases, key='vector_tab1_vector_databases_options')
    vector_numbers_percent_options = vector_widget_2.selectbox('Growth units', ('Count', '% Growth'), key='vector_tab1_vector_numbers_percent_options')

    #df_vector_databases = add_cumulative_column_usage_trends(df, 'SUBDOMAIN', vector_databases_options)

    
    if vector_numbers_percent_options=='Count':
       y_var = "WEEKLY_APP_COUNT"
       y_title = "Weekly App Count"
    if vector_numbers_percent_options=='% Growth':
       y_var = "WEEKLY_PCT"
       y_title = "Relative Weekly App Growth (%)"
        
    df_vector_db = df[df.LLM_MODEL.isin(vector_databases_options)]
    df_vector_databases =  calculate_weekly_app_count(df_vector_db)
    df_vector_databases_sort_list = sort_LLM_tools(df_vector_databases, y_var)  

    
    # df_llm.LLM_MODEL = [x.replace('"', '') for x in df_llm.LLM_MODEL]
    # df_vector_databases = df_llm[df_llm.LLM_MODEL.isin(vector_databases_options)]

    # df_vector_databases = prepare_data_for_trends_plot(df_llm, vector_databases_options)
    
    
    # Plotting
    # Create a selection that chooses the nearest point & selects based on x-value
    vector1_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['WEEK_START'], empty='none')
    
    # The basic line
    vector1_line = alt.Chart(df_vector_databases).mark_line(interpolate='linear').encode(
        x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
        y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
        color=alt.Color('LLM_MODEL:N',
                         scale=alt.Scale(domain=df_vector_databases_sort_list),
                         legend=alt.Legend(title=" ") ))
    
    # Transparent selectors across the chart. This is what tells us the x-value of the cursor
    vector1_selectors = alt.Chart(df_vector_databases).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(vector1_nearest)
    
    # Draw points on the line, and highlight based on selection
    vector1_points = vector1_line.mark_point().encode(opacity=alt.condition(vector1_nearest, alt.value(1), alt.value(0)))
    
    # Draw text labels near the points, and highlight based on selection
    vector1_text = vector1_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(vector1_nearest, f'{y_var}:Q', alt.value(' ')))
    
    # Draw a rule at the location of the selection
    vector1_rules = alt.Chart(df_vector_databases).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(vector1_nearest)
    
    # Put the five layers into a chart and bind the data
    vector1_app_count = alt.layer(vector1_line, vector1_selectors, vector1_points, vector1_rules, vector1_text
    ).properties(width=800, height=500
    ).configure(padding=20, background="#111111"
    ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
    ).configure_axis(labelFontSize=14)
    
    with line1:
        st.altair_chart(vector1_app_count, use_container_width=True)

    with line2:
        st.markdown(f'''
            **Vector databases** are designed to store vectorized data, which are numerical representations of data objects (also known as vector embeddings). 
            
            They use indexing and other storing techniques to speed up the retrieval process and enable efficient searching in high-dimensional spaces.
            - :orange[**{pinecone_pct}%**] Pinecone
            - :orange[**{chromadb_pct}%**] ChromaDB
            - :orange[**{qdrant_client_pct}%**] Qdrant
            - :orange[**{weaviate_pct}%**] Weaviate
            ''')


with vector_tab3:

    vector_widget_1, vector_widget_2 = st.columns((5,1))
    line1, line2 = st.columns((2,1), gap="large")
    
    # Load and prepare data
    vector_search_engines = ['elasticsearch', 'faiss', 'pgvector', 'qdrant_client']

    vector_search_options = vector_widget_1.multiselect('Select vector search', vector_search_engines, vector_search_engines, key='vector_tab2_vector_search_engines_options')
    vector_search_numbers_percent_options = vector_widget_2.selectbox('Growth units', ('Count', '% Growth'), key='vector_tab2_vector_search_numbers_percent_options')

    #df_vector_search_engines = add_cumulative_column_usage_trends(df, 'SUBDOMAIN', vector_search_options)

    
    if vector_search_numbers_percent_options=='Count':
       y_var = "WEEKLY_APP_COUNT"
       y_title = "Weekly App Count"
    if vector_search_numbers_percent_options=='% Growth':
       y_var = "WEEKLY_PCT"
       y_title = "Relative Weekly App Growth (%)"

    df_vector_search = df[df.LLM_MODEL.isin(vector_search_options)]
    df_vector_search_engines =  calculate_weekly_app_count(df_vector_search)
    df_vector_search_engines_sort_list = sort_LLM_tools(df_vector_search_engines, y_var)
    
    
    # df_llm.LLM_MODEL = [x.replace('"', '') for x in df_llm.LLM_MODEL]
    # df_vector_search_engines = df_llm[df_llm.LLM_MODEL.isin(vector_search_options)]

    # df_vector_search_engines = prepare_data_for_trends_plot(df_llm, vector_search_options)
    
    
    # Plotting
    
    # Create a selection that chooses the nearest point & selects based on x-value
    vector3_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['WEEK_START'], empty='none')
    
    # The basic line
    vector3_line = alt.Chart(df_vector_search_engines).mark_line(interpolate='linear').encode(
        x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
        y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
        color=alt.Color('LLM_MODEL:N',
                         scale=alt.Scale(domain=df_vector_search_engines_sort_list),
                         legend=alt.Legend(title=" ") ))
    
    # Transparent selectors across the chart. This is what tells us the x-value of the cursor
    vector3_selectors = alt.Chart(df_vector_search_engines).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(vector3_nearest)
    
    # Draw points on the line, and highlight based on selection
    vector3_points = vector3_line.mark_point().encode(opacity=alt.condition(vector3_nearest, alt.value(1), alt.value(0)))
    
    # Draw text labels near the points, and highlight based on selection
    vector3_text = vector3_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(vector3_nearest, f'{y_var}:Q', alt.value(' ')))
    
    # Draw a rule at the location of the selection
    vector3_rules = alt.Chart(df_vector_search_engines).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(vector3_nearest)
    
    # Put the five layers into a chart and bind the data
    vector3_app_count = alt.layer(vector3_line, vector3_selectors, vector3_points, vector3_rules, vector3_text
    ).properties(width=800, height=500
    ).configure(padding=20, background="#111111"
    ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
    ).configure_axis(labelFontSize=14)
    
    with line1:
        st.altair_chart(vector3_app_count, use_container_width=True)
    with line2:
        st.markdown(f'''  
            **Vector search** tools transform unstructured datasets into numerical representations so the algorithm can efficiently retrieve contextual, similar data.
            - :orange[**{faiss_pct}%**] Faiss
            - :orange[**{qdrant_client_pct}%**] Qdrant
            - :orange[**{elasticsearch_pct}%**] Elasticsearch
            - :orange[**{pgvector_pct}%**] Pgvector
            ''')


with vector_tab4:
    
    vector_line_widget_1, vector_line_widget_2, vector_line_widget_3 = st.columns((2,4,1))
    line1, line2 = st.columns((2,1), gap="large")
    
    # Load and prepare data
    
    # Prepare proprietary vs open source data
    proprietary_vector_databases = ['pinecone', 'elasticsearch']
    opensource_vector_databases = ['chromadb', 'weaviate', 'faiss', 'pgvector', 'qdrant_client']
    
    proprietary_vector_options = vector_line_widget_1.multiselect('Select proprietary tools', proprietary_vector_databases, proprietary_vector_databases)
    opensource_vector_options = vector_line_widget_2.multiselect('Select open source tools', opensource_vector_databases, opensource_vector_databases)
    vector_numbers_percent_options = vector_line_widget_3.selectbox('Growth units', ('Count', '% Growth'), key='vector_tab3_vector_numbers_percent_options')

    all_vector_databases = proprietary_vector_options + opensource_vector_options
 
    
    if vector_numbers_percent_options=='Count':
       y_var = "WEEKLY_APP_COUNT"
       y_title = "Weekly App Count"
    if vector_numbers_percent_options=='% Growth':
       y_var = "WEEKLY_PCT"
       y_title = "Relative Weekly App Growth (%)"


    df_vector_proprietary_opensource_type = add_cumulative_column_proprietary_opensource(df, 'SUBDOMAIN', proprietary_vector_options, opensource_vector_options, 'tools')

    df_vector_proprietary_opensource_type['TOTAL_WEEKLY_APP_COUNT'] = df_vector_proprietary_opensource_type.groupby('WEEK_START').sum().reset_index()['WEEKLY_APP_COUNT']
    df_vector_proprietary_opensource_type['WEEKLY_PCT'] = (df_vector_proprietary_opensource_type['WEEKLY_APP_COUNT']/df_vector_proprietary_opensource_type['TOTAL_WEEKLY_APP_COUNT']) * 100
    df_vector_proprietary_opensource_type['WEEKLY_PCT'] = df_vector_proprietary_opensource_type['WEEKLY_PCT'].astype('int')
    
    df_vector_proprietary_opensource_type_sort_list = sort_opensource_tools(df_vector_proprietary_opensource_type, y_var)
    
       #df_vector_growth = df_vector_databases_type.copy()
       #df_vector_growth = df_vector_growth.groupby(['MODEL_TYPE', 'WEEK_START']).sum().reset_index()
       ## df_vector_growth.drop(['WEEK_OVER_WEEK_USER_GROWTH'], axis=1)
       #df_vector_growth['WEEK_OVER_WEEK_APP_GROWTH'] = (df_vector_growth['WEEKLY_APP_COUNT'].pct_change()*100)
       #df_vector_databases_type = df_vector_growth.copy()
    
    ## df_llm.LLM_MODEL = [x.replace('"', '') for x in df_llm.LLM_MODEL]
    #df_vector_databases = df_llm[df_llm.LLM_MODEL.isin(all_vector_databases)]

    ## df_vector_databases['MODEL_TYPE'] = 'open source'
    #df_vector_databases = df_vector_databases.assign(MODEL_TYPE='open source')
    #df_vector_databases.loc[df_vector_databases['LLM_MODEL'].isin(proprietary_vector_options), 'MODEL_TYPE'] = 'proprietary'
    
    #df_vector_databases_agg = df_vector_databases.groupby(['MODEL_TYPE', 'WEEK_START']).sum()
    #df_vector_databases_type = df_vector_databases_agg.add_suffix('').reset_index()

   
    # Plotting
    # Create a selection that chooses the nearest point & selects based on x-value
    vector4_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['WEEK_START'], empty='none')
    
    # The basic line
    vector4_line = alt.Chart(df_vector_proprietary_opensource_type).mark_line(interpolate='linear').encode(
        x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
        y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
        color=alt.Color('MODEL_TYPE:N',
                         scale=alt.Scale(domain=df_vector_proprietary_opensource_type_sort_list),
                         legend=alt.Legend(title=" ") ))
    
    # Transparent selectors across the chart. This is what tells us the x-value of the cursor
    vector4_selectors = alt.Chart(df_vector_proprietary_opensource_type).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(vector4_nearest)
    
    # Draw points on the line, and highlight based on selection
    vector4_points = vector4_line.mark_point().encode(opacity=alt.condition(vector4_nearest, alt.value(1), alt.value(0)))
    
    # Draw text labels near the points, and highlight based on selection
    vector4_text = vector4_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(vector4_nearest, f'{y_var}:Q', alt.value(' ')))
    
    # Draw a rule at the location of the selection
    vector4_rules = alt.Chart(df_vector_proprietary_opensource_type).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(vector4_nearest)
    
    # Put the five layers into a chart and bind the data
    vector4_app_count = alt.layer(vector4_line, vector4_selectors, vector4_points, vector4_rules, vector4_text
    ).properties(width=800, height=500
    ).configure(padding=20, background="#111111"
    ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
    ).configure_axis(labelFontSize=14)
    
    with line1:
        st.altair_chart(vector4_app_count, use_container_width=True)
    
    with line2:
        st.markdown(f'''
            ### :orange[**{vector_opensource_pct}%**] use open source
            - **Open-source tools** are lower cost, typically allowing for more flexibility and community support.
            - **Proprietary tools** are usually higher cost, and include managed support, integrated solutions, or specific features not readily available in open-source alternatives.
            ''')


st.markdown("""
<div class="container">
  <a data-scroll href="#seventh">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)

############################################################

st.markdown("<div id='seventh'></div>", unsafe_allow_html=True)

colored_header(
label="Are chatbots the future?",
description="",
color_name="light-blue-70",
)

def calculate_chat_most_recent_week_pct(input_df):
    chat_most_recent_week = input_df['WEEK_START'].max()
    most_recent_df = input_df[input_df.WEEK_START == chat_most_recent_week]
    return most_recent_df.iloc[0]['WEEKLY_APP_PCT']
recent_chat_apps_pct = calculate_chat_most_recent_week_pct(df_weekly_chat_app)

usage_col = st.columns((2,1), gap="large")

with usage_col[0]:
    chat_numbers_percent_options = st.selectbox('Growth units', ('% Growth', 'Count'), key='key_chat_numbers_percent_options')

    if chat_numbers_percent_options=='% Growth':
       y_var = "WEEKLY_APP_PCT"
       y_title = "Relative Weekly App Growth (%)"
        
    if chat_numbers_percent_options=='Count':
       y_var = "WEEKLY_APP_COUNT"
       y_title = "Weekly App Count"

    # Plotting
    # Create a selection that chooses the nearest point & selects based on x-value
    chat_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                                fields=['WEEK_START'], empty='none')
        
    # The basic line
    chat_line = alt.Chart(df_weekly_chat_app).mark_line(interpolate='linear').encode(
                    x=alt.X("WEEK_START:T", axis=alt.Axis(title="Week Start", titlePadding=15, titleFontSize=20, titleFontWeight=900, labelAngle=-90), scale=alt.Scale(padding=32)),
                    y=alt.Y(f"{y_var}:Q", axis=alt.Axis(title=y_title, titlePadding=15, titleFontSize=20, titleFontWeight=900)),
                    # y=alt.Y("WEEKLY_APP_COUNT:Q", axis=alt.Axis(title="Weekly App Count", titlePadding=15, titleFontSize=20, titleFontWeight=900)),
                    color=alt.Color('APP_TYPE:N',
                                     scale=alt.Scale(domain=['single text input', 'chat']),
                                     legend=alt.Legend(title=" ") ))
        
    # Transparent selectors across the chart. This is what tells us the x-value of the cursor
    chat_selectors = alt.Chart(df_weekly_chat_app).mark_point().encode(x='WEEK_START:T', opacity=alt.value(0),).add_selection(chat_nearest)
        
    # Draw points on the line, and highlight based on selection
    chat_points = chat_line.mark_point().encode(opacity=alt.condition(chat_nearest, alt.value(1), alt.value(0)))
        
    # Draw text labels near the points, and highlight based on selection
    chat_text = chat_line.mark_text(align='left', dx=0, dy=-15, fontSize=16).encode(text=alt.condition(chat_nearest, f'{y_var}:Q', alt.value(' ')))
        
    # Draw a rule at the location of the selection
    chat_rules = alt.Chart(df_weekly_chat_app).mark_rule(color='gray').encode(x='WEEK_START:T',).transform_filter(chat_nearest)
        
    # Put the five layers into a chart and bind the data
    chat_count = alt.layer(chat_line, chat_selectors, chat_points, chat_rules, chat_text
        ).properties(width=800, height=500
        ).configure(padding=20, background="#111111"
        ).configure_legend(orient='bottom', titleFontSize=16, labelFontSize=14, titlePadding=0
        ).configure_axis(labelFontSize=14)
    
    st.altair_chart(chat_count, use_container_width=True)


with usage_col[1]:
    st.markdown(f"""
        ### :orange[**{st_chat_input_and_chat_message_pct}**%] of total apps (and growing weekly) are chatbots

        Chatbots let users iteratively refine answers, leaving room for fluid, human-like conversations with the LLM. Chatbots are also on the rise as indicated by their weekly growth to :orange[**{recent_chat_apps_pct}%**].

        Conversely, :orange[**{100-st_chat_input_and_chat_message_pct}**%] of total apps use text inputs with a single objective, generally not allowing for conversational refinement.

        Explore the [gallery](#gallery-of-llm-apps) below to see examples of apps using these different mode of accepting text input (_e.g._ single text input or chat input).
    """)

add_vertical_space(2)


st.markdown("""
<div class="container">
  <a data-scroll href="#eighth">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)


############################################################

st.markdown("<div id='eighth'></div>", unsafe_allow_html=True)


colored_header(
    label="Gallery of LLM apps",
    description="",
    color_name="light-blue-70",
)

with st.expander('Expand to see instructions'):
    st.markdown('''
        #### How to use the search
        - **Step 1.** Select LLM libraries of interest in the `Select LLM library` multi-select widget.
        - **Step 2.** Select app type of interest in the `Select app type` multi-select widget.
        - **Step 3.** Query results should appear after a short page refresh.
    ''')
    add_vertical_space(2)

#if "filtered_df" not in st.session_state:
#    df_gallery_data = pd.DataFrame()


llm_tool_list = ['openai',
                  'anthropic',
                  'cohere',
                  'huggingface_hub',
                  'transformers',
                  'llama_cpp',
                  'faiss',
                  'qdrant_client',
                  'pinecone',
                  'chromadb',
                  'weaviate',
                  'pgvector',
                  'langchain',
                  'llama_index']
                #'diffusers',
                # 'nomic', 'See All'
# st_chat_elements = ['st.chat_input', 'st.chat_message', 'st.text_input', 'st.text_area']

topapps_widget = st.columns(2)

with topapps_widget[0]:
    # selected_tool_topapps = st.selectbox('Select an LLM Library', llm_tool_list)
    selected_tool_topapps = st.multiselect('Select LLM library', llm_tool_list, []) # ['openai', 'langchain', 'pinecone']

#with topapps_widget[1]:
#    tool_boolean_search = st.selectbox('Search Option (Tool)', ('OR', 'AND'))

#with topapps_widget[1]:
#    selected_chat_elements = st.multiselect('Select input widget', st_chat_elements, []) # st_chat_elements[:2]

app_type = ['chat', 'single text input']
#app_type = ['chat', 'non-chat']
with topapps_widget[1]:
    selected_app_type = st.selectbox('Select app type', app_type, index=None)

#with topapps_widget[3]:
#    chat_boolean_search = st.selectbox('Search Option (Chat)', ('OR', 'AND'))


# Create a function to apply the boolean search
def perform_boolean_search(row):
    return all(tool in row for tool in selected_tool_topapps)
    
    #if tool_boolean_search == 'AND':
    #    return all(tool in row for tool in selected_tool_topapps)
    #elif tool_boolean_search == 'OR':
    #    return any(tool in row for tool in selected_tool_topapps)
    #else:
    #    return False
        
def perform_chat_boolean_search(row):
    return all(tool in row for tool in selected_app_type)
    #if chat_boolean_search == 'AND':
    #    return all(tool in row for tool in selected_chat_elements)
    #elif chat_boolean_search == 'OR':
    #    return any(tool in row for tool in selected_chat_elements)
    #else:
    #    return False


# df_gallery_data = top_apps[ top_apps['TOOL_LIST'].apply(perform_boolean_search) & top_apps['ST_CHAT_ELEMENTS'].apply(perform_chat_boolean_search)]



#if selected_tool_topapps != []:
#     tool_mask = top_apps['TOOL_LIST'].apply(perform_boolean_search)
#     df_gallery_data = top_apps[top_apps['TOOL_LIST'].apply(perform_boolean_search)]
#if selected_chat_elements!= []:
#     chat_mask = top_apps['ST_CHAT_ELEMENTS'].apply(perform_chat_boolean_search)
#     df_gallery_data = top_apps[top_apps['ST_CHAT_ELEMENTS'].apply(perform_chat_boolean_search)]

#df_gallery_data = top_apps[tool_mask]
#df_gallery_data = top_apps[chat_mask]



# Working
mask_tool = top_apps['TOOL_LIST'].apply(perform_boolean_search)

# mask_app_type = top_apps['APP_TYPE'].apply(perform_chat_boolean_search)
# df_gallery_data = top_apps[mask_tool & mask_chat]

if selected_app_type == None:
    df_gallery_data = top_apps[mask_tool].sort_values(by='VIEWS_CUMULATIVE', ascending=False)
else:
    mask_app_type = (top_apps.APP_TYPE == selected_app_type)
    df_gallery_data = top_apps[mask_tool & mask_app_type].sort_values(by='VIEWS_CUMULATIVE', ascending=False)


df_gallery_data['TOOL_LIST'] = df_gallery_data['TOOL_LIST'].apply(lambda x: [item for item in x if item in llm_tool_list])

# Show only public apps
df_public_apps_list = list(df_daily[df_daily.IS_PRIVATE == True].SUBDOMAIN.unique())
df_gallery_data = df_gallery_data[~df_gallery_data.SUBDOMAIN.isin(df_public_apps_list)]



#df_topapps = df_top_apps.copy()
## df_topapps['REPOSITORY'] = [f'https://github.com/{x}' for x in df_topapps['REPOSITORY']]

#df_topapps['SUBDOMAIN'] = [f'https://{x}.streamlit.app' for x in df_topapps['SUBDOMAIN']]
#df_topapps = df_topapps.drop('TOOL', axis=1)
#df_topapps.index = np.arange(1, len(df_topapps) + 1)


# Top apps
app_col = st.columns(5, gap="medium")
#with app_col[0]:
    #show_apps(df_gallery_data, 0)
    
#with app_col[1]:
    #show_apps(df_gallery_data, 1)
    
#with app_col[2]:
    #show_apps(df_gallery_data, 2)
    
#with app_col[3]:
    #show_apps(df_gallery_data, 3)
    
#with app_col[4]:
    #show_apps(df_gallery_data, 4)

# Show the top 5 rows individually if they exist
for i in range(5):
    if i < len(df_gallery_data):
        with app_col[i]:
            show_apps(df_gallery_data, i)

add_vertical_space(1)


st.dataframe(
    # df_topapps,
    # df_top_apps3,
    # top_apps,
    df_gallery_data,
    column_config={
         #"OWNER": st.column_config.TextColumn("Owner", help="App Creator"),
         #"GH_URL": st.column_config.LinkColumn(
         #   "GitHub Repository", 
         #   help="App GitHub Repository",
         #   width="medium",
         #),
         "SUBDOMAIN": st.column_config.TextColumn(
            "App Name", 
            help="App Name", 
            max_chars=100,
         ),
         "APP_URL": st.column_config.LinkColumn(
            "App URL", 
            help="App URL", 
            validate="^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
         ),
         "VIEWS_CUMULATIVE": st.column_config.ProgressColumn(
            "Cumulative Views",
            help="Total number of views since created",
            format="%f",
            min_value=0,
            max_value=df_gallery_data['VIEWS_CUMULATIVE'].max(),
         ),
         "TOOL_LIST": st.column_config.ListColumn(
            "Tool List", 
            help="LLM Tool List",
         ),
         "ST_CHAT_ELEMENTS": st.column_config.ListColumn(
            "Streamlit Chat Elements", 
            help="Streamlit Chat Elements",
         ),
         "APP_TYPE": st.column_config.ListColumn(
            "App Type", 
            help="App Type",
         ),
     },
     column_order=("SUBDOMAIN", "APP_URL", "VIEWS_CUMULATIVE", "GH_URL", "TOOL_LIST", "APP_TYPE"), # "ST_CHAT_ELEMENTS"
     height=210,
     use_container_width=True,
     hide_index=True
)

st.write(f"Results: `{len(df_gallery_data):,}` public apps")


st.markdown("""
<div class="container">
  <a data-scroll href="#ninth">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)

############################################################

st.markdown("<div id='ninth'></div>", unsafe_allow_html=True)

# :orange[Trust] is the biggest concern
colored_header(
    label="Concerns building with LLMs",
    description="",
    color_name="light-blue-70",
)

st.write('***What is your biggest concern when building LLM apps?*** In June 2023, :orange[**980**] respondents from the Streamlit community answered:')

# https://twitter.com/streamlit/status/1686816512324358144
# https://www.youtube.com/post/UgkxDfU29sWpCVF3RpPgV9XejmK2hFU5ZCTK
# https://www.linkedin.com/posts/streamlit_activity-7092582839195500546-S1nw

concerns_col = st.columns(4, gap="medium")
with concerns_col[0]:
   st.markdown(f"""
       <div style='text-align: center;'>
           <h3>Trust</h3>
           Is the LLM response <br> accurate?
       </div>
   """, unsafe_allow_html=True)
   add_vertical_space(1)

   @st.cache_data
   def display_trust_chart():
       st.altair_chart(make_donut(36, 'Trust', 'blue'), use_container_width=True)
   display_trust_chart()
   #st.altair_chart(make_donut_chart(36, 'Trust', 'blue'), use_container_width=True)

with concerns_col[1]:
   st.markdown("""
       <div style='text-align: center;'>
           <h3>Privacy</h3>
           Is my data <br> safe?
       </div>
   """, unsafe_allow_html=True)
   add_vertical_space(1)

   @st.cache_data
   def display_privacy_chart():
       st.altair_chart(make_donut(28, 'Privacy', 'green'), use_container_width=True)
   display_privacy_chart()
   #st.altair_chart(make_donut_chart(28, 'Privacy', 'green'), use_container_width=True)
    
with concerns_col[2]:
   st.markdown("""
       <div style='text-align: center;'>
           <h3>Cost</h3>
           AI ain't <br> cheap!
       </div>
   """, unsafe_allow_html=True)
   add_vertical_space(1)

   @st.cache_data
   def display_cost_chart():
       st.altair_chart(make_donut(19, 'Cost', 'orange'), use_container_width=True)
   display_cost_chart()
   #st.altair_chart(make_donut_chart(19, 'Cost', 'orange'), use_container_width=True)
   
with concerns_col[3]:
   st.markdown("""
       <div style='text-align: center;'>
           <h3>Skills</h3>
           I'm still <br> learning.
       </div>
   """, unsafe_allow_html=True)
   add_vertical_space(1)

   @st.cache_data
   def display_skills_chart():
       st.altair_chart(make_donut(17, 'Skills', 'red'), use_container_width=True)
   display_skills_chart()
   #st.altair_chart(make_donut_chart(17, 'Skills', 'red'), use_container_width=True)
   

st.markdown("""
<div class="container">
  <a data-scroll href="#tenth">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)


############################################################
st.markdown("<div id='tenth'></div>", unsafe_allow_html=True)

colored_header(
    label="LLM app architecture",
    description="",
    color_name="light-blue-70",
)

add_vertical_space(2)

# Drawn in Excalidraw
# Version 1: https://excalidraw.com/#json=DjFC86l72phtwfac29B2V,NIvz2cKtoQXkLCdp-4Cjow
# Version 2: https://excalidraw.com/#json=Dm-Rw4aXPuypV-MbDpZk5,Fs_3m2dV4bHP93pTbAkMpw

img_col = st.columns((1, 10, 1))
with img_col[1]:

    @st.cache_data
    def display_app_architecture():
        st.markdown('<img src="https://github.com/dataprofessor/streamlit-for-llm/blob/master/img/LLM-app-architecture.png?raw=true" width="100%">', unsafe_allow_html=True)
    display_app_architecture()
    
add_vertical_space(2)


with st.expander('Expand to see definitions of LLM architecture components'):

    about_col = st.columns(3, gap="large")
    
    with about_col[0]:
        st.markdown('''
            #### :blue[Large Language Models]
            Large Language Models (LLMs) are deep learning AI algorithms designed to understand and generate human-like text for a variety of practical purposes. 
    
            LLMs are pre-trained by processing natural language patterns from vast text datasets. Fine-tuning refines a pre-trained model to adapt to specific tasks by training it on specialized data.
    
            The most popular use cases for LLMs include content generation or summarization, language translation, chatbots/virtual assistants, or insights from data analysis.
    
        ''')
    with about_col[1]:
        st.markdown('''
            #### :blue[LLM Orchestration]
            LLMs rely on the data they're trained on, which can come with limitations, biases, and may be incomplete.
    
            Orchestration frameworks, like LangChain or LlamaIndex, allow developers to enhance their LLM and accomplish more, through AI agents and tools. This control layer helps improve the reliability, scalability, and accuracy of the responses generated.
    
            This unlocks the potential for LLMs to reveal their reasoning, continually “self-ask” the LLM to gather more relevant information for the task, make external API calls to other data sources, and much more.
    
        ''')
    with about_col[2]:
        st.markdown('''
            #### :blue[Vector Retrieval]
            Vector retrieval tools enable fast, efficient search from unstructured datasets (including text, images, video, or audio). These tools are commonly used in search or recommendation engines.
            - **Vector search** tools (_e.g._ FAISS or Elasticsearch) transform unstructured datasets into numerical representations so the algorithm can locate similar data.
            - **Vector databases** (_e.g._ Pinecone or Chroma) use indexing and other techniques to speed up the retrieval process. 
        ''')
    
    st.markdown('''
            #### :red[Visual UI]
            Streamlit\'s free, open-source Python library enables data scientists and developers to easily create interactive web apps with a user-friendly visual UI for data exploration and visualization. 
    ''')

add_vertical_space(2)


st.markdown("""
<div class="container">
  <a data-scroll href="#eleventh">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)

############################################################

st.markdown("<div id='eleventh'></div>", unsafe_allow_html=True)


colored_header(
    label="About Streamlit",
    description="",
    color_name="light-blue-70",
)

st.markdown(f"""
[Streamlit](https://streamlit.io/) is a faster way to build interactive data apps. (Like this one!)

It's simple, yet powerful, open source Python library enables developers and data scientists to easily create and share beautiful, custom web apps. 

With the rise of Large Language Models (LLMs), Streamlit has rapidly become the visual UI of choice for LLM-powered apps, including chatbots, sentiment analysis tools, content summarizers, and more.
""")
# As of {update_date}, over :orange[**{total_developer_number:,}**] app developers have created more than :orange[**{total_app_number:,}**] LLM-powered apps on Streamlit Community Cloud, with that number growing every day. In this report, we'll share key insights into the latest adoption trends and tools used in LLM app development.

add_vertical_space(2)

st.markdown("""
<div class="container">
  <a data-scroll href="#twelfth">
      <div class="arrow"></div>
  </a>
</div>
""", unsafe_allow_html=True)

############################################################

st.markdown("<div id='twelfth'></div>", unsafe_allow_html=True)

colored_header(
    label="Methodology",
    description="",
    color_name="light-blue-70",
)


# 1 container
st.caption('''
### Usage data from Streamlit Community Cloud
All data was collected in accordance with our [Community Cloud Terms of Use](https://streamlit.io/deployment-terms-of-use) and [Privacy Notice](https://streamlit.io/privacy-policy#3.-how-do-we-use-your-information?). 

The Streamlit software (the Python library) is open-sourced under the Apache 2.0 license, but Streamlit Community Cloud (the free hosting and deployment service) is proprietary to Snowflake Inc.  

### Aggregated data only
This Streamlit app does not provide direct query access to raw data, and ensures that no individual-level data can be extracted from the dataset. 

Streamlit may use personal information and other information to create de-identified and/or aggregated information. 

### No PII included
The dataset associated with this Streamlit app contains no personally identifiable information (PII). This includes but is not limited to names, addresses, phone numbers, email addresses, or any other sensitive personal information.

### Data Analysis and Visualization
Data wrangling is performed using `pandas`, visualized using `altair` and interactive widgets are provided herein to allow users to retrieve data subsets or views of interest. App screenshots used in the gallery section are dynamically retrieved via `selenium` and `chromium`.

''')


# Original 5 columns
# methodology_col = st.columns(5, gap="large")
# with methodology_col[0]:
#    st.subheader('1. Telemetry data')
#    st.write('As mentioned during the installation process, Streamlit collects usage statistics. You can find out more by reading our [Privacy Notice](https://streamlit.io/privacy-policy#3.-how-do-we-use-your-information?), but the high-level summary is that although we collect telemetry data we cannot see and do not store information contained in Streamlit apps.')    
#with methodology_col[1]:
#    st.subheader('2. No PII Included')
#    st.write('This dataset contains no Personally Identifiable Information (PII). This includes but is not limited to names, addresses, phone numbers, email addresses, or any other sensitive personal information.')    
#with methodology_col[2]:
#    st.subheader('3. Aggregated Data Only')
#    st.write('We are not providing direct query access to our database or any raw data. This approach ensures that no individual-level data can be extracted from the dataset, further reducing the risk of data breaches or privacy violations.')    
#with methodology_col[3]:
#    st.subheader('4. Data Analysis and Visualization')
#    st.write('Data wrangling is performed using `pandas`, visualized using `altair` and interactive widgets are provided herein to allow users to retrieve data subsets or views of interest.')
#with methodology_col[4]:
#    st.subheader('5. Data Retention and Refresh')
#    st.write('It is intended that the data will be refreshed every quarter/month.')


############################################################ 


# Up arrow
st.markdown("""
<div class="container">
  <a data-scroll href="#top">
      <div class="up-arrow"></div>
  </a>
</div>
<div align="center" style="color:#888;">Back to top</div>
""", unsafe_allow_html=True)
