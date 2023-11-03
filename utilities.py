import streamlit as st
import pandas as pd
import altair as alt
import time
import psutil
import random
import os
import sys
import base64
import textwrap
from PIL import Image, ImageDraw, ImageOps
from PIL.Image import Resampling
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from os.path import exists
import base64
from pathlib import Path


def load_bootstrap():
    return st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

@st.cache_data
def usage_in_app(input_df, input_tool):
  pct_usage = (input_df[input_df.LLM_MODEL == input_tool].SUBDOMAIN.nunique()/input_df.SUBDOMAIN.nunique()) * 100
  return round(pct_usage, 1)

@st.cache_data
def usage_number_in_app(input_df, input_tool):
    n_usage = input_df[input_df.LLM_MODEL == input_tool].SUBDOMAIN.nunique()
    return n_usage

@st.cache_data
def df_usage_in_app(input_df):
  llm_models = ['openai', 'anthropic', 'huggingface_hub', 'cohere', 'llama_cpp', 'diffusers', 'transformers']
  vector_databases = ['pgvector', 'faiss', 'qdrant_client', 'elasticsearch', 'pinecone', 'chromadb', 'weaviate']
  orchestration_tooling = ['langchain', 'llama_index']
  
  llm_models_results = []
  vector_databases_results = []
  orchestration_tooling_results = []

  llm_tool_dict = {
      'openai' : 'OpenAI',
      'anthropic': 'Anthrophic',
      'huggingface_hub' : 'HuggingFace Hub',
      'cohere': 'Cohere',
      'llama_cpp': 'Llama.cpp',
      'pyllamacpp': 'PyLlama.cpp',
      'diffusers': 'Diffusers',
      'transformers': 'Transformers',
      'pgvector' : 'pgvector',
      'faiss': 'FAISS',
      'qdrant_client': 'Qdrant Client',
      'elasticsearch': 'Elastic Search',
      'pinecone': 'Pinecone',
      'chromadb': 'ChromaDB',
      'weaviate': 'Weaviate',
      'langchain': 'LangChain',
      'llama_index': 'Llama Index'
      }
    
  for tool in llm_models:
    pct_usage = usage_in_app(input_df, tool)
    pct_dict = f'{{"TOOL": ["{tool}"], "PCT_USAGE": [{pct_usage}], "LLM_CATEGORY": ["LLM models"]}}' # llm_models
    results = pd.DataFrame.from_dict(eval(pct_dict))
    llm_models_results.append(results)

  for tool in vector_databases:
    pct_usage = usage_in_app(input_df, tool)
    pct_dict = f'{{"TOOL": ["{tool}"], "PCT_USAGE": [{pct_usage}], "LLM_CATEGORY": ["Vector retrieval"]}}' # vector_retrieval
    results = pd.DataFrame.from_dict(eval(pct_dict))
    vector_databases_results.append(results)

  for tool in orchestration_tooling:
    pct_usage = usage_in_app(input_df, tool)
    pct_dict = f'{{"TOOL": ["{tool}"], "PCT_USAGE": [{pct_usage}], "LLM_CATEGORY": ["LLM orchestration"]}}' # llm_orchestration
    results = pd.DataFrame.from_dict(eval(pct_dict))
    orchestration_tooling_results.append(results)

  all_models = llm_models_results + vector_databases_results + orchestration_tooling_results
  df_all_models = pd.concat(all_models, axis=0).sort_values(by=['PCT_USAGE'], ascending=False).reset_index().drop('index', axis=1)
  df_all_models['NAME'] = [llm_tool_dict[x] for x in df_all_models['TOOL']]
  return df_all_models

@st.cache_data
def get_top_apps(input_df, input_tool, input_daily):
  # Tool subset
  df_subset = input_df[input_df.LLM_MODEL == input_tool]

  # Views for apps using selected tool
  # df_subset_views = input_df.groupby(['LLM_MODEL']).get_group(input_tool).groupby('SUBDOMAIN')['SUBDOMAIN','TOTAL_VIEWS'].sum().sort_values(by=['TOTAL_VIEWS'], ascending=False).reset_index()
  df_subset_views = input_df.groupby(['LLM_MODEL']).get_group(input_tool).groupby('SUBDOMAIN')['TOTAL_VIEWS'].sum().reset_index().sort_values(by=['TOTAL_VIEWS'], ascending=False)

    
  # Convert DataFrame to a dictionary
  df_owner_subdomain_dict = df_subset[['OWNER', 'SUBDOMAIN']].set_index('SUBDOMAIN').to_dict()

  # Add OWNER column
  df_subset_views['OWNER'] = [df_owner_subdomain_dict['OWNER'].get(x) for x in df_subset_views.SUBDOMAIN]
  df_subset_views.rename(columns = {'TOTAL_VIEWS':'VIEWS_CUMULATIVE'}, inplace = True)

  # Convert DataFrame to a dictionary
  df_repository_subdomain_dict = input_daily[['REPOSITORY', 'SUBDOMAIN']].set_index('SUBDOMAIN').to_dict()
    
  # Add REPOSITORY column
  df_subset_views['REPOSITORY'] = [df_repository_subdomain_dict['REPOSITORY'].get(x) for x in df_subset_views.SUBDOMAIN]

  # Reorder column
  df_subset_views = df_subset_views.reindex(sorted(df_subset_views.columns), axis=1)
  return df_subset_views


# Prepare gallery data
# @st.cache_data
def prepare_gallery_data(input_df, input_df_daily):
  llm_tool_list = ['openai',
                  'anthropic',
                  'cohere',
                  #'diffusers',
                  'huggingface_hub',
                  'transformers',
                  'llama_cpp',
                  'faiss',
                  'qdrant_client',
                  'pinecone',
                  'chromadb',
                  'weaviate',
                  #'nomic',
                  'pgvector',
                  'langchain',
                  'llama_index']
  df_tool_list = []
  for tool in llm_tool_list:
    df_tool = get_top_apps(input_df, tool, input_df_daily)
    df_tool_list.append(df_tool)

  df_tools = pd.concat(df_tool_list, axis=0)
    
  df_tools = df_tools.drop_duplicates(subset=['SUBDOMAIN'])
    
  #df_tools['TOOL_LIST'] = input_df.groupby('SUBDOMAIN')['LLM_MODEL'].unique().agg(list).reindex(df_tools['SUBDOMAIN']).tolist()
  #df_tools['USES_CHAT_INPUT'] = input_df.groupby('SUBDOMAIN')['USES_CHAT_INPUT'].unique().agg(lambda x: x[0]).reindex(df_tools['SUBDOMAIN']).tolist()
  #df_tools['USES_CHAT_MESSAGE'] = input_df.groupby('SUBDOMAIN')['USES_CHAT_MESSAGE'].unique().agg(lambda x: x[0]).reindex(df_tools['SUBDOMAIN']).tolist()
  #df_tools['USES_TEXT_INPUT'] = input_df.groupby('SUBDOMAIN')['USES_TEXT_INPUT'].unique().agg(lambda x: x[0]).reindex(df_tools['SUBDOMAIN']).tolist()
  #df_tools['USES_TEXT_AREA'] = input_df.groupby('SUBDOMAIN')['USES_TEXT_AREA'].unique().agg(lambda x: x[0]).reindex(df_tools['SUBDOMAIN']).tolist()

  df_tools['TOOL_LIST'] = input_df.groupby('SUBDOMAIN')['LLM_MODEL'].unique().transform(list).reindex(df_tools['SUBDOMAIN']).tolist()
  df_tools['USES_CHAT_INPUT'] = input_df.groupby('SUBDOMAIN')['USES_CHAT_INPUT'].unique().transform(lambda x: x[0]).reindex(df_tools['SUBDOMAIN']).tolist()
  df_tools['USES_CHAT_MESSAGE'] = input_df.groupby('SUBDOMAIN')['USES_CHAT_MESSAGE'].unique().transform(lambda x: x[0]).reindex(df_tools['SUBDOMAIN']).tolist()
  df_tools['USES_STREAMLIT_CHAT'] = input_df.groupby('SUBDOMAIN')['USES_STREAMLIT_CHAT'].unique().transform(lambda x: x[0]).reindex(df_tools['SUBDOMAIN']).tolist()
  df_tools['USES_TEXT_INPUT'] = input_df.groupby('SUBDOMAIN')['USES_TEXT_INPUT'].unique().transform(lambda x: x[0]).reindex(df_tools['SUBDOMAIN']).tolist()
  df_tools['USES_TEXT_AREA'] = input_df.groupby('SUBDOMAIN')['USES_TEXT_AREA'].unique().transform(lambda x: x[0]).reindex(df_tools['SUBDOMAIN']).tolist()

  return df_tools

# prepare_gallery_data(df, df_daily)



def weekly_growth_plot(input_df, input_tool, input_color):
  return alt.Chart(input_df[input_df.LLM_MODEL==input_tool]).mark_line(strokeWidth=3.5).encode(
      x=alt.X("WEEK_START:T", axis=None),
      # y=alt.Y("WEEKLY_APP_COUNT:Q", axis=None, scale=alt.Scale(domain=[0, input_df.WEEKLY_APP_COUNT.max()])),
      y=alt.Y("WEEKLY_APP_COUNT:Q", axis=None),
      color=alt.Color('LLM_MODEL:N', legend=None, scale=alt.Scale(scheme=input_color)),
      tooltip=[alt.Tooltip('monthdate(WEEK_START):O', title='Week Start'),
              alt.Tooltip('max(WEEKLY_APP_COUNT):Q', title='Max Weekly App Count')]
      ).properties(height=200).configure(background='#1C2833').configure_view(strokeOpacity=0)


@st.cache_resource
def get_driver():
    width = 900 # 1000
    height = 486 # 540
    options = webdriver.ChromeOptions()
    
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    options.add_argument(f"--window-size={width}x{height}")
    
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    
    return webdriver.Chrome(service=service, options=options)

@st.cache_data
def get_screenshot(subdomain_name):
    driver = get_driver()
    driver.get(f"https://{subdomain_name}.streamlit.app/~/+/")
            
    time.sleep(3)
            
    # Explicitly wait for an essential element to ensure content is loaded
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            
    # Get scroll height and width
    #scroll_width = driver.execute_script('return document.body.parentNode.scrollWidth')
    #scroll_height = driver.execute_script('return document.body.parentNode.scrollHeight')
            
    # Set window size
    #driver.set_window_size(scroll_width, scroll_height)
            
    # Now, capture the screenshot
    driver.save_screenshot(f'{subdomain_name}_screenshot.png')


def add_corners(im, rad):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2 - 1, rad * 2 - 1), fill=255)
    
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    
    # Apply rounded corners only to the top
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    
    im.putalpha(alpha)
    return im

@st.cache_data
def generate_app_image(subdomain_name):
    bg_random = random.randint(1,100)
    if bg_random < 10:
        bg_random = '0' + str(bg_random)
    bg_img = Image.open(f'background/background-{bg_random}.jpeg')
    app_img = Image.open(f'{subdomain_name}_screenshot.png')

    # Create a blank white rectangle
    w, h = app_img.width, app_img.height
    img = Image.new('RGB', (w, h), color='white')
      
    # Create a drawing object
    draw = ImageDraw.Draw(img)
        
    # Define the coordinates of the rectangle (left, top, right, bottom)
    rectangle_coordinates = [(0, 0), (w + 50, h + 0)]
        
    # Draw the white rectangle
    draw.rectangle(rectangle_coordinates, fill='#FFFFFF')
    img = add_corners(img, 24)
    img.save('rect.png')
    ###
    # Resize app image
    image_resize = 0.95
    new_width = int(img.width * image_resize)
    new_height = int(img.height * image_resize)
    resized_app_img = app_img.resize((new_width, new_height))
    
    # Crop top portion of app_img
    border = (0, 4, 0, 0) # left, top, right, bottom
    resized_app_img = ImageOps.crop(resized_app_img, border)
    
    # Add corners
    resized_app_img = add_corners(resized_app_img, 24)
    
    img.paste(resized_app_img, (int(resized_app_img.width*0.025),int(resized_app_img.width*0.035)), resized_app_img)
    img.save('app_rect.png')

    ###
    # Resize app image
    image_resize_2 = 0.9
    new_width_2 = int(bg_img.width * image_resize_2)
    new_height_2 = int(bg_img.height * image_resize_2)
    resized_img = img.resize((new_width_2, new_height_2))
    
    bg_img.paste(resized_img, ( int(bg_img.width*0.05), int(bg_img.width*0.06) ), resized_img)

    # Add Streamlit logo
    logo_width = 100
    logo_vertical_placement = 670
    logo_horizontal_placement = 80

    logo_img = Image.open('static/streamlit-logo.png').convert('RGBA')
    logo_img.thumbnail([sys.maxsize, logo_width], Resampling.LANCZOS)
    # bg_img.paste(logo_img, (logo_horizontal_placement, logo_vertical_placement), logo_img)
    bg_img_file_name = f'{subdomain_name}_final.png'
    bg_img.save(bg_img_file_name)
    
    # st.image(bg_img)


    img_path = bg_img_file_name

    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    def img_to_html(img_path):
        img_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='100%'>".format(
          img_to_bytes(img_path)
        )
        return f'<a href="https://{subdomain_name}.streamlit.app">{img_html}</a>'

    st.markdown(img_to_html(img_path), unsafe_allow_html=True)

  
    
    #st.markdown(f'''
    #  <a href="https://{subdomain_name}.streamlit.app">
    #    <img src="{bg_img_file_name}">
    #  </a>
    #''', unsafe_allow_html=True,
    #)

    #with Image.open('final.png') as image:
    #    st.image(image)


def show_apps(input_df, input_position):
    app = input_df['SUBDOMAIN'].iloc[input_position]
    app_name = input_df['SUBDOMAIN'].iloc[input_position]
    if len(app_name) > 48:
        app_name = textwrap.shorten(app_name, width=42, placeholder=" ...")
    owner = input_df['OWNER'].iloc[input_position]
    repo = input_df['REPOSITORY'].iloc[input_position]
    # views = input_df['NUM_VIEWS_LAST_30_DAYS'].iloc[input_position]
    views = int(input_df['VIEWS_CUMULATIVE'].iloc[input_position])
    get_screenshot(app)
    generate_app_image(app)
    return st.markdown(f'''
        <a href="https://{app}.streamlit.app" style="font-size: 18px; font-weight: 900;">{app_name}</a>
        <br>
        <a href="https://github.com/{owner}/{repo}" target="_blank" style="text-decoration: none;">
            View source &nbsp;
        </a> →
        <br>
        Cumulative Views: `{views:,}`
    ''', unsafe_allow_html=True)

        #<br>
        #<a href="https://github.com/{owner}" target="_blank" style="text-decoration: none;">
        #    &nbsp;<img src="https://github.com/{owner}.png" alt="{owner}" class="avatar">&nbsp;
        #    {owner}
        #</a>

def make_donut(input_response, input_text, input_color):
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']
    
  source = pd.DataFrame({
      # "category": ['B', 'A'],
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      #"category": ['B', 'A'],
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=70, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  )
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=38, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=70, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  )
  return plot_bg + plot + text


def make_donut_chart(input_response, input_text, input_color):

   if input_color == 'blue':
       chart_color = ['#29b5e8', '#155F7A']
   if input_color == 'green':
       chart_color = ['#27AE60', '#12783D']
   if input_color == 'orange':
       chart_color = ['#F39C12', '#875A12']
   if input_color == 'red':
       chart_color = ['#E74C3C', '#781F16']
    
   source = pd.DataFrame({"Topic": ['Other', input_text], "Percent Response": [100-input_response, input_response]})
   source_bg = pd.DataFrame({"Topic": ['Other', input_text],"Percent Response": [100, 0]})
    
   selector = alt.selection_single(on='mouseover')
    
   donut = alt.Chart(source).mark_arc(innerRadius=70, cornerRadius=25).encode(
             theta=alt.Theta(field="Percent Response", type="quantitative", stack=True),
             color=alt.condition(selector, 'Topic', alt.value('#111111'), scale=alt.Scale(domain=[input_text, 'Other'], range=chart_color), legend=None),
             tooltip=['Topic', 'Percent Response']
   ).add_selection(selector).encode(text='Topic')

   #####
   plot = alt.Chart(source).mark_arc(innerRadius=70, cornerRadius=25).encode(
      theta=alt.Theta(field="Percent Response", type="quantitative", stack=True),
      color=alt.condition(selector, 'Topic', alt.value('#111111'), scale=alt.Scale(domain=[input_text, 'Other'], range=chart_color), legend=None),
   )
   #####
    
   plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=70, cornerRadius=20).encode(
             theta=alt.Theta(field="Percent Response", type="quantitative", stack=True),
             color=alt.condition(selector, 'Topic', alt.value('#111111'), scale=alt.Scale(domain=[input_text, 'Other'], range=chart_color), legend=None),
             tooltip=['Topic', 'Percent Response']
   )

   text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=38, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
   return plot_bg + donut + text


@st.cache_data
def df_tools_usage(input_df):
  llm_models = ['openai', 'anthropic', 'huggingface_hub', 'cohere', 'llama_cpp', 'diffusers', 'transformers']
  vector_retrieval = ['pgvector', 'faiss', 'qdrant_client', 'elasticsearch', 'pinecone', 'chromadb', 'weaviate']
  llm_orchestration = ['langchain', 'llama_index']
  all_tools = llm_models + vector_retrieval + llm_orchestration

  llm_models_results = []
  vector_retrieval_results = []
  llm_orchestration_results = []

  for i in input_df.index:
    # Is the app using LLM model?
    if any(x in llm_models for x in eval(input_df.ATTRIBUTIONS[i])):
      llm_models_results.append(1)
    else:
      llm_models_results.append(0)

    # Is the app using Vector Retrieval?
    if any(x in vector_retrieval for x in eval(input_df.ATTRIBUTIONS[i])):
      vector_retrieval_results.append(1)
    else:
      vector_retrieval_results.append(0)

    # Is the app using LLM Orchestration?
    if any(x in llm_orchestration for x in eval(input_df.ATTRIBUTIONS[i])):
      llm_orchestration_results.append(1)
    else:
      llm_orchestration_results.append(0)

  input_df['USE_LLM_MODELS'] = llm_models_results
  input_df['USE_VECTOR_RETRIEVAL'] = vector_retrieval_results
  input_df['USE_LLM_ORCHESTRATION'] = llm_orchestration_results

  input_df['USE_SUM'] = input_df['USE_LLM_MODELS'] + input_df['USE_VECTOR_RETRIEVAL'] + input_df['USE_LLM_ORCHESTRATION']
  return input_df

@st.cache_data
def calculate_tools_usage(input_df):
  df_subdomain = input_df.groupby('SUBDOMAIN').first().reset_index()
  total_apps = df_subdomain.shape[0]
  use_llm_models = df_subdomain[df_subdomain.USE_LLM_MODELS == 1].shape[0]
  use_vector_retrieval = df_subdomain[df_subdomain.USE_VECTOR_RETRIEVAL == 1].shape[0]
  use_llm_orchestration = df_subdomain[df_subdomain.USE_LLM_ORCHESTRATION == 1].shape[0]
  pct_llm_models = round((use_llm_models/total_apps)*100)
  pct_vector_retrieval = round((use_vector_retrieval/total_apps)*100)
  pct_llm_orchestration = round((use_llm_orchestration/total_apps)*100)
  return pct_llm_models, pct_vector_retrieval, pct_llm_orchestration

@st.cache_data
def calculate_use_sum(input_df):
  df_subdomain = input_df.groupby('SUBDOMAIN').first().reset_index()
  total_apps = df_subdomain.shape[0]
  
  use_1 = df_subdomain[df_subdomain.USE_SUM == 1].shape[0]
  use_2 = df_subdomain[df_subdomain.USE_SUM == 2].shape[0]
  use_3 = df_subdomain[df_subdomain.USE_SUM == 3].shape[0]

  pct_use_1 = round((use_1/total_apps)*100)
  pct_use_2 = round((use_2/total_apps)*100)
  pct_use_3 = round((use_3/total_apps)*100)
  return pct_use_1, pct_use_2, pct_use_3

####################################
####################################
# Prepare daily data to weekly
def prep_daily_data_to_week(input_df, input_type):
  # input_df['week'] = pd.DatetimeIndex(input_df['RECORD_DATE']).week
  input_df['WEEK_START'] = pd.to_datetime(input_df['RECORD_DATE']).dt.to_period('W').dt.start_time
  input_df['WEEK_START'] = pd.to_datetime(input_df['WEEK_START'].astype(str), format='%Y-%m-%d')

  # week_list = list(input_df.sort_values('week')['week'].unique())
  week_list = list(input_df.sort_values('WEEK_START')['WEEK_START'].unique())
  
  # Retain only full weeks
  for i in week_list:
    num_days = input_df[input_df['WEEK_START'] == i].RECORD_DATE.nunique()
    keep_weeks = []
    if num_days == 7:
      keep_weeks.append(i)
    else:
      week_list.remove(i)

  # Number of developers
  num_list = []
  for i in week_list:
    num_count = input_df[input_df['WEEK_START'] == i][input_type].nunique()
    num_list.append(num_count)

  # month_list, num_apps_list
  df = pd.DataFrame({'WEEK_START': week_list, 'WEEKLY_COUNT': num_list})
  df['WEEK_START'] = df['WEEK_START'].dt.strftime('%m/%d/%Y')

  #cumulative_list = []
  #for x in df['WEEK_START'].unique():
  #  subdomains_count = input_df[input_df.WEEK_START.isin(input_df['WEEK_START'].unique()[:input_df['WEEK_START'].unique().tolist().index(x) + 1])]['SUBDOMAIN'].nunique()
  #  cumulative_list.append(subdomains_count)
  #df['WEEKLY_CUMULATIVE_COUNT'] = cumulative_list
    
  return df


####################################
# Prepare daily data to monthly
def prep_daily_data_to_month(input_df, input_type):
  input_df['month'] = pd.DatetimeIndex(input_df['RECORD_DATE']).month
  month_list = list(input_df.sort_values('month')['month'].unique())
  
  # Retain only full months
  for i in month_list:
    num_days = input_df[input_df.month == i].RECORD_DATE.nunique()
    keep_months = []
    if num_days >= 28:
      keep_months.append(i)
    else:
      month_list.remove(i)

  # Number of developers
  num_list = []
  for i in month_list:
    num_count = input_df[input_df.month == i][input_type].nunique()
    num_list.append(num_count)

  # month_list, num_apps_list
  df = pd.DataFrame({'MONTH': month_list, 'MONTHLY_COUNT': num_list})
  df['MONTH'] = [(str(x) + '-2023' ) for x in df['MONTH']]
  df['MONTH'] = pd.to_datetime(df['MONTH']).apply(lambda x: x.strftime('%b-%Y'))
  # df['MONTH'] = pd.to_datetime( df['MONTH'], format='%m').dt.month_name()
  df['MONTHLY_CUMULATIVE_COUNT'] = df['MONTHLY_COUNT'].cumsum()
  return df


#weekly_dev_count = prep_daily_data_to_week(df_daily, 'OWNER')
#monthly_dev_count = prep_daily_data_to_month(df_daily, 'OWNER')
#weekly_app_count = prep_daily_data_to_week(df_daily, 'SUBDOMAIN')
#monthly_app_count = prep_daily_data_to_month(df_daily, 'SUBDOMAIN')
####################################
####################################

@st.cache_data
def get_weekly_cumulative_app_count(input_df):
  weekly_app_count = [(input_df[input_df.WEEK_START == x]['SUBDOMAIN'].nunique()) for x in input_df['WEEK_START'].unique()]
  df_weekly_app_count = pd.DataFrame({'WEEK_START': input_df['WEEK_START'].unique(), 'WEEKLY_APP_COUNT': weekly_app_count})

  #cumulative_list = []
  #for x in input_df['WEEK_START'].unique():
  #    subdomains_count = input_df[input_df.WEEK_START.isin(input_df['WEEK_START'].unique()[:input_df['WEEK_START'].unique().tolist().index(x) + 1])]['SUBDOMAIN'].nunique()
  #    cumulative_list.append(subdomains_count)
  #df_weekly_app_count['CUMULATIVE_WEEKLY_APP_COUNT'] = cumulative_list

  df_weekly_app_count.rename(columns = {'WEEKLY_APP_COUNT':'WEEKLY_COUNT'}, inplace = True)
  df_weekly_app_count['COUNT_TYPE'] = 'Apps created'
    
  return df_weekly_app_count

# get_weekly_cumulative_app_count(df)

@st.cache_data
def get_weekly_cumulative_developer_count(input_df):
  weekly_developer_count = [(input_df[input_df.WEEK_START == x]['OWNER'].nunique()) for x in input_df['WEEK_START'].unique()]
  df_weekly_developer_count = pd.DataFrame({'WEEK_START': input_df['WEEK_START'].unique(), 'WEEKLY_DEVELOPER_COUNT': weekly_developer_count})

  #cumulative_list = []
  #for x in input_df['WEEK_START'].unique():
  #    developer_count = input_df[input_df.WEEK_START.isin(input_df['WEEK_START'].unique()[:input_df['WEEK_START'].unique().tolist().index(x) + 1])]['OWNER'].nunique()
  #    cumulative_list.append(developer_count)
  #df_weekly_developer_count['CUMULATIVE_WEEKLY_DEVELOPER_COUNT'] = cumulative_list

  df_weekly_developer_count.rename(columns = {'WEEKLY_DEVELOPER_COUNT':'WEEKLY_COUNT'}, inplace = True)
  df_weekly_developer_count['COUNT_TYPE'] = 'Unique developers'
    
  return df_weekly_developer_count

# get_weekly_cumulative_developer_count(df)


# Add cumulative column
def add_cumulative_column_bak(input_df, input_category, input_type, llm_models, vector_databases, orchestration_tooling):
  #embedded_models = ['openai', 'anthropic', 'huggingface_hub', 'cohere', 'llama_cpp', 'pyllamacpp', 'diffusers', 'transformers']
  #vector_databases = ['pgvector', 'faiss', 'pinecone', 'chromadb', 'weaviate', 'qdrant_client']
  #orchestration_tooling = ['langchain', 'llama_index', 'nomic', 'torch', 'tensorflow']

  input_df['LLM_CATEGORY'] = 'NA'
  input_df.loc[input_df['LLM_MODEL'].isin(llm_models), 'LLM_CATEGORY'] = 'llm_models'
  input_df.loc[input_df['LLM_MODEL'].isin(vector_databases), 'LLM_CATEGORY'] = 'vector_databases'
  input_df.loc[input_df['LLM_MODEL'].isin(orchestration_tooling), 'LLM_CATEGORY'] = 'orchestration_tooling'

  input_df = input_df.loc[input_df['LLM_CATEGORY'] != 'NA']
  input_df = input_df.loc[input_df['LLM_CATEGORY'] == input_category]

  # Add a cumulative app count column
  cumulative_list = []
  subdomains_list = []
  for x in input_df['WEEK_START'].unique():
    subdomains_count = input_df[input_df.WEEK_START.isin([x])][input_type].nunique()
    subdomains_cumulative_count = input_df[input_df.WEEK_START.isin(input_df['WEEK_START'].unique()[:input_df['WEEK_START'].unique().tolist().index(x) + 1])][input_type].nunique()
    subdomains_list.append(subdomains_count)
    cumulative_list.append(subdomains_cumulative_count)
  
  df = pd.DataFrame({'WEEK_START': input_df['WEEK_START'].unique(), 'WEEKLY_APP_COUNT': subdomains_list, 'CUMULATIVE_WEEKLY_APP_COUNT': cumulative_list})
  df['LLM_CATEGORY'] = input_category

  return df


#df_llm_models_app_count = add_cumulative_column(df, 'llm_models', 'SUBDOMAIN')



def add_cumulative_column(input_df, input_type, llm_models, vector_retrieval, llm_orchestration):
    #input_category = ['llm_models', 'vector_retrieval', 'llm_orchestration']
    input_category = ['LLM models', 'Vector retrieval', 'LLM orchestration']

    df_list = []
    for category in input_category:
      #llm_models = ['openai', 'anthropic', 'huggingface_hub', 'cohere', 'llama_cpp', 'pyllamacpp', 'diffusers', 'transformers']
      #vector_retrieval = ['pgvector', 'faiss', 'pinecone', 'chromadb', 'weaviate', 'qdrant_client']
      #llm_orchestration = ['langchain', 'llama_index', 'nomic', 'torch', 'tensorflow']
    
      input_df['LLM_CATEGORY'] = 'NA'
      input_df.loc[input_df['LLM_MODEL'].isin(llm_models), 'LLM_CATEGORY'] = 'LLM models' # llm_models
      input_df.loc[input_df['LLM_MODEL'].isin(vector_retrieval), 'LLM_CATEGORY'] = 'Vector retrieval' # vector_retrieval
      input_df.loc[input_df['LLM_MODEL'].isin(llm_orchestration), 'LLM_CATEGORY'] = 'LLM orchestration' # llm_orchestration
    
      df = input_df.loc[input_df['LLM_CATEGORY'] != 'NA']
      df0 = df.loc[df['LLM_CATEGORY'] == category]
    
      # Add a cumulative app count column
      #cumulative_list = []
      subdomains_list = []
      for x in df0['WEEK_START'].unique():
        subdomains_count = df0[df0.WEEK_START.isin([x])][input_type].nunique()
        #subdomains_cumulative_count = df0[df0.WEEK_START.isin(df0['WEEK_START'].unique()[:df0['WEEK_START'].unique().tolist().index(x) + 1])][input_type].nunique()
        subdomains_list.append(subdomains_count)
        #cumulative_list.append(subdomains_cumulative_count)

      # 'CUMULATIVE_WEEKLY_APP_COUNT': cumulative_list
      df1 = pd.DataFrame({'WEEK_START': df0['WEEK_START'].unique(), 'WEEKLY_APP_COUNT': subdomains_list})
      df1['LLM_CATEGORY'] = category
      df_list.append(df1)
    df2 = pd.concat(df_list, axis=0)
    return df2



def add_cumulative_column_proprietary_opensource(input_df, input_type, proprietary_models, opensource_models, input_tool_type):
    # input_category = ['proprietary_models', 'opensource_models']
    input_category = [f'Proprietary {input_tool_type}', f'Open source {input_tool_type}']

    df_list = []
    for category in input_category:

      # Prepare proprietary vs open source data
      input_df['MODEL_TYPE'] = 'NA'
      input_df.loc[input_df['LLM_MODEL'].isin(proprietary_models), 'MODEL_TYPE'] = f'Proprietary {input_tool_type}' # proprietary_models
      input_df.loc[input_df['LLM_MODEL'].isin(opensource_models), 'MODEL_TYPE'] = f'Open source {input_tool_type}' # opensource_models
    
      df = input_df.loc[input_df['MODEL_TYPE'] != 'NA']
      df0 = df.loc[df['MODEL_TYPE'] == category]
    
      # Add a cumulative app count column
      #cumulative_list = []
      subdomains_list = []
      for x in df0['WEEK_START'].unique():
        subdomains_count = df0[df0.WEEK_START.isin([x])][input_type].nunique()
        #subdomains_cumulative_count = df0[df0.WEEK_START.isin(df0['WEEK_START'].unique()[:df0['WEEK_START'].unique().tolist().index(x) + 1])][input_type].nunique()
        subdomains_list.append(subdomains_count)
        #cumulative_list.append(subdomains_cumulative_count)

      # 'CUMULATIVE_WEEKLY_APP_COUNT': cumulative_list,
      df1 = pd.DataFrame({'WEEK_START': df0['WEEK_START'].unique(), 'WEEKLY_APP_COUNT': subdomains_list, 'WEEK_OVER_WEEK_APP_GROWTH': (pd.Series(subdomains_list).pct_change()*100).fillna(0)})
      df1['MODEL_TYPE'] = category
      df_list.append(df1)
    df2 = pd.concat(df_list, axis=0)
    return df2
# add_cumulative_column_proprietary_opensource(df, 'SUBDOMAIN', proprietary_models, opensource_models)


# Prepare data for LLM proprietary vs open source data
def prepare_llm_data(input_df, input_type, proprietary_models, opensource_models, input_tool_type):
    # input_category = ['proprietary_models', 'opensource_models']
    input_category = [f'Proprietary {input_tool_type}', f'Open source {input_tool_type}']

    df_list = []
    for category in input_category:

      # Prepare proprietary vs open source data
      input_df['MODEL_TYPE'] = 'NA'
      input_df.loc[input_df['LLM_MODEL'].isin(proprietary_models), 'MODEL_TYPE'] = f'Proprietary {input_tool_type}' # proprietary_models
      input_df.loc[input_df['LLM_MODEL'].isin(opensource_models), 'MODEL_TYPE'] = f'Open source {input_tool_type}' # opensource_models
    
      df = input_df.loc[input_df['MODEL_TYPE'] != 'NA']
      df0 = df.loc[df['MODEL_TYPE'] == category]

      df_list.append(df0)
    df2 = pd.concat(df_list, axis=0)
    return df2



# 
def prepare_data_for_trends_plot(input_df, tool_selection):
  df_list = []

  for i in tool_selection:
      df0 = input_df[input_df.LLM_MODEL == i]
      df1 = pd.DataFrame({'WEEK_START': df0['WEEK_START'], 'WEEKLY_APP_COUNT': df0['WEEKLY_APP_COUNT'], 'CUMULATIVE_WEEKLY_APP_COUNT': df0['WEEKLY_APP_COUNT'].cumsum(), 'WEEK_OVER_WEEK_APP_GROWTH': df0['WEEK_OVER_WEEK_APP_GROWTH'] })
      df1['LLM_MODEL'] = i
      df_list.append(df1)
  df2 = pd.concat(df_list, axis=0)

  return df2


def prepare_data_for_trends_opensource_plot(input_df, all_llm_models, proprietary_models_options, opensource_models_options):    

    df_llm_models = input_df[input_df.LLM_MODEL.isin(all_llm_models)]

    # df_llm_models['MODEL_TYPE'] = 'open source'
    df_llm_models = df_llm_models.assign(MODEL_TYPE='open source')
    df_llm_models.loc[df_llm_models['LLM_MODEL'].isin(proprietary_models_options), 'MODEL_TYPE'] = 'proprietary'
    
    df_llm_models_agg = df_llm_models.groupby(['MODEL_TYPE', 'WEEK_START']).sum()
    df_llm_model_type = df_llm_models_agg.add_suffix('').reset_index()

    
    #####
    #df_llm_models = input_df[input_df.LLM_MODEL.isin(all_llm_models)]
    
    ## df_llm_models['MODEL_TYPE'] = 'open source'
    #df_llm_models = df_llm_models.assign(MODEL_TYPE='open source')
    #df_llm_models.loc[df_llm_models['LLM_MODEL'].isin(proprietary_models_options), 'MODEL_TYPE'] = 'proprietary'
        
    #df_llm_models_agg = df_llm_models.groupby(['MODEL_TYPE', 'WEEK_START']).sum()
    #df0 = df_llm_models_agg.add_suffix('').reset_index()
        
    #df0['CUMULATIVE_WEEKLY_APP_COUNT'] = df0['WEEKLY_APP_COUNT'].cumsum()
    #####
 
    # Add a cumulative app count column
    #cumulative_list = []
    #subdomains_list = []
    #for x in df0['WEEK_START'].unique():
    #    subdomains_count = df0[df0.WEEK_START.isin([x])]['SUBDOMAIN'].nunique()
    #    subdomains_cumulative_count = df0[df0.WEEK_START.isin(df0['WEEK_START'].unique()[:df0['WEEK_START'].unique().tolist().index(x) + 1])]['SUBDOMAIN'].nunique()
    #    subdomains_list.append(subdomains_count)
    #    cumulative_list.append(subdomains_cumulative_count)
      
    #    df1 = pd.DataFrame({'WEEK_START': df0['WEEK_START'].unique(), 'WEEKLY_APP_COUNT': subdomains_list, 'CUMULATIVE_WEEKLY_APP_COUNT': cumulative_list})
    #    df1['LLM_CATEGORY'] = category
    #    df_list.append(df1)
    #df2 = pd.concat(df_list, axis=0)
    
    return df_llm_model_type

# prepare_data_for_trends_opensource_plot(df_llm, all_llm_models, proprietary_models_options, opensource_models_options)


# Prepare the data for Usage Trends
def add_cumulative_column_usage_trends(input_df, input_type, models_selections):
  df_list = []
  for model in models_selections:
    df0 = input_df.loc[input_df['LLM_MODEL'] == model]
    # Add a cumulative app count column
    #cumulative_list = []
    subdomains_list = []
    models_list = []
    for x in df0['WEEK_START'].unique():
      subdomains_count = df0[df0.WEEK_START.isin([x])][input_type].nunique()
      #subdomains_cumulative_count = df0[df0.WEEK_START.isin(df0['WEEK_START'].unique()[:df0['WEEK_START'].unique().tolist().index(x) + 1])][input_type].nunique()
      subdomains_list.append(subdomains_count)
      #cumulative_list.append(subdomains_cumulative_count)
      models_list.append(model)

    # 'CUMULATIVE_WEEKLY_APP_COUNT': cumulative_list, 
    df1 = pd.DataFrame({'LLM_MODEL': models_list, 'WEEK_START': df0['WEEK_START'].unique(), 'WEEKLY_APP_COUNT': subdomains_list, 'WEEK_OVER_WEEK_APP_GROWTH': (pd.Series(subdomains_list).pct_change()*100).fillna(0) })
    df_list.append(df1)

  df2 = pd.concat(df_list, axis=0)
  return df2
# add_cumulative_column(df, 'SUBDOMAIN', llm_models)


# Prepare weekly chat/single text input app data
@st.cache_data
def load_weekly_chat_app(input_df):
  most_recent_start_week = input_df['WEEK_START'].max()
  input_df = input_df[input_df['WEEK_START'] < most_recent_start_week]
    
  df_list = []
  for i in ['chat', 'single text input']:
    df = input_df.loc[input_df['APP_TYPE'] == i]
    weekly_app_count = [(df[df.WEEK_START == x]['SUBDOMAIN'].nunique()) for x in df['WEEK_START'].unique()]
    week_list = df['WEEK_START'].unique()
    # df2 = pd.DataFrame({'WEEK_START': week_list, 'WEEKLY_APP_COUNT': weekly_app_count})

    # Add a cumulative app count column
    #cumulative_list = []
    subdomains_list = []
    for x in df['WEEK_START'].unique():
      subdomains_count = df[df.WEEK_START.isin([x])]['SUBDOMAIN'].nunique()
    #  subdomains_cumulative_count = df[df.WEEK_START.isin(df['WEEK_START'].unique()[:df['WEEK_START'].unique().tolist().index(x) + 1])]['SUBDOMAIN'].nunique()
      subdomains_list.append(subdomains_count)
    #  cumulative_list.append(subdomains_cumulative_count)

    total_weekly_app_count = [(input_df[input_df.WEEK_START == x]['SUBDOMAIN'].nunique()) for x in input_df['WEEK_START'].unique()]
    df2 = pd.DataFrame({'WEEK_START': week_list, 'WEEKLY_APP_COUNT': subdomains_list, 'TOTAL_WEEKLY_APP_COUNT': total_weekly_app_count}) # 'CUMULATIVE_WEEKLY_APP_COUNT': cumulative_list,
    df2['WEEKLY_APP_PCT'] = df2['WEEKLY_APP_COUNT']/df2['TOTAL_WEEKLY_APP_COUNT']
    df2['WEEKLY_APP_PCT'] = df2['WEEKLY_APP_PCT'].apply(lambda x: x*100)
    df2['WEEKLY_APP_PCT'] = df2['WEEKLY_APP_PCT'].astype('int')
    df2['APP_TYPE'] = i
    df_list.append(df2)
  df3 = pd.concat(df_list, axis=0)
  return df3

#load_weekly_chat_app(df_weekly_chat_data)


def sort_LLM_tools(input_df, input_column):
    recent_weeks = list(input_df['WEEK_START'].unique())[-4:]
    df_recent_weeks = input_df[input_df['WEEK_START'].isin(recent_weeks)]
    average_weekly_app_count = df_recent_weeks.groupby('LLM_MODEL')[input_column].mean()
    sorted_models = average_weekly_app_count.sort_values(ascending=False)
    return sorted_models.reset_index().iloc[:,0].to_list()

def sort_opensource_tools(input_df, input_column):
    recent_weeks = list(input_df['WEEK_START'].unique())[-4:]
    df_recent_weeks = input_df[input_df['WEEK_START'].isin(recent_weeks)]
    average_weekly_app_count = df_recent_weeks.groupby('MODEL_TYPE')[input_column].mean()
    sorted_models = average_weekly_app_count.sort_values(ascending=False)
    return sorted_models.reset_index().iloc[:,0].to_list()
    
def calculate_weekly_app_count(input_df):
    # Step 1: Calculate WEEKLY_APP_COUNT for each LLM_MODEL
    weekly_app_count_df = input_df.groupby(['LLM_MODEL', 'WEEK_START'])['SUBDOMAIN'].nunique().reset_index()
    weekly_app_count_df.rename(columns={'SUBDOMAIN': 'WEEKLY_APP_COUNT'}, inplace=True)

    # Step 2: Calculate TOTAL_WEEKLY_APP_COUNT for each LLM_MODEL and WEEK_START
    total_weekly_app_count_df = input_df.groupby(['LLM_MODEL', 'WEEK_START'])['SUBDOMAIN'].nunique().groupby('LLM_MODEL').transform('sum').reset_index()
    total_weekly_app_count_df.rename(columns={'SUBDOMAIN': 'TOTAL_WEEKLY_APP_COUNT'}, inplace=True)

    # Step 3: Replace missing values in the TOTAL_WEEKLY_APP_COUNT column with 'NA'
    total_weekly_app_count_df['TOTAL_WEEKLY_APP_COUNT'].fillna('NA', inplace=True)

    # Merge the two DataFrames
    result_df = weekly_app_count_df.merge(total_weekly_app_count_df, on=['LLM_MODEL', 'WEEK_START'], how='right')

    # If you want to replace NaN values in WEEKLY_APP_COUNT with 'NA' as well:
    result_df['WEEKLY_APP_COUNT'].fillna('NA', inplace=True)
    result_df['WEEKLY_PCT'] = (result_df['WEEKLY_APP_COUNT']/result_df['TOTAL_WEEKLY_APP_COUNT']) * 100
    result_df = result_df.round({'WEEKLY_PCT': 2})

    return result_df

# calculate_weekly_app_count(df_llm)


def redirect_button(text: str= None, url: str, color="#F63366"):
    st.markdown(
    f'''
    <a href="{url}" target="_self">
        <div style="
            text-align: center;
            display: inline-block;
            padding: 0.5em 1em;
            color: #FFFFFF;
            background-color: {color};
            border-radius: 6px;
            text-decoration: none;">
            {text}
        </div>
    </a>
    ''',
    unsafe_allow_html=True)
# redirect_button("Go to Top models", "#top-models")
