# Import python lib
import streamlit as st
import time
import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise import KNNBaseline

# Import wine dataframes
df_wine_model = pd.read_pickle('./data/df_wine_us_rate.pkl')
df_wine_combi = pd.read_pickle('./data/df_wine_combi.pkl')

# Instantiate the list of wine traits
all_traits = ['almond', 'anise', 'apple', 'apricot', 'baked', 'baking_spices', 'berry', 'black_cherry', 'black_currant', 'black_pepper', 'black_tea', 'blackberry', 'blueberry', 
              'boysenberry', 'bramble', 'bright', 'butter', 'candy', 'caramel', 'cardamom', 'cassis', 'cedar', 'chalk', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean', 'closed',
              'clove', 'cocoa', 'coffee', 'cola', 'complex', 'concentrated', 'cranberry', 'cream', 'crisp', 'dark', 'dark_chocolate', 'dense', 'depth', 'dried_herb', 'dry', 'dust',
              'earth', 'edgy', 'elderberry', 'elegant', 'fennel', 'firm', 'flower', 'forest_floor', 'french_oak', 'fresh', 'fruit', 'full_bodied', 'game', 'grapefruit', 'graphite',
              'green', 'gripping', 'grippy', 'hearty', 'herb', 'honey', 'honeysuckle', 'jam', 'juicy', 'lavender', 'leafy', 'lean', 'leather', 'lemon', 'lemon_peel', 'length', 'licorice',
              'light_bodied', 'lime', 'lush', 'meaty', 'medium_bodied', 'melon', 'milk_chocolate', 'minerality', 'mint', 'nutmeg', 'oak', 'olive', 'orange', 'orange_peel', 'peach',
              'pear', 'pencil_lead', 'pepper', 'pine', 'pineapple', 'plum', 'plush', 'polished', 'pomegranate', 'powerful', 'purple', 'purple_flower', 'raspberry', 'refreshing',
              'restrained', 'rich', 'ripe', 'robust', 'rose', 'round', 'sage', 'salt', 'savory', 'sharp', 'silky', 'smoke', 'smoked_meat', 'smooth', 'soft', 'sparkling', 'spice',
              'steel', 'stone', 'strawberry', 'succulent', 'supple', 'sweet', 'tangy', 'tannin', 'tar', 'tart', 'tea', 'thick', 'thyme', 'tight', 'toast', 'tobacco', 'tropical_fruit',
              'vanilla', 'velvety', 'vibrant', 'violet', 'warm', 'weight', 'wet_rocks', 'white', 'white_pepper', 'wood']

#---------------------------------------------------------------------------------------------------------

# Function to instantiate the model & return the est recsys scores
def recommend_scores():
    
    # Instantiate reader & data for surprise
    reader = Reader(rating_scale=(88, 100))
    data = Dataset.load_from_df(df_wine_model, reader)
    
    # Instantiate recsys model
    sim_options={'name':'cosine'}
    model = KNNBaseline(k=35, min_k=1, sim_options=sim_options, verbose=False)

    # Train & fit the data into model
    train=data.build_full_trainset()
    model.fit(train)

    # Start the model to compute the best estimate match score on wine list
    recommend_list = []
    user_wines = df_wine_model[df_wine_model.taster_name == 'mockuser']['title'].unique()
    not_user_wines = []
    
    for wine in df_wine_model['title'].unique():
        if wine not in user_wines:
            not_user_wines.append(wine)

    for wine in not_user_wines:
        wine_compatibility = []
        prediction = model.predict(uid='mockuser', iid=wine)
        wine_compatibility.append(prediction.iid)
        wine_compatibility.append(prediction.est)
        recommend_list.append(wine_compatibility)
        
    result_df = pd.DataFrame(recommend_list, columns = ['title', 'est_match_pts'])
    
    return result_df

# Function for background image
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        
        [data-testid="stAppViewContainer"] {{
        background-image: url("https://images.pexels.com/photos/391213/pexels-photo-391213.jpeg");
        background-attachment: fixed;
        background-size: cover      
        }}
        
        [data-testid="stVerticalBlock"] {{
        background-color: rgba(255,255,255,0.5)
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )

#----------------------------------------------------------------------------------------------------------

st.title("Which wine should I get?")
st.write("By Lee Wan Xian")
st.write("[GitHub](https://github.com/leewanxian) | [LinkedIn](https://www.linkedin.com/in/wanxianlee)")
st.write("You can type the wine traits that you want in the dropdown list below")
add_bg_from_url()

select_temptrait = st.multiselect(label = " ", options = all_traits, label_visibility = "collapsed")

if st.button('Show me the wines!'):
    with st.spinner('Should you have some wine now?'):
        
        time.sleep(2)
        # Instantiate selected wine traits
        if len(select_temptrait) == 0:
            selected_traits = all_traits
        else:
            selected_traits = select_temptrait

        # Run recommender model
        recommend_df = recommend_scores()
    
        # Instantiate traits filter
        trait_filter = ['title']

        # Add on any traits selected by user
        trait_filter.extend(selected_traits)

        # Create dataframe for wine name and traits
        df_temp_traits = df_wine_combi.drop(columns=['taster_name', 'points', 'variety', 'designation', 'winery', 'country', 'province', 'region_1', 'region_2', 'price', 'description',
                                                     'desc_wd_count', 'traits'])

        # Code to start filtering out wines with either one of the selected traits
        df_temp_traits = df_temp_traits[trait_filter]
        df_temp_traits['sum'] = df_temp_traits.sum(axis=1, numeric_only=True)
        df_temp_traits = df_temp_traits[df_temp_traits['sum'] != 0]

        # Merge the selected wines traits with recommend scores
        df_selectrec_temp = df_temp_traits.merge(recommend_df, on='title', how='left')

        # Merge the selected wines with recommendations with df on details
        df_selectrec_detail = df_selectrec_temp.merge(df_wine_combi, on='title', how='left')
        df_selectrec_detail.drop_duplicates(inplace=True)

        # Pull out the top 10 recommendations (raw)
        df_rec_raw = df_selectrec_detail.sort_values('est_match_pts', ascending=False).head(10)
        
        # Prepare the display for the top 10 recommendations
        df_rec_final = df_rec_raw[['title', 'points', 'price', 'variety', 'country', 'province', 'winery', 'description', 'traits']].reset_index(drop=True)
        df_rec_final.index = df_rec_final.index + 1
        df_rec_final['traits']=df_rec_final['traits'].str.replace(" ", " | ")
        df_rec_final.rename(columns={'title':'Name',
                                     'country':'Country',
                                     'province':'State/Province',
                                     'variety':'Type',
                                     'winery':'Winery',
                                     'points':'Rating (Out of 100)',
                                     'price':'Price',
                                     'description':'Review',
                                     'traits':'Key Traits'}, inplace=True)
        st.balloons()
        st.dataframe(df_rec_final.style.format({"Price": "${:,.2f}"}))
