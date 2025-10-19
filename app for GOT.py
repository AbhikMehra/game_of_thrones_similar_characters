import streamlit as st
import requests
import pandas as pd
import numpy as np

# ----------------------------------------------------
# 1️⃣ Fetch Game of Thrones character data from API
# ----------------------------------------------------
api_data = requests.get("https://thronesapi.com/api/v2/Characters").json()

# Convert API data to DataFrame
df = pd.DataFrame(api_data)

# Keep relevant columns only (you can add more if needed)
# fullName: character name
# imageUrl: image link
# title: character title
# family: family name
df = df[['id', 'fullName', 'title', 'family', 'imageUrl']]

# Rename for simplicity
df.rename(columns={'fullName': 'character'}, inplace=True)

# ----------------------------------------------------
# 2️⃣ Basic cleaning and transformations
# ----------------------------------------------------
df['character'] = df['character'].replace({
    'Jaime Lannister': 'Jamie Lannister',
    'Lord Varys': 'Varys',
    'Bronn': 'Lord Bronn',
    'Sandor Clegane': 'The Hound',
    'Robb Stark': 'Rob Stark'
})

# For demonstration, we’ll generate random “x” and “y” features
# (In your real project, these might come from embeddings or PCA)
np.random.seed(42)
df['x'] = np.random.rand(len(df))
df['y'] = np.random.rand(len(df))

# Limit to first 25 characters (optional)
df = df.head(25)

# ----------------------------------------------------
# 3️⃣ Streamlit App UI
# ----------------------------------------------------
st.title("🧙‍♂️ Game of Thrones Personality Matcher")

characters = df['character'].values
selected_character = st.selectbox("Select a character", characters)

# ----------------------------------------------------
# 4️⃣ Helper: Fetch image
# ----------------------------------------------------
def fetch_image(name):
    row = df[df['character'] == name]
    if not row.empty:
        return row.iloc[0]['imageUrl']
    return None

# ----------------------------------------------------
# 5️⃣ Compute similarity based on Euclidean distance
# ----------------------------------------------------
character_id = np.where(df['character'].values == selected_character)[0][0]
x = df[['x', 'y']].values

distances = []
for i in range(len(x)):
    distances.append(np.linalg.norm(x[character_id] - x[i]))

# Find closest match (excluding self)
recommended_id = sorted(list(enumerate(distances)), key=lambda x: x[1])[1][0]
recommended_character = df['character'].values[recommended_id]

# ----------------------------------------------------
# 6️⃣ Display Results
# ----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(selected_character)
    st.image(fetch_image(selected_character), use_container_width=True)

with col2:
    st.subheader(f"Similar: {recommended_character}")
    st.image(fetch_image(recommended_character), use_container_width=True)
