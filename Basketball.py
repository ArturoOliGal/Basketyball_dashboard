import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
st.title('NBA player Stats Explorer')
st.markdown("""
This app performs simple webscraping of NBA player stats data!
* **Python libraries:** streamlit, pandas, bs4
* **Data source:** [Basketball-reference](https://www.basketball-reference.com/)
""")
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2024))))

@st.cache_data

def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_"+str(selected_year)+"_per_game.html#per_game_stats"
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw=df.drop(df[df.Age == 'Age'].index)
    raw=raw.fillna(0)
    playerstats=raw.drop(['Rk'],axis=1)
    return playerstats
playerstats=load_data(selected_year)


#sorted_unique_team=sorted(playerstats.Tm.unique())
sorted_unique_team = sorted(playerstats['Team'].astype(str).unique())

selected_team=st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

unique_pos=['C','PF','SF','PG','SG']
selected_pos=st.sidebar.multiselect('Position',unique_pos, unique_pos)

#df_selected_team=playerstats[(playerstats.Tm.isin(selected_team)&(playerstats.Pos.isin(selected_pos)))]
df_selected_team = playerstats[(playerstats['Team'].isin(selected_team)) & (playerstats['Pos'].isin(selected_pos))]


st.header('Display player stats of selected Team')
st.write("Data dimension:"+str(df_selected_team.shape[0])+"rows and"+str(df_selected_team.shape[1])+"columns")
st.dataframe(df_selected_team)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)
if st.button('Intercorraltion Heatmap'):
    st.header('Intercorrational Matrix Heatmap')
    df_selected_team=df_selected_team.drop(['Player','Pos','Team', 'Awards'],axis=1)
    corr=df_selected_team.corr()
    mask=np.zeros_like(corr)
    mask[np.triu_indices_from(mask)]=True
    with sns.axes_style("white"):
        f, ax= plt.subplots(figsize=(7, 5))
        ax= sns.heatmap(corr, mask=mask, square=True)
    st.pyplot(f)


if st.button('Sactterplot'):
    st.header('sactterplot')
    x=df_selected_team['Age']
    y=df_selected_team['PTS']
    plt.scatter(x, y)
    plt.xlabel('Edad')
    plt.ylabel('Puntos')
#    plt.grid(True)  # Agregar cuadrícula
    plt.tight_layout()  # Ajustar el diseño
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()