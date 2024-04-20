import pandas as pd
import streamlit as st
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import transformer as trf
import numpy as np

st.image("IPL-2023-Winner-Prediction.jpg", use_column_width=True)

st.title('IPL Win Probability')

team_list  = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

city_list = ['Delhi',
 'Mumbai',
 'Hyderabad',
 'Kolkata',
 'Bangalore',
 'Centurion',
 'Chennai',
 'Chandigarh',
 'Abu Dhabi',
 'Jaipur',
 'Durban',
 'Indore',
 'Mohali',
 'Port Elizabeth',
 'Cape Town',
 'Ahmedabad',
 'Nagpur',
 'Sharjah',
 'Ranchi',
 'Cuttack',
 'Visakhapatnam',
 'Bloemfontein',
 'East London',
 'Bengaluru',
 'Raipur',
 'Kimberley',
 'Pune',
 'Dharamsala',
 'Johannesburg']

model = pickle.load(open('model.pkl','rb'))
# trf = pickle.load(open('ohe.pkl','rb'))


col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team',sorted(team_list))

with col2:
    bowling_team =st.selectbox('Select Bowling Team',sorted(team_list))

selected_city = st.selectbox('Select City',sorted(city_list))

target = st.number_input('Target', value=0, step=1)

col3, col4, col5 = st.columns(3)

with col3:
 score = st.number_input('Score', value=0, step=1)

with col4:
 overs = st.number_input('Overs completed', value=0, step=1)

with col5:
 wicket = st.number_input('Wickets out', value=0, step=1)

if st.button("Predict Probability"):
  runs_left = int(target - score)
  balls_left = int(120 - (overs * 6))
  wickets = int((10 - wicket))

  crr = (score/overs)
  rrr = ((runs_left*6)/balls_left)

  input_df = pd.DataFrame({'batting_team': [batting_team],'bowling_team': [bowling_team],'city':[selected_city], 'runs_left':[runs_left], 'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

  X_test_trf = trf.transform(input_df)

  # st.table((input_df))

  result = np.round(model.predict_proba(X_test_trf)[0],2)

  # st.text(result)


  # st.text(f'{batting_team} Winning Probability : {result[1]}')
  # st.text(f'{bowling_team} Winning Probability : {result[0]}')

  col6, col7 = st.columns(2)
  with col6:
      st.markdown(
          f"<p style='font-family: Arial; font-size: 16px; color: grey;'><b>{batting_team}</b> Winning Probability: <b>{result[1]} %</b></p>",
          unsafe_allow_html=True)
  with col7:
      st.markdown(
      f"<p style='font-family: Arial; font-size: 16px; color: green;'><b>{bowling_team}</b> Winning Probability: <b>{result[0]} %</b></p>",
      unsafe_allow_html=True)

st.image("IPL_Poster.jpg", use_column_width=True)









