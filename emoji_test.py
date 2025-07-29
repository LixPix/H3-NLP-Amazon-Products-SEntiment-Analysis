import streamlit as st

st.title("Emoji Test Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😊 Positive", key="test_positive"):
        st.write("Positive clicked!")

with col2:
    if st.button("😞 Negative", key="test_negative"):
        st.write("Negative clicked!")

with col3:
    if st.button("😐 Neutral", key="test_neutral"):
        st.write("Neutral clicked!")

st.write("If you see question marks in diamonds instead of emojis, there's an encoding issue.")
