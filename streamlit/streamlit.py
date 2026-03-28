import streamlit.streamlit as st

st.title("Hello Streamlit!")
st.write("This is a simple Streamlit app.")
st.write("You can use Streamlit to create interactive web applications with Python.")
st.write("the current time is: ", __import__('datetime').datetime.now())

st.write("Here's an example of a simple input form:")
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")