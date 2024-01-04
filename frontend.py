import streamlit as st
from llm_stuff import create_qa_chain

qa_chain = create_qa_chain()
def process_llm_response(user_input):
    llm_response = qa_chain(user_input)
    return llm_response['result']


def main():
    st.title("VIT QA System")
    user_input = st.text_area("Enter Query")

    if st.button("Get response"):
        result = process_llm_response(user_input)
        st.write("Response:")
        # make text bigger
        st.write(f"**{result}**")
main()