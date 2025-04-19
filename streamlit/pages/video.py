import streamlit as st

def init():
    st.session_state["cleanup_function"] = cleanup

def cleanup():
    pass

def video():
    st.title("📷 Video Drowning Detection")

if __name__ == "__main__":
    init()
    video()
