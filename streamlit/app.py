import streamlit as st
import sys
import os
from dotenv import load_dotenv

# streamlit run streamlit\app.py

pages = {
    "Main Menu": [
        st.Page(os.path.join('pages', 'home.py'), title="Home")
    ],
    "Detection Sites": [
        st.Page(os.path.join('pages', 'live.py'), title="Live Detection"),
        st.Page(os.path.join('pages', 'video.py'), title="Video Detection"),
    ],
}


def init():
    load_dotenv()
    sys.path.append(os.getenv('SRC_DIR'))


def main():
    pg = st.navigation(pages, position="sidebar")
    pg.run()


if __name__ == "__main__":
    init()
    main()
