import streamlit as st
import os

pages = {
    "Main Menu": [
        st.Page(os.path.join(os.path.dirname(__file__),
                'pages', 'home.py'), title="Home")
    ],
    "Your account": [
        st.Page(os.path.join(os.path.dirname(__file__),
                'pages', 'a.py'), title="Create your account")
    ],
}


def main():
    pg = st.navigation(pages, position="sidebar")
    pg.run()


if __name__ == "__main__":
    main()
