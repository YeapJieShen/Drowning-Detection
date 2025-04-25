import streamlit as st
from pathlib import Path
import sys
##########################################################

if "previous_page" not in st.session_state:
    st.session_state["previous_page"] = None

pages_dir = Path("pages")

pages = {
    "Main Menu": [
        st.Page(pages_dir / "home.py", title="Home")
    ],
    "Detection Sites": [
        st.Page(pages_dir / "live.py", title="Live Detection"),
        st.Page(pages_dir / "video.py", title="Video Detection")
    ]
}

def init():
    if 'app_init' not in st.session_state:
        st.session_state['app_init'] = True

        sys.path.append(str(Path(__file__).parent.parent.absolute()))

def cleanup_page_state(current_page):
    previous_page = st.session_state["previous_page"]
    if previous_page is not None and previous_page != current_page:
        st.session_state['cleanup_function']()

def app():
    pg = st.navigation(pages, position="sidebar")

    cleanup_page_state(pg.title)
    st.session_state["previous_page"] = pg.title
    pg.run()

if __name__ == "__main__":
    init()
    app()