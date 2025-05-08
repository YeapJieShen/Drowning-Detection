import streamlit as st

def init():
    st.session_state["cleanup_function"] = cleanup

def cleanup():
    pass

def home():
    st.title("🏊‍♂️ Drowning Detection System")
    
    st.markdown("""
    ### 📚 Project Description
    This project is developed under the **BMCS2133 Image Processing** course as part of the final semester project.

    It aims to detect and classify human activities in swimming pools, specifically focusing on identifying:
    - **Swimming**
    - **Treading Water**
    - **Drowning**

    Using a combination of **YOLOv11** for human detection and a custom **CNN classifier**, we extract ROIs (Regions of Interest) and classify the person's activity for real-time drowning detection.

    ### 👥 Project Members
    - **Yeap Jie Shen**
    - **Gan Yee Jing**
    - **Jerome Subash A/L Joseph**

    From:
    > Bachelor in Data Science (Honours)  
    > Year 3 Semester 2 – **Group 4**

    ### 🙏 Acknowledgements
    Special thanks to **Assoc. Prof. Ts. Dr Tan Chi Wee** for his continuous guidance and support throughout this project.

    ### 🔗 GitHub Repository
    View our project code on [GitHub](https://github.com/YeapJieShen/Drowning-Detection)
    """)

if __name__ == "__main__":
    init()
    home()