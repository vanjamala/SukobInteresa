# auth.py

import streamlit as st

def authenticate():
    st.write("auth")
    users = st.secrets["users"]

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔐 Login Required")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}!")
                st.rerun()
                return
            else:
                st.error("Invalid username or password")
        st.stop()


def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
