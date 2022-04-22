import streamlit as st

# Source: https://github.com/dataprofessor/multi-page-app/blob/main/multiapp.py
class Router:
    def __init__(self):
        self.routes = []
    
    def add_route(self, title, function):
        self.routes.append({
            "title": title,
            "function": function,
        })

    def run(self):
        app = st.sidebar.selectbox(
            'Navigation',
            self.routes,
            format_func=lambda app: app['title'])

        app['function']()