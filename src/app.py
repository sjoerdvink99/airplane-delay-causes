import streamlit as st
from router import Router
from pages import introduction, classification, clustering

# Setting the page config
st.set_page_config(page_title="Airline delay causes")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Defining the router
app = Router()

# Adding the routes
app.add_route("Introduction", introduction.app)
app.add_route("Clustering", clustering.app)
app.add_route("Classification", classification.app)

# Running the app
app.run()