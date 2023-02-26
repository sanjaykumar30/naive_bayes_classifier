import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.title("Naive Bayes Classifier")
st.markdown("""
👋 Welcome ! \n
Naive Bayes is a simple technique for constructing classifiers: models that assign class labels
 to problem instances, represented as vectors of feature values, where the class labels are drawn 
 from some finite set.\n
To train the Bayesian model, \n
👉Upload the Dataset \n
👉Set the feature and target variables\n
👉Set the testing dataset ratio\n
👉Download the final predicted output\n

**Note**: The input dataset should not contain any Null values and should not require any transformations\n.
"""
)
if st.button("Get Started"):
    switch_page("Page 1")

