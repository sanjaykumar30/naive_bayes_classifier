import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page


def main(df):
    import streamlit as st
    import pandas as pd
    
    with st.sidebar:
        
        feature_var=st.multiselect("Select the feature variables",df.columns,key=1)
        target_var=st.selectbox("Select the target variable",df.columns,key=2)
        
        if st.button('Go'):

            if feature_var==[] or target_var==[]:
                st.error("Atleast select one feature or target variable")
            else:
                st.session_state["df"]=df[feature_var+[target_var]]
                st.session_state["target_var"]=target_var
                switch_page("page 2")
                
                
    


if __name__=='__main__':
    
    st.write("""
      # Naive Bayes Classifier
      
       👈 Upload the Excel file 
       """)
    
    try:
        uploaded_file=st.sidebar.file_uploader("Choose a file",type=['xls','xlsx'],help='File size should not exceeds 200 MB')  
        
    except:
        st.sidebar.error('Error While Uploading File')

    if uploaded_file is not None:
            df=pd.read_excel(uploaded_file)
            
            st.subheader('Preview')
            st.dataframe(df)
            main(df)
    
            
            

     