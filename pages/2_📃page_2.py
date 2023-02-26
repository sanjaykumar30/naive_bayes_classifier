import streamlit as st
import pandas as pd
from sklearn.naive_bayes import CategoricalNB  
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn 
from sklearn.metrics import confusion_matrix



@st.cache_data
def convert_df_to_csv(df):
  
  return df.to_csv().encode('utf-8')

def cm(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sn.heatmap(cm, annot=True,cmap='Blues',cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    st.pyplot(plt)

if __name__=='__main__':
    
    st.title("Naive Bayes Classifier")
    df=st.session_state.df # Accessing dataframe from previous page 

    naive_bayes=CategoricalNB(force_alpha=True)

    #Coverting string categorical to numerical categorical value
    encoder=OrdinalEncoder( dtype=int)
    encoder.fit(df)
    columns=df.columns
    df=encoder.transform(df)
    df=pd.DataFrame(df,columns=columns)

    X=df.drop(st.session_state.target_var,axis=1) #Feature variable

    y=df[st.session_state.target_var] #Target variable

    test_ratio=st.sidebar.slider('Percentage of validation dataset',1.0,20.0,1.0,0.5)

    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=test_ratio/100,random_state=2) #Splitting the dataset

    st.write(f"""
    👉Total rows={df.shape[0]} \n
    👉Training dataset rows={x_train.shape[0]} \n
    👉Validation dataset rows={x_test.shape[0]} \n
    """)


    if st.sidebar.button("Train") :

        # Training the model
        naive_bayes.fit(x_train,y_train)
        y_pred=naive_bayes.predict(x_test)

        st.balloons()

        # Tranforming and Joining the datasets
        test_df=x_test.assign(Actual=y_test)
        test_df=pd.DataFrame(encoder.inverse_transform((test_df)),columns=columns)
        pred_df=x_test.assign(Predicted=y_pred)
        pred_df=pd.DataFrame(encoder.inverse_transform((pred_df)),columns=columns)
        final_df=test_df.assign(Predicted=pred_df[st.session_state.target_var])
        
        st.sidebar.success("Tained Successfully")
        st.metric(label="Predicted Accuracy",value=str(naive_bayes.score(x_test,y_test)*100) + " %" )
        # Plotting the confution matrix
        st.subheader("Confusion Matrix")
        cm(y_test,y_pred)

        st.subheader("Output Preview")
        st.dataframe(final_df)

        # Download Button 
        st.sidebar.download_button(
            "📥 Download",
            data=convert_df_to_csv(final_df),
            file_name="Predicted_dataset.csv",  
            mime="text/csv")



    
                
                
        


