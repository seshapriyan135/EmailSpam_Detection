import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

model = pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))

st.title("Email Spam Classification Application")
st.write("Build with Streamlit & Python")
activites=["Classification","About","Algorithm","calculation"]
choices=st.sidebar.selectbox("Select Activities",activites)
if choices=="Classification":
	st.subheader("Classification")
	msg=st.text_input("Enter a text")
	if st.button("Process"):
		print(msg)
		print(type(msg))
		data=[msg]
		print(data)
		vec=cv.transform(data).toarray()
		result=model.predict(vec)
		if result[0]==0:
			st.success("This is Not A Spam Email")
		else:
			st.error("This is A Spam Email")
if choices=="About":
	st.image("img1.jpg")
	st.info("Project Done By")
	st.markdown("Sheshapriyan-RA2011030010013")
	st.markdown("Gokul M K-RA2011030010023")
	st.markdown("Adhin Jibil-RA2011030010031")

	sel=st.selectbox("select",["Dataset"])
	if sel == 'Dataset':
		data=pd.read_csv("spam.csv",encoding='latin-1')
		st.dataframe(data,width=700,height=500)
if choices=="Algorithm":
	st.write("The Multinomial Naive Bayes algorithm")
	st.write("Steps for vectorization")
	st.info('1.Import')
	st.info('2.Instantiate')
	st.info('3.Fit')
	st.info('4.Transform')
	st.info('5.Build Model')
    

if choices=="calculation":
	st.write('This example shows how multinomialNB model works inorder to predict the given text is Ham/Spam')
	st.image("1.png")
	st.image("2.png")			