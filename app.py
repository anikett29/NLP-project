import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')
nltk.download('punkt')

clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))


def clean_resume(text):
  CleanText = re.sub('http\S+\s',' ',text)
  CleanText = re.sub('RT|cc',' ',CleanText)
  CleanText = re.sub('#\S+\s',' ',CleanText)
  CleanText = re.sub('@\S+',' ',CleanText)
  CleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',CleanText)
  CleanText = re.sub(r'[^\x00-\x7f]',' ',CleanText)
  CleanText = re.sub('\s+',' ',CleanText)
  return CleanText

# web app title
def main():
    st.title("ScanRes App")
    uploaded_file = st.file_uploader("Upload your resume", type=['pdf','txt'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode("utf-8")
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode("latin-1")

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer", 
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name)
# Python main
if  __name__ == '__main__':
    main()