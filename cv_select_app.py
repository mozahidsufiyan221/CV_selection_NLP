# Resume Phrase Matcher code
# importing all required libraries
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io, zipfile, sys
import spacy
import PyPDF2
import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher

# Function to read resumes from the folder one by one
mypath = 'C:\\a_work\\a_iot\\CV_selection_NLP\\application_analyst_SP_Rec'  # enter your path here where you saved the resumes
onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

# PDF miner function call
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password,
                                  caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text
# PDF iner end here

def pdfextract(file):
    fileReader = PyPDF2.PdfFileReader(open(file, 'rb'))
    countpage = fileReader.getNumPages()
    count = 0
    text = []
    while count < countpage:
        pageObj = fileReader.getPage(count)
        count += 1
        t = pageObj.extractText()
        text.append(t)
        print('*******print the text******', text)
    return text

# function to read resume ends


# function that does phrase matching and builds a candidate profile
def create_profile(file):
    # text = pdfextract(file)  #pdf convert from PyPDF2
    text = convert_pdf_to_txt(file) #pdf convert from pdfMiner
    # text = get_tika_data(file) #pdf convert from pdfMiner
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()
    # print(text)
    # below is the csv where we have all the keywords, you can customize your own
    keyword_dict = pd.read_csv('C:\\a_work\\a_iot\\CV_selection_NLP\\skill_set_copy.csv')
    stats_words = [nlp(text) for text in keyword_dict['PM-BI'].dropna(axis=0)]
    NLP_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis=0)]
    ML_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis=0)]
    DL_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis=0)]
    R_words = [nlp(text) for text in keyword_dict['R Language'].dropna(axis=0)]
    python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis=0)]
    Data_Engineering_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis=0)]


    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('PM-BI', None, *stats_words)
    matcher.add('NLP', None, *NLP_words)
    matcher.add('MachineLearning', None, *ML_words)
    matcher.add('DeepLearning', None, *DL_words)
    matcher.add('R', None, *R_words)
    matcher.add('Python', None, *python_words)
    matcher.add('DataEngg', None, *Data_Engineering_words)
    doc = nlp(text)

    d = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start: end]  # get the matched slice of the doc
        d.append((rule_id, span.text))
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i, j in Counter(d).items())

    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords), names=['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ', 1).tolist(), columns=['Subject', 'Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(', 1).tolist(), columns=['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'], df2['Keyword'], df2['Count']], axis=1)
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))

    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]

    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2), names=['Candidate Name'])
    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis=1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace=True)

    return (dataf)


# function ends

# code to execute/call the above functions

final_database = pd.DataFrame()
i = 0
while i < len(onlyfiles):
    file = onlyfiles[i]
    dat = create_profile(file)
    # print(dat)
    final_database = final_database.append(dat)
    i += 1
    # print(final_database)
print(final_database)
sample3=final_database.to_csv('CVdatabase.csv')
# code to count words under each category and visulaize it through Matplotlib

final_database2 = final_database['Keyword'].groupby(
    [final_database['Candidate Name'], final_database['Subject']]).count().unstack()
final_database2.reset_index(inplace=True)
final_database2.fillna(0, inplace=True)
new_data = final_database2.iloc[:, 1:]
new_data.index = final_database2['Candidate Name']


# execute the below line if you want to see the candidate profile in a csv format
sample2=new_data.to_csv('sample.csv')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})
ax = new_data.plot.barh(title="Resume keywords by category", legend=False, figsize=(25, 7), stacked=True)
labels = []
for j in new_data.columns:
    for i in new_data.index:
        label = str(j) + ": " + str(new_data.loc[i][j])
        labels.append(label)
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width / 2., y + height / 2., label, ha='center', va='center')
plt.show()
