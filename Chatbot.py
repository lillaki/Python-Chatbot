import random
import pandas as pd
import nltk
import re  # to do regular expression matching operations
import database  # Add funny questions and answers to the database
from nltk.stem import wordnet  # to perform lemmitization
from sklearn.feature_extraction.text import TfidfVectorizer  # to perform tfidf
from nltk import pos_tag  # for parts of speech
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

df = pd.read_excel('dialog_talk_agent.xlsx')
# df.shape[0]  # returns the number of rows in dataset
df.ffill(axis=0, inplace=True)  # fills the null value with the previous value
# print(df)
df = database.add_answers(df)

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "hi there", "hey")
GREETING_RESPONSES = ["hi", "hey", "Sup?", "hi there", "hello", "I'm glad you are talking to me", "Sup?"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# function that performs text normalization steps
def text_normalization(sentence):
    sentence = str(sentence).lower()  # text to lower case
    spl_char_text = re.sub(r'[^ a-z]', '', sentence)  # removing special characters
    tokens = nltk.word_tokenize(spl_char_text)  # word tokenizing
    lema = wordnet.WordNetLemmatizer()  # initializing lemmatization
    tags_list = pos_tag(tokens, tagset=None)  # parts of speech
    lema_words = []  # empty list
    for token, pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val = 'v'
        elif pos_token.startswith('J'):  # Adjective
            pos_val = 'a'
        elif pos_token.startswith('R'):  # Adverb
            pos_val = 'r'
        else:
            pos_val = 'n'  # Noun
        lema_token = lema.lemmatize(token, pos_val)  # performing lemmatization
        lema_words.append(lema_token)  # appending the lemmatized token into a list

    return " ".join(lema_words)  # returns the lemmatized tokens as a sentence


def chat(text):  # defining a function that returns response to query using tf-idf
    tfidf_vectorizer = TfidfVectorizer()
    similarity_list = []
    lemme_sentences = []

    for s in df['Context']:
        lemme_sentences.append(text_normalization(s))

    for i in lemme_sentences:
        z = [text_normalization(text), i]
        tfidf_matrix = tfidf_vectorizer.fit_transform(z)
        cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
        similarity_list.append(cos_similarity[0][1])
        index = lemme_sentences.index(i)

    # print('\nSimilarity list\n', similarity_list)
    max_similarity = [similarity_list.index(max(similarity_list)), max(similarity_list)]
    # print('\nMax similarity\n', max_similarity)

    return df['Text Response'][max_similarity[0]]


"""*************** MAIN **************** """
flag = True

print("Lyrica: My name is Dot. You can chat with me! If you want to exit, type 'See ya'")
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'see ya':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("Lyrica: You are welcome...")
        else:
            if greeting(user_response) is not None:
                print("Chatbot: " + greeting(user_response))
            else:
                print("Chatbot: ", end="")
                print(chat(user_response))
    else:
        flag = False
        print("Chatbot: Bye! Take care")