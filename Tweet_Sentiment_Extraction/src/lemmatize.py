import nltk
from nltk import stem
from nltk.stem import WordNetLemmatizer
from dataloader import load_data

nltk.download('wordnet')

def get_jaccard_sim(str1, str2):
    str1_set = set(str1.split())
    str2_set = set(str2.split())
    lemmatized_str1 = lemmatize_set(str1_set)
    lemmatized_str2 = lemmatize_set(str2_set)
    # import pdb;pdb.set_trace()
    co_occurrence = lemmatized_str1.intersection(lemmatized_str2)
    return float(len(co_occurrence)) / (len(lemmatized_str1) + len(lemmatized_str2) - len(co_occurrence))

# def lemmatizer(word):
#     lemmatizer = WordNetLemmatizer(word)
#     return lemmatizer

def lemmatize_set(words):
    stemmer1 = stem.PorterStemmer()
    stemmer2 = stem.LancasterStemmer()
    stemmer3 = stem.SnowballStemmer(language='english')
    lemmatizer = WordNetLemmatizer()
    return set([lemmatizer.lemmatize(stemmer3.stem(s.lower())) for s in words])

if __name__ == '__main__':
    str1 = 'AI is our friend and it has been friendly'
    str2 = 'AI and humans have always been friendly'
    print(get_jaccard_sim(str1, str2))


