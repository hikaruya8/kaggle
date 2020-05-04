from nltk.stem import WordNetLemmatizer
from dataloader import load_data

def get_jaccard_sim(str1, str2):
    str1_set = set(str1.split())
    str2_set = set(str2.split())
    co_occurrence = str1_set.intersection(str2_set)
    return float(len(co_occurrence)) / (len(str1_set) + len(str2_set) - len(co_occurrence))

def lemmatize(word):
    lemmatizer = WordNetLemmatizer(word)

if __name__ == '__main__':
    str1 = 'AI is our friend and it has been friendly'
    str2 = 'AI and humans have always been friendly'
    print(get_jaccard_sim(str1, str2))


