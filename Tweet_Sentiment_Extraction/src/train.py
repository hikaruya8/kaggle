


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

if __name__ == '__main__':
    str1 = 'AI is our friend and it has been friendly'
    str2 = 'AI and humans have always been friendly'
    print(get_jaccard_sim(str1, str2))


