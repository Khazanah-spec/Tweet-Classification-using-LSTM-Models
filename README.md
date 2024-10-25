# Building a Tweet Classification Model
This model classifies tweets by American Politicans. It predicts the political affiliation of the writer.
## Preprocessing
``` python
url_pattern = re.compile(r'https?://\S+|www\.\S+')

# Punctuation to be removed
english_punctuations = string.punctuation + '’' + '"' + '”' + '“' + "–" + "—"

def clean_data(data, colname):
    # Make all text lowercase
    data_holder = data[colname].str.lower()
    tweet = []
    # Remove empty items and split tweet text
    for i in data_holder:
        if type(i) == str: tweet.append(i.split())
    #remove hashtags and links
    for i in range(len(tweet)):
        lst = []
        for j in range(len(tweet[i])):
            if tweet[i][j][0] != "@" and tweet[i][j][0] != '#':
                lst.append(url_pattern.sub("", tweet[i][j]))
        tweet[i] = lst
    # Remove all punctuation and numbers
    for i in range(len(tweet)):
        for j in range(len(tweet[i])):
            for k in english_punctuations:
                tweet[i][j] = tweet[i][j].replace(k, "")
            tweet[i][j] = ''.join([char for char in tweet[i][j] if not char.isdigit()])
    # Remove all stop words
    stop_words = stopwords.words('english')
    for i in range(len(stop_words)):
        for k in english_punctuations:
            stop_words[i] = stop_words[i].replace(k, "")
    for i in range(len(tweet)):
        tweet[i] = [j for j in tweet[i] if j not in stop_words]
    # Apply stemming and lemmatizer
    st = nltk.PorterStemmer()
    lm = nltk.WordNetLemmatizer()
    for i in range(len(tweet)):
        tweet[i] = [st.stem(word) for word in tweet[i]]
        tweet[i] = [lm.lemmatize(word) for word in tweet[i]]
    # Remove ''
    for i in range(len(tweet)):
        tweet[i] = [j for j in tweet[i] if j not in stop_words]
        tweet[i] = [j for j in tweet[i] if (j != '' and j not in english_punctuations)]
    return [' '.join(t) for t in tweet]
```
