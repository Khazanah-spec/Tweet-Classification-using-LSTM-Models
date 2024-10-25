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

## LSTM Model
``` python
# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform input size
max_length = 150
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

# Split the data into training and testing sets, none of the new tweets are included
X_train, X_test, y_train, y_test = train_test_split(X_padded[0:len(t)], y_encoded[0:len(t)], test_size=0.2, random_state=42)

# Define the model
embedding_dim = 128
model = Sequential([
    Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    layers.GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class classification
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])

# Print the model summary
model.summary()

# Set up early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
```
