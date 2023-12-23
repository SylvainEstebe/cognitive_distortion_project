import re

def clean_text(text):
    import re

        """
        Summary: turns text input string into list of cleaned word tokens
        
        Arguments:
            text: str of text
        
        Returns:
            lemmatized_words: str, lemmatized words from original text after cleaning
        """
        stop_words = set(stopwords.words('english'))

        # remove numbers
        clean_text = re.sub(r'[0-9]+', '', text)
        
        # remove punctuation
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        
        # convert everything to lowercase
        clean_text = clean_text.lower()
        
        # tokenize
        wt = WhitespaceTokenizer()
        words = wt.tokenize(clean_text)
        
        # remove stop words
        cleaned_words = []
        for w in words:
            if w not in stop_words:
                cleaned_words.append(w)
                
        # lemmatize words
        wnl = WordNetLemmatizer()
        wnl_lemmatized_tokens = []
        for token in cleaned_words:
            wnl_lemmatized_tokens.append(wnl.lemmatize(token))
        
        lemmatized_words = ' '.join(wnl_lemmatized_tokens)
        
        return lemmatized_words