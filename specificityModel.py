import json
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class SpecificityCalculator():

    def __init__(self, verbose=False):
        self.verbose = verbose
        with open("default_values.json", "r") as file:
            values = json.load(file)
            self.default_uniqueness_percentiles = values["default_uniqueness"]
            self.words_dict = values["words_dict"]
        np.random.seed(0)
        self.default_lengths_distribution = np.random.normal(loc=60, scale=20, size=300)

    def process_response(self, response):
        tokenized = nltk.word_tokenize(response)
        tokenized = [i for i in tokenized if i.isalpha()]
        tokenized = [i for i in tokenized if i.lower() not in stopwords.words("english")]
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(i.lower()) for i in tokenized]
        return text

    def get_uniqueness_score(self, response):
        if len(response) < 3:
            return
        total_count = 0
        adj_factor = 0
        for word in response:
            if self.words_dict.get(word, 0):
                total_count += self.words_dict[word]
            else:
                adj_factor += 1
        if adj_factor == len(response):
            return
        return max(int(total_count / (len(response) - adj_factor)), 0)

    def get_percentile(self, nums, target):
        smaller_nums = [i for i in nums if target >= i]
        return round(len(smaller_nums) / len(nums), 4)

    def get_uniqueness_percentile(self, response):
        response = self.process_response(response)
        uniqueness_score = self.get_uniqueness_score(response)
        if uniqueness_score:
            uniqueness_percentile = 1 - self.get_percentile(self.default_uniqueness_percentiles, uniqueness_score)
            if self.verbose:
                print(f"Uniqueness percentile: {uniqueness_percentile}")
            return uniqueness_percentile
        return
    
    def get_length_percentile(self, response):
        response = self.process_response(response)
        text = " ".join(response)
        if self.verbose:
            print(f"Length percentile: {self.get_percentile(self.default_lengths_distribution, len(text))}, length: {len(text)}")
        return self.get_percentile(self.default_lengths_distribution, len(text))

    def get_specificity_score(self, response, uniqueness_weight=0.5):
        length_weight = 1 - uniqueness_weight
        uniqueness_percentile = self.get_uniqueness_percentile(response)
        if uniqueness_percentile:
            return round(sum([uniqueness_percentile * uniqueness_weight, self.get_length_percentile(response)*length_weight]), 2)