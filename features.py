import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
import numpy as np

# Download necessary NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

class PsychoLinguisticExtractor:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def get_features(self, text, use_sentiment=False, use_complexity=False, use_pronouns=False):
        features = []
        
        # 1. Sentiment (Negative Emotion & Intensity)
        if use_sentiment:
            ss = self.sid.polarity_scores(text)
            # We care about negativity and compound score (intensity)
            features.extend([ss['neg'], ss['compound']])

        # 2. Complexity (Cognitive Certainty/patterning)
        if use_complexity:
            # Flesch Reading Ease: Lower usually implies simpler, more aggressive text
            # OR sometimes conspiracy theories are overly convoluted (very low score)
            try:
                score = textstat.flesch_reading_ease(text)
            except:
                score = 50.0 # fallback
            features.append(score / 100.0) # Normalize roughly

        # 3. Pronoun Ratio (Us vs Them / Othering)
        if use_pronouns:
            tokens = text.lower().split()
            n = len(tokens) if len(tokens) > 0 else 1
            
            # Us: we, us, our, ours
            # Them: they, them, their, theirs
            us_count = sum(1 for t in tokens if t in ['we', 'us', 'our', 'ours'])
            them_count = sum(1 for t in tokens if t in ['they', 'them', 'their', 'theirs'])
            
            features.append(us_count / n)
            features.append(them_count / n)

        return np.array(features, dtype=np.float32)

    def get_feature_dim(self, use_sentiment, use_complexity, use_pronouns):
        dim = 0
        if use_sentiment: dim += 2
        if use_complexity: dim += 1
        if use_pronouns: dim += 2
        return dim