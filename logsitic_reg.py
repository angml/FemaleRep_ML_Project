import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("TMDB_female_representation.csv") 
unneccessary_cols = ['tmdb_id', 'title', 'release_date', 'decade', 'overview', 'original_language', 'genre_names', 'primary_country', 
                     'primary_company', 'language_count', 'lead_gender', 'lead_actor_name', 'lead_actor_age', 'top5_female', 'top5_male', 
                     'pct_female_top5', 'cast_female', 'cast_male', 'cast_total_known','pct_female_cast', 'director_gender', 'director_name', 
                     'director_age', 'num_directors', 'pct_female_writers', 'pct_female_producers','female_writers', 'female_producers', 'vote_count']
clean_df = df.drop(columns = unneccessary_cols)
X = clean_df.drop(columns = ["representation_tier"])
y = clean_df["representation_tier"]

# Encode y
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Log transform skewed features
X['budget'] = np.log1p(X['budget'])
X['revenue'] = np.log1p(X['revenue'])
X['popularity'] = np.log1p(X['popularity'])

# Standardize numeric features
scaler = StandardScaler()
X[['popularity', 'budget', 'revenue', 'runtime']] = scaler.fit_transform(X[['popularity', 'budget', 'revenue', 'runtime']])

X['vote_average'] = X['vote_average'] / 10

# Split, train
X_train, X_test, y_train, y_test = train_test_split( X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))





