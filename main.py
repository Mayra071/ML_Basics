import pickle
with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

print(preprocessor)
print(preprocessor.named_transformers_)

