import pickle
with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

print(model)
# print(preprocessor.named_transformers_)

