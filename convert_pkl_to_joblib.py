import os
import pickle
import joblib

FOLDER_PATH = "."   # current directory (change if needed)

for file in os.listdir(FOLDER_PATH):
    if file.endswith(".pkl"):
        pkl_path = os.path.join(FOLDER_PATH, file)
        joblib_path = pkl_path.replace(".pkl", ".joblib")

        print(f"Converting: {file} → {os.path.basename(joblib_path)}")

        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        joblib.dump(obj, joblib_path, compress=3)

print("✅ All PKL files converted to JOBLIB")