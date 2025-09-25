# clean_newlines.py
path = "iris_classifier.py"

with open(path, "r", encoding="utf-8") as f:
    lines = f.read().rstrip() + "\n"

with open(path, "w", encoding="utf-8") as f:
    f.write(lines)
