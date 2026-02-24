# from googletrans import Translator
# import numpy as np 
# import pickle


# # path = "/home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/preprocess/CSL-Daily/train_info.npy"
# path = "/home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/preprocess/phoenix2014-T/train_info.npy"

# with open(path, 'rb') as f:
#     data = np.load(f, allow_pickle=True).item()

# print(data[100])

# exit()

# # Initialize the translator
# translator = Translator()

# # Example sentences
# sentences = [
#     "Deep learning has revolutionized computer vision.",
#     "Sign language recognition is a multimodal problem.",
#     "Machine translation bridges linguistic barriers."
# ]

# # Target languages (ISO 639-1 codes)
# languages = {
#     "es": "Spanish",
#     "fr": "French",
#     "en": "English"
# }

# for i in range(len(data)):
#     sentence = data[i]['original_info'].split("|")[-2]
#     print(f"\nOriginal: {sentence}")
#     for lang_code, lang_name in languages.items():
#         translation = translator.translate(sentence, dest=lang_code)
#         print(f"{lang_code}_text: {translation.text}")

#     print()


from googletrans import Translator
import numpy as np
import os
from tqdm import tqdm

# Input path
path = "/Users/gufran/Developer/Projects/AI/USTM/preprocess/CSL-Daily/train_info.npy"

# Load the data
with open(path, 'rb') as f:
    data = np.load(f, allow_pickle=True).item()

# Initialize the translator
translator = Translator()

# Target languages (ISO 639-1 codes)
languages = {
    "es": "Spanish",
    "fr": "French",
    "en": "English"
}

# Translate and add new keys
for i in tqdm(range(len(data))):
    sentence = data[i]['original_info'].split("|")[-2]
    data[i]['text'] = sentence  # store original text

    tqdm.write(f"\nOriginal: {sentence}")
    for lang_code in languages.keys():
        try:
            translation = translator.translate(sentence, dest=lang_code)
            data[i][f"{lang_code}_text"] = translation.text
            tqdm.write(f"{lang_code}_text: {translation.text}")
        except Exception as e:
            tqdm.write(f"Error translating index {i} to {lang_code}: {e}")
            data[i][f"{lang_code}_text"] = None

    tqdm.write("\n")

# Save the new version
save_path = path.replace(".npy", "_translated.npy")

with open(save_path, 'wb') as f:
    np.save(f, data, allow_pickle=True)

print(f"\n✅ Saved translated data to: {save_path}")
