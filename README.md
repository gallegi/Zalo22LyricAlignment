# 1. About the project:
- Basic solution for Zalo AI Challenge 2022, track "Lyric Alignment": https://challenge.zalo.ai/portal/lyric-alignment
- Using a pretrained Vietnamese wav2vec model to predict lyric and corresponding time span of each word
- Main dependencies:
  - transformers
  - huggingface

# 3. How to run:
## 3.0. Data preparation
- Unzip Lyrice Alignment data (train.zip and public_test.zip) inside data/ folder
- The structure should look like this
  - ----- data
  - --------|-- train.zip
  - --------|-- public_test.zip
  
## 3.1. Evaluate on the whole train set:
- Run predict_on_train.ipynb. The notebooks output folder: valid_predictions/250h_pretrained/
- Run compute_score.ipynb. The notebook output validation score
## 3.2. Predict:
- Run predict_on_test.ipynb. The notebooks output folder: submissions/250h_pretrained/
