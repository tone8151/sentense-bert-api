from huggingface_hub import snapshot_download
from transformers import BertJapaneseTokenizer, BertModel

# download_path = snapshot_download(repo_id="sonoisa/sentence-bert-base-ja-mean-tokens-v2")

download_path = 'sonoisa/sentence-bert-base-ja-mean-tokens-v2'
dir_name = './model'

# パイプラインの準備
print("downloading...")
model = BertModel.from_pretrained(download_path)
tokenizer = BertJapaneseTokenizer.from_pretrained(download_path) 

#モデルのsave
print("saving...")
model.save_pretrained(dir_name)
tokenizer.save_pretrained(dir_name)