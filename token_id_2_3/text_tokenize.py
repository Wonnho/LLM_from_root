import urllib.request
url=("https://raw.githubusercontent.com/rickiepark/llm-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")

file_path="the-verdict.txt"
urllib.request.urlretrieve(url,file_path)
with open("the-verdict.txt",'r',encoding="utf-8") as file:
   raw_text=file.read()
print("total words:",len(raw_text))
print(raw_text[:199])

