import re
file_name='the-verdict.txt'
try:
   with open(file_name,'r') as file:
      content=file.read()
      #token=re.split(r'([,.]|\s)',content) #step 1
      token=re.split(r'([,.:;?_!"()\`]|--|\s)',content) #step2
      token=[item for item in token if item.strip()]
      #print(content)
      #print(token)
except FileNotFoundError:
   print(f"erroråç{file_name} is Not Found")

#print('tokenized text:',len(token))

all_works=sorted(set(token))
vocab_size=len(all_works)
#print(vocab_size)

vocab={token:integer for integer,token in enumerate(all_works)}
# for k,item in enumerate(vocab.items()):
#    print(item)
#    if k>=50:
#       break

# ✅ 특수 토큰 추가
vocab['<|endoftext|>'] = len(vocab)
vocab['<|unk|>'] = len(vocab)