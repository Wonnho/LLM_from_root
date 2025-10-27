import re
file_name='the-verdict.txt'
try:
   with open(file_name,'r') as file:
      content=file.read()
      #token=re.split(r'([,.]|\s)',content) #step 1
      token=re.split(r'([,.:;?_!"()\`]|--|\s)',content) #step2
      token=[item for item in token if item.strip()]
      print(content)
      print(token)
except FileNotFoundError:
   print(f"erroråç{file_name} is Not Found")

print('tokenized text:',len(token))   