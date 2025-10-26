from encode_decode_2_3 import SimpletokenizerV1
from token_id import vocab

tokenizer=SimpletokenizerV1(vocab)
text="""
It is the last he painted, you know, Mrs. Gisburn said with pardonable price.
"""
ids=tokenizer.encode(text)
print('token',ids)

print('translate to text:',tokenizer.decode(ids))