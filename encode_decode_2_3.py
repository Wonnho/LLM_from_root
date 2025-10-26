import re

class SimpletokenizerV1:
   def __init__(self,vocab):
      self.str_to_int=vocab
      self.int_to_str={k:s for s, k in vocab.items()}

   def encode(self,text):
      preprocessed=re.split(r'([.,?_!"()\']|--|\s)',text)
      preprocessed=[
         item.strip() for item in preprocessed if item.strip()

      ]
      ids=[self.str_to_int[s] for s in preprocessed]
      return ids
   
   def decode(self,ids):
      text=" ".join([self.int_to_str[k] for k in ids])
      text=re.sub(r'\s+([.,?!"()])','r\1',text)
      return text
   
"""
self의 역할
self = 객체 자신을 가리키는 참조
pythonclass SimpletokenizerV1:
   def __init__(self, vocab):
      self.str_to_int = vocab  # 인스턴스 변수

self 사용 이유
1. 인스턴스 변수로 저장
python# self 사용 ✅
def __init__(self, vocab):
   self.str_to_int = vocab  # 객체에 저장됨

# self 없이 ❌
def __init__(self, vocab):
   str_to_int = vocab  # 지역 변수 - 메서드 종료 시 사라짐
2. 다른 메서드에서 접근 가능
pythonclass SimpletokenizerV1:
   def __init__(self, vocab):
      self.str_to_int = vocab  # 저장
   
   def encode(self, text):
      # 다른 메서드에서 사용 가능
      return self.str_to_int[text]  
   
   def get_vocab_size(self):
      return len(self.str_to_int)  # 여기서도 접근
3. 객체마다 독립적인 데이터
pythonvocab1 = {'a': 0, 'b': 1}
vocab2 = {'x': 0, 'y': 1}

tokenizer1 = SimpletokenizerV1(vocab1)
tokenizer2 = SimpletokenizerV1(vocab2)

# 각 객체가 독립적인 데이터 보유
tokenizer1.str_to_int  # {'a': 0, 'b': 1}
tokenizer2.str_to_int  # {'x': 0, 'y': 1}

비유
python# self는 "내 것"을 의미
class Person:
   def __init__(self, name):
      self.name = name  # "내 이름"
핵심: self로 저장해야 객체가 데이터를 "기억"하고 다른 메서드에서도 사용
"""