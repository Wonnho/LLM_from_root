import tiktoken
from .data_loader_2_5 import GPTDatasetV1 # " . => 현재 패키지 내부의 data_loader_2_5 모듈에서 가져와라"
from torch.utils.data import Dataset,DataLoader

def create_dataloader_v1(txt,batch_size=4,man_length=256,stride=128,
                            shuffle=True,drop_last=True,num_workers=0):
     tokenizer=tiktoken.get_encoding("gpt2") 
     dataset=GPTDatasetV1(txt,tokenizer,man_length,stride)
     dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
                              drop_last=drop_last,num_workers=num_workers)
     return dataloader

"""
함수 시그니처:

txt: 학습할 원본 텍스트
batch_size=4: 배치당 샘플 수
man_length=256: 시퀀스 최대 길이 (오타: max_length)
stride=128: 슬라이딩 윈도우 이동 간격
shuffle=True: 에포크마다 데이터 섞기
drop_last=True: 마지막 불완전 배치 버리기
num_workers=0: 데이터 로딩 병렬 프로세스 수


Step 1: 토크나이저 로드
pythontokenizer = tiktoken.get_encoding("gpt-2")

OpenAI의 GPT-2 BPE 토크나이저 가져오기
어휘 크기: 50,257개 토큰


Step 2: Dataset 생성
pythondataset = GPTDatasetV1(txt, tokenizer, man_length, stride)

텍스트를 슬라이딩 윈도우로 분할
(input, target) 쌍 생성


Step 3: DataLoader 생성
pythondataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       drop_last=drop_last, num_workers=num_workers)

배치 단위로 데이터 제공
shuffle=True: 학습 시 overfitting 방지
drop_last=True: 배치 크기 일관성 유지


반환: 학습 루프에서 바로 사용 가능한 DataLoader재시도
"""

with open("the-verdict.txt","r",encoding="utf-8") as file:
     raw_text=file.read()
dataloader=create_dataloader_v1(raw_text,batch_size=1,man_length=4,stride=1,shuffle=False)

data_iter=iter(dataloader)
first_batch=next(data_iter)
print(first_batch)

second_batch=next(data_iter)
print(second_batch)

print('====batch size increases 1 to 8====')
dataloader=create_dataloader_v1(raw_text,batch_size=8,man_length=4,stride=4,shuffle=False)

data_iter=iter(dataloader)
inputs,targets=next(data_iter)
print("input:\n",inputs)
print("target:\n",targets)

"""
why increase stride=1 to 4 diminishes overfitting?

Stride와 Overfitting 관계:
Stride = 1 (높은 중복):
텍스트: [A, B, C, D, E, F, G, H]
max_length = 4, stride = 1

Window 1: [A, B, C, D]
Window 2: [B, C, D, E]  ← A,B,C,D 중 B,C,D 재사용
Window 3: [C, D, E, F]  ← B,C,D,E 중 C,D,E 재사용
Window 4: [D, E, F, G]  ← C,D,E,F 중 D,E,F 재사용
Window 5: [E, F, G, H]

중복률: 75% (4개 중 3개 중복)
```

**Stride = 4 (중복 없음)**:
```
텍스트: [A, B, C, D, E, F, G, H]
max_length = 4, stride = 4

Window 1: [A, B, C, D]
Window 2: [E, F, G, H]  ← 완전히 새로운 데이터

중복률: 0%
Overfitting 감소 이유:
Stride    중복 문제 효과
1    높음 모델이 같은 패턴 반복 학습 → 암기  Overfitting
4    없음 모델이 독립적인 패턴 학습 → 일반화 Generalization
결론: Stride ↑ → 중복 ↓ → 데이터 다양성 ↑ → Overfitting ↓
Trade-off: Stride가 너무 크면 학습 데이터가 줄어듭니다 (8개 → 2개)

"""
