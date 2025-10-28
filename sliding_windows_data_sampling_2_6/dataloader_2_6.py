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
