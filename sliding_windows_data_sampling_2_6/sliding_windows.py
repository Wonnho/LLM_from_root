#from byte_pair_encoding_2_5 import byte_pair_encoding
# byte_pair_encoding_2_5 폴더에서 byte_pair_encoding을 임포트
import tiktoken

# BytePairEncoding을 활용해서 tokenizer 인스턴스 생성
tokenizer = tiktoken.get_encoding("gpt2")
"""
tiktoken 라이브러리는 실제로 Byte Pair Encoding(BPE) 기반 토크나이저이며,
코드 흐름과 해설이 GPT-2 및 많은 LLM들이 사용하는 BPE 원리와 정확히 일치합니다.

코드와 동작 방식 요약
tiktoken.get_encoding("gpt2")를 통해 BPE 기반 인코더를 가져옵니다.

encode 메서드는 텍스트를 정수 토큰 시퀀스로 변환합니다(실제 BPE로).

decode는 다시 그 토큰 목록을 문자열로 복원합니다.

BPE 핵심 원리와 목적
코멘트로 주신 설명(가장 빈번한 문자 쌍을 반복 병합, Out-of-Vocabulary 해결, 효율적인 vocabulary, 다국어 처리 등)은 BPE의 주요 특성과 완전히 일치합니다.
GPT-2, tiktoken, BERT(WordPiece 변형), LLaMA(SentencePiece 기반) 등 LLM 토크나이저의 실제 동작구조와 이론적 근거 모두 올바르게 기술되어 있습니다.

"""
with open("the-verdict.txt","r",encoding="utf-8") as file:
   raw_text=file.read()
   #print("✅ raw_text preview:", raw_text[:200])

enc_text=tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample=enc_text[50:]

context_size=4
x=enc_sample[:context_size]
y=enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:     {y}")


for k in range(1,context_size+1):
   context=enc_sample[:k]
   desired=enc_sample[k]
   print(context,"====>",desired)
   