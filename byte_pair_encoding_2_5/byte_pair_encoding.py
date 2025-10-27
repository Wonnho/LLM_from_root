from importlib.metadata import version
import tiktoken

#print("tiktoken version:",version("tiktoken"))
tokenizer=tiktoken.get_encoding("gpt2")

text=(
   "This is a cryptocurrency ranking dashboard built with Next.js 15.2.4 and React 19.<|endoftext|>"
   " It displays information about the top cryptocurrencies including price, market cap, volume, and 24h change.<|endoftext|>"
   " Currently uses mock data; real-time API integration is planned."
)
integers=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
#print(integers)
strings=tokenizer.decode(integers)
print(strings)
"""
Byte Pair Encoding (BPE):
가장 자주 나오는 문자 쌍을 반복적으로 병합하여 vocabulary를 만드는 토큰화 알고리즘입니다.
동작 원리:
python# 초기 텍스트
"low low low lower"

# 1단계: 문자 단위로 분리
['l', 'o', 'w', ' ', 'l', 'o', 'w', ...]

# 2단계: 가장 빈번한 쌍 찾기 → "lo" (3번)
# 병합: "lo"를 하나의 토큰으로

# 3단계: 다음 빈번한 쌍 → "low" (3번)
# 병합: "low"를 하나의 토큰으로

# 최종 vocabulary
['l', 'o', 'w', 'e', 'r', 'lo', 'low', 'lower']
목적:
1. OOV(Out-of-Vocabulary) 문제 해결
python# 기존 방식 (단어 단위)
vocab = {"running": 1, "walk": 2}
"runner" → ❌ 없음 (OOV)

# BPE
vocab = {"run", "ning", "er", "walk"}
"runner" → ["run", "er"] ✅ 분해 가능
2. 효율적인 vocabulary 크기

너무 작음: 긴 subword → 비효율
너무 큼: 메모리 낭비
BPE: 최적 균형

3. 다국어 처리
python# 한국어+영어 혼용
"안녕하세요hello" → ["안녕", "하세", "요", "hello"]
실제 사용:

GPT-2/3/4: BPE 사용
BERT: WordPiece (BPE 변형)
LLaMA: SentencePiece (BPE 기반)

핵심: 자주 나오는 패턴을 학습하여 모든 단어를 표현 가능한 최소 vocabulary 생성
"""