#import sys
#from pathlib import Path
#sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Python이 모듈을 찾는 방식 때문입니다.
Python의 모듈 검색 경로 (sys.path):

실행한 스크립트가 있는 디렉토리
PYTHONPATH 환경변수
표준 라이브러리 경로

문제 상황:
scratch_llm/                    <- 여기가 필요
├── token_id_2_3/
│   └── token_id.py
└── context_token_2_4/
    └── tokenizer_app.py        <- 여기서 실행

python context_token_2_4/tokenizer_app.py 실행 시:

sys.path[0] = /Users/wonnho/scratch_llm/context_token_2_4/
token_id_2_3는 상위 폴더에 있어서 찾을 수 없음

해결:
sys.path.insert(0, str(Path(__file__).parent.parent))
# sys.path에 /Users/wonnho/scratch_llm 추가

Java/Spring Boot와 비교:

Java: classpath 설정과 유사
Python: 실행 위치가 중요, 명시적으로 경로 추가 필요

더 나은 방법: 프로젝트를 패키지로 설치

pip install -e .  # setup.py 사용
"""

from .tokenizer import SimpleTokenizerV2
from token_id_2_3.token_id import vocab

tokenizer=SimpleTokenizerV2(vocab)
text1="Would you like a cup of tea"
text2="In the sunlit terraces of the palce?"
text="<|endoftext|> ".join((text1,text2))
#print(text)
print(tokenizer.encode(text))
print('decode to text',tokenizer.decode(tokenizer.encode(text)))