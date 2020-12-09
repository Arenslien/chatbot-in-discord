# const.py의 모든 코드는 github의 코드를 가져온 코드입니다.
# 하나의 코드로 된 것을 분리한 코드입니다.
# https://github.com/deepseasw/seq2seq_chatbot/blob/master/Seq2Seq%20Chatbot.ipynb

# 단어 사전을 생성하고나서 첫 인덱스부터 들어갈 태그입니다.
PAD = "<PADDING>"   # 패딩
STA = "<START>"     # 시작
END = "<END>"       # 끝
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3

# 데이터 타입
ENCODER_INPUT  = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2

# 한 문장에서 단어 시퀀스의 최대 개수
max_sequences = 30

# 임베딩 벡터 차원
embedding_dim = 100

# LSTM 히든레이어 차원
lstm_hidden_dim = 128

# 정규 표현식 필터
from re import compile
RE_FILTER = compile("[.,!?\"':;~()]")