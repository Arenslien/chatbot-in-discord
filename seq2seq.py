from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import preprocessing
from const import embedding_dim, lstm_hidden_dim

#--------------------------------------------
# Seq2Seq 훈련 모델의 인코더 정의
#--------------------------------------------
class Encoder:
    def __init__(self, len_of_words):
        self.len_of_words = len_of_words

        # 입력 문장의 인덱스 시퀀스를 입력으로 받음
        self.inputs = layers.Input(shape=(None,))

        # 임베딩 레이어
        self.outputs = layers.Embedding(len_of_words, embedding_dim)(self.inputs)

        # return_state가 True면 상태값 리턴
        # LSTM은 state_h(hidden state)와 state_c(cell state) 2개의 상태 존재
        self.outputs, self.state_h, self.state_c = layers.LSTM(
            lstm_hidden_dim,
            dropout=0.1,
            recurrent_dropout=0.5,
            return_state=True
        )(self.outputs)

        # 히든상태와 셀 상태를 하나로 묶음
        self.states = [self.state_h, self.state_c]

#--------------------------------------------
# Seq2Seq 훈련 모델의 디코더 정의
#--------------------------------------------
class Decoder:
    def __init__(self, encoder_state, len_of_words):
        # 연결될 인코더 선언
        self.encoder_state = encoder_state

        #목표 문장의 인덱스 시퀀스를 입력으로 받음
        self.inputs = layers.Input(shape=(None,))

        # 임베딩 레이어
        self.embedding = layers.Embedding(len_of_words, embedding_dim)
        self.outputs = self.embedding(self.inputs)

        # 인코더와 달리 return_sequences를 True로 설정하여 모든 타임 스텝 출력값 리턴
        # 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리하기 위함
        self.lstm = layers.LSTM(
            lstm_hidden_dim,
            dropout=0.1,
            recurrent_dropout=0.5,
            return_state=True,
            return_sequences=True
        )

        # initial_state를 인코더의 상태로 초기화
        self.outputs, _, _ = self.lstm(
            self.outputs,
            initial_state=self.encoder_state
        )

        # 단어의 개수만큼 노드의 개수를 설정하여 원핫 형식으로 각 단어 인덱스를 출력
        self.dense = layers.Dense(len_of_words, activation='softmax')
        self.outputs = self.dense(self.outputs)
