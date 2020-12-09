# seq2seq.py의 일부 코드는 github의 코드를 가져온 코드입니다.
# 하나의 코드로 된 것을 분리한 코드입니다.
# 모델을 객체화한 코드 입니다.
# https://github.com/deepseasw/seq2seq_chatbot/blob/master/Seq2Seq%20Chatbot.ipynb


from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import preprocessing
from const import embedding_dim, lstm_hidden_dim
import pickle


# 모델을 객체화한 코드입니다.
# 기존의 코드를 객체로 변환시키는 작업을 했습니다.

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

    # 예측 모델을 정의 하는 부분을 함수화한 부분입니다.
    # 기존의 코드와 거의 똑같습니다.
    def get_predict_model(self):
        #--------------------------------------------
        #  예측 모델 인코더 정의
        #--------------------------------------------

        # 훈련 모델의 인코더 상태를 사용하여 예측 모델 인코더 설정
        self.predict_model = models.Model(self.inputs, self.states)

        return self.predict_model


# 모델을 객체화한 코드입니다.
# 기존의 코드를 객체로 변환시키는 작업을 했습니다.

#--------------------------------------------
# Seq2Seq 훈련 모델의 디코더 정의
#--------------------------------------------
class Decoder:
    def __init__(self, encoder_states, len_of_words):
        # 연결될 인코더 선언
        self.encoder_states = encoder_states

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
            initial_state=self.encoder_states
        )

        # 단어의 개수만큼 노드의 개수를 설정하여 원핫 형식으로 각 단어 인덱스를 출력
        self.dense = layers.Dense(len_of_words, activation='softmax')
        self.outputs = self.dense(self.outputs)

    # 예측 모델을 정의 하는 부분을 함수화한 부분입니다.
    # 기존의 코드와 거의 똑같습니다.
    def get_predict_model(self):
        #--------------------------------------------
        # 예측 모델 디코더 정의
        #--------------------------------------------
        
        # 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행
        # 매번 이전 디코더 상태를 입력으로 받아서 새로 설정
        self.decoder_state_input_h = layers.Input(shape=(lstm_hidden_dim,))
        self.decoder_state_input_c = layers.Input(shape=(lstm_hidden_dim,))
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]    
        
        # 임베딩 레이어
        self.outputs = self.embedding(self.inputs)
        
        # LSTM 레이어
        self.outputs, self.state_h, self.state_c = self.lstm(
            self.outputs,
            initial_state=self.decoder_states_inputs
        )
        
        # 히든 상태와 셀 상태를 하나로 묶음
        self.states = [self.state_h, self.state_c]
        
        # Dense 레이어를 통해 원핫 형식으로 각 단어 인덱스를 출력
        self.outputs = self.dense(self.outputs)
        
        # 예측 모델 디코더 설정
        self.predict_model = models.Model(
            [self.inputs] + self.decoder_states_inputs,
            [self.outputs] + self.states
        )

        return self.predict_model