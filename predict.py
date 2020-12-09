# predict.py의 일부 코드는 github의 코드를 가져온 코드입니다.
# 하나의 코드로 된 것을 분리한 코드입니다.
# 예측 모델을 기반으로 입력된 문장을 출력하기 위한 기능을 담당하는 코드입니다.
# https://github.com/deepseasw/seq2seq_chatbot/blob/master/Seq2Seq%20Chatbot.ipynb

from const import ENCODER_INPUT, STA_INDEX, END_INDEX, max_sequences, lstm_hidden_dim
from preprocess import Preprocessor
from keras import layers
from keras import models
import numpy as np
import pickle

# 전역 변수(객체) 선언
preprocessor = Preprocessor()

# 예측을 위한 입력 생성
def make_predict_input(sentence):
    """
    함수: make_predict_input(sentence)\n
    입력: 문장을 요소로 갖는 리스트\n
    출력: 리스트\n
    목적: 모델의 입력 값을 구하기 위한 함수 => 단어 임베딩 과정
    """
    sentences = []
    sentences.append(sentence)
    sentences = preprocessor.tokenize_ko(sentences)
    input_seq = preprocessor.convert_text_to_index(sentences, get_word_to_index(), ENCODER_INPUT)
    return input_seq

# 텍스트 생성
def generate_text(input_seq):
    """
    함수: generate_text(input_seq)\n
    입력: 문장이 임베딩된 리스트\n
    출력: 모델이 예측한 문장\n
    목적: 임베딩 된 문장을 기반으로 예측 모델의 출력 값을 구하기 위한 함수    
    """

    # 직접 추가한 코드
    # 기존의 코드는 예측 모델을 일회용으로 사용하기에 저장하지 않았습니다.
    # 챗봇 모델은 모델을 계속 저장하여 사용하므로 저장된 모델을 가져오는 코드입니다.
    encoder_model = models.load_model('./model/encoder_model.h5')
    decoder_model = models.load_model('./model/decoder_model.h5')

    # 이 부분부터는 github의 코드입니다.
    # 입력을 인코더에 넣어 마지막 상태 구함
    states = encoder_model.predict(input_seq)

    # 목표 시퀀스 초기화
    target_seq = np.zeros((1, 1))
    
    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0, 0] = STA_INDEX
    
    # 인덱스 초기화
    indexs = []
    
    # 디코더 타임 스텝 반복
    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict([target_seq] + states)

        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)
        
        # 종료 검사
        if index == END_INDEX or len(indexs) >= max_sequences:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index
        
        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = preprocessor.convert_index_to_text(indexs, get_index_to_word())
        
    return sentence


# 여기서부터는 직접 짠 코드입니다.
def get_word_to_index():
    """
    함수: get_word_to_index()\n
    입력: 없음\n
    출력: 딕셔너리 파일(단어 사전)\n
    목적: 저장된 단어 사전 파일을 불러오기 위한 함수  
    """

    # pickle 라이브러리를 활용해서 데이터를 불러옴
    with open('./vocabulary/word_to_index.txt', 'rb') as f:
        data = pickle.load(f)

    return data

def get_index_to_word():
    """
    함수: get_index_to_word()\n
    입력: 없음\n
    출력: 딕셔너리 파일(단어 사전)\n
    목적: 저장된 단어 사전 파일을 불러오기 위한 함수  
    """

    # pickle 라이브러리를 활용해서 데이터를 불러옴
    with open('./vocabulary/index_to_word.txt', 'rb') as f:
        data = pickle.load(f)

    return data
