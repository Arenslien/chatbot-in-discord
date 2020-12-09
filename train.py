# train.py의 일부 코드는 github의 코드를 가져온 코드입니다.
# 하나의 코드로 된 것을 분리한 코드입니다.
# 모델 훈련과 관련된 기능을 수행하는 코드 입니다.
# https://github.com/deepseasw/seq2seq_chatbot/blob/master/Seq2Seq%20Chatbot.ipynb

from preprocess import Preprocessor
from seq2seq import Encoder, Decoder
from predict import make_predict_input, generate_text, get_word_to_index, get_index_to_word
from const import ENCODER_INPUT, DECODER_INPUT, DECODER_TARGET
from keras import models
from keras import layers
import numpy as np
import pickle
import os.path

# 전체적인 코드의 내용은 github의 코드와 유사하지만
# 하나의 파일을 여러개의 파일로 분리해서 사용하기 때문에
# 객체를 사용하는 부분들이 조금씩 변경되었습니다.

if __name__ == '__main__':
    # 전처리 객체 선언
    # 전처리와 관련된 부분을 다른 파일로 구성했기 때문에 가져오는 작업입니다.
    preprocessor = Preprocessor()

    # 데이터 불러오기
    question, answer = preprocessor.load_data('./dataset/ChatbotData.csv')

    # 데이터의 일부만 학습에 사용
    # 전체 데이터, 5000개, 4000개, 3000개에 대한 학습 시간이 너무 길어서 2000개로 하였습니다.
    question = question[:2000]
    answer = answer[:2000]

    # 데이터에 토큰화 함수 적용
    # 다른 파일에 토큰화 함수가 있기에 조금 변경
    # 수행하는 결과는 github 코드와 동일합니다.
    question = preprocessor.tokenize_ko(question)
    answer = preprocessor.tokenize_ko(answer)

    # sentences 리스트 = 질문과 대답 리스트를 합친 것
    sentences = []
    sentences.extend(question)
    sentences.extend(answer)


    # 단어와 인덱스의 딕셔너리 생성
    # 다른 파일에 단어 생성 함수가 있기에 조금 변경
    # 수행하는 결과는 github 코드와 동일합니다.
    word_to_index, index_to_word = preprocessor.build_vocab(sentences)

    # 다른 파일에 단어 생성 함수가 있기에 조금 변경
    # 수행하는 결과는 github 코드와 동일합니다.
    # 인코더 입력 인덱스 변환
    x_encoder = preprocessor.convert_text_to_index(question, word_to_index, ENCODER_INPUT)

    # 디코더 입력 인덱스 변환
    x_decoder = preprocessor.convert_text_to_index(answer, word_to_index, DECODER_INPUT)
    
    # 디코더 목표 인덱스 변환
    y_decoder = preprocessor.convert_text_to_index(answer, word_to_index, DECODER_TARGET)
    
    # 원핫 인코딩
    y_decoder = preprocessor.one_hot_encode(y_decoder)

    # 훈련 모델 인코더, 디코더 정의
    # 모델을 객체화 했기 때문에 객체로 가져오는 코드
    encoder = Encoder(len(preprocessor.words))
    decoder = Decoder(encoder.states, encoder.len_of_words)

    # github코드와 거의 동일
    # 훈련 모델 정의
    model = models.Model([encoder.inputs, decoder.inputs], decoder.outputs)

    # github코드와 동일
    # 학습 방법 설정
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # github코드와 같은 결과를 내보냄
    # 예측 모델을 생성하는 것을 객체의 함수로 만들었기 때문에 다음과 같이 코딩했습니다.
    # 예측 모델 정의
    encoder_model = encoder.get_predict_model()
    decoder_model = decoder.get_predict_model()


    # 에폭 훈련 과정은 github 코드와 동일합니다.
    # 에폭 반복
    for epoch in range(20):
        print('Total Epoch :', epoch + 1)

        # 훈련 시작
        history = model.fit(
            [x_encoder, x_decoder],
            y_decoder,
            epochs=100,
            batch_size=64,
            verbose=0
        )
        
        # 정확도와 손실 출력
        print('accuracy :', history.history['acc'][-1])
        print('loss :', history.history['loss'][-1])
        
        # 문장 예측 테스트
        # (3 박 4일 놀러 가고 싶다) -> (여행 은 언제나 좋죠)
        input_encoder = x_encoder[2].reshape(1, x_encoder[2].shape[0])
        input_decoder = x_decoder[2].reshape(1, x_decoder[2].shape[0])
        results = model.predict([input_encoder, input_decoder])
        
        # 결과의 원핫인코딩 형식을 인덱스로 변환
        # 1축을 기준으로 가장 높은 값의 위치를 구함
        indexs = np.argmax(results[0], 1) 
        
        # 인덱스를 문장으로 변환
        sentence = preprocessor.convert_index_to_text(indexs, index_to_word)
        print(sentence)
        print()


    # User Warning을 사라지게 하기 위해서 추가한 코드
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 직접 추가한 코드
    # 기존의 코드는 예측 모델을 일회용으로 사용하기에 저장하지 않았습니다.
    # 해당 챗봇은 모델을 계속 사용하기에 예측 모델을 저장하는 코드입니다.
    # 훈련 & 예측 모델 저장

    model.save('./model/train_model.h5')
    encoder_model.save('./model/encoder_model.h5')
    decoder_model.save('./model/decoder_model.h5')


    # 마지막 테스트 한 번은 github 코드와 매우 유사합니다.
    print('저장완료')
    print('예측 모델 테스트')
    print('Q: 3박4일 놀러가고 싶다.')
    test = make_predict_input('3박4일 놀러가고 싶다.')
    result = generate_text(test)
    print(f'A: {result}') 

