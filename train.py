from preprocess import Preprocessor
from seq2seq import Encoder, Decoder
from const import ENCODER_INPUT, DECODER_INPUT, DECODER_TARGET
from keras import models
from keras import layers
import numpy as np

if __name__ == '__main__':
    # 전처리 객체 선언
    preprocessor = Preprocessor()

    # 데이터 불러오기
    question, answer = preprocessor.load_data('./dataset/chatbot/ChatbotData.csv')

    # 데이터의 일부만 학습에 사용
    question = question[:100]
    answer = answer[:100]

    # 데이터에 토큰화 함수 적용
    question = preprocessor.tokenize_ko(question)
    answer = preprocessor.tokenize_ko(answer)

    # sentences 리스트 = 질문과 대답 리스트를 합친 것
    sentences = []
    sentences.extend(question)
    sentences.extend(answer)

    # 단어와 인덱스의 딕셔너리 생성
    word_to_index, index_to_word = preprocessor.build_vocab(sentences)

    # 인코더 입력 인덱스 변환
    x_encoder = preprocessor.convert_text_to_index(question, word_to_index, ENCODER_INPUT)

    # 디코더 입력 인덱스 변환
    x_decoder = preprocessor.convert_text_to_index(answer, word_to_index, DECODER_INPUT)
    
    # 디코더 목표 인덱스 변환
    y_decoder = preprocessor.convert_text_to_index(answer, word_to_index, DECODER_TARGET)
    
    # 원핫 인코딩
    y_decoder = preprocessor.one_hot_encode(y_decoder)

    # 훈련 모델 인코더, 디코더 정의
    encoder = Encoder(len(preprocessor.words))
    decoder = Decoder(encoder.states, encoder.len_of_words)

    # 훈련 모델 정의
    model = models.Model([encoder.inputs, decoder.inputs], decoder.outputs)

    # 학습 방법 설정
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

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

    model.save('./model/chatbot_model.h5')
