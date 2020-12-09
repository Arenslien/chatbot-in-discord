# preprocess.py의 일부 코드는 github의 코드를 가져온 코드입니다.
# 하나의 코드로 된 것을 분리한 코드입니다.
# 전처리와 관련된 기능들을 수행하는 코드 입니다.
# https://github.com/deepseasw/seq2seq_chatbot/blob/master/Seq2Seq%20Chatbot.ipynb

from const import PAD, STA, END, OOV, ENCODER_INPUT, DECODER_INPUT, DECODER_TARGET, max_sequences, RE_FILTER, END_INDEX, OOV_INDEX
import pandas as pd
import numpy as np
from re import sub
from konlpy.tag import Okt
import pickle


# 해당 클래스는 github 코드의 전처리에 해당되는 부분을 클래스화 한 것입니다.
# 전처리에 해당되는 코드를 구조화한 것이 직접 한 부분들입니다.
# 세부 코드들은 github의 코드입니다.
class Preprocessor:
    def __init__(self):
        pass

    # 학습 데이터를 로드하기 위한 함수로 구조화만 했습니다.
    # 해당 코드는 github의 코드입니다.
    def load_data(self, path):
        # 챗봇 데이터 로드
        self.chatbot_data = pd.read_csv(path, encoding='utf-8')
        question, answer = list(self.chatbot_data['Q']), list(self.chatbot_data['A'])

        return question, answer

    # 입력되는 문장 리스트를 형태소 분석하기 위한 함수로 구조화만 했습니다.
    # 해당 코드는 github의 코드입니다.
    def tokenize_ko(self, sentences):
        # KoNLPy 형태소 분석기 설정
        tagger = Okt()

        # 문장 품사 변수 초기화
        tokenized_sentences = []

        # 모든 문장 반복
        for sentence in sentences:
            # 특수기호 제거
            sentence = sub(RE_FILTER, "", sentence)

            # 형태소 분석한 것을 띄어쓰기로 구분하여 붙임
            sentence = " ".join(tagger.morphs(sentence))
            tokenized_sentences.append(sentence)

        return tokenized_sentences

    # 형태소 분석이 된 리스트를 바탕으로 단어 사전을 만들기 위한 함수로 구조화만 했습니다.
    # 해당 코드는 github의 코드입니다.
    def build_vocab(self, sentences):
        # 단어 배열 선언
        self.words = []

        # 단어 배열 생성
        for sentence in sentences:
            for word in sentence.split():
                self.words.append(word)

        # 길이가 0인 단어는 삭제
        self.words = [word for word in self.words if len(word) > 0]

        # 중복 단어 삭제
        self.words = list(set(self.words))

        # 제일 앞에 태그 단어 삽입
        self.words[:0] = [PAD, STA, END, OOV]

        # 단어와 인덱스의 딕셔너리 생성
        word_to_index = {word: index for index, word in enumerate(self.words)}
        index_to_word = {index: word for index, word in enumerate(self.words)}

        # 직접 추가한 코드
        # 기존의 코드는 예측 모델을 일회용으로 사용하기에 저장하지 않았습니다.
        # 챗봇 모델은 단어 사전을 계속 사용하기에 단어 사전을 저장하는 코드입니다.
        # word_to_index, index_to_word 저장
        with open('./vocabulary/word_to_index.txt', 'wb') as f:
            pickle.dump(word_to_index, f)
        with open('./vocabulary/index_to_word.txt', 'wb') as f:
            pickle.dump(index_to_word, f)

        return word_to_index, index_to_word

    # 모델의 입력값을 위해 형태소 분석된 문장을 임베딩 하기 위한 함수로 구조화만 했습니다.
    # 해당 코드는 github의 코드입니다.
    def convert_text_to_index(self, sentences, vocabulary, type):
        sentences_index = []

        # 모든 문장에 대해서 반복
        for sentence in sentences:
            sentence_index = []

            # 디코더 입력일 경우 맨 앞에 START 태그 추가
            if type == DECODER_INPUT:
                sentence_index.extend([vocabulary[STA]])
            
            # 문장의 단어들을 띄어쓰기로 분리
            for word in sentence.split():
                if vocabulary.get(word) is not None:
                    # 사전에 있는 단어면 해당 인덱스를 추가
                    sentence_index.extend([vocabulary[word]])
                else:
                    # 사전에 없는 단어면 OOV 인덱스를 추가
                    sentence_index.extend([vocabulary[OOV]])

            # 최대 길이 검사
            if type == DECODER_TARGET:
                # 디코더 목표일 경우 맨 뒤에 END 태그 추가
                if len(sentence_index) >= max_sequences:
                    sentence_index = sentence_index[:max_sequences-1] + [vocabulary[END]]
                else:
                    sentence_index += [vocabulary[END]]
            else:
                if len(sentence_index) > max_sequences:
                    sentence_index = sentence_index[:max_sequences]
            
            # 최대 길이에 없는 공간은 패딩 인덱스로 채움
            sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]
            
            # 문장의 인덱스 배열을 추가
            sentences_index.append(sentence_index)

        return np.asarray(sentences_index)

    # 원 핫 인코딩을 위한 함수로 구조화만 했습니다.
    # 해당 코드는 github의 코드입니다.
    def one_hot_encode(self, y_decoder):
        # 원핫인코딩 데이터 초기화
        one_hot_data = np.zeros((len(y_decoder), max_sequences, len(self.words)))

        # 디코더 목표 => 원핫인코딩으로 변환
        # 학습시 입력은 인덱스이지만, 출력은 원핫인코딩 형식임
        for i, sequence in enumerate(y_decoder):
            for j, index in enumerate(sequence):
                one_hot_data[i, j, index] = 1

        return one_hot_data

    # 인덱스화된 문장을 다시 문자열로 변환하는 함수로 구조화만 했습니다.
    # 해당 코드는 github의 코드입니다.    
    # 인덱스를 문장으로 변환
    def convert_index_to_text(self, indexs, vocabulary): 
        
        sentence = ''
        
        # 모든 문장에 대해서 반복
        for index in indexs:
            if index == END_INDEX:
                # 종료 인덱스면 중지
                break
            if vocabulary.get(index) is not None:
                # 사전에 있는 인덱스면 해당 단어를 추가
                sentence += vocabulary[index]
            else:
                # 사전에 없는 인덱스면 OOV 단어를 추가
                sentence.extend([vocabulary[OOV_INDEX]])
                
            # 빈칸 추가
            sentence += ' '
    
        return sentence