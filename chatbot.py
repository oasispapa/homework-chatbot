import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LevenshteinChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)        

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers
    
    # 모든 질문과 사용자의 질문을 비교하여 거리를 구하고 distance_array 에 담는다. 
    def collect_distance(self, input_sentence):
        distance_array = []
        for question in self.questions:
            distance = calc_distance(input_sentence, question)
            distance_array.append(distance)
        return distance_array.index(min(distance_array))  # 거리가 가장 짧은 질문의 인덱스를 반환

    def answer(self, input_sentence):
        # 1. 입력 문장과 가장 유사한 질문을 찾기
        best_match_index = self.collect_distance(input_sentence)
        # 2. 찾은 질문의 인덱스를 사용해 답변 리턴
        return self.answers[best_match_index]


class TfidfChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)  # 질문을 TF-IDF로 변환

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers

    def answer(self, input_sentence):
        input_vector = self.vectorizer.transform([input_sentence])
        similarities = cosine_similarity(input_vector, self.question_vectors) # 코사인 유사도 값들을 저장        
        best_match_index = similarities.argmax()   # 유사도 값이 가장 큰 값의 인덱스를 반환
        return self.answers[best_match_index]
    

# 레벤슈타인 거리 구하기
def calc_distance(a, b):
    ''' 레벤슈타인 거리 계산하기 '''
    if a == b: return 0 # 같으면 0을 반환
    a_len = len(a) # a 길이
    b_len = len(b) # b 길이
    if a == "": return b_len
    if b == "": return a_len
    # 2차원 표 (a_len+1, b_len+1) 준비하기 --- (※1)
    # matrix 초기화의 예 : [[0, 1, 2, 3], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0]]
    # [0, 1, 2, 3]
    # [1, 0, 0, 0]
    # [2, 0, 0, 0]
    # [3, 0, 0, 0] 
    matrix = [[] for i in range(a_len+1)] # 리스트 컴프리헨션을 사용하여 1차원 초기화
    for i in range(a_len+1): # 0으로 초기화
        matrix[i] = [0 for j in range(b_len+1)]  # 리스트 컴프리헨션을 사용하여 2차원 초기화
    # 0일 때 초깃값을 설정
    for i in range(a_len+1):
        matrix[i][0] = i
    for j in range(b_len+1):
        matrix[0][j] = j
    # 표 채우기 --- (※2)
    # print(matrix,'----------')
    for i in range(1, a_len+1):
        ac = a[i-1]
        # print(ac,'=============')
        for j in range(1, b_len+1):
            bc = b[j-1] 
            # print(bc)
            cost = 0 if (ac == bc) else 1  #  파이썬 조건 표현식 예:) result = value1 if condition else value2
            matrix[i][j] = min([
                matrix[i-1][j] + 1,     # 문자 제거: 위쪽에서 +1
                matrix[i][j-1] + 1,     # 문자 삽입: 왼쪽 수에서 +1   
                matrix[i-1][j-1] + cost # 문자 변경: 대각선에서 +1, 문자가 동일하면 대각선 숫자 복사
            ])
            # print(matrix)
        # print(matrix,'----------끝')
    return matrix[a_len][b_len]


# Main 
if __name__ == '__main__':
    # CSV 파일 경로
    filepath = 'ChatbotData.csv'

    # 챗봇 인스턴스 생성
    print("> '종료'라고 말씀하시면 챗봇과의 대화가 종료됩니다.")
    chatbotType = input('> 어떤 챗봇이 응답하길 원하시나요? 1.레벤슈타인챗봇 / 2.TfIDF챗봇 : ')
    if chatbotType == '1':
        chatbot = LevenshteinChatBot(filepath)
    elif chatbotType == '2':
        chatbot = TfidfChatBot(filepath)
    else:
        print('잘못된 입력입니다. 레벤슈타인챗봇을 사용합니다.')
        chatbot = LevenshteinChatBot(filepath)

    # '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복
    while True:
        input_sentence = input('> You: ')
        if input_sentence.lower() == '종료':
            break
        response = chatbot.answer(input_sentence)
        print('Chatbot:', response)
    
