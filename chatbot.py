# chatbot.py의 모든 코드는 DISCORD API DOCUMENTATION을 활용하여 직접 작성했습니다.

# discord 모듈 => 디스코드 API를 활용하기 위한 라이브러리
import discord
# 챗봇 입력 값, 출력값을 위한 라이브러리
from predict import make_predict_input, generate_text

class chatbot(discord.Client):
    # 프로그램이 처음 실행되었을 때 초기 구성
    async def on_ready(self):
        # 상태 메시지 설정
        # 종류는 3가지: Game, Streaming, CustomActivity
        # Nothing은 CustomActivity에 해당된다.
        game = discord.Game("Nothing")

        # 계정 상태를 변경한다.
        # 온라인 상태, game 중으로 설정
        await client.change_presence(status=discord.Status.online, activity=game)

        # 준비가 완료되면 콘솔 창에 "Ready!"라고 표시
        print("READY")

    # 봇에 메시지가 오면 수행 될 액션
    async def on_message(self, message):
        # SENDER가 BOT일 경우 반응을 하지 않도록 한다.
        if message.author.bot:
            return None

        # channel: 메세지가 입력된 채널
        channel = message.channel

        # message.content: 메세지 내용
        # 메세지 내용을 모델의 입력 값으로 변환
        input_seq = make_predict_input(message.content)

        # generate_text 함수를 통해 모델의 출력 값을 얻음
        msg = generate_text(input_seq)

        # 메세지가 들어온 채널로 모델의 출력 값을 전달해줌.
        await channel.send(msg)


# 프로그램이 실행되면 제일 처음으로 실행되는 함수
if __name__ == "__main__":
    # 객체를 생성
    client = chatbot()
    # TOKEN 값을 통해 로그인하고 봇을 실행
    # TOKEN 값은 디스코드 채팅 서버마다 값이 다름.
    client.run("NzgwNjcwNjIyNDg5MTE2Njcz.X7yeLA.uCULxsbAAI8fnG-xw7ocP5f8v7U")
