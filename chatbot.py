# discord 라이브러리 사용 선언
import discord
from keras.models import load_model
from test import make_predict_input, generate_text

model = load_model('./model/chatbot_model.h5')


class chatbot(discord.Client):
    # 프로그램이 처음 실행되었을 때 초기 구성
    async def on_ready(self):
        # 상태 메시지 설정
        # 종류는 3가지: Game, Streaming, CustomActivity
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

        # message.content = message의 내용
        channel = message.channel
        input_seq = make_predict_input(message.content)

        msg = generate_text(input_seq)
        await channel.send(msg)

# 프로그램이 실행되면 제일 처음으로 실행되는 함수
if __name__ == "__main__":
    # 객체를 생성
    client = chatbot()
    # TOKEN 값을 통해 로그인하고 봇을 실행
    client.run("NzgwNjcwNjIyNDg5MTE2Njcz.X7yeLA.uCULxsbAAI8fnG-xw7ocP5f8v7U")


