import asyncio
import json
import os
import urllib
import telebot
from langchain import requests

bot = telebot.TeleBot('api-key')

@bot.message_handler(commands=['start'])
# 使得交互时候输入start，自动响应 您好！
def start_message(message):
    bot.reply_to(message, '您好!')


@bot.message_handler(func=lambda message: True)
# 将用户发送的文本原样回复给用户，相当于复制粘贴
# 实现和app.py文件对接
def echo_all(message):
    # bot.reply_to(message, message.text)
    try:
        encoded_text = urllib.parse.quote(message.text)
        response = requests.post('http://127.0.0.1:8000/chat?query' + encoded_text, timeout=100)
        if response.status_code == 200:
            aisay = json.loads(response.text)
            if 'msg' in aisay:
                bot.reply_to(message, aisay["msg"]["output"])
                # 将前面微软语音服务生成的音频文件以MP3的形式保存在服务器上
                audio_path = f"{aisay['id']}.mp3"
                asyncio.run(check_audio(message, audio_path))
            else:
                bot.reply_to(message, '对不起,我不知道怎么回答你')
        else:
            bot.reply_to(message, '服务器错误')
    except requests.RequestException as e:
        bot.reply_to(message, '服务器错误')


"""循环检查前面的音频文件是否已经生成好，不能耽误主线程的进展所以采用异步的方式"""
async def check_audio(message, audio_path):
    while True:
        if os.path.exists(audio_path):
            with open(audio_path, 'rb') as f:
                bot.send_audio(message.chat.id, f)
            os.remove(audio_path)
            break
        else:
            print("waiting")
            # 让其休眠一秒
            await asyncio.sleep(1)


"""
作用：用于让机器人持续运行并接收来自 Telegram 服务器的消息
    1.持续监听：它会无限期地轮询 Telegram 服务器，检查是否有新的更新（消息、命令等）
    2.自动重连：当网络连接中断或出现异常时，会自动重新连接，保持机器人在线
    3.阻塞运行：该方法会阻塞程序执行，使程序保持运行状态以响应事件"""
bot.infinity_polling()
