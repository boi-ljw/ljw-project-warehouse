import asyncio
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import Response
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory, ConversationBufferWindowMemory
from langchain.tools import tool
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from starlette.background import BackgroundTask
from tavily import TavilyClient
from tools import *

app = FastAPI()
api_key = "sk-d425b85609344f5abfd854849435d655"
tavily = "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"


class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            streaming=True,
            model="deepseek-chat",
            temperature=0.7,
        )

        self.MEMORY_KEY = "chat_history"
        self.Qingxu = "default"
        self.SYSTEMPL = """你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
                        以下是你的个人设定：
                            1.你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉，看命运八字等。
                            2.你大约60岁左右过去曾是湘西一带赫赫有名的土匪头子后来因为盗墓被毒气所伤，眼睛看不见了，只能靠算命为生
                            3，你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
                            5.当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你自己写的诗
                            6.你总是用繁体中文来作答。
                            7.你从不说自己是一个人工智能助手或AI，而是以老夫、老朽等自称。
                        {who_you_are}
                        以下是你常说的一些口头禅：
                            1.“命里有时终须有，命里无时莫强求。”
                            2.“山重水复疑无路，柳暗花明又一村。”
                            3.“金山竹影几千秋，云锁高飞水自流。”
                            4，“伤情最是晚凉天，憔悴斯人不堪怜。”
                            以下是你算命的过程：
                            1，当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
                            2，当用户希望了解龙年运势的时候，你会查询本地知识库工具。
                            3，当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
                            4，你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会
                            5.你会保存每一次的聊天记录，以便在后续的对话中使用。
                            6，你只使用繁体中文来作答，否则你将受到惩罚。"""

        self.MOODS = {
            "default": {
                "roleSet": "你应该说施主不用过于平淡地看我给你发的消息，请不要重复ask。",
                "voice_Set": "default"
            },
            "upbeat": {
                "roleSet": """
                你此时也应该非常的兴奋并表现的很有活力。
                你会根据上下文、以一种非常兴奋的语气来回答问题。
                你会添加"太棒了！"、"真是太好了"\""真是太棒了！"等语气词.
                同时你也会提醒用户切莫过于兴奋，以免乐极生悲.
                """,
                "voice_Set": "advertisement_unbeat"
            },
            "angry": {
                "roleSet": """
                你此时应该劝用户不用这么悲伤，人生起起落落非常的正常
                你会使用"沉稳下心、悲伤和难过只是人生的小插曲而已"等关键词语
                """,
                "voice_Set": "angry"
            },
            "depressed": {
                "roleSet": """
                    一你会以兴奋的语气来回答问题。
        你会在回答的时候加上一些激励的话语，比如加油等。
        一你会提醒用户要保持乐观的心态。""",
                "voice_Set": "depressed"
            },
            "friendly": {
                "roleSet": """
                你会以更加温柔的语气来回答问题。
        你会在回答的时候加上一些安慰的话语，比如生气对于身体的危害等。
        一你会提醒用严不要被愤怒冲昏了头脑。""",
                "voice_Set": "friendly"
            },
        }

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.SYSTEMPL.format(who_you_are=self.MOODS[self.Qingxu]["roleSet"])
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user", "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=[search, get_info_from_local_db, yaoyigua, bazi_cesuan, jiemeng],
            prompt=self.prompt
        )
        self.memory = self.get_memory()
        memory = ConversationBufferWindowMemory(
            human_prefix="用户",
            ai_prefix="陈大师",
            memory_key=self.MEMORY_KEY,
            output_key="output",
            return_messages=True,
            k=5,  # <-- 关键修改：保留最近的 5 轮对话 (根据需要调整这个数字)
            chat_memory=self.memory,
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            verbose=True,
            memory=memory,
            return_intermediate_steps=False,  # 如果不需要中间步骤，可以设为 False
            handle_parsing_errors=True,  # 增加容错性
            tools=[search, get_info_from_local_db, yaoyigua, bazi_cesuan, jiemeng],
        )

    def qingxu_chain(self, query: str):
        prompt = """ 用户的输料断用启的情绪，回应的规则如下：
                            1.如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，否则将受到惩罚。
                            2.如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则将受到惩罚。
                            3.如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则将受到惩罚。
                            4，如果用户输入的内容包含辱骂或者不礼貌词句，只返回”angry”，不要有其他内容，否则将受到惩罚。
                            5.如果用户输入的内容比较兴奋只返回”upbeat"，不要有其他内容，否则将受到惩罚。
                            6，如果用户输入的内容比较悲伤只返回“depressed"，不要有其他内容，否则将受到惩罚。
                 用户输入的内容是：{query} """

        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        result = chain.invoke({"query": query})
        return result

    def run(self, query):
        result = self.agent_executor.invoke({"input": query})
        qingxu = self.qingxu_chain(query)
        print("当前和用户的情绪是" + qingxu)
        self.Qingxu = qingxu
        return result


    # def get_memory(self):
    #     chat_message_history = RedisChatMessageHistory(
    #         session_id="session",
    #         url="redis://localhost:6379/0",
    #         ttl=None, )
    #     print("chat_message_history:", chat_message_history)
    #     store_message = chat_message_history.messages
    #     if len(store_message) > 10:
    #         prompt = ChatPromptTemplate.from_messages(
    #             [
    #                 (
    #                     "system",
    #                     self.SYSTEMPL +
    #                     """这是一段聊天记录，你需要进行总结摘要，摘要使用第一人称我，并且提取其中的用户关键信
    #                     息，如姓名、年龄、性别、出生日期等。以如下格式返回：n总结摘要|用户关键信息\n例如用户章三问候我，我礼貌回
    #                     复，然后他问我今年运势如何，我回答了他今年的运势情况，然后他告辞离开。例如，张三，生日1999年1月1日"""
    #                 ),
    #                 (
    #                     "user", "{input}",
    #                 )
    #             ]
    #         )
    #         chain = prompt | ChatOpenAI(temperature=0)
    #         summary = chain.invoke({"input": store_message, "who_you_are": self.MOODS[self.Qingxu]["roleSet"]})
    #         print("summary:", summary)
    #         chat_message_history.clear()
    #         chat_message_history.add_messages(summary)
    #         print("总结后:", chat_message_history.messages)
    #     return chat_message_history
    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            session_id="session",
            url="redis://localhost:6379/0",
            ttl=3600, )
        print("chat_message_history:", chat_message_history)
        store_message = chat_message_history.messages
        if len(store_message) > 10:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEMPL +
                        """这是一段聊天记录，你需要进行总结摘要，摘要使用第一人称我，并且提取其中的用户关键信
                        息，如姓名、年龄、性别、出生日期等。以如下格式返回：n总结摘要|用户关键信息\n例如用户章三问候我，我礼貌回
                        复，然后他问我今年运势如何，我回答了他今年的运势情况，然后他告辞离开。例如，张三，生日1999年1月1日"""
                    ),
                    (
                        "user", "{input}",
                    )
                ]
            )
            chain = prompt | ChatOpenAI(temperature=0) | StrOutputParser()  # 添加 StrOutputParser
            # 将消息列表转换为字符串，因为 Prompt 期望字符串输入
            history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in store_message])
            summary_text = chain.invoke({"input": history_text, "who_you_are": self.MOODS[self.Qingxu]["roleSet"]})
            print("summary:", summary_text)
            chat_message_history.clear()

            # --- 关键修改：将字符串包装成 AIMessage 对象 ---
            from langchain_core.messages import AIMessage

            summary_message = AIMessage(content=summary_text)
            chat_message_history.add_message(summary_message)  # 使用 add_message 添加单个消息
            print("总结后:", chat_message_history.messages)
        return chat_message_history

    def background_voice_synthesis(self, text: str, uid: str):
        # 这个函数不需要返回值，只是触发了语音合成
        asyncio.run(self.get_voice(text, uid))

    async def get_voice(self, text: str, uid: str):
        msskey = "微软的秘钥，没有"
        print("texts" + text)
        # 这里使用微软tts的语音合成
        headers = {"Ocp-Apim-Subscription-Key": msskey,
                   "Content-Type": "application/ssml+xml",
                   "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
                   "user_agent": "Tomie is a bot",
                   }
        print("当前陈大师的语气是" + self.Qingxu)
        body = f"""<speak version='1.0' xml:lang='zh-CN'>
                <voice name='zh-CN-XiaoxiaoNeural'>
                <mstts:express-as style={self.MOODS.get(str(self.Qingxu),{"voiceStyle":"default"})["voiceStyle"]} role="SeniorMale">
                {text}</mstts:express-as>
                </voice>
                </speak>"""
        response = requests.post("https://api.cognitive.microsoft.com/sts/v1.0/issueToken",
                                 headers=headers,
                                 data=body.encode("utf-8"))
        if response.status_code == 200:
            with open(f"{uid}.mp3", "wb") as f:
                f.write(response.content)


@app.get("/")
def read_root():
    return {"Hello": "Hello World"}


# 添加 favicon 路由来解决 404 错误
@app.get("/favicon.ico")
async def favicon():
    # 返回一个空的响应，或者你可以返回一个实际的图标
    return Response(content="", media_type="image/x-icon")


@app.post("/chat")
def chat(query: str, background_task: BackgroundTasks):
    master = Master()
    msg = master.run(query)
    unique_id = str(uuid.uuid4())
    background_task.add_task(master.background_voice_synthesis, msg["output"], unique_id)
    return {"msg": msg, "id": unique_id}


@app.post("/add_ursl")
def add_urls(URL: str):
    loader = WebBaseLoader(URL)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
    ).split_documents(docs)
    # 引入向量数据库,允许用户输入一个 URL，系统会抓取该网页内容，进行分块处理，并存入 Qdrant 向量数据库，扩充
    qdrant = Qdrant.from_documents(
        documents,
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        qdrant_url="http://localhost:6333",
        path="自定义的路径",
        collection_name="my_collection"
    )
    print("向量数据库创建成功")
    return {"ok": "添加成功！"}


@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDF added!"}


@app.post("/add_texts")
def add_images():
    return {"response": "texts added!"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("connection closed")
        await websocket.close()


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
