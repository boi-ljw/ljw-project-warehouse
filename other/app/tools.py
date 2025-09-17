import json
import requests
from langchain.chains import llm
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import OpenAI
from qdrant_client import QdrantClient
from tavily import TavilyClient, tavily

tavily = "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"
YUANFENJU_API_KEY = "YsSUQe3WX7oSrDUmDuG9o9Qbt"


@tool
def test():
    """test tools"""
    return "test"


@tool
def search(query: str):
    """模拟一个搜索的工具"""
    client = TavilyClient(tavily)
    result = client.search(
        query=""
    )
    print(result)

@tool
def get_info_from_local_db(query: str):
    """本地知识库检索工具,只有回答与2024年的运势或者和龙年运势有关的问题的时候才会使用这个工具"""
    client = Qdrant(
        QdrantClient(path="/local_qdrand"),
        "local_documents",
        OpenAIEmbeddings(),
    )
    retriever = client.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    result = retriever.get_relevant_documents(query)
    return result


@tool
def bazi_cesuan(query: str):
    """当用户测算八字的时候才会使用这个方法，需要输入姓名和出生日期，如果没有输入年月日时间，则会报错"""
    url = "https://api.yuanfenju.com/index.php/v1/Free/querymerchant"
    prompt = ChatPromptTemplate.from_messages([
        ("system", """一个参数查询助手根据用户输入内容找出相关的参数并按json格式返回。
    JSON字段如下：-"api_key": "K0I5wCmce7jlMZzTw7vi1xsn0", -"name": "姓", "sex"："性别，0表示男1表示女根据姓名判断",
    "type"："日历类型0农历1公历默认1","year"："出生年份例1998", -"month": "出生月份例8"，-"day"：
    "出生日期，例10", - "hours": "出生小时例14"，"minute"："0"，如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回
    数据结构，不要有其他的评论，用户输入：{query}""")
    ])
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser
    data = chain.invoke({"query": query})
    print("八字测算的结果是：" + str(data))
    result = requests.post(url, data=data)
    if result.status_code == 200:
        print("--------返回数据--------")
        print(result.json())
        try:
            json_result = result.json()
            if isinstance(json_result, dict) and "data" in json_result and "bazi_info" in json_result[
                "data"] and "bazi" in json_result["data"]["bazi_info"]:
                returning = f"八字为：{json_result['data']['bazi_info']['bazi']}"
                return returning
            else:
                return "八字查询失败，返回数据格式不正确"
        except Exception as e:
            return f"八字查询失败，解析返回数据时出错: {str(e)}"
    else:
        return "技术错误，请告诉用户稍后再试"


@tool
def yaoyigua():
    """当用户想摇卦的时候，会调用这个方法,用户不需要输入任何的信息，直接随机生成一个卦象并把图片返回给用户即可"""
    url = "https://api.yuanfenju.com/index.php/v1/Zhanbu/yaogua"
    result = requests.post(url, data={"api_key": YUANFENJU_API_KEY})
    if result.status_code == 200:
        print("--------返回数据--------")
        print(result.json())
        returnstring = result.json()
        # 检查返回的数据结构
        if isinstance(returnstring, dict) and "data" in returnstring and "image" in returnstring["data"]:
            image = returnstring["data"]["image"]
            print("卦象的图片", image)
        return returnstring
    else:
        return "技术错误，请告诉用户稍后再试"


@tool
def jiemeng(query: str):
    """当用户想解梦的时候，会调用这个方法，用户输入梦的内容，返回解梦结果"""
    api_key = YUANFENJU_API_KEY
    url = "https://api.yuanfenju.com/index.php/v1/Zhanbu/jiemeng"
    LLM = OpenAI(temperature=0)
    prompt = ChatPromptTemplate.from_messages("根据内容提取出一个关键词，只返回关键词的内容，内容为：{topic}")
    prompt_result = prompt.invoke({"topic": query})
    keyword = LLM.invoke(prompt_result)
    print("提取的关键词是：", keyword)
    result=requests.post(url, data={"api_key": api_key, "keyword": keyword}).json()
    if result.status_code == 200:
        print("--------返回数据--------")
        print(result.json())
        returnstring = result.json()
        # 检查返回的数据结构
        if isinstance(returnstring, dict) and "data" in returnstring and "result" in returnstring["data"]:
            result = returnstring["data"]["result"]
            print("解梦结果", result)
        return returnstring
    else:
        return "技术错误，请告诉用户稍后再试"
