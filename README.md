# ljw-project-warehouse
这是关于一个基于 FastAPI 的 Web 服务 + LangChain 代理（Agent）应用

一、 整体系统架构
核心功能: 模拟一个名为“陈玉楼”的AI算命先生，通过工具调用处理用户的玄学咨询（八字测算、解梦、摇卦、运势查询等），具备记忆、情绪识别和语音合成功能。
核心技术栈: FastAPI, LangChain, OpenAI/DeepSeek API, Qdrant, Redis, Tavily, 微软 TTS API（语音合成涉及到微软TTS的密钥功能，测试完毕后会生成一个唯一的一个序列号，只需要接入正常的密钥即可使用）

系统架构流程图
二、 模块与功能详细分析
File 1: tools.py - 工具函数库
此文件定义了所有被主Agent调用的功能函数（使用 @tool 装饰器）。

1. test()
功能: 简单的测试工具，用于验证工具链是否正常工作。
逻辑: 无参数，直接返回字符串 "test"。
衔接过程: 被Agent调用后立即返回固定结果。

3. search(query: str)
功能: 模拟网络搜索工具。
逻辑:
初始化 TavilyClient。
尝试进行搜索（当前存在Bug: query 参数未传入搜索函数，实际搜索内容为空）。
打印结果。
衔接过程: Agent在遇到未知知识时调用此工具获取实时信息。当前实现无效。

3. get_info_from_local_db(query: str)
功能: 从本地Qdrant向量数据库检索信息。
触发条件: 用户问题与“2024年运势”或“龙年运势”高度相关。
逻辑:
连接至本地路径 (/local_qdrand) 的 Qdrant 客户端。
配置检索器 (as_retriever)，使用最大边际相关性算法 (MMR)，返回最相关的3个文档片段。
执行检索并返回结果。
衔接过程: Agent判断用户问题属于运势范畴后调用，将检索到的知识片段融入最终回答。

5. bazi_cesuan(query: str)
功能: 调用第三方API进行八字测算。
触发条件: 用户Input中包含姓名和出生日期信息。
逻辑与流程:
参数提取: 使用一个小型LangChain链 (Prompt → LLM → JsonOutputParser)，从用户非结构化的输入中提取出API所需的结构化参数（姓名、性别、公历/农历、年月日时等）。
API调用: 向 api.yuanfenju.com 发送POST请求，携带提取出的参数。
结果解析:
检查HTTP状态码。
解析返回的JSON，层层检查数据结构 (data → bazi_info → bazi)。
成功则返回八字结果，失败则返回格式化错误信息。
衔接过程: Agent识别出算命意图后调用此工具，工具负责复杂的参数处理和API交互，最终将标准化结果返回给Agent用于组织语言。

5. yaoyigua()
功能: 随机生成一个卦象。
触发条件: 用户表达摇卦意愿。
逻辑:
向固定API端点发送POST请求，仅需提供 api_key。
解析返回的JSON，提取卦象图片URL。
返回整个结果集。
衔接过程: Agent直接调用，并将返回的图片URL等信息描述给用户。

7. jiemeng(query: str)
功能: 对用户描述的梦境进行解读。
逻辑与流程:
关键词提取:
使用一个简单的Prompt（“根据内容提取出一个关键词”）和OpenAI LLM，从梦境描述中提取核心关键词。
API调用: 将关键词发送至解梦API。
结果解析: 解析API返回结果，提取解梦内容。（当前存在Bug: 在检查 result.status_code 之前已经执行了 .json()，导致 result 变为字典，后续状态码检查会抛出异常）。
衔接过程: Agent识别解梦意图后调用，工具先对输入进行预处理（提取关键词），再调用API获取结果。

File 2: app.py - 主应用与核心逻辑
1. Master Class - 系统大脑
1.1 初始化 (__init__)
LLM配置: 连接到DeepSeek的聊天模型 (deepseek-chat)，开启流式传输，设置创造性温度。
角色设定 (SYSTEMPL): 极其详细地定义了“陈大师”的人设、背景、规则和行为准则（必须使用繁体中文、特定口头禅、自称等）。
情绪系统 (MOODS): 一个字典，定义了5种情绪状态（default, upbeat, angry, depressed, friendly），每种状态包含对应的“角色设定片段”和“语音合成风格参数”。
代理构建:
Prompt模板: 组合了系统消息（包含动态情绪角色设定）、聊天历史占位符、用户输入占位符、Agent思考占位符。
工具绑定: 将 tools.py 中定义的所有功能工具加载到Agent中。
记忆系统: 使用 RedisChatMessageHistory 作为持久化存储，并用 ConversationBufferWindowMemory(k=5) 进行包装，只保留最近5轮对话，有效管理上下文长度。
代理执行器 (AgentExecutor): 将以上所有组件（LLM, Tools, Prompt, Memory）组装成可自动运行的核心引擎。

1.2 情绪分析链 (qingxu_chain)
功能: 判断用户输入的情绪倾向。
逻辑: 一个独立的小型LangChain链（Prompt → LLM → StrOutputParser）。Prompt规则清晰：根据输入内容判断其情绪属性（正面、负面、中性、骂人、兴奋、悲伤），并严格返回对应的英文关键词。
衔接过程: 在每次处理用户输入时首先被调用，其输出结果 (qingxu) 用于动态修改系统Prompt中的角色设定部分 (who_you_are)，从而影响LLM本次回答的语气和风格。

1.3 记忆管理系统 (get_memory)
功能: 管理聊天历史，防止无限增长。
核心逻辑 - 摘要总结:
从Redis中获取当前会话的所有消息。
当消息数量 > 10 时，触发摘要流程：
使用另一个LLM链，以所有历史消息为输入，生成一个总结摘要。
摘要要求: 使用第一人称、提取用户关键信息（姓名、生日等）。
格式化输出: “总结摘要|用户关键信息”。
清空Redis中的现有历史。
将生成的一条摘要消息作为新的“压缩后的历史”存回Redis。
衔接过程: 在Master初始化时调用，为Agent执行器提供记忆上下文。这是一个后台管理过程，确保传递给LLM的上下文窗口总是简洁且包含关键信息。

1.4 运行入口 (run)
功能: 处理用户查询的主流程。
衔接流程:
调用 qingxu_chain 分析情绪，更新实例状态 self.Qingxu。
将用户输入和当前所有状态（记忆、情绪）交给 agent_executor.invoke() 执行。
等待Agent完成工具调用和思考，得到最终输出。
返回结果。

1.5 语音合成系统 (background_voice_synthesis, get_voice)
功能: 将文本回复合成为语音文件。
逻辑:
构造符合微软TTS API要求的SSML报文。
关键: 在SSML中，使用 mstts:express-as 标签，其 style 属性由当前情绪状态决定 (self.MOODS[self.Qingxu]["voice_Set"])。
调用API合成语音，并保存为以UUID命名的MP3文件。
衔接过程: 作为后台任务执行。主线程在HTTP响应中返回文本和语音文件ID，客户端可根据ID异步获取或播放语音文件。

2. FastAPI 路由 (HTTP Endpoints)
POST /chat: 主接口。接收用户输入，实例化Master，调用 run()，启动后台语音任务，返回JSON响应。
POST /add_urls: 知识库管理接口。接收一个URL，使用 WebBaseLoader 抓取内容，使用 RecursiveCharacterTextSplitter 进行文本分块，最后存入Qdrant向量数据库。这是 get_info_from_local_db 工具的数据来源。
POST /add_pdfs / POST /add_texts: 预留接口，功能未实现。
WebSocket /ws: 一个基础的WebSocket回声测试端点，为未来实现流式文本传输预留。
