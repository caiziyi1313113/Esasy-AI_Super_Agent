# Esasy 1.0 项目

## 1. 项目概述

Esasy 1.0 是一个旨在提供文档处理和智能问答服务的应用。它结合了后端API和前端界面，支持PDF解析、AI问答以及用户管理功能。

## 2. 项目配置与运行

### 2.1. 环境依赖

*   **Python 3.11**: 后端运行环境。
*   **Node.js (LTS)**: 前端运行环境。
*   **Poppler**: PDF解析库，**需手动安装并配置环境变量。**请参考 [Poppler 官网](https://poppler.freedesktop.org/)。
*   **Tesseract OCR**: OCR引擎，**需手动安装并配置环境变量。**请参考 [Tesseract OCR 官网](https://tesseract-ocr.github.io/)。

### 2.2. 后端配置与启动

1.  **安装Python依赖**：

	进入项目根目录 `esasy1.0`，创建并激活Conda环境，然后安装 `requirements.txt` 中的依赖：

	```bash
	conda create --name esasy python=3.11
	conda activate esasy
	pip install -r requirements.txt
	```

2.  **配置**：

	主要配置文件为 `src/configs.py`，您可以在此配置数据库连接、API密钥等。

3.  **启动后端服务**：

	```bash
	cd src
	python main.py
	```

	后端服务通常运行在 http://127.0.0.1:8000

	可以通过此链接查看后端网络路由接口 http://localhost:8000/docs

### 2.3. 前端配置与启动

1.  **安装Node.js依赖**：

	进入前端目录 `node-site`，安装 `package.json` 中的依赖：

	```bash
	cd node-site
	npm install
	```

2.  **配置**：

	在.env中配置自己的大模型api密钥。

3.  **启动前端服务**：

	```bash
	node main.js
	```

	前端服务通常运行在 `http://127.0.0.1:3000`。

## 3. 项目结构

```python
esasy1.0/
├── data/           # 资源文件目录，运行时会自动创建和管理
│   ├── uploads/           # 论文pdf源文件上传保存的目录
│   ├── parsed_results     # 保存parser解析pdf源文件产生的json结构化文件等
│   ├── chroma_db/         # 向量化存储的rag数据库，用于问答系统（待重构优化）
│   └── essay.db           # ai初步解析得到的内容保存到的数据库（待重构优化）
├── node-site/      # 前端应用目录
│   ├── main.js            # 前端入口文件，负责启动Node.js服务和路由
│   ├── package.json       # 前端依赖管理
│   └── static/            # 静态资源文件 (HTML, CSS, JS, 字体等；其中index是主页，paper是论文显示单页)
├── requirements.txt      
├── README.md  
├── .env            # 你的api配置文件，格式为"DASHSCOPE_API_KEY = your api key"
└── src/            # 后端应用目录
    ├── configs.py         # 项目配置文件，主要是标记了通用的项目目录
    ├── main.py            # 后端入口文件，启动FastAPI应用
    ├── models/            # 数据库模型定义 (如User, Paper)
    │   ├── db.py          # 数据库连接和会话管理
    │   ├── paper.py       # Paper模型定义
    │   └── user.py        # User模型定义
    ├── routes/            # API路由定义 (如用户认证、文档操作)——————重要！
    │   ├── paper_routes.py   # 文档相关API路由
    │   └── user_routes.py    # 用户相关API路由
    ├── schemas/           # 数据验证和序列化模型 (Pydantic)
    │   ├── chat_schemas.py   # 聊天相关数据模型
    │   ├── paper_schemas.py   # 文档相关数据模型
    │   └── user_schemas.py   # 用户相关数据模型
    └── services/          # 核心业务逻辑和外部服务集成————————————重要！
        ├── ai_service.py  # 调用大模型接口处理智能分析、问答逻辑————目前的功能还非常浅陋，都是用于演示的demo
        └── pdf_parser.py  # PDF解析————目前的已经能比较全面地提取pdf内容，但是详细的分块解析方法还有待优化
```

> **论文传进来，先将上传到项目目录，然后用pdf_parser解析成结构化的json文件。分析阶段，一方面，用大模型接口根据上述结构化json生成初步的分析报告，另一方面，用rag方法构建向量数据库，后面问答阶段依赖此向量数据库（暂时是这样，但是整个后端AI模块目前都比较混乱，后面需要全面重构一下。**
>
> **目前为止主要的成果是基本搭起来了项目框架，能够进行比较有效的样例运行，但是核心算法部分、核心展示部分的具体划分和展示效果、扩展内容（如登陆系统、外部数据库等），以及前端交互都需要重构或者优化一下**



## 4. 关键代码文件内容概述

*   **`src/main.py`**: 后端应用的入口点。它初始化FastAPI应用，加载配置，并注册所有API路由 (`user_routes.py`, `paper_routes.py`)。这是整个后端服务的核心启动文件。

*   **`src/configs.py`**: 集中管理后端应用的各项配置，包括数据库连接字符串、外部服务（如AI服务）的API密钥、端口设置等。修改此文件可以快速调整应用行为。

*   **`src/models/db.py`**: 负责数据库的初始化、会话管理以及ORM（SQLAlchemy）的配置。它定义了如何连接数据库，并提供了获取数据库会话的方法，供其他模块进行数据操作。

*   **`src/services/pdf_parser.py`**: 实现了PDF文档的解析逻辑。它利用Poppler和Tesseract等工具，从PDF文件中提取文本内容，为后续的AI问答提供数据基础。

*   **`src/services/ai_service.py`**: 封装了与AI模型交互的逻辑。它接收用户的问题和文档内容，调用AI接口进行处理，并返回智能问答的结果。

*   **`node-site/main.js`**: 前端Node.js服务的入口文件。它负责启动一个简单的HTTP服务器，处理前端路由，并提供静态文件服务。同时，它也可能包含与后端API交互的逻辑。

## 5. 使用方法

1.  确保后端和前端服务均已成功启动。
2.  在浏览器中访问前端地址  http://127.0.0.1:3000
3.  通过前端界面上传PDF文档。
4.  在界面上输入问题，与AI进行文档内容的智能问答。

## 6. 注意事项

*   **环境变量配置**: 务必确保Poppler和Tesseract的可执行文件路径已正确添加到系统环境变量中，否则PDF解析功能将无法正常工作。
*   **端口冲突**: 如果启动服务时遇到端口被占用的错误，请检查 `src/main.py` 和 `node-site/main.js` 中的端口配置，并修改为可用端口。

