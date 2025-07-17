import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from sqlalchemy.orm import Session

from models.db import get_db_session
from models.paper import Paper, ChatSession
from services.pdf_parser import PDFParser
from services.ai_service import AIService

from pydantic import BaseModel
from schemas.paper_schemas import *
from schemas.chat_schemas import *

from configs import DATA_DIR

import json

router = APIRouter(prefix="/papers", tags=["papers"])

UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.get("/env-check")
def env_check():
    import sys
    import os
    return {
        "python_path": sys.executable,
        "env_path": os.environ.get("PATH"),
        "tesseract_path": os.environ.get("TESSDATA_PREFIX"),
    }


# ========== 上传论文 ==========
@router.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    user_id: int = Form(1),
    db: Session = Depends(get_db_session),
):
    print(f"[UPLOAD] 用户 user_id={user_id} 上传文件 filename={file.filename}")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    paper = Paper(
        filename=filename,
        original_filename=file.filename,
        file_path=file_path,
        user_id=user_id,
        processing_status="uploaded"
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)

    print(f"[UPLOAD] 文件已保存到 file_path={paper.file_path}, paper_id={paper.id}")

    return {
        "message": "File uploaded successfully",
        "paper_id": paper.id,
        "filename": paper.original_filename
    }


# ========== 分析论文 ==========
#**AIService 类本身并不直接“接收”整个JSON文件。**
#真正的“接收”和“分发”工作，是在我们之前看过的 **papers 路由文件中的 analyze_paper 函数**里完成的。
#这个函数扮演着一个**“总指挥”**或**“协调者”**的角色。
#在paper_analyze函数中，我们首先实例化了PDFParser类，用于解析PDF文件。
#然后，我们调用PDFParser的parse_pdf方法，将解析后的数据存储在parsed_data变量中。
#接着，我们调用PDFParser的extract_key_sections方法，提取论文的关键章节。
#然后，我们实例化了AIService类，用于进行与AI的交互。
#最后，我们调用AIService的generate_summary方法，生成论文的摘要。
@router.post("/{paper_id}/analyze", response_model=PaperAnalysisResult)
async def analyze_paper(paper_id: int, db: Session = Depends(get_db_session)):
    print(f"[ANALYZE] 开始分析论文 paper_id={paper_id}")
    paper = db.query(Paper).get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if paper.processing_status == 'processing':
        raise HTTPException(status_code=400, detail="Paper is already being processed")

    paper.processing_status = 'processing'
    db.commit()

    try:
        #实例化PDFParser类，用于解析PDF文件
        #调用我们创建的pdf分析工具
        parser = PDFParser()
        parsed_data = parser.parse_pdf(paper.id, paper.file_path)
        print(f"[ANALYZE] PDF parse finished")

        key_sections = parser.extract_key_sections(parsed_data)
        print(f"[ANALYZE] Extracted key sections: {list(key_sections.keys())}")

        def _format_sections(sections: Dict[str, str]) -> str:
            return "\n\n".join([f"## {k.capitalize()}\n{v.strip()}" for k, v in sections.items() if v.strip()])

        key_sections_text = _format_sections(key_sections)

        #实例化AIService类，用于进行与AI的交互
        #在AIService类中，我们定义了多个方法，用于与AI进行交互。
        #这些方法包括：
        #generate_summary: 生成论文的摘要
        #extract_key_content: 提取论文的关键内容
        #translate_text: 翻译论文的摘要
        #explain_terminology: 解释论文的术语
        #analyze_research_context: 分析论文的研究背景
        ai_service = AIService()

        # 提取论文的标题、作者、摘要,从parsed_data 中，
        paper.title = parsed_data.get('title', paper.original_filename)
        paper.authors = parsed_data.get('authors', '')
        paper.abstract = parsed_data.get('abstract', '')

        print(f"[ANALYZE] Title: {paper.title}, Authors: {paper.authors}")
        
        #进行pdf和aiservice的互动！
        
        if paper.abstract and paper.title:
            paper.summary = ai_service.generate_summary(paper.abstract, paper.title)
            print(f"[ANALYZE] Summary generated")
        else:
            paper.summary = f"这是一篇关于{paper.title}的学术论文。"

        if key_sections_text:
            paper.key_content = ai_service.extract_key_content(key_sections_text, paper.title)
            print(f"[ANALYZE] Key content extracted")
        else:
            paper.key_content = "关键内容提取功能需要更完整的论文结构。"

        if paper.abstract:
            paper.translation = ai_service.translate_text(paper.abstract)
            print(f"[ANALYZE] Abstract translated")
        else:
            paper.translation = "未找到摘要内容进行翻译。"

        full_text = parsed_data.get('full_text', '')
        if full_text:
            paper.terminology = ai_service.explain_terminology(full_text[:2000])
            print(f"[ANALYZE] Terminology explained")
        else:
            paper.terminology = "术语解释功能需要更完整的文本内容。"

        paper.research_context = ai_service.analyze_research_context(
            paper.title, paper.abstract, paper.key_content
        )
        print(f"[ANALYZE] Research context analyzed")

        # --- 新增：调用Semantic Scholar服务 ---
        print(f"[ANALYZE] Fetching related papers from Semantic Scholar for title: {paper.title}")
        related_data = ai_service.fetch_related_papers(paper.title)
        paper.s2_id = related_data.get('s2_id')
        paper.related_papers_json = related_data.get('related_papers_json')
        print(f"[ANALYZE] Semantic Scholar data fetched. S2 ID: {paper.s2_id}")

        # rag 构建
        if full_text:
            ai_service.setup_rag(full_text, paper.id)
            print(f"[ANALYZE] RAG setup completed")

        paper.processing_status = 'completed'
        db.commit()

        print(f"[ANALYZE] Paper {paper_id} analysis completed")

        return PaperAnalysisResult(
            message="Paper analysis completed",
            paper=PaperResponse.model_validate(paper),
            parsed_data=ParsedDataSummary(
                sections_count=len(parsed_data.get('sections', [])),
                tables_count=len(parsed_data.get('tables', [])),
                images_count=len(parsed_data.get('images', [])),
                formulas_count=len(parsed_data.get('formulas', [])),
                references_count=len(parsed_data.get('references', [])),
            )
        )

    except Exception as e:
        paper.processing_status = 'failed'
        db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ========== 获取单篇论文 ==========
@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: int, db: Session = Depends(get_db_session)):
    paper = db.query(Paper).get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperResponse.model_validate(paper)


# ========== 获取论文列表 ==========
@router.get("/", response_model=List[PaperResponse])
async def get_papers(user_id: int = 1, db: Session = Depends(get_db_session)):
    papers = db.query(Paper).filter_by(user_id=user_id).order_by(Paper.upload_time.desc()).all()
    return [PaperResponse.model_validate(p) for p in papers]


# ========== 聊天问答 ==========
#使用json处理llm问答得到的答案
@router.post("/{paper_id}/chat", response_model=ChatResponse)
async def chat_with_paper(paper_id: int, request_data: QuestionRequest, db: Session = Depends(get_db_session)):
    print(f"[CHAT] user_id={request_data.user_id} asking: question='{request_data.question}' for paper_id={paper_id}")
    paper = db.query(Paper).get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    ai_service = AIService()
    try:
        # --- 修改点: 调用新的 agentic_answer 方法 ---
        # 这个方法会返回一个字典，可能包含图表信息
        result_dict = ai_service.agentic_answer(request_data.question, paper)
        print(f"[CHAT] Agent result received: {result_dict}")

        # --- 修改点: 将结果字典序列化为 JSON 字符串存入数据库 ---
        # 这样前端就能收到完整的结构化信息
        answer_content = json.dumps(result_dict, ensure_ascii=False)

        chat_session = ChatSession(
            paper_id=paper_id,
            user_id=request_data.user_id,
            question=request_data.question,
            answer=answer_content  # 存储 JSON 字符串
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)

        print(f"[CHAT] Saved chat session id={chat_session.id}")

        return ChatResponse(
            id=chat_session.id,
            paper_id=paper_id,
            user_id=request_data.user_id,
            question=request_data.question,
            answer=answer_content, # 返回 JSON 字符串
            timestamp=chat_session.timestamp,
        )

    except Exception as e:
        # 打印详细的异常信息，便于调试
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ========== 聊天历史记录 ==========
@router.get("/{paper_id}/chat/history", response_model=List[ChatResponse])
async def get_chat_history(paper_id: int, user_id: int = 1, db: Session = Depends(get_db_session)):
    chats = db.query(ChatSession).filter_by(paper_id=paper_id, user_id=user_id).order_by(ChatSession.timestamp.asc()).all()
    return [ChatResponse.model_validate(chat) for chat in chats]
