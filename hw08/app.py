import streamlit as st
import os

# 修正所有导入路径
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# 页面基础配置
st.set_page_config(page_title="课程知识库RAG问答系统", layout="wide")
st.title("📚 课程知识库智能问答Web应用")
st.markdown("基于RAG技术，上传课程PDF/Word/TXT资料，精准检索知识点答疑")

# 向量数据库保存文件夹
CHROMA_PATH = "chroma_db"
# 加载开源向量化模型
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 侧边栏：文档上传模块
with st.sidebar:
    st.header("📂 知识库管理")
    uploaded_files = st.file_uploader("上传课程文档", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files and st.button("构建知识库"):
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        for file in uploaded_files:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())
            # 根据文件类型选择加载器
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            elif file.name.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(temp_path)
            else:
                loader = TextLoader(temp_path)
            docs = loader.load_and_split(text_splitter=text_splitter)
            all_docs.extend(docs)
            os.remove(temp_path)
        # 存入向量数据库
        Chroma.from_documents(all_docs, embedding_model, persist_directory=CHROMA_PATH)
        st.success(f"✅ 知识库构建完成，共加载{len(all_docs)}段课程文本")

# 聊天历史初始化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 渲染历史对话
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# 用户提问输入框
user_question = st.chat_input("请针对上传的课程资料提问...")
if user_question:
    st.chat_message("user").write(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # 从向量库检索相关知识点
    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
        search_docs = db.similarity_search(user_question, k=3)
        context_text = "\n\n".join([doc.page_content for doc in search_docs])

        answer = f"""### 基于课程知识库的检索结果
【参考资料片段】
{context_text[:1000]}

> 当前为检索演示，下一步接入大模型即可生成完整知识点解答。"""
    else:
        answer = "⚠️ 请先在左侧上传课程文档，点击【构建知识库】后再提问！"

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
