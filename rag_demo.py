import streamlit as st
import os
import tempfile
import hashlib
from llama_cpp import Llama
import numpy as np
from typing import List
import jieba
import gc
import PyPDF2
import docx
import markdown
from bs4 import BeautifulSoup



# 预设的嵌入模型和语言模型
PRESET_EMBED_MODELS = [
    {"name": "bge-small-en-v1_5-f16.gguf", "path": "./bge-small-en-v1_5-f16.gguf"},
    # 可以添加更多的预设模型
]
PRESET_LLM_MODELS = [
    {"name": "gemma-2-2B-it-Q4_0_4_4.gguf", "path": "./gemma-2-2B-it-Q4_0_4_4.gguf"},
    # 可以添加更多的预设模型
]

# 处理文件上传
def handle_file_upload(uploaded_files):
    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        return temp_dir
    return None

# 计算上传文件的哈希值
def get_files_hash(files):
    hash_md5 = hashlib.md5()
    for file in files:
        file_bytes = file.read()
        file.seek(0)  # 重置文件指针
        hash_md5.update(file_bytes)
    return hash_md5.hexdigest()



############################################################
def extract_text_from_pdf(file_path: str):
    """Extract text content from a PDF file."""
    contents = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text().strip()
            raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
            new_text = ''
            for text in raw_text:
                new_text += text
                if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」', '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                    contents.append(new_text)
                    new_text = ''
            if new_text:
                contents.append(new_text)
    return contents

def extract_text_from_txt(file_path: str):
    """Extract text content from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = [text.strip() for text in f.readlines() if text.strip()]
    return contents

def extract_text_from_docx(file_path: str):
    """Extract text content from a DOCX file."""
    document = docx.Document(file_path)
    contents = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return contents

def extract_text_from_markdown(file_path: str):
    """Extract text content from a Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, 'html.parser')
    contents = [text.strip() for text in soup.get_text().splitlines() if text.strip()]
    return contents

def preprocess_chunks(doc_file: str, chunks, start_index=0):
    chunks_with_info = []
    if len(chunks) < 1:
        return chunks_with_info
    index = start_index
    for chunk in chunks:
        current = {}
        current["metadata"] = {}
        current["metadata"]["source"] = doc_file
        current["metadata"]["chunk_id"] = index
        current["content"] = chunk
        chunks_with_info.append(current)
        index = index + 1
    return chunks_with_info

class SentenceSplitter:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if self._is_has_chinese(text):
            return self._split_chinese_text(text)
        else:
            return self._split_english_text(text)

    def _split_chinese_text(self, text: str) -> List[str]:
        sentence_endings = {'\n', '。', '！', '？', '；', '…'}  # 句末标点符号
        chunks, current_chunk = [], ''
        for word in jieba.cut(text):
            if len(current_chunk) + len(word) > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += word
            if word[-1] in sentence_endings and len(current_chunk) > self.chunk_size - self.chunk_overlap:
                chunks.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            chunks.append(current_chunk.strip())
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)
        return chunks

    def _split_english_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        chunks, current_chunk = [], ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size or not current_chunk:
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:  # Add the last chunk
            chunks.append(current_chunk)
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)
        return chunks

    def _is_has_chinese(self, text: str) -> bool:
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return True
        else:
            return False

    def _handle_overlap(self, chunks: List[str]) -> List[str]:
        overlapped_chunks = []
        for i in range(len(chunks) - 1):
            chunk = chunks[i] + ' ' + chunks[i + 1][:self.chunk_overlap]
            overlapped_chunks.append(chunk.strip())
        overlapped_chunks.append(chunks[-1])
        return overlapped_chunks


class Kb:
    # 初始化知识库，从多个文件中读取内容
    def __init__(self, filepaths, llm_embed, chunk_size=250, chunk_overlap=50):
        self.docs = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 检查 filepaths 是否为 None
        if filepaths is not None:
            for filepath in filepaths:
                if filepath.endswith('.pdf'):
                    content = extract_text_from_pdf(filepath)
                elif filepath.endswith('.docx'):
                    content = extract_text_from_docx(filepath)
                elif filepath.endswith('.md'):
                    content = extract_text_from_markdown(filepath)
                else:
                    content = extract_text_from_txt(filepath)
                self.docs.extend(self.split_content(content))
            self.embeds = self.embed(self.docs, llm_embed)
        else:
            # filepaths 为 None 时的处理逻辑
            # 例如，可以打印一条消息，或者初始化为空的知识库
            print("No file paths provided. The knowledge base will be empty.")

    # 根据上下文最大长度，将知识库内容分割成多个chunks
    def split_content(self, content):
        text_splitter = SentenceSplitter(self.chunk_size, self.chunk_overlap)  # 使用实例变量
        chunks = text_splitter.split_text('\n'.join(content))
        return chunks

    # 为了方便文档chunks进行检索，所以会使用embedding模型将chunks转换为向量
    def embed(self, texts, llm_embed):
        embeds = []
        for text in texts:
            embeddings = llm_embed.create_embedding(text)
            if 'data' in embeddings and len(embeddings['data']) > 0:
                embed_vector = embeddings['data'][0]['embedding']
                embeds.append(np.array(embed_vector))
            else:
                raise ValueError("嵌入字典中没有找到 'data' 或 'embedding' 键")
        return np.array(embeds)

    # 计算余弦相似度
    @staticmethod
    def similarity(A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        cosine_sim = dot_product / (norm_A * norm_B)
        return cosine_sim

    def search(self, query, llm_embed):
        max_similarity = 0
        max_similarity_index = 0
        query_embed = self.embed([query], llm_embed)[0]
        for idx, te in enumerate(self.embeds):
            similarity = self.similarity(query_embed, te)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = idx
        return self.docs[max_similarity_index]

############################################################
# 初始化会话状态
if 'llm_query' not in st.session_state:
    st.session_state.llm_query = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_llm_model' not in st.session_state:
    st.session_state.selected_llm_model = None
if 'n_thread' not in st.session_state:
    st.session_state.n_thread = 8
if 'n_ctx' not in st.session_state:
    st.session_state.n_ctx = 2048
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 512
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'top_p' not in st.session_state:
    st.session_state.top_p = 0.8
if 'top_k' not in st.session_state:
    st.session_state.top_k = 100
if 'llm_config' not in st.session_state:
    st.session_state.llm_config = None

if 'llm_embed' not in st.session_state:
    st.session_state.llm_embed = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'selected_embed_model' not in st.session_state:
    st.session_state.selected_embed_model = None
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 250
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 50
if 'embed_config' not in st.session_state:
    st.session_state.embed_config = None


def render_llm_config_ui():
    with st.sidebar:
        st.sidebar.header("Select models")
        selected_llm_model = st.selectbox("选择语言模型", [model["path"] for model in PRESET_LLM_MODELS], key="llm_model_select")
        st.sidebar.header("Parameters")
        n_thread = st.slider("N_thread", 1, 12, 8, step=1, key="n_thread_slider")
        n_ctx = st.slider("N_ctx", 0, 4096, 2048, step=2, key="n_ctx_slider")
        max_tokens = st.slider("Max_tokens", 0, 1024, 512, step=1, key="max_tokens_slider")
        top_p = st.slider("Top_p", 0.0, 1.0, 0.8, step=0.01, key="top_p_slider")
        top_k = st.slider("Top_k", 0, 100, 100, step=1, key="top_k_slider")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01, key="temperature_slider")
        st.session_state.selected_llm_model = selected_llm_model
        st.session_state.n_thread = n_thread
        st.session_state.n_ctx = n_ctx
        st.session_state.max_tokens = max_tokens
        st.session_state.top_p = top_p
        st.session_state.top_k = top_k
        st.session_state.temperature = temperature

def get_llm_config():
    selected_llm_model = st.session_state.selected_llm_model
    n_thread = st.session_state.n_thread
    n_ctx = st.session_state.n_ctx
    max_tokens = st.session_state.max_tokens
    top_p = st.session_state.top_p
    top_k = st.session_state.top_k
    temperature = st.session_state.temperature
    llm_config = {
        'n_thread': n_thread,
        'n_ctx': n_ctx,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'selected_llm_model': selected_llm_model,
    }
    return llm_config

def render_embed_config_ui():
    # 设置划分chunks时的参数
    with st.sidebar:
        st.sidebar.header("Select models")
        selected_embed_model = st.selectbox("选择Embedding模型", [model["path"] for model in PRESET_EMBED_MODELS], key="embed_model_select")
        st.sidebar.header("Split Chunks Parameters")
        chunk_size = st.slider("chunk_size", 50, 500, 250, step=1, key="chunk_size_slider")
        chunk_overlap = st.slider("Chunk_overlap", 0, 100, 50, step=1, key="chunk_overlap_slider")
        st.session_state.selected_embed_model = selected_embed_model
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

def get_embed_config():
    chunk_size = st.session_state.chunk_size
    chunk_overlap = st.session_state.chunk_overlap
    selected_embed_model = st.session_state.selected_embed_model
    embed_config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'selected_embed_model': selected_embed_model,
    }
    return embed_config

@st.cache_resource
def init_llm_embed(embed_config):
    llm_embed = Llama(
        model_path=embed_config.get('selected_embed_model'),
        embedding=True,
    )
    return llm_embed

@st.cache_resource
def init_llm_query(llm_config):
    llm_query = Llama(
        model_path=llm_config.get('selected_llm_model'),
        chat_format="chatml",
        n_ctx=llm_config['n_ctx'],
        n_threads=llm_config.get('n_threads'),
    )
    return llm_query




# 创建 Kb 实例
@st.cache_resource
def create_knowledge_base(temp_dir, _llm_embed, chunk_size, chunk_overlap):
    # 检查 temp_dir 是否为 None
    if temp_dir is None:
        # 如果 temp_dir 是 None，可以选择返回 None 或执行其他操作
        return None
    
    # 如果 temp_dir 不是 None，继续处理文件路径
    filepaths = [os.path.join(temp_dir, file_name) for file_name in os.listdir(temp_dir)]
    return Kb(filepaths, _llm_embed, chunk_size, chunk_overlap)

# 创建一个生成器函数，用于提取和处理每个 chunk 中的 content
def extract_content(response):
    generated_text = ""
    for chunk in response:
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                # 去除换行符
                # content = content.replace('\n', ' ')
                generated_text += content
                yield content
    return generated_text





st.set_page_config(page_title="RAG Application", page_icon="🔍")
st.title("💻 Local RAG Chatbot 🤖")
st.caption("🚀 A RAG chatbot powered by llama-cpp-python 🦙.")



# 文件上传
st.sidebar.header("Upload Data")
uploaded_files = st.sidebar.file_uploader("Upload your data files:", type=["txt", "pdf", "docx"], accept_multiple_files=True)
buttonClean = st.sidebar.button("Clear chat history", key="clean")


render_embed_config_ui()
# 渲染参数配置 UI
render_llm_config_ui()

# 初始化llm模型，因为要使得get_llm_config()返回值不是none，所以st.session_state.llm_config = get_llm_config()必须放到render_llm_config_ui()后面
if st.session_state.llm_config is None:
    st.session_state.llm_config = get_llm_config()
    if st.session_state.llm_query is None:
        st.session_state.llm_query = init_llm_query(st.session_state.llm_config)
# 初始化embed模型
if st.session_state.embed_config is None:
    st.session_state.embed_config = get_embed_config()
    if st.session_state.llm_embed is None:
        st.session_state.llm_embed = init_llm_embed(st.session_state.embed_config)

# 检查参数变化进而重新初始化模型
if st.session_state.llm_config != get_llm_config():
    st.session_state.llm_config = get_llm_config()
    if st.session_state.llm_query is not None:
        del st.session_state.llm_query
        st.cache_resource.clear()
        gc.collect()  # 强制垃圾回收
    st.session_state.llm_query = init_llm_query(st.session_state.llm_config)


# 初始化文件哈希值
current_files_hash = get_files_hash(uploaded_files) if uploaded_files else None

# 检测文件是否发生变化并初始化模型
if 'files_hash' in st.session_state:
    if st.session_state['files_hash'] != current_files_hash:
        st.session_state['files_hash'] = current_files_hash
        if uploaded_files:
            st.session_state['temp_dir'] = handle_file_upload(uploaded_files)
            st.sidebar.success("Files uploaded successfully.")
            if st.session_state.llm_embed is None:
                st.session_state.llm_embed = init_llm_embed(st.session_state.embed_config)
            # 清理旧的知识库实例
            if 'kb' in st.session_state:
                del st.session_state['kb']  # 删除旧的知识库实例
                # 清理 create_knowledge_base 函数的缓存
                create_knowledge_base.clear()
            st.session_state['kb'] = create_knowledge_base(st.session_state['temp_dir'], st.session_state.llm_embed, st.session_state.embed_config.get('chunk_size'), st.session_state.embed_config.get('chunk_overlap'))
        else:
            st.sidebar.error("No uploaded files.")
else:
    if uploaded_files:
        st.session_state['files_hash'] = current_files_hash
        st.session_state['temp_dir'] = handle_file_upload(uploaded_files)
        st.sidebar.success("Files uploaded successfully.")
        if st.session_state.llm_embed is None:
            st.session_state.llm_embed = init_llm_embed(st.session_state.embed_config)
        # 清理旧的知识库实例
        if 'kb' in st.session_state:
            del st.session_state['kb']  # 删除旧的知识库实例
            # 清理 create_knowledge_base 函数的缓存
            create_knowledge_base.clear()
        st.session_state['kb'] = create_knowledge_base(st.session_state['temp_dir'], st.session_state.llm_embed, st.session_state.embed_config.get('chunk_size'), st.session_state.embed_config.get('chunk_overlap'))
    else:
        st.sidebar.error("No uploaded files.")

# 检查嵌入配置是否发生变化
if st.session_state.embed_config != get_embed_config():
    st.session_state.embed_config = get_embed_config()  # 更新配置

    # 清除旧的嵌入模型和知识库实例
    if 'llm_embed' in st.session_state:
        del st.session_state.llm_embed  # 删除旧的嵌入模型
    if 'kb' in st.session_state:
        del st.session_state['kb']  # 删除旧的知识库实例

    # 重新初始化嵌入模型
    st.session_state.llm_embed = init_llm_embed(st.session_state.embed_config)

    # 如果有已上传的文件，重新构建知识库
    if 'temp_dir' in st.session_state and st.session_state['temp_dir'] is not None:
        st.session_state['kb'] = create_knowledge_base(st.session_state['temp_dir'], st.session_state.llm_embed, st.session_state.embed_config.get('chunk_size'), st.session_state.embed_config.get('chunk_overlap'))
    elif 'files_hash' in st.session_state:
        if st.session_state['files_hash'] != current_files_hash:
            st.session_state['files_hash'] = current_files_hash
            if uploaded_files:
                st.session_state['temp_dir'] = handle_file_upload(uploaded_files)
                st.sidebar.success("Files uploaded successfully.")
                st.session_state['kb'] = create_knowledge_base(st.session_state['temp_dir'], st.session_state.llm_embed, st.session_state.embed_config.get('chunk_size'), st.session_state.embed_config.get('chunk_overlap'))
            else:
                st.sidebar.error("No uploaded files.")
    else:
        if uploaded_files:
            st.session_state['files_hash'] = current_files_hash
            st.session_state['temp_dir'] = handle_file_upload(uploaded_files)
            st.sidebar.success("Files uploaded successfully.")
            st.session_state['kb'] = create_knowledge_base(st.session_state['temp_dir'], st.session_state.llm_embed, st.session_state.embed_config.get('chunk_size'), st.session_state.embed_config.get('chunk_overlap'))
        else:
            st.sidebar.error("No uploaded files.")
        

# User and assistant names
U_NAME = "User"
A_NAME = "Assistant"

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            if message["content"] is not None:
                st.markdown(message["content"])
    else:
        with st.chat_message(name="model", avatar="assistant"):
            st.markdown(message["content"])

if buttonClean:
    st.session_state.chat_history = []
    st.session_state.response = ""
    st.rerun()

# 检查是否已上传文件
if 'kb' not in st.session_state or st.session_state['kb'] is None:
    st.warning("Please upload files to start a chat conversation.")
else:
    # User input box
    user_text = st.chat_input("Enter your question")
    if user_text:
        with st.chat_message(U_NAME, avatar="user"):
            st.session_state.chat_history.append({"role": "user", "content": user_text})
            st.markdown(f"{U_NAME}: {user_text}")

        with st.chat_message(A_NAME, avatar="assistant"):
            # 设置Prompt
            prompt_template = """
            【检索增强型生成（RAG）对话系统 - 简体中文回答】

            ### 用户问题
            {user_question}

            ### 背景信息
            基于以下背景信息：
            {context}

            ### 回答要求
            1. **语言**：请使用标准简体中文书写，避免繁体字或方言。
            2. **准确性**：确保答案基于提供的背景信息，不引入未经证实的信息。
            3. **完整性**：
            - 如果问题是封闭式的（如是/否问题），请直接给出明确的答案，并解释原因。
            - 如果问题是开放式的，请提供详尽但不过于冗长的回答，包括所有必要的信息。
            4. **引用**：当引用背景信息中的具体内容时，请使用引号标注，并指明出处或位置（例如，“根据...的数据”）。
            5. **补充信息**：如果背景信息不足以完整回答问题，可以提及这一点，并说明需要哪些额外信息来提供更完整的答案。
            """

            # 在知识库中查找上下文
            context = st.session_state['kb'].search(user_text, st.session_state.llm_embed)
            # 使用 str.format() 方法来填充模板
            prompt = prompt_template.format(user_question=user_text, context=context)
            # 创建消息列表
            msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # 创建聊天完成请求
            response = st.session_state.llm_query.create_chat_completion(
                messages=msgs,
                stream=True,
            )
            # 使用 st.write_stream 进行流式输出
            generated_text = st.write_stream(extract_content(response))
            # 将生成的完整文本添加到会话历史记录中
        st.session_state.chat_history.append({"role": "model", "content": generated_text})

        st.divider()

