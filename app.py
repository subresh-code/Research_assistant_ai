# app.py - Research Paper Reading Assistant
# Required packages:
# pip install streamlit langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pypdf python-dotenv

import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

# ────────────────────────────────────────────────
# PAGE CONFIGURATION - MUST BE FIRST
# ────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Paper Assistant", 
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# SESSION STATE INITIALIZATION - MUST BE EARLY
# ────────────────────────────────────────────────

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "paper_analysis" not in st.session_state:
        st.session_state.paper_analysis = None
    if "paper_text" not in st.session_state:
        st.session_state.paper_text = ""
    if "reading_level" not in st.session_state:
        st.session_state.reading_level = "Intermediate"
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

# Call initialization IMMEDIATELY
initialize_session_state()

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("⚠️ OPENROUTER_API_KEY not found. Add it to .env or Streamlit secrets.")
    st.stop()

LLM_MODEL = "meta-llama/llama-3.1-70b-instruct"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# ────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────

def extract_text_from_pdfs(pdf_files):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n\n"
        except Exception as e:
            st.warning(f"⚠️ Error reading {pdf.name}: {e}")
    return text.strip()


def create_vectorstore(text):
    """Split text into chunks and create FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        return None
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model=EMBEDDING_MODEL,
    )
    
    return FAISS.from_texts(chunks, embedding=embeddings)


def create_rag_chain(vectorstore, mode="chat"):
    """Create RAG chain with LLM and retriever."""
    llm = ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=LLM_MODEL,
        temperature=0.3,
        max_tokens=3000,
    )

    if mode == "explain":
        system_prompt = f"""You are an expert research paper explainer helping someone at the {st.session_state.reading_level} level.

Your task:
1. Explain concepts clearly and simply
2. Use analogies and examples when helpful
3. Break down complex ideas into digestible parts
4. Define technical terms in simple language
5. Connect ideas to real-world applications

Context from the paper:
{{context}}"""
    else:
        system_prompt = """You are a research paper assistant helping users understand academic papers.

Answer based on the provided context. Be clear, accurate, and helpful.
If information is not in the paper, clearly say so.

Context from the paper:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    def get_chat_history():
        """Safely get chat history"""
        messages = st.session_state.get("messages", [])
        if len(messages) > 10:
            return messages[-10:]
        return messages

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
            "chat_history": lambda _: get_chat_history()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def analyze_paper(text):
    """Perform initial analysis of the research paper."""
    llm = ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=LLM_MODEL,
        temperature=0.2,
        max_tokens=2000,
    )
    
    # Take first 8000 characters for analysis
    sample_text = text[:8000]
    
    prompt = f"""Analyze this research paper excerpt and provide:

1. **Title & Authors**: Extract if visible
2. **Main Topic**: What is this paper about? (2-3 sentences)
3. **Key Contribution**: What's the main innovation or finding?
4. **Research Field**: What area of study (e.g., Machine Learning, NLP, Computer Vision)
5. **Difficulty Level**: Beginner/Intermediate/Advanced
6. **Key Terms**: List 5-7 important technical terms used

Paper excerpt:
{sample_text}

Format your response clearly with headers."""
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Analysis failed: {str(e)}"


# ────────────────────────────────────────────────
# QUICK ACTION FUNCTIONS
# ────────────────────────────────────────────────

def generate_summary(detail_level="medium"):
    """Generate paper summary at different detail levels."""
    prompts = {
        "brief": "Provide a 3-4 sentence summary of this paper's main contribution and findings.",
        "medium": "Provide a comprehensive summary covering: 1) Main problem addressed, 2) Proposed solution/method, 3) Key results, 4) Significance. Use 2-3 paragraphs.",
        "detailed": "Provide a detailed summary covering: 1) Background and motivation, 2) Problem statement, 3) Methodology and approach, 4) Experiments and results, 5) Conclusions and impact. Use 4-5 paragraphs."
    }
    
    chain = st.session_state.get("chain")
    if chain:
        try:
            response = chain.invoke(prompts[detail_level])
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    return "Please process the paper first."


def explain_like_im(level):
    """Explain paper at different comprehension levels."""
    prompts = {
        "5": "Explain this research paper as if I'm 5 years old. Use very simple words and fun examples.",
        "high_school": "Explain this paper at a high school level. Use clear language and relatable examples.",
        "undergrad": "Explain this paper at an undergraduate level. Include technical concepts but explain them clearly.",
        "expert": "Provide an expert-level technical explanation with full details."
    }
    
    chain = st.session_state.get("chain")
    if chain:
        try:
            response = chain.invoke(prompts[level])
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    return "Please process the paper first."


def ask_question(question):
    """Ask a question and add to chat history"""
    st.session_state.messages.append(HumanMessage(content=question))
    
    chain = st.session_state.get("chain")
    if chain:
        try:
            with st.spinner("Thinking..."):
                response = chain.invoke(question)
                st.session_state.messages.append(AIMessage(content=response))
                return True
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.messages.append(AIMessage(content=error_msg))
            return False
    return False


# ────────────────────────────────────────────────
# STREAMLIT UI
# ────────────────────────────────────────────────

# Custom CSS
st.markdown("""
<style>
    .quick-action-btn {
        margin: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Research Paper Reading Assistant")
st.caption("Understand complex research papers with AI assistance")

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────

with st.sidebar:
    st.header("📄 Upload Paper")
    
    pdf_files = st.file_uploader(
        "Upload research paper (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research papers"
    )
    
    # Reading level selector
    st.session_state.reading_level = st.select_slider(
        "📊 Your Knowledge Level",
        options=["Beginner", "Intermediate", "Advanced", "Expert"],
        value=st.session_state.reading_level,
        help="Adjust explanations to your level"
    )
    
    if st.button("🚀 Process Paper", type="primary", use_container_width=True):
        if not pdf_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("📖 Reading and analyzing paper..."):
                try:
                    text = extract_text_from_pdfs(pdf_files)
                    
                    if not text:
                        st.error("❌ Could not extract text from PDF.")
                    else:
                        st.session_state.paper_text = text
                        
                        # Create vectorstore
                        vectorstore = create_vectorstore(text)
                        
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.chain = create_rag_chain(vectorstore)
                            st.session_state.processed_files = [f.name for f in pdf_files]
                            
                            # Perform initial analysis
                            st.session_state.paper_analysis = analyze_paper(text)
                            
                            # Clear previous messages
                            st.session_state.messages = []
                            
                            st.success(f"✅ Paper processed! {len(vectorstore.docstore._dict)} sections indexed.")
                            st.balloons()
                        else:
                            st.error("❌ Failed to process paper.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Quick Actions
    if st.session_state.vectorstore:
        st.subheader("⚡ Quick Actions")
        
        with st.expander("📝 Generate Summary", expanded=False):
            if st.button("Brief (3-4 sentences)", use_container_width=True, key="brief_btn"):
                summary = generate_summary("brief")
                st.session_state.messages.append(AIMessage(content=f"**Brief Summary:**\n\n{summary}"))
                st.rerun()
            
            if st.button("Medium (2-3 paragraphs)", use_container_width=True, key="medium_btn"):
                summary = generate_summary("medium")
                st.session_state.messages.append(AIMessage(content=f"**Summary:**\n\n{summary}"))
                st.rerun()
            
            if st.button("Detailed (4-5 paragraphs)", use_container_width=True, key="detailed_btn"):
                summary = generate_summary("detailed")
                st.session_state.messages.append(AIMessage(content=f"**Detailed Summary:**\n\n{summary}"))
                st.rerun()
        
        with st.expander("🎯 Explain Like I'm...", expanded=False):
            if st.button("👶 5 Years Old", use_container_width=True, key="eli5_btn"):
                explanation = explain_like_im("5")
                st.session_state.messages.append(AIMessage(content=f"**ELI5 Explanation:**\n\n{explanation}"))
                st.rerun()
            
            if st.button("🎓 High School Student", use_container_width=True, key="hs_btn"):
                explanation = explain_like_im("high_school")
                st.session_state.messages.append(AIMessage(content=f"**High School Level:**\n\n{explanation}"))
                st.rerun()
            
            if st.button("🎯 Undergraduate", use_container_width=True, key="undergrad_btn"):
                explanation = explain_like_im("undergrad")
                st.session_state.messages.append(AIMessage(content=f"**Undergraduate Level:**\n\n{explanation}"))
                st.rerun()
        
        st.divider()
        
        # Suggested questions based on paper type
        st.subheader("💡 Ask About")
        
        questions = [
            "What problem does this paper solve?",
            "What is the main contribution?",
            "How does the proposed method work?",
            "What are the key results?",
            "What are the limitations?",
            "How does this compare to previous work?",
            "What are the practical applications?",
            "What mathematical concepts are used?",
            "Explain the experimental setup",
            "What future work is suggested?"
        ]
        
        for i, q in enumerate(questions):
            if st.button(q, use_container_width=True, key=f"q_{i}"):
                if ask_question(q):
                    st.rerun()
    
    st.divider()
    
    # Show processed files
    if st.session_state.processed_files:
        st.subheader("📋 Processed Files")
        for filename in st.session_state.processed_files:
            st.text(f"• {filename}")
    
    st.divider()
    
    # Export and clear options
    if st.session_state.messages:
        chat_text = "\n\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in st.session_state.messages
        ])
        st.download_button(
            "💾 Export Chat",
            chat_text,
            file_name="paper_discussion.txt",
            use_container_width=True
        )
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ────────────────────────────────────────────────
# MAIN CONTENT AREA
# ────────────────────────────────────────────────

# Show paper analysis in an expander
if st.session_state.paper_analysis:
    with st.expander("📊 Paper Overview", expanded=True):
        st.markdown(st.session_state.paper_analysis)

# Create tabs for different views
if st.session_state.vectorstore:
    tab1, tab2 = st.tabs(["💬 Chat", "📚 Study Guide"])
    
    with tab1:
        # Chat interface
        if not st.session_state.messages:
            st.info("👋 Ask me anything about the research paper! Try the suggested questions in the sidebar or ask your own.")
        
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)
        
        # Chat input
        if prompt := st.chat_input("Ask about the paper..."):
            st.session_state.messages.append(HumanMessage(content=prompt))
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    chain = st.session_state.get("chain")
                    if chain:
                        try:
                            response = chain.invoke(prompt)
                            st.session_state.messages.append(AIMessage(content=response))
                            st.write(response)
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.session_state.messages.append(AIMessage(content=error_msg))
                            st.write(error_msg)
                    else:
                        error_msg = "⚠️ Please process the paper first."
                        st.write(error_msg)
    
    with tab2:
        st.subheader("📚 Study Guide Generator")
        st.write("Generate structured study materials from the paper")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Key Concepts List", use_container_width=True, key="concepts_btn"):
                chain = st.session_state.get("chain")
                if chain:
                    with st.spinner("Extracting key concepts..."):
                        response = chain.invoke(
                            "List and briefly explain the 10 most important concepts, terms, or ideas in this paper. Format as a numbered list."
                        )
                        st.markdown(response)
            
            if st.button("❓ Practice Questions", use_container_width=True, key="practice_btn"):
                chain = st.session_state.get("chain")
                if chain:
                    with st.spinner("Generating questions..."):
                        response = chain.invoke(
                            "Generate 10 thought-provoking questions that would help someone deeply understand this paper. Include a mix of conceptual and technical questions."
                        )
                        st.markdown(response)
        
        with col2:
            if st.button("🎯 Main Takeaways", use_container_width=True, key="takeaways_btn"):
                chain = st.session_state.get("chain")
                if chain:
                    with st.spinner("Identifying takeaways..."):
                        response = chain.invoke(
                            "What are the 5-7 main takeaways someone should remember from this paper? Be specific and actionable."
                        )
                        st.markdown(response)
            
            if st.button("🔗 Prerequisite Knowledge", use_container_width=True, key="prereq_btn"):
                chain = st.session_state.get("chain")
                if chain:
                    with st.spinner("Analyzing prerequisites..."):
                        response = chain.invoke(
                            "What background knowledge, concepts, or papers should someone be familiar with before reading this paper?"
                        )
                        st.markdown(response)

else:
    # Welcome screen
    st.info("👋 Welcome! Upload a research paper in the sidebar to get started.")
    
    st.markdown("""
    ### Features:
    - 📝 **Smart Summaries** - Brief, medium, or detailed summaries
    - 🎯 **Multi-Level Explanations** - From ELI5 to expert level
    - 💡 **Suggested Questions** - Common questions to explore
    - 📊 **Automatic Analysis** - Quick overview of the paper
    - 💬 **Interactive Chat** - Ask anything about the paper
    - 📚 **Study Guide** - Generate practice questions and key concepts
    - 💾 **Export Chat** - Save your discussion for later
    
    ### Perfect for understanding papers like:
    - "Attention Is All You Need" (Transformers)
    - "BERT: Pre-training of Deep Bidirectional Transformers"
    - "GPT-3: Language Models are Few-Shot Learners"
    - And any other research paper!
    """)