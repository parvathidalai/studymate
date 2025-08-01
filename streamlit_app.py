import streamlit as st
import os
from dotenv import load_dotenv
from backend.pdf_processor import PDFProcessor
from backend.embedding_engine import EmbeddingEngine
from backend.llm_handler import GraniteLLMHandler

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="StudyMate - AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f8fafc;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .answer-card {
        background-color: #e0f2fe;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0277bd;
    }
    .reference-section {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pdf_processor' not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if 'embedding_engine' not in st.session_state:
    st.session_state.embedding_engine = EmbeddingEngine()
if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = GraniteLLMHandler()
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 StudyMate</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #64748b;">AI-Powered PDF Study Assistant with IBM Granite</p>', unsafe_allow_html=True)
    
    # PDF Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📄 Upload Your Study Materials")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files containing your study materials"
    )
    
    if uploaded_files and not st.session_state.documents_processed:
        with st.spinner("Processing PDFs and building knowledge base..."):
            try:
                # Process PDFs
                chunks = st.session_state.pdf_processor.process_multiple_pdfs(uploaded_files)
                
                if chunks:
                    # Build embeddings and FAISS index
                    st.session_state.embedding_engine.build_faiss_index(chunks)
                    st.session_state.documents_processed = True
                    
                    st.success(f"✅ Successfully processed {len(uploaded_files)} PDF(s) with {len(chunks)} text chunks!")
                else:
                    st.error("❌ No text could be extracted from the uploaded PDFs.")
            
            except Exception as e:
                st.error(f"❌ Error processing PDFs: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question Input Section
    if st.session_state.documents_processed:
        st.subheader("🤔 Ask Your Questions")
        
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is overfitting in machine learning?",
            help="Ask any question related to your uploaded documents"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Ask", type="primary")
        
        if ask_button and question:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # Retrieve relevant chunks
                    relevant_chunks = st.session_state.embedding_engine.search_similar_chunks(question, k=3)
                    
                    if relevant_chunks:
                        # Generate answer using Granite
                        answer = st.session_state.llm_handler.generate_answer(question, relevant_chunks)
                        
                        # Display answer
                        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
                        st.markdown("**Answer:**")
                        st.write(answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display references
                        with st.expander("📖 Referenced Paragraphs"):
                            for i, chunk in enumerate(relevant_chunks):
                                st.markdown(f"**Source {i+1}:** {chunk['source']}")
                                st.markdown('<div class="reference-section">', unsafe_allow_html=True)
                                st.write(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add to history
                        st.session_state.qa_history.append({
                            "question": question,
                            "answer": answer,
                            "references": relevant_chunks
                        })
                        
                    else:
                        st.warning("⚠️ No relevant information found in the uploaded documents.")
                
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
    
    # Q&A History Section
    if st.session_state.qa_history:
        st.subheader("📝 Q&A History")
        
        # Download button
        history_text = ""
        for i, qa in enumerate(st.session_state.qa_history):
            history_text += f"Q{i+1}: {qa['question']}\n"
            history_text += f"A{i+1}: {qa['answer']}\n"
            history_text += "-" * 50 + "\n\n"
        
        st.download_button(
            label="💾 Download Q&A History",
            data=history_text,
            file_name="studymate_qa_history.txt",
            mime="text/plain"
        )
        
        # Display history
        for i, qa in enumerate(reversed(st.session_state.qa_history)):
            with st.expander(f"Q: {qa['question'][:100]}..."):
                st.markdown("**Question:**")
                st.write(qa['question'])
                st.markdown("**Answer:**")
                st.write(qa['answer'])

if __name__ == "__main__":
    main()