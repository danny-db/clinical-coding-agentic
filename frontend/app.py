import logging
import os
import streamlit as st
from model_serving_utils import query_endpoint
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

# Configure page layout and theme
st.set_page_config(
    page_title="Clinical Coding Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4dadbd 0%, #4d838c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .info-box {
        background-color: #f0f8ff;
        border-left: 4px solid #4dadbd;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .user-info {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #f0f8ff;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

user_info = get_user_info()

# Initialize session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🏥 Clinical Coding Assistant")
    st.markdown("---")
    
    # User Information
    if user_info.get("user_name"):
        st.markdown(f"**User:** {user_info['user_name']}")
    if user_info.get("user_email"):
        st.markdown(f"**Email:** {user_info['user_email']}")
    
    st.markdown("---")
    
    # Session Information
    st.markdown("### 📊 Session Info")
    st.markdown(f"**Started:** {st.session_state.session_start.strftime('%H:%M:%S')}")
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📋 Export Chat", use_container_width=True):
        chat_export = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        st.download_button(
            "Download Chat Log",
            chat_export,
            file_name=f"clinical_coding_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    st.markdown("---")
    
    # Help Section
    with st.expander("❓ How to Use"):
        st.markdown("""
        **This AI assistant helps with:**
        - ICD-10 code lookups
        - Clinical documentation queries  
        - Coding guidelines clarification
        - Medical terminology assistance
        
        **Tips:**
        - Be specific with your queries
        - Provide relevant clinical context
        - Ask for code explanations
        """)
    
    with st.expander("📚 Common Queries"):
        st.markdown("""
        Try asking:
        - "What's the ICD-10 code for diabetes?"
        - "Explain the coding for pneumonia"
        - "Help me code this clinical note..."
        - "What are the guidelines for..."
        """)

# Main Content Area
col1, col2 = st.columns([3, 1])

with col1:
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Clinical Coding Assistant</h1>
        <p>AI-powered support for ICD-10 coding and clinical documentation</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # System Status
    st.markdown("### 🟢 System Status")
    st.success("Online")
    st.markdown(f"**Endpoint:** Connected")
    st.markdown(f"**Model:** Ready")

# Information Box
st.markdown("""
<div class="info-box">
    <strong>🎯 Purpose:</strong> This intelligent assistant leverages Databricks' agentic AI workflows to help healthcare professionals with clinical coding tasks. 
    Ask questions about ICD-10 codes, clinical documentation, or coding guidelines.
</div>
""", unsafe_allow_html=True)

# Chat Interface
st.markdown("### 💬 Chat Interface")

# Display welcome message if no chat history
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Welcome to the Clinical Coding Assistant!**
        
        I'm here to help you with:
        - **ICD-10 code lookups** and explanations
        - **Clinical documentation** guidance  
        - **Coding guidelines** and best practices
        - **Medical terminology** clarification
        
        What would you like help with today?
        """)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about ICD-10 codes, clinical documentation, or coding guidelines..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your query..."):
            try:
                assistant_response = query_endpoint(
                    endpoint_name=os.getenv("SERVING_ENDPOINT"),
                    messages=st.session_state.messages,
                    max_tokens=400,
                )["content"]
                st.markdown(assistant_response)
            except Exception as e:
                st.error("❌ Sorry, I encountered an error processing your request. Please try again.")
                logger.error(f"Error querying endpoint: {e}")
                assistant_response = "I apologize, but I'm currently unable to process your request. Please try again later."

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔒 Privacy:** Your conversations are secure")

with col2:
    st.markdown("**⚡ Powered by:** Databricks Mosaic AI")

with col3:
    st.markdown("**📞 Contact us:** danny.wong@databricks.com")
