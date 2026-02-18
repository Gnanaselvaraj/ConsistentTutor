import streamlit as st
import tempfile, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.rag_engine import ConsistentTutorRAG

st.set_page_config("ConsistentTutor - Universal AI Tutor", layout="wide", page_icon="🎓")

# Header
st.title("🎓 ConsistentTutor - Universal AI Tutor")
st.markdown("""
**Supports all subjects, all classes, all boards, and college education!**  
Upload your textbooks and get intelligent, context-aware answers from your study materials.
""")

@st.cache_resource
def load_tutor():
    # Enable multi-model architecture for superior intelligence
    return ConsistentTutorRAG(use_multi_model=True, use_meta_prompting=True)

tutor = load_tutor()

# ---------- HELPER FUNCTIONS ----------
def get_kb_stats(subject_dir: str) -> dict:
    """Get statistics about a knowledge base"""
    import pickle
    from datetime import datetime
    
    stats = {
        'is_multimodal': False,
        'num_texts': 0,
        'num_images': 0,
        'created_date': None,
        'size_mb': 0,
        'files': []
    }
    
    if not os.path.exists(subject_dir):
        return stats
    
    # Check if multimodal
    stats['is_multimodal'] = os.path.exists(os.path.join(subject_dir, "text_index.faiss"))
    
    # Count texts
    texts_file = os.path.join(subject_dir, "texts.pkl")
    if os.path.exists(texts_file):
        try:
            with open(texts_file, 'rb') as f:
                texts = pickle.load(f)
                stats['num_texts'] = len(texts) if texts else 0
        except:
            pass
    
    # Count images (multimodal only)
    if stats['is_multimodal']:
        img_meta_file = os.path.join(subject_dir, "image_metadata.pkl")
        if os.path.exists(img_meta_file):
            try:
                with open(img_meta_file, 'rb') as f:
                    img_meta = pickle.load(f)
                    stats['num_images'] = len(img_meta) if img_meta else 0
            except:
                pass
    
    # Get creation date
    try:
        creation_time = os.path.getctime(subject_dir)
        stats['created_date'] = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M")
    except:
        pass
    
    # Calculate total size
    total_size = 0
    for root, dirs, files in os.walk(subject_dir):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                stats['files'].append(file)
            except:
                pass
    stats['size_mb'] = total_size / (1024 * 1024)
    
    return stats

def delete_knowledge_base(subject: str):
    """Delete a knowledge base directory"""
    import shutil
    subject_dir = os.path.join("vector_db", subject)
    if os.path.exists(subject_dir):
        try:
            shutil.rmtree(subject_dir)
            return True
        except Exception as e:
            st.error(f"Error deleting knowledge base: {e}")
            return False
    return False

def export_all_stats():
    """Export all KB statistics to JSON for download"""
    import json
    from datetime import datetime
    
    all_stats = {}
    for subject in st.session_state.subjects:
        subject_dir = os.path.join("vector_db", subject)
        stats = get_kb_stats(subject_dir)
        all_stats[subject] = stats
    
    # Add metadata
    export_data = {
        "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_subjects": len(st.session_state.subjects),
        "subjects": all_stats
    }
    
    # Create download button
    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="💾 Download Stats JSON",
        data=json_str,
        file_name=f"consistenttutor_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# ---------- SESSION ----------
if "subjects" not in st.session_state:
    st.session_state.subjects = os.listdir("vector_db") if os.path.exists("vector_db") else []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Knowledge Base" if not st.session_state.subjects else "Tutor Chat"
if "kb_subtab" not in st.session_state:
    st.session_state.kb_subtab = "Add New Subject"

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("📊 Knowledge Base Stats")
    if st.session_state.subjects:
        # Calculate total stats
        total_size = 0
        total_multimodal = 0
        for subj in st.session_state.subjects:
            stats = get_kb_stats(os.path.join("vector_db", subj))
            total_size += stats['size_mb']
            if stats['is_multimodal']:
                total_multimodal += 1
        
        col1, col2 = st.columns(2)
        col1.metric("📚 Subjects", len(st.session_state.subjects))
        col2.metric("💾 Total Size", f"{total_size:.1f} MB")
        
        st.write(f"🖼️ Multimodal: {total_multimodal}/{len(st.session_state.subjects)}")
        
        with st.expander("📚 Quick View"):
            for i, subj in enumerate(st.session_state.subjects, 1):
                stats = get_kb_stats(os.path.join("vector_db", subj))
                icon = "🖼️" if stats['is_multimodal'] else "📝"
                st.write(f"{i}. {icon} {subj} ({stats['num_texts']} chunks)")
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🗄️ Manage Knowledge Bases", use_container_width=True):
            st.session_state.active_tab = "Knowledge Base"
            st.session_state.kb_subtab = "Manage Existing"
            st.rerun()
    else:
        st.info("No knowledge bases yet")
        if st.button("➕ Add First Subject", use_container_width=True, type="primary"):
            st.session_state.active_tab = "Knowledge Base"
            st.session_state.kb_subtab = "Add New Subject"
            st.rerun()
    
    st.markdown("---")
    st.subheader("ℹ️ Features")
    st.markdown("""
    ✅ **Multi-subject support**  
    ✅ **Multimodal (text + images)**  
    ✅ **Source citations**  
    ✅ **Practice questions**  
    ✅ **Context-aware chat**  
    ✅ **On-device privacy**
    """)

# ---------- MAIN TABS ----------
tab_options = ["📚 Knowledge Base", "🎓 Tutor Chat", "📝 Practice Questions"]
selected_tab = st.radio(
    "Navigation",
    tab_options,
    index=tab_options.index(st.session_state.active_tab) if st.session_state.active_tab in tab_options else 0,
    horizontal=True,
    label_visibility="collapsed"
)
st.session_state.active_tab = selected_tab

# Clean tab name (remove emoji)
active_tab_clean = selected_tab.replace("📚 ", "").replace("🎓 ", "").replace("📝 ", "")

# ---------- KB TAB ----------
if active_tab_clean == "Knowledge Base":
    # Two columns: Management and Add New
    kb_management, kb_add_new = st.tabs(["📋 Manage Existing", "➕ Add New Subject"])
    
    # ===== KNOWLEDGE BASE MANAGEMENT =====
    with kb_management:
        st.subheader("📋 Knowledge Base Management")
        
        if not st.session_state.subjects:
            st.info("📚 No knowledge bases found. Add your first subject using the 'Add New Subject' tab.")
        else:
            # Bulk operations
            col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
            with col_header1:
                st.markdown(f"**Total Subjects:** {len(st.session_state.subjects)}")
            with col_header2:
                if st.button("📊 Export Stats", help="Export all KB statistics to JSON"):
                    export_all_stats()
            with col_header3:
                if st.button("🗑️ Delete All", type="secondary", help="Delete all knowledge bases"):
                    st.session_state['confirm_delete_all'] = True
            
            # Confirm delete all
            if st.session_state.get('confirm_delete_all', False):
                st.error("⚠️ **WARNING:** This will delete ALL knowledge bases!")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ Yes, Delete Everything", type="primary"):
                        for subj in list(st.session_state.subjects):
                            delete_knowledge_base(subj)
                        st.session_state.subjects = []
                        st.session_state['confirm_delete_all'] = False
                        st.success("✅ All knowledge bases deleted")
                        st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel"):
                        st.session_state['confirm_delete_all'] = False
                        st.rerun()
            
            st.markdown("---")
            
            # Display each subject with management options
            for subject in st.session_state.subjects:
                subject_dir = os.path.join("vector_db", subject)
                
                # Get KB statistics
                kb_stats = get_kb_stats(subject_dir)
                
                with st.expander(f"📖 {subject}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**📊 Statistics:**")
                        if kb_stats['is_multimodal']:
                            st.write(f"✅ Multimodal KB")
                            st.write(f"📝 Text chunks: {kb_stats['num_texts']}")
                            st.write(f"🖼️ Images: {kb_stats['num_images']}")
                        else:
                            st.write(f"📝 Text-only KB")
                            st.write(f"📝 Text chunks: {kb_stats['num_texts']}")
                        
                        if kb_stats['created_date']:
                            st.write(f"📅 Created: {kb_stats['created_date']}")
                        
                        st.write(f"💾 Size: {kb_stats['size_mb']:.2f} MB")
                    
                    with col2:
                        # Rebuild button
                        if st.button(f"🔄 Rebuild", key=f"rebuild_{subject}"):
                            st.session_state[f'confirm_rebuild_{subject}'] = True
                        
                        # View details button
                        if st.button(f"ℹ️ Details", key=f"details_{subject}"):
                            st.session_state[f'show_details_{subject}'] = not st.session_state.get(f'show_details_{subject}', False)
                    
                    with col3:
                        # Delete button
                        if st.button(f"🗑️ Delete", key=f"delete_{subject}", type="secondary"):
                            st.session_state[f'confirm_delete_{subject}'] = True
                    
                    # Confirmation dialogs
                    if st.session_state.get(f'confirm_delete_{subject}', False):
                        st.warning(f"⚠️ Are you sure you want to delete **{subject}**? This cannot be undone!")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button(f"✅ Yes, Delete", key=f"confirm_yes_{subject}", type="primary"):
                                delete_knowledge_base(subject)
                                st.session_state.subjects.remove(subject)
                                st.session_state[f'confirm_delete_{subject}'] = False
                                st.success(f"✅ Deleted: {subject}")
                                st.rerun()
                        with col_no:
                            if st.button(f"❌ Cancel", key=f"confirm_no_{subject}"):
                                st.session_state[f'confirm_delete_{subject}'] = False
                                st.rerun()
                    
                    # Rebuild confirmation
                    if st.session_state.get(f'confirm_rebuild_{subject}', False):
                        st.info(f"🔄 To rebuild **{subject}**, please upload new PDF files below:")
                        rebuild_files = st.file_uploader(
                            "Upload PDFs for rebuild",
                            type="pdf",
                            accept_multiple_files=True,
                            key=f"rebuild_upload_{subject}"
                        )
                        
                        rebuild_multimodal = st.checkbox(
                            "🖼️ Enable Multimodal",
                            value=kb_stats['is_multimodal'],
                            key=f"rebuild_multimodal_{subject}"
                        )
                        
                        col_rebuild, col_cancel_rebuild = st.columns(2)
                        
                        with col_rebuild:
                            if st.button(f"🔄 Start Rebuild", key=f"start_rebuild_{subject}", type="primary"):
                                if rebuild_files:
                                    # Save uploaded files
                                    rebuild_paths = []
                                    for f in rebuild_files:
                                        t = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                                        t.write(f.getbuffer())
                                        rebuild_paths.append(t.name)
                                    
                                    # Delete old KB
                                    delete_knowledge_base(subject)
                                    
                                    # Rebuild with new files
                                    with st.spinner(f"Rebuilding {subject}..."):
                                        rebuild_bar = st.progress(0)
                                        rebuild_msg = st.empty()
                                        
                                        if rebuild_multimodal:
                                            num_texts, num_images = tutor.ingest_pdfs_multimodal(
                                                rebuild_paths, subject, extract_images=True,
                                                cb=lambda p, m: (rebuild_bar.progress(int(p * 100)), rebuild_msg.write(m))
                                            )
                                            st.success(f"✅ Rebuilt: {num_texts} text chunks, {num_images} images")
                                        else:
                                            num_texts = tutor.ingest_pdfs(
                                                rebuild_paths, subject,
                                                cb=lambda p, m: (rebuild_bar.progress(int(p * 100)), rebuild_msg.write(m))
                                            )
                                            st.success(f"✅ Rebuilt: {num_texts} text chunks")
                                        
                                        st.session_state[f'confirm_rebuild_{subject}'] = False
                                        st.balloons()
                                        st.rerun()
                                else:
                                    st.warning("Please upload at least one PDF file")
                        
                        with col_cancel_rebuild:
                            if st.button(f"❌ Cancel Rebuild", key=f"cancel_rebuild_{subject}"):
                                st.session_state[f'confirm_rebuild_{subject}'] = False
                                st.rerun()
                    
                    # Show details if requested
                    if st.session_state.get(f'show_details_{subject}', False):
                        st.markdown("---")
                        st.markdown("**🔍 Detailed Information:**")
                        st.json({
                            "Subject": subject,
                            "Type": "Multimodal" if kb_stats['is_multimodal'] else "Text-only",
                            "Text Chunks": kb_stats['num_texts'],
                            "Images": kb_stats['num_images'],
                            "Files": kb_stats['files'],
                            "Size (MB)": round(kb_stats['size_mb'], 2)
                        })
    
    # ===== ADD NEW KNOWLEDGE BASE =====
    with kb_add_new:
        st.subheader("➕ Add New Subject")
    
    # Multimodal toggle
    multimodal_enabled = st.checkbox(
        "🖼️ Enable Multimodal (Extract & Index Diagrams)", 
        value=True,
        help="Extract diagrams from PDFs and enable image-based retrieval using CLIP embeddings"
    )
    
    files = st.file_uploader("Upload textbook PDFs", type="pdf", accept_multiple_files=True)
    if files:
        paths = []
        original_names = []
        for f in files:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            t.write(f.getbuffer())
            paths.append(t.name)
            original_names.append(f.name)

        # Intelligent subject detection using LLM
        def detect_subject_intelligently(filename: str, pdf_path: str) -> str:
            """Use LLM to analyze filename and PDF content intelligently"""
            try:
                # STEP 1: Extract sample content from first few pages
                from pypdf import PdfReader
                reader = PdfReader(pdf_path)
                sample_text = ""
                # Read first 3-5 pages
                for i in range(min(5, len(reader.pages))):
                    page_text = reader.pages[i].extract_text()
                    sample_text += page_text[:2000] if page_text else ""
                
                # STEP 2: Intelligent analysis with FILENAME as PRIMARY source
                prompt = f"""Analyze this document intelligently to create a descriptive name.

📄 FILENAME (PRIMARY SOURCE - highest priority): {filename}
📖 Content sample (for validation and details):
{sample_text[:1500]}

CRITICAL RULES:
1. FILENAME is the PRIMARY and MOST RELIABLE source for subject identification
2. If filename clearly indicates "Computer Science" → use "Computer Science" (not Commerce!)
3. If filename clearly indicates "Commerce" → use "Commerce" (not Computer Science!)
4. Content should only ENHANCE the filename analysis, not override it
5. Document types can be: textbook, study notes, tutor notes, practice material, reference guide, etc.

OUTPUT FORMAT (exactly 3 parts separated by |):
[Subject from filename] | [Class/Level from content or filename] | [Board/Source from content or "General"]

EXAMPLES:
File: "Class_12_Computer_Science_English_Medium-2024.pdf" → Computer Science | Class 12 | General
File: "Class_12_Commerce_English_Medium-2024.pdf" → Commerce | Class 12 | General  
File: "Physics_Notes_Grade_11_CBSE.pdf" → Physics | Grade 11 | CBSE
File: "Accountancy_Study_Material.pdf" → Accountancy | General | General

RESPOND WITH ONLY THE 3-PART FORMAT:"""
                
                response = tutor.llm.invoke(prompt).strip()
                
                # Parse the response
                lines = [l.strip() for l in response.split('\n') if l.strip()]
                
                import re
                
                for line in lines:
                    # Skip explanation/instruction lines
                    if any(skip in line.lower() for skip in ['example', 'note', 'format', 'respond', 'output', 'with only', 'file:', '→']):
                        continue
                    
                    # Look for the formatted line (contains |)
                    if '|' in line:
                        parts = [p.strip() for p in line.split('|')]
                        
                        if len(parts) >= 2:
                            # Extract parts
                            subject_part = parts[0].strip()
                            level_part = parts[1].strip() if len(parts) > 1 else "General"
                            board_part = parts[2].strip() if len(parts) > 2 else "General"
                            
                            # Clean common LLM mistakes
                            subject_part = re.sub(r'^\d+\.?\s+', '', subject_part)
                            subject_part = re.sub(r'^(Subject|Class|Board|Part)\s*:?\s*', '', subject_part, flags=re.IGNORECASE)
                            
                            level_part = re.sub(r'^\d+\.?\s+', '', level_part)
                            level_part = re.sub(r'^(Subject|Class|Board|Level|Part)\s*:?\s*', '', level_part, flags=re.IGNORECASE)
                            
                            board_part = re.sub(r'^\d+\.?\s+', '', board_part)
                            board_part = re.sub(r'^(Subject|Class|Board|Part)\s*:?\s*', '', board_part, flags=re.IGNORECASE)
                            
                            # Validate and build result
                            if subject_part and len(subject_part) > 2:
                                clean_subject = f"{subject_part} - {level_part} - {board_part}"
                                
                                # Sanitize for filesystem
                                clean_subject = clean_subject.replace('\n', ' ').replace('\r', ' ')
                                clean_subject = clean_subject.replace('/', '-').replace('\\', '-')
                                clean_subject = clean_subject.replace(':', '-').replace('*', '').replace('?', '')
                                clean_subject = clean_subject.replace('"', '').replace('<', '').replace('>', '')
                                clean_subject = clean_subject.replace('|', '-')
                                clean_subject = ''.join(c for c in clean_subject if c.isalnum() or c in ' -_()')
                                clean_subject = ' '.join(clean_subject.split())
                                
                                return clean_subject[:100].strip()
                
                # Fallback: Simple filename-based name
                fallback = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
                fallback = ''.join(c for c in fallback if c.isalnum() or c in ' ')
                return ' '.join(fallback.split())[:60].strip() + " - General - General"
                    
            except Exception as e:
                st.warning(f"Could not auto-detect subject details: {e}")
                return "Unknown"
        
        # Auto-detect with progress indicator
        if files and "detected_subject" not in st.session_state:
            with st.spinner("🔍 Analyzing textbook to detect subject, class, and board..."):
                detected_subject = detect_subject_intelligently(files[0].name, paths[0])
                st.session_state.detected_subject = detected_subject
        else:
            detected_subject = st.session_state.get("detected_subject", "Unknown")
        
        subject = st.text_input("Subject Identifier", value=detected_subject, 
                               help="Format: Subject - Class/Level - Board. Edit if needed.")
        
        # Check for duplicates
        existing_subjects = st.session_state.subjects
        duplicate_found = None
        similar_found = []
        
        if subject and subject.strip():
            # Exact match check
            if subject in existing_subjects:
                duplicate_found = subject
            
            # Similar name check (fuzzy matching)
            import difflib
            for existing in existing_subjects:
                similarity = difflib.SequenceMatcher(None, subject.lower(), existing.lower()).ratio()
                if similarity > 0.7 and similarity < 1.0:  # 70-99% similar
                    similar_found.append((existing, similarity))
        
        # Show warnings if duplicates detected
        if duplicate_found:
            st.error(f"⚠️ **Duplicate Detected!** A knowledge base with the name **'{duplicate_found}'** already exists.")
            st.warning("**Options:**\n- Change the subject identifier above to create a new KB\n- Or use the option below to overwrite")
            
            overwrite_confirm = st.checkbox("🔄 I want to **OVERWRITE** the existing knowledge base (this will delete the old one)", key="overwrite_checkbox")
            
        elif similar_found:
            st.warning(f"⚠️ **Similar Knowledge Base Found!**")
            for sim_name, sim_score in similar_found:
                st.write(f"- **{sim_name}** ({int(sim_score*100)}% similar)")
            st.info("💡 **Tip:** If this is the same subject, consider using the exact same name to overwrite, or choose a more distinct name.")
        
        bar = st.progress(0)
        msg = st.empty()

        if st.button("Build Knowledge Base", type="primary"):
            if subject == "Unknown" or not subject.strip():
                st.error("Please enter a valid subject identifier")
            elif duplicate_found and not st.session_state.get("overwrite_checkbox", False):
                st.error("❌ **Cannot create duplicate!** Please check the overwrite option above or change the subject name.")
            else:
                # Handle overwrite case - delete existing KB
                if duplicate_found and st.session_state.get("overwrite_checkbox", False):
                    import shutil
                    old_kb_path = os.path.join(tutor.db_dir, subject)
                    if os.path.exists(old_kb_path):
                        try:
                            shutil.rmtree(old_kb_path)
                            st.info(f"🗑️ Deleted existing KB: {subject}")
                        except Exception as e:
                            st.error(f"Failed to delete old KB: {e}")
                            st.stop()
                
                with st.spinner("Building knowledge base..."):
                    if multimodal_enabled:
                        # Multimodal ingestion
                        num_texts, num_images = tutor.ingest_pdfs_multimodal(
                            paths,
                            subject,
                            extract_images=True,
                            cb=lambda p, m: (bar.progress(int(p * 100)), msg.write(m))
                        )
                        success_msg = f"✅ Multimodal KB built: {num_texts} text chunks, {num_images} diagrams"
                    else:
                        # Text-only ingestion
                        num_texts = tutor.ingest_pdfs(
                            paths,
                            subject,
                            cb=lambda p, m: (bar.progress(int(p * 100)), msg.write(m))
                        )
                        success_msg = f"✅ Text-only KB built: {num_texts} chunks"
                    
                    # Add overwrite indicator if applicable
                    if duplicate_found:
                        success_msg = "🔄 " + success_msg + " (Overwritten)"
                    
                    if subject not in st.session_state.subjects:
                        st.session_state.subjects.append(subject)
                    # Clear detection cache and overwrite flag
                    if "detected_subject" in st.session_state:
                        del st.session_state.detected_subject
                    if "overwrite_checkbox" in st.session_state:
                        del st.session_state.overwrite_checkbox
                    st.success(success_msg)
                    if multimodal_enabled and num_images > 0:
                        st.info(f"🖼️ Enhanced with {num_images} visual aids for better learning!")
                    st.balloons()

# ---------- CHAT TAB ----------
elif active_tab_clean == "Tutor Chat":
    if not st.session_state.subjects:
        st.info("📚 Please build a knowledge base first by uploading your textbooks in the Knowledge Base tab.")
    else:
        subject = st.selectbox("📖 Select Subject/Course", st.session_state.subjects, 
                              help="Choose the subject or course you want to study")
        
        st.markdown(f"**Current Topic:** `{subject}`")
        st.markdown("---")
        
        # Display chat history
        if st.session_state.chat:
            st.subheader("💬 Conversation History")
            for turn in st.session_state.chat:
                st.markdown(f"**You:** {turn['user']}")
                st.markdown(turn['assistant'], unsafe_allow_html=True)
                st.markdown("---")
        
        q = st.text_input("💭 Ask a question", placeholder="e.g., Explain recursion in programming...")

        col1, col2 = st.columns([3, 1])
        with col1:
            ask_button = st.button("Ask", type="primary")
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat = []
                st.session_state.summary = ""
                st.rerun()

        if ask_button:
            if not q.strip():
                st.warning("Please enter a question")
            else:
                # Use streaming for real-time token-by-token display
                answer_placeholder = st.empty()
                answer_chunks = []
                
                status_text = st.empty()
                status_text.text("🔍 Analyzing question...")
                
                try:
                    for chunk in tutor.answer_stream(q, subject, st.session_state.chat, st.session_state.summary):
                        answer_chunks.append(chunk)
                        # Update status based on chunk content
                        if "Stage 0:" in chunk:
                            status_text.text("⚡ Loading from cache...")
                        elif "Stage 2:" in chunk:
                            status_text.text("🧠 Analyzing question...")
                        elif "Stage 3:" in chunk:
                            status_text.text("🔍 Searching knowledge base...")
                        elif "Stage 4:" in chunk or "Stage 5:" in chunk:
                            status_text.text("✅ Validating results...")
                        elif "Stage 6:" in chunk:
                            status_text.text("✍️ Generating answer...")
                        # Real-time display of accumulated answer
                        answer_placeholder.markdown(''.join(answer_chunks), unsafe_allow_html=True)
                    
                    # Clear status after completion
                    status_text.empty()
                    
                    # Store complete answer with subject for context filtering
                    complete_answer = ''.join(answer_chunks)
                    st.session_state.chat.append({
                        "user": q, 
                        "assistant": complete_answer,
                        "subject": subject  # Track subject for topic-switch detection
                    })

                    if len(st.session_state.chat) > 6:
                        with st.spinner("📝 Updating memory..."):
                            tutor.memory.update_summary(tutor.llm, callback=None)
                        st.session_state.chat = st.session_state.chat[-4:]

                    st.rerun()  # Refresh to show the new message in history
                
                except Exception as e:
                    status_text.empty()
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("🔍 Debug Info"):
                        st.code(traceback.format_exc())

# ---------- PRACTICE QUESTIONS TAB ----------
elif active_tab_clean == "Practice Questions":
    if not st.session_state.subjects:
        st.info("📚 Please build a knowledge base first by uploading your textbooks in the Knowledge Base tab.")
    else:
        st.subheader("📝 Practice Question Generator")
        st.markdown("""
        Generate practice questions from your textbooks to test your understanding and prepare for exams.
        """)
        
        subject = st.selectbox("📖 Select Subject/Course", st.session_state.subjects, key="practice_subject",
                              help="Choose the subject for practice questions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            topic = st.text_input("🎯 Topic", placeholder="e.g., Recursion, Photosynthesis, Supply and Demand",
                                help="Enter the specific topic you want to practice")
        
        with col2:
            num_questions = st.number_input("# Questions", min_value=1, max_value=10, value=5)
        
        difficulty = st.select_slider("Difficulty Level", 
                                     options=["easy", "medium", "hard"], 
                                     value="medium")
        
        if st.button("🎲 Generate Practice Questions", type="primary"):
            if not topic.strip():
                st.warning("Please enter a topic name")
            else:
                with st.spinner(f"🤔 Generating {num_questions} {difficulty} questions on '{topic}'..."):
                    questions_html = tutor.generate_practice_questions(
                        topic=topic,
                        subject=subject,
                        num_questions=num_questions,
                        difficulty=difficulty
                    )
                    st.markdown(questions_html, unsafe_allow_html=True)
                    st.success(f"✅ Generated {num_questions} practice questions!")
