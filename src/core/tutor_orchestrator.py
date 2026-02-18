"""
tutor_orchestrator.py: Clean orchestrator pattern for robust question handling
Replaces fragile prompt engineering with structured decision-making
"""
from typing import List, Dict, Any, Optional, Tuple
import os


class TutorOrchestrator:
    """
    Orchestrates the entire tutoring flow with structured decision-making.
    Prevents the "fix one, break another" problem by maintaining clear separation of concerns.
    """    
    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.llm = rag_engine.llm
        self.current_context = {}  # Stores current conversation state
    
    def process_question(self, question: str, subject: str, chat_history: List[Dict]) -> str:
        """
        Main processing pipeline with clear stages.
        
        Pipeline:
        1. Context Resolution → Handle follow-ups
        2. Academic Validation → Check if appropriate
        3. KB Search → Retrieve relevant content
        4. Subject Validation → Ensure content matches subject
        5. Answer Generation → Create response
        
        Args:
            question: Raw user input
            subject: Selected subject
            chat_history: Previous conversation [{user:..., assistant:...}, ...]
            
        Returns:
            Formatted HTML answer
        """        
        try:
            # STAGE 1: Context Resolution
            resolved_q = self._resolve_with_context(question, chat_history)
            
            # STAGE 2: Academic Validation
            if not self._is_academic_appropriate(resolved_q):
                return self._render_non_academic_response()
            
            # STAGE 3: KB Search & Load
            search_results = self._search_knowledge_base(resolved_q, subject)
            
            if not search_results['texts']:
                return self._render_no_results_response(question, subject)
            
            # STAGE 4: Subject Validation - REMOVED (architectural guarantee)
            # File system isolation (vector_db/Subject/) guarantees subject purity
            # if self._check_subject_mismatch(resolved_q, subject, search_results['texts'][0][0]):
            #     return self._render_subject_mismatch_response(question, subject)
            
            # STAGE 5: Generate Answer
            return self._generate_answer(
                original_q=question,
                resolved_q=resolved_q,
                subject=subject,
                search_results=search_results,
                chat_history=chat_history
            )
            
        except Exception as e:
            return self._render_error_response(str(e))
    
    def _resolve_with_context(self, question: str, chat_history: List[Dict]) -> str:
        """Stage 1: Resolve follow-ups and ambiguous questions"""
        if not chat_history:
            return question
        
        # Build context string
        context_lines = []
        for turn in chat_history[-3:]:  # Last 3 turns
            if 'user' in turn:
                context_lines.append(f"User: {turn['user']}")
            if 'assistant' in turn:
                # Extract plain text from HTML
                import re
                plain = re.sub(r'<[^>]+>', '', turn['assistant'])
                context_lines.append(f"Assistant: {plain[:3000]}...")
        
        context_str = "\n".join(context_lines)
        
        # Use LLM to resolve
        prompt = f"""Given this conversation, clarify what the user is asking.

CONTEXT:
{context_str}

NEW INPUT: "{question}"

RULES:
- If follow-up (more, it, that, continue, explain further): extract the topic from context
- If affirmation (yes, sure, interested, ok): identify what they're confirming
- If new complete question: return as-is
- Reply with ONLY the clarified question

Clarified question:"""
        
        try:
            resolved = self.llm.invoke(prompt).strip()
            # Validate response isn't too long or empty
            if resolved and len(resolved) < 200:
                return resolved
        except:
            pass
        
        return question  # Fallback to original
    
    def _is_academic_appropriate(self, question: str) -> bool:
        """Stage 2: Check if question is academic/appropriate"""
        prompt = f"""Is this an academic or educational question?

Question: "{question}"

Academic includes: subject concepts, explanations, homework, study help, diagrams, 
examples, calculations, theory, follow-ups, affirmations (yes, more, continue, tell me).

Non-academic: entertainment gossip, sports scores, personal life, celebrity news.

Reply ONLY "yes" or "no":"""
        
        try:
            response = self.llm.invoke(prompt).strip().lower()
            return any(word in response for word in ['yes', 'academic', 'educational'])
        except:
            return True  # Default to allowing
    
    def _search_knowledge_base(self, question: str, subject: str) -> Dict[str, Any]:
        """Stage 3: Search the KB with multimodal support"""
        # Ensure correct KB loaded
        if not self.rag.vector_store or self.rag.current_subject != subject:
            self.rag.load_subject(subject)
            self.rag.current_subject = subject
        
        from .embeddings import embed_texts_batched
        from .image_embeddings import embed_text_for_image_search
        from .multimodal_vector_store import MultimodalVectorStore
        
        is_multimodal = isinstance(self.rag.vector_store, MultimodalVectorStore)
        
        if is_multimodal:
            # Check if wants visuals
            visual_kw = ['show', 'visualize', 'diagram', 'illustration', 'picture', 
                        'image', 'looks like', 'structure', 'architecture', 'flowchart']
            wants_visual = any(kw in question.lower() for kw in visual_kw)
            
            # Search both text and images
            q_vec_text = embed_texts_batched([question])
            q_vec_image = embed_text_for_image_search(question)
            k_images = 8 if wants_visual else 3
            
            results = self.rag.vector_store.search_multimodal(
                q_vec_text, q_vec_image, k_text=15, k_images=k_images
            )
            
            return {
                'texts': results['texts'],
                'images': results['images'],
                'has_images': results['has_visual']
            }
        else:
            # Text-only
            q_vec = embed_texts_batched([question])
            text_results = self.rag.vector_store.search(q_vec, k=50, threshold=0.5)
            
            return {
                'texts': text_results,
                'images': [],
                'has_images': False
            }
    
    def _check_subject_mismatch_DEPRECATED(self, question: str, subject: str, sample_content: str) -> bool:
        """Stage 4: Validate content matches subject"""
        subject_name = subject.split('-')[0].strip()
        
        prompt = f"""Does this content belong to {subject_name}?

Question: {question}
Subject: {subject_name}
Sample: {sample_content[:3000]}

Examples:
- CS subject with stock market content → no
- CS subject with programming content → yes
- Commerce with market content → yes

Reply ONLY "yes" or "no":"""
        
        try:
            response = self.llm.invoke(prompt).strip().lower()
            return 'no' in response  # True if mismatch
        except:
            return False  # Assume match on error
    
    def _generate_answer(self, original_q: str, resolved_q: str, subject: str, 
                         search_results: Dict, chat_history: List[Dict]) -> str:
        """Stage 5: Generate the final answer"""
        
        texts = search_results['texts']
        images = search_results['images']
        has_images = search_results['has_images']
        
        # Calculate confidence
        scores = [r[1] for r in texts]
        max_conf = max(scores) if scores else 0
        avg_conf = sum(scores) / len(scores) if scores else 0
        
        # Build context
        syllabus_text = "\n\n".join([r[0] for r in texts])
        
        # Build conversation context
        conv_context = ""
        if chat_history:
            conv_context = "Previous discussion:\n"
            for turn in chat_history[-2:]:
                if 'user' in turn:
                    conv_context += f"User: {turn['user']}\n"
        
        # Generate answer with LLM
        prompt = f"""You are a friendly tutor helping a student.

{conv_context}

Current Question: "{original_q}"

Textbook Content:
{syllabus_text}

Instructions:
- Answer the current question clearly and thoroughly
- Use conversation context only to understand references (it, that, more)
- For general questions, provide comprehensive overview
- For specific questions, focus on that aspect
- Be conversational and educational
- Use examples when helpful

Your answer:"""
        
        answer_text = self.llm.invoke(prompt)
        
        # Format response
        return self._render_answer_response(
            question=original_q,
            answer=answer_text,
            subject=subject,
            confidence=max_conf,
            num_sources=len(texts),
            sources=texts,
            images=images if has_images else None
        )
    
    def _render_answer_response(self, question: str, answer: str, subject: str,
                                confidence: float, num_sources: int, sources: List,
                                images: Optional[List] = None) -> str:
        """Render successful answer as HTML"""
        # Confidence badge
        badge = ""
        if confidence < 0.55:
            badge = "⚠️ <span style='color:#d48806'><b>Low Confidence</b></span><br>"
        elif confidence < 0.65:
            badge = "⚠️ <span style='color:#d48806'><b>Moderate Confidence</b></span><br>"
        elif confidence >= 0.75:
            badge = "✅ <span style='color:#52c41a'><b>High Confidence</b></span><br>"
        
        # Source citations
        citations = self._format_sources([r[2] for r in sources])
        
        # Image section
        image_html = ""
        if images:
            image_html = self._format_images(images)
        
        html = f"""<div style='background:#f6f8fa;border-radius:8px;padding:16px 20px;margin-bottom:8px;'>
<h4 style='color:#2d6cdf;margin-top:0'>📘 {subject} Tutor Answer</h4>
{badge}
<b>Question:</b> {question}<br>
<b>Answer:</b><br>
<div style='margin-left:1em'>{answer}</div>
<div style='margin-top:8px;font-size:0.85em;color:#6a737d;'>
Confidence: {confidence:.2f} | Sources: {num_sources} chunks{' | 🖼️ ' + str(len(images)) + ' diagrams' if images else ''}
</div>
{citations}
{image_html}
</div>"""
        
        return html
    
    def _render_no_results_response(self, question: str, subject: str) -> str:
        """Render no results found"""
        return f"""<div style='background:#fff4e6;border:2px solid #fa8c16;border-radius:8px;padding:16px 20px;margin-bottom:8px;'>
<h4 style='color:#d48806;margin-top:0'>⚠️ Insufficient Knowledge Grounding</h4>
<b>Question:</b> {question}<br>
<b>Status:</b> No relevant content found in the <b>{subject}</b> knowledge base (0 matches above threshold).<br><br>
<b>Options:</b><br>
1. This topic may not be covered in your uploaded textbooks<br>
2. Try rephrasing your question with different keywords<br>
3. Upload additional materials covering this topic<br><br>
<i style='color:#8c8c8c;'>As per paper principles: Refusing to guess is better than hallucinating.</i>
</div>"""
    
    def _render_subject_mismatch_response(self, question: str, subject: str) -> str:
        """Render subject mismatch warning"""
        return f"""<div style='background:#fff4e6;border:2px solid #fa8c16;border-radius:8px;padding:16px 20px;margin-bottom:8px;'>
<h4 style='color:#d48806;margin-top:0'>⚠️ Subject Mismatch Detected</h4>
<b>Your Question:</b> {question}<br>
<b>Selected Subject:</b> {subject}<br><br>
<b>Issue:</b> The retrieved content appears to be from a <b>different subject area</b>.<br><br>
<b>Possible Causes:</b><br>
1. Wrong subject selected from dropdown<br>
2. Question is outside this subject's scope<br>
3. Wrong PDF uploaded to this knowledge base<br><br>
<b>💡 Suggestion:</b> Check your subject selection or upload the correct textbooks.
</div>"""
    
    def _render_non_academic_response(self) -> str:
        """Render non-academic question rejection"""
        return "❌ <span style='color:red'><b>Sorry, as your tutor, I can only answer academic questions related to your studies.</b></span>"
    
    def _render_error_response(self, error: str) -> str:
        """Render error"""
        return f"<div style='color:red;padding:10px;background:#fff1f0;border-radius:4px;'><b>Error:</b> {error}</div>"
    
    def _format_sources(self, metadata_list: List[Dict]) -> str:
        """Format source citations"""
        if not metadata_list:
            return ""
        
        from collections import defaultdict
        source_pages = defaultdict(set)
        
        for meta in metadata_list[:5]:
            if meta and 'source' in meta and 'page' in meta:
                source_file = os.path.basename(meta['source'])
                source_pages[source_file].add(meta['page'])
        
        if not source_pages:
            return ""
        
        citations = []
        for source_file, pages in source_pages.items():
            sorted_pages = sorted(list(pages))
            page_str = ", ".join(str(p) for p in sorted_pages[:3])
            if len(sorted_pages) > 3:
                page_str += ", ..."
            citations.append(f"{source_file} (p. {page_str})")
        
        return f"<div style='margin-top:10px;font-size:0.85em;color:#8c8c8c;'>📚 <b>Sources:</b> {' | '.join(citations)}</div>"
    
    def _format_images(self, images: List) -> str:
        """Format and display images inline"""
        if not images:
            return ""
        
        import base64
        from io import BytesIO
        
        html_parts = []
        html_parts.append(
            "<div style='margin-top:16px;padding:12px;background:#f0f7ff;border-left:4px solid #1890ff;border-radius:4px;'>"
            "<h4 style='margin-top:0;color:#1890ff;'>🖼️ Relevant Diagrams from Textbook</h4>"
        )
        
        for img_data, meta, score in images[:3]:
            try:
                pil_image = img_data.get('image') if isinstance(img_data, dict) else None
                
                if pil_image:
                    buffered = BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    source_file = os.path.basename(meta.get('source', 'Unknown'))
                    page = meta.get('page', '?')
                    confidence = int(score * 100)
                    
                    html_parts.append(
                        f"<div style='margin:10px 0;padding:8px;background:white;border:1px solid #d9d9d9;border-radius:4px;'>"
                        f"<img src='data:image/png;base64,{img_str}' style='max-width:100%;height:auto;border-radius:4px;'/>"
                        f"<div style='margin-top:6px;font-size:0.85em;color:#595959;'>"
                        f"📚 <b>{source_file}</b> - Page {page} | Relevance: {confidence}%"
                        f"</div>"
                        f"</div>"
                    )
            except Exception as e:
                source_file = os.path.basename(meta.get('source', 'Unknown'))
                page = meta.get('page', '?')
                html_parts.append(
                    f"<div style='padding:4px;font-size:0.9em;'>📚 {source_file} - Page {page}</div>"
                )
        
        html_parts.append("</div>")
        return "".join(html_parts)
