"""
rag_engine.py: Orchestrates all core modules for RAG logic with advanced AI agents
"""
import logging
from typing import List, Any, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from .llm import OllamaLLM
from .multi_model_llm import MultiModelLLM, TaskType

logger = logging.getLogger(__name__)
from .embeddings import embed_texts_batched
from .vector_store import VectorStore
from .multimodal_vector_store import MultimodalVectorStore
from .image_embeddings import embed_images_batched, embed_text_for_image_search
from .image_extractor import extract_images_from_pdf
from .pdf_loader import load_pdfs
from .memory import SessionMemory
from .logger import ConversationLogger
from .practice_questions import PracticeQuestionGenerator
from .agent_orchestrator import AgentOrchestrator, QuestionType
from .retrieval_agent import RetrievalStrategyAgent
from .cache_manager import CacheManager
from .student_profile import StudentProfile
from .subject_cache import SubjectKnowledgeCache
from .agent_orchestrator import QuestionType, AnalyzedQuestion  # Import at module level

class ConsistentTutorRAG:
    def __init__(self, db_dir="vector_db", log_dir="logs", multimodal: bool = True, use_meta_prompting: bool = True, use_multi_model: bool = True):
        # Initialize LLM system - multi-model for task-specialized intelligence
        if use_multi_model:
            self.llm = MultiModelLLM(
                reasoning_model="qwen2.5:14b-instruct-q4_K_M",  # 14B quantized, 9GB - complex reasoning
                generation_model="llama3.1:8b",  # 8B, 5GB - generation and quick checks
                temperature=0.7
            )
            self.use_multi_model = True
            logger.info("🎯 2-Model architecture: Qwen2.5-14B (reasoning, 9GB) + Llama3.1-8B (generation+checks, 5GB) = 14GB total")
        else:
            self.llm = OllamaLLM()
            self.use_multi_model = False
            logger.info("📦 Single-model mode: Using Llama3")
        
        self.memory = SessionMemory()
        self.logger = ConversationLogger(log_dir=log_dir)
        self.db_dir = db_dir
        self.vector_store = None
        self.current_subject = None  # Track currently loaded subject for isolation
        self.use_meta_prompting = use_meta_prompting  # Meta-prompting for adaptive answer generation
        self.multimodal = multimodal  # Enable multimodal by default
        
        # Sub-components receive a simple LLM interface (backward compatibility)
        # For multi-model, they'll use the generation model by default
        compat_llm = OllamaLLM() if not use_multi_model else self.llm
        self.question_generator = PracticeQuestionGenerator(compat_llm)
        
        # NEW: Advanced AI systems
        self.agent = AgentOrchestrator(compat_llm)
        self.cache = CacheManager()
        self.student = StudentProfile()
        self.retrieval_agent = RetrievalStrategyAgent(compat_llm, self.student)  # Memory-aware adaptive retrieval
        self.student.start_session()
        
        # Subject knowledge base cache - pre-load vector DBs in RAM for sub-20s response times
        # MEMORY OPTIMIZATION (v2): Increased from 10 to 30 subjects to fully utilize 64GB RAM
        self.subject_cache = SubjectKnowledgeCache(max_subjects=30, text_dim=384, image_dim=512, db_dir=db_dir)
        logger.info("🚀 Subject cache ready: Expanded to 30 subjects, leveraging 64GB RAM for instant vector DB access")

    def ingest_pdfs(self, pdf_paths: List[str], subject: str, cb=None):
        docs = load_pdfs(pdf_paths)
        # ... splitting logic here ...
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        texts = [c.page_content for c in chunks]
        
        # Preserve metadata (page numbers, source files)
        metadata = []
        for c in chunks:
            meta = {
                'page': c.metadata.get('page', 0) + 1,  # 1-indexed for display
                'source': c.metadata.get('source', 'Unknown'),
            }
            metadata.append(meta)
        
        vectors = embed_texts_batched(texts)
        self.vector_store = VectorStore(vectors.shape[1], self.db_dir)
        self.vector_store.add(vectors, texts, metadata)
        self.vector_store.save(subject)
        if cb:
            cb(1.0, "Knowledge base ready")
        return len(texts)

    def ingest_pdfs_multimodal(self, pdf_paths: List[str], subject: str, extract_images: bool = True, cb=None):
        """
        Ingest PDFs with multimodal support (text + images).
        
        Args:
            pdf_paths: List of PDF file paths
            subject: Subject name
            extract_images: Whether to extract and index images
            cb: Progress callback
        """
        # 1. Extract text chunks (same as before)
        docs = load_pdfs(pdf_paths)
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        texts = [c.page_content for c in chunks]
        
        # Preserve metadata (page numbers, source files)
        text_metadata = []
        for c in chunks:
            meta = {
                'page': c.metadata.get('page', 0) + 1,  # 1-indexed for display
                'source': c.metadata.get('source', 'Unknown'),
                'type': 'text'
            }
            text_metadata.append(meta)
        
        if cb:
            cb(0.2, "Text extraction complete")
        
        # 2. Embed text using regular embeddings
        text_vectors = embed_texts_batched(texts)
        
        if cb:
            cb(0.4, "Text embeddings generated")
        
        # 3. Extract and embed images if enabled
        image_metadata_list = []
        image_vectors = None
        
        if extract_images and self.multimodal:
            all_images = []
            for pdf_path in pdf_paths:
                try:
                    images = extract_images_from_pdf(pdf_path)
                    for img_data in images:
                        all_images.append(img_data)
                        image_metadata_list.append({
                            'page': img_data['page'],
                            'source': img_data['source'],
                            'type': 'image',
                            'width': img_data['width'],
                            'height': img_data['height'],
                        })
                except Exception as e:
                    if cb:
                        cb(0.5, f"Warning: Could not extract images from {pdf_path}")
            
            if cb:
                cb(0.6, f"Extracted {len(all_images)} images")
            
            # Embed images using CLIP
            if all_images:
                pil_images = [img_data['image'] for img_data in all_images]
                image_vectors = embed_images_batched(pil_images)
                if cb:
                    cb(0.8, "Image embeddings generated")
        
        # 4. Create multimodal vector store
        text_dim = text_vectors.shape[1]  # 384 for all-MiniLM-L6-v2
        image_dim = 512  # CLIP ViT-B-32 uses 512
        self.vector_store = MultimodalVectorStore(text_dim, image_dim, self.db_dir)
        self.vector_store.add_texts(text_vectors, texts, text_metadata)
        
        if image_vectors is not None and len(image_vectors) > 0:
            self.vector_store.add_images(image_vectors, all_images, image_metadata_list)
        
        self.vector_store.save(subject)
        
        if cb:
            cb(1.0, f"✅ Multimodal KB ready: {len(texts)} text chunks, {len(image_metadata_list)} images")
        
        return len(texts), len(image_metadata_list)

    def load_subject(self, subject: str):
        """Load subject KB using intelligent cache - eliminates disk I/O on every query"""
        import os
        subject_dir = os.path.join(self.db_dir, subject)
        
        # Check if multimodal store exists
        has_text_index = os.path.exists(f"{subject_dir}/text_index.faiss")
        has_old_index = os.path.exists(f"{subject_dir}/index.faiss")
        
        if has_text_index:
            # Multimodal store - use cache for instant access
            self.vector_store = self.subject_cache.get_vector_store(subject)
            if not self.vector_store:
                raise FileNotFoundError(f"Failed to load knowledge base for subject: {subject}")
        elif has_old_index:
            # Legacy text-only store (no caching for old format)
            logger.warning(f"⚠️ Using legacy text-only format for {subject} - consider regenerating with multimodal")
            self.vector_store = VectorStore(384, self.db_dir)
            self.vector_store.load(subject)
        else:
            raise FileNotFoundError(f"No knowledge base found for subject: {subject}")
    
    def preload_all_subjects(self):
        """
        Pre-load all available subjects into cache on app startup.
        
        Benefits:
        - First query on any subject is instant (no disk I/O)
        - With 64GB RAM, can cache all subjects simultaneously
        - Reduces response time from 30s to <20s
        """
        import os
        if not os.path.exists(self.db_dir):
            logger.warning(f"⚠️ Knowledge base directory not found: {self.db_dir}")
            return
        
        # Find all subjects (directories with FAISS indices)
        subjects = []
        for subject_name in os.listdir(self.db_dir):
            subject_path = os.path.join(self.db_dir, subject_name)
            if os.path.isdir(subject_path):
                # Check for multimodal index
                if os.path.exists(os.path.join(subject_path, "text_index.faiss")):
                    subjects.append(subject_name)
        
        if subjects:
            logger.info(f"🚀 Pre-loading {len(subjects)} subjects into cache...")
            self.subject_cache.preload_subjects(subjects)
            stats = self.subject_cache.get_cache_stats()
            logger.info(f"✅ Cache ready: {stats['cached_subjects']} subjects, hit_rate={stats['hit_rate']}")
        else:
            logger.warning("⚠️ No subjects found to pre-load")

    def answer(self, question: str, subject: str, chat_history: List[Any], summary: str, image_query: Optional[bytes] = None) -> str:
        """
        Answer a question using advanced AI agents (non-streaming).
        
        Args:
            question: Text question
            subject: Subject name  
            chat_history: Conversation history
            summary: Session summary
            image_query: Optional uploaded image
            
        Returns:
            Complete formatted HTML answer
        """
        import time
        query_start = time.time()
        
        # Use streaming internally but collect all chunks
        answer_parts = []
        for chunk in self.answer_stream(question, subject, chat_history, summary, image_query):
            answer_parts.append(chunk)
        
        result = "".join(answer_parts)
        
        query_time = time.time() - query_start
        print(f"\n✅ TOTAL QUERY TIME: {query_time:.2f}s\n")
        
        # Cache for similar future questions
        if result and len(result) > 100:  # Only cache valid responses
            self.cache.set_response(question, subject, result)
        
        return result
    
    def answer_stream(self, question: str, subject: str, chat_history: List[Any], summary: str, 
                     image_query: Optional[bytes] = None) -> Generator[str, None, None]:
        """
        Stream answer using advanced AI agent system.
        
        Yields HTML chunks as they're generated for real-time display.
        """
        import time
        stage_times = {}
        total_start = time.time()
        
        try:
            # STAGE 0: Check cache for instant responses
            # Helpful when students rephrase or revisit concepts
            stage_start = time.time()
            cached = self.cache.get_response(question, subject)
            if cached:
                yield cached
                return
            stage_times['0_cache_check'] = time.time() - stage_start
            
            # CRITICAL: Ensure correct subject KB is loaded
            stage_start = time.time()
            if not self.vector_store or self.current_subject != subject:
                self.load_subject(subject)
                self.current_subject = subject
            stage_times['0b_subject_loading'] = time.time() - stage_start
            
            # STAGE 1: Build FULL conversation context for analysis
            stage_start = time.time()
            full_conversation_context = self._build_conversation_context(chat_history, max_turns=3)
            stage_times['1_context_building'] = time.time() - stage_start
            
            # STAGE 2: Analyze question using intelligent agent
            # This is important - don't skip! Provides context understanding
            stage_start = time.time()
            analysis = self.agent.analyze_question(question, full_conversation_context, subject)
            stage_times['2_question_analysis'] = time.time() - stage_start
            
            # STAGE 2.5: Filter context based on question type (prevent context pollution)
            # NEW_TOPIC or subject switch → Clear context to avoid confusion
            # FOLLOW_UP/CLARIFICATION → Keep context for continuity
            conversation_context = self._filter_context_by_type(
                full_conversation_context, 
                analysis.type,
                subject,
                chat_history
            )
            
            # Handle off-topic questions
            if analysis.type == QuestionType.OFF_TOPIC:
                if not self.agent.is_academic(question):
                    response = self._render_non_academic()
                    yield response
                    return
            
            # STAGE 3: Search knowledge base - Adaptive intelligent retrieval
            # LLM determines optimal k, thresholds based on question + student history
            stage_start = time.time()
            try:
                results, image_results, has_images = self._search_kb(
                    analysis.expanded, 
                    image_query,
                    analyzed_question=analysis,
                    subject=subject
                )
            except KeyError as e:
                # Handle missing expected keys gracefully
                yield self._render_error(f"Search configuration error: {str(e)}. Please try rephrasing your question.")
                return
            except Exception as e:
                yield self._render_error(f"Search error: {str(e)}")
                return
            stage_times['3_kb_search'] = time.time() - stage_start
            
            if not results:
                response = self._render_no_results(question, subject)
                yield response
                return
            
            # Extract content
            syllabus_chunks = [r[0] for r in results]
            syllabus_text = "\n\n".join(syllabus_chunks)
            sources = [r[2] for r in results] if results else []
            max_confidence = max((r[1] for r in results), default=0.0)
            
            # STAGE 4: Intelligent relevance check + meta-prompting (preserved!)
            # All intelligence features active - quality first, speed second
            stage_start = time.time()
            relevance_result = {'is_relevant': True}
            dynamic_prompt_result = {'prompt': None}
            
            if self.use_meta_prompting:
                # PARALLEL: Relevance check + dynamic prompt generation
                logger.info("🔄 PARALLEL: Relevance check + meta-prompting")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Use fast relevance check method
                    future_relevance = executor.submit(
                        self._is_relevant_answer_fast, 
                        analysis.expanded, 
                        syllabus_text[:1500]
                    )
                    # Generate dynamic prompt tailored to question
                    future_prompt = executor.submit(
                        self._generate_dynamic_prompt,
                        question=question,
                        analysis=analysis,
                        conversation_context=conversation_context,
                        syllabus_sample=syllabus_text[:2000],
                        has_images=has_images,
                        image_count=len(image_results) if image_results else 0
                    )
                    
                    relevance_result['is_relevant'] = future_relevance.result()
                    dynamic_prompt_result['prompt'] = future_prompt.result()
                logger.info(f"✅ Intelligence checks complete: relevant={relevance_result['is_relevant']}")
            else:
                # Fallback: Just relevance check
                relevance_result['is_relevant'] = self._is_relevant_answer_fast(analysis.expanded, syllabus_text[:1500])
            
            stage_times['4_intelligent_checks'] = time.time() - stage_start
            
            # Check if content is relevant
            if not relevance_result['is_relevant']:
                response = self._render_insufficient_grounding(question, subject, len(results))
                yield response
                return
            
            # Log to student profile
            stage_start = time.time()
            self.student.log_question(
                question=question,
                topic=analysis.topic,
                subject=subject,
                confidence=max_confidence,
                answered=True
            )
            stage_times['5_student_logging'] = time.time() - stage_start
            
            # STAGE 6: Generate answer with streaming
            stage_start = time.time()
            answer_chunks = []
            for chunk in self._stream_answer(
                question=question,
                analysis=analysis,
                subject=subject,
                conversation_context=conversation_context,
                syllabus_text=syllabus_text,
                sources=sources,
                max_confidence=max_confidence,
                image_results=image_results,
                has_images=has_images,
                dynamic_prompt=dynamic_prompt_result['prompt'] if self.use_meta_prompting else None
            ):
                answer_chunks.append(chunk)
                yield chunk
            
            # Capture FULL answer generation time (including all streaming)
            stage_times['6_answer_generation'] = time.time() - stage_start
            
            stage_times['TOTAL'] = time.time() - total_start
            
            # Print timing breakdown
            print("\n" + "="*80)
            print("⏱️  DETAILED STAGE TIMING BREAKDOWN")
            print("="*80)
            for stage, duration in stage_times.items():
                print(f"{stage:.<50} {duration:>8.2f}s")
            print("="*80 + "\n")
            
        except Exception as e:
            import traceback
            print(f"\n❌ ERROR in answer_stream: {e}")
            traceback.print_exc()
            yield self._render_error(str(e))
    
    def _search_kb(self, query: str, image_query: Optional[bytes], 
                   analyzed_question=None, subject: str = None) -> tuple:
        """
        Adaptive semantic search - LLM determines optimal strategy.
        Memory-aware: adapts based on student's learning patterns.
        Returns: (text_results, image_results, has_images)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        is_multimodal_store = isinstance(self.vector_store, MultimodalVectorStore)
        has_images = False
        image_results = []
        
        logger.info(f"🔍 is_multimodal_store: {is_multimodal_store}, vector_store type: {type(self.vector_store).__name__}")
        
        # OPTIMIZATION: Determine retrieval strategy + thresholds in ONE LLM call (save ~2-3s)
        # This is carefully tuned - preserves quality while being faster
        sub_times = {}  # Track sub-stage timing to find bottlenecks
        sub_start = time.time()
        
        if analyzed_question and subject:
            logger.info(f"🧠 Using adaptive retrieval strategy (LLM + student memory)")
            strategy, thresholds = self.retrieval_agent.determine_complete_strategy(query, analyzed_question, subject)
            sub_times['strategy_determination'] = time.time() - sub_start
            
            # Map strategy to parameters
            params = {
                'text_threshold': thresholds['text'],
                'image_threshold': thresholds['image'],
                'k_text': strategy.context_window,  # LLM decided based on complexity + student history
                'k_images': 15 if strategy.multimodal_priority == 'visual_heavy' else 10 if strategy.multimodal_priority == 'balanced' else 5
            }
            logger.info(f"📊 Intelligent Strategy: {strategy}")
            logger.info(f"⚙️  Adaptive params: k_text={params['k_text']}, text_threshold={params['text_threshold']:.2f}")
        else:
            # Fallback: Use balanced defaults
            params = {
                'text_threshold': 0.25,
                'image_threshold': 0.20,
                'k_text': 15,
                'k_images': 8
            }
            logger.info(f"Using fallback params: {params}")
        
        if is_multimodal_store:
            logger.info(f"✅ Entering multimodal search path")
            
            # TIMING: Track embedding generation
            sub_start = time.time()
            
            # Use separate embeddings for text and image search
            if image_query:
                # Image-to-text/image search - PARALLELIZE embedding generation
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_query))
                
                logger.info("🔄 PARALLEL: Generating text + image embeddings")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_text = executor.submit(embed_texts_batched, [query])
                    future_image = executor.submit(embed_images_batched, [img])
                    q_vec_text = future_text.result()
                    q_vec_image = future_image.result()
                logger.info("✅ PARALLEL embeddings complete")
            else:
                # OPTIMIZATION: Only load CLIP model if question explicitly needs visuals
                # This saves ~10s from loading CLIP unnecessarily
                visual_keywords = ['show', 'diagram', 'image', 'picture', 'visualize', 'illustration', 'graph', 'chart', 'structure']
                needs_visuals = any(kw in query.lower() for kw in visual_keywords)
                
                # Check cache first for text embedding
                cached_emb = self.cache.get_embedding(query)
                
                if cached_emb is not None:
                    q_vec_text = cached_emb.reshape(1, -1)
                    # Only generate image embedding if actually needed
                    if needs_visuals:
                        q_vec_image = embed_text_for_image_search(query)
                    else:
                        # Skip CLIP loading - use zero vector (won't match anything, which is fine)
                        import numpy as np
                        q_vec_image = np.zeros((1, 512), dtype=np.float32)
                else:
                    # Generate text embedding
                    q_vec_text = embed_texts_batched([query])
                    self.cache.set_embedding(query, q_vec_text[0])
                    
                    # Generate image embedding only if needed
                    if needs_visuals:
                        q_vec_image = embed_text_for_image_search(query)
                    else:
                        import numpy as np
                        q_vec_image = np.zeros((1, 512), dtype=np.float32)
            
            # Track embedding time
            sub_times['embedding_generation'] = time.time() - sub_start
            
            # Use intelligent retrieval parameters
            logger.info(f"Multimodal search with k_text={params['k_text']}, k_images={params['k_images']}, text_thresh={params['text_threshold']}, image_thresh={params['image_threshold']}")
            
            # TIMING: Track FAISS search
            sub_start = time.time()
            
            results_dict = self.vector_store.search_multimodal(
                q_vec_text, q_vec_image, 
                k_text=params['k_text'], 
                k_images=params['k_images'],
                text_threshold=params['text_threshold'],
                image_threshold=params['image_threshold']
            )
            results = results_dict['texts']
            image_results = results_dict['images']
            has_images = results_dict['has_visual']
            
            # Track FAISS search time
            sub_times['faiss_search'] = time.time() - sub_start
            
            logger.info(f"📊 MULTIMODAL RESULTS: {len(results)} text chunks, {len(image_results)} images")
            logger.info(f"📸  has_visual flag: {has_images}")
            logger.info(f"🎯 image_results type: {type(image_results)}, empty: {len(image_results) == 0}")
            
            # Print sub-stage timing breakdown
            if sub_times:
                logger.info(f"\n📊 Stage 3 Sub-Timing: {', '.join(f'{k}={v:.2f}s' for k, v in sub_times.items())}")
        else:
            logger.info(f"📝 Text-only search path (not multimodal)")
            
            # TIMING: Track embedding generation
            sub_start = time.time()
            
            # Text-only search with caching
            cached_emb = self.cache.get_embedding(query)
            if cached_emb is not None:
                q_vec = cached_emb.reshape(1, -1)
            else:
                q_vec = embed_texts_batched([query])
                self.cache.set_embedding(query, q_vec[0])
            
            sub_times['embedding_generation'] = time.time() - sub_start
            
            # TIMING: Track FAISS search
            sub_start = time.time()
            
            # Use intelligent parameters
            logger.info(f"Text-only search with k={params['k_text']}, threshold={params['text_threshold']}")
            results = self.vector_store.search(q_vec, k=params['k_text'], threshold=params['text_threshold'])
            
            sub_times['faiss_search'] = time.time() - sub_start
            
            logger.info(f"Retrieved: {len(results)} text chunks")
            
            # Print sub-stage timing breakdown
            if sub_times:
                logger.info(f"\n📊 Stage 3 Sub-Timing: {', '.join(f'{k}={v:.2f}s' for k, v in sub_times.items())}")
        
        return results, image_results, has_images
    
    def _stream_answer(self, question: str, analysis, subject: str, conversation_context: str,
                      syllabus_text: str, sources: list, max_confidence: float,
                      image_results: list, has_images: bool, dynamic_prompt: str = None) -> Generator[str, None, None]:
        """
        Stream the answer generation with all metadata.
        Uses dynamic LLM-generated prompt if provided, otherwise falls back to static template.
        """
        # Use dynamic prompt if available, otherwise fall back to static template
        if dynamic_prompt:
            logger.info("✅ Using LLM-generated dynamic prompt")
            prompt = self._build_final_prompt_from_template(
                dynamic_prompt_template=dynamic_prompt,
                question=question,
                conversation_context=conversation_context,
                syllabus_text=syllabus_text,
                has_images=has_images,
                image_count=len(image_results) if image_results else 0
            )
        else:
            logger.info("⚠️ Falling back to static prompt template")
            prompt = self._build_smart_prompt(
                question=question,
                conversation_context=conversation_context,
                syllabus_text=syllabus_text,
                has_images=has_images,
                image_count=len(image_results) if image_results else 0
            )
        
        # Confidence badge
        confidence_badge = ""
        if max_confidence < 0.55:
            confidence_badge = "⚠️ <span style='color:#d48806'><b>Low Confidence:</b> Limited relevant content found.</span><br>"
        elif max_confidence < 0.65:
            confidence_badge = "⚠️ <span style='color:#d48806'><b>Moderate Confidence:</b> Some relevant content found.</span><br>"
        elif max_confidence >= 0.75:
            confidence_badge = "✅ <span style='color:#52c41a'><b>High Confidence:</b> Strong match found.</span><br>"
        
        # Start HTML wrapper
        yield "<div style='background:#f6f8fa;border-radius:8px;padding:16px 20px;margin-bottom:8px;'>"
        yield f"<h4 style='color:#2d6cdf;margin-top:0'>📘 {subject} Tutor Answer</h4>"
        yield confidence_badge
        yield f"<b>Question:</b> {question}<br>"
        yield "<b>Answer:</b><br>"
        yield "<div style='margin-left:1em'>"
        
        # Stream the actual answer
        full_answer = []
        # Use specialized generation model for answers (if multi-model enabled)
        if self.use_multi_model:
            for chunk in self.llm.stream(prompt, task_type=TaskType.ANSWER_GENERATION):
                full_answer.append(chunk)
                yield chunk
        else:
            for chunk in self.llm.stream(prompt):
                full_answer.append(chunk)
                yield chunk
        
        yield "</div>"  # Close answer div
        
        # Add metadata
        yield f"<div style='margin-top:8px;font-size:0.85em;color:#6a737d;'>"
        yield f"Confidence: {max_confidence:.2f} | Sources: {len(sources)} chunks"
        if has_images:
            yield f" | 🖼️ {len(image_results)} relevant diagrams"
        yield "</div>"
        
        # Add source citations
        yield self._format_sources(sources)
        
        # Add images if available
        logger.info(f"🖼️  Image display check: has_images={has_images}, image_results len={len(image_results) if image_results else 0}")
        if has_images and image_results:
            logger.info(f"✅ Displaying {len(image_results)} images")
            yield self._format_image_references(image_results)
        else:
            logger.info(f"❌ NOT displaying images - has_images={has_images}, image_results empty={not image_results}")
        
        yield "</div>"  # Close main wrapper
        
        # NO response caching - students don't repeat questions!
        # All performance comes from KB/embedding/subject caching
    
    def _render_non_academic(self) -> str:
        """Render response for non-academic questions"""
        return (
            "❌ <span style='color:red'><b>Sorry, as your tutor, I can only answer academic questions related to your studies.</b></span> "
            "If you have a question about your subject, please ask!"
        )
    
    def _render_no_results(self, question: str, subject: str) -> str:
        """Render response when no relevant content found"""
        return (
            "<div style='background:#fff4e6;border:2px solid #fa8c16;border-radius:8px;padding:16px 20px;margin-bottom:8px;'>"
            "<h4 style='color:#d48806;margin-top:0'>⚠️ Insufficient Knowledge Grounding</h4>"
            f"<b>Question:</b> {question}<br>"
            f"<b>Status:</b> No relevant content found in the <b>{subject}</b> knowledge base (0 matches above threshold).<br><br>"
            "<b>Options:</b><br>"
            "1. This topic may not be covered in your uploaded textbooks<br>"
            "2. Try rephrasing your question with different keywords<br>"
            "3. Upload additional materials covering this topic<br><br>"
            "<i style='color:#8c8c8c;'>Refusing to guess is better than hallucinating.</i>"
            "</div>"
        )
    
    # _render_subject_mismatch removed - redundant due to file system isolation
    
    def _render_insufficient_grounding(self, question: str, subject: str, num_results: int) -> str:
        """Render response when grounding is weak"""
        return (
            "<div style='background:#fffbe6;border-radius:8px;padding:16px 20px;margin-bottom:8px;'>"
            "<h4 style='color:#d48806;margin-top:0'>⚠️ Weak Knowledge Grounding</h4>"
            f"<b>Question:</b> {question}<br>"
            f"<b>Note:</b> This topic has weak coverage in the <b>{subject}</b> textbook ({num_results} weak matches).<br><br>"
            "<b>Recommendation:</b> Upload more comprehensive materials on this topic for better answers."
            "</div>"
        )
    
    def _render_error(self, error_msg: str) -> str:
        """Render error message"""
        return (
            "<div style='background:#fff1f0;border:2px solid #ff4d4f;border-radius:8px;padding:16px 20px;'>"
            "<h4 style='color:#ff4d4f;margin-top:0'>❌ Error Occurred</h4>"
            f"<b>Error:</b> {error_msg}<br><br>"
            "Please try again or rephrase your question."
            "</div>"
        )
        
        # Log the interaction
        self.logger.log(question, answer_html)
        
        return answer_html
    
    # OLD COMPLEX METHOD - Kept for reference, can be removed after validation
    def answer_old_complex(self, question: str, subject: str, chat_history: List[Any], summary: str, image_query: Optional[bytes] = None):
        """
        Answer a question with multimodal support.
        
        Args:
            question: Text question
            subject: Subject name
            chat_history: Conversation history
            summary: Session summary
            image_query: Optional uploaded image for visual question answering
        """
        # CRITICAL: Always ensure correct subject KB is loaded (for proper isolation)
        if not self.vector_store or self.current_subject != subject:
            self.load_subject(subject)
            self.current_subject = subject
        
        # Build conversational context from recent chat history (last 3 exchanges)
        conversation_context = self._build_conversation_context(chat_history, max_turns=3)
        
        # Resolve the question using chat history context (handle follow-ups like "more disadvantages", "explain it further", etc.)
        resolved_question = self._resolve_question_with_context(question, conversation_context)
        
        # Determine if this is a text or multimodal query
        is_multimodal_store = isinstance(self.vector_store, MultimodalVectorStore)
        has_images = False
        image_results = []
        
        if is_multimodal_store:
            # Detect if user wants visual/diagram content
            visual_keywords = ['show', 'visualize', 'diagram', 'illustration', 'picture', 'image', 'looks like', 'structure', 'architecture', 'flowchart', 'graph', 'chart']
            wants_visual = any(keyword in resolved_question.lower() for keyword in visual_keywords)
            
            # Use separate embeddings for text and image search
            if image_query:
                # Image-to-text/image search
                from .image_embeddings import embed_images_batched
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_query))
                q_vec_image = embed_images_batched([img])
                q_vec_text = embed_texts_batched([resolved_question])  # Also embed text
            else:
                # Text-to-text and text-to-image search
                q_vec_text = embed_texts_batched([resolved_question])  # For text index (384-dim)
                q_vec_image = embed_text_for_image_search(resolved_question)  # For image index (512-dim)
            
            # Always retrieve relevant diagrams: 5 by default, 10 when explicitly requested
            k_images = 10 if wants_visual else 5
            results_dict = self.vector_store.search_multimodal(
                q_vec_text, q_vec_image, 
                k_text=50, k_images=k_images
                # Using function defaults for thresholds (0.5 text, 0.4 image)
            )
            results = results_dict['texts']
            image_results = results_dict['images']
            has_images = results_dict['has_visual']
        else:
            # Text-only search
            q_vec = embed_texts_batched([resolved_question])
            results = self.vector_store.search(q_vec, k=50)  # Using default threshold=0.5
        
        # Calculate confidence metrics
        avg_confidence = sum(r[1] for r in results) / len(results) if results else 0.0
        max_confidence = max((r[1] for r in results), default=0.0)
        
        # Extract text, scores, and metadata
        syllabus_chunks = [r[0] for r in results]
        syllabus_text = "\n\n".join(syllabus_chunks) if results else ""
        sources = [r[2] for r in results] if results else []  # Metadata with page numbers
        ctx = summary  # For simplicity, use summary as context

        # NOTE: Subject match check removed - file system isolation (vector_db/Subject/)
        # already guarantees all retrieved content is from the correct subject.
        # Architecture provides guarantee, no runtime check needed.

        # Check if question is relevant using LLM
        if False:  # Disabled - architectural guarantee replaces this check
            # Content doesn't match the subject - warn user
            system_output = (
                "<div style='background:#fff4e6;border:2px solid #fa8c16;border-radius:8px;padding:16px 20px;margin-bottom:8px;'>"
                "<h4 style='color:#d48806;margin-top:0'>⚠️ Subject Mismatch Detected</h4>"
                f"<b>Your Question:</b> {question}<br>"
                f"<b>Selected Subject:</b> {subject}<br><br>"
                f"<b>Issue:</b> The retrieved content appears to be from a <b>different subject area</b> than '{subject}'.<br><br>"
                "<b>Possible Causes:</b><br>"
                "1. You uploaded the wrong PDF to this knowledge base<br>"
                "2. The question is outside the scope of this subject<br>"
                "3. Try selecting a different subject from the dropdown<br><br>"
                "<b>💡 Suggestion:</b> Check your knowledge base and ensure the correct textbooks are uploaded for each subject."
                "</div>"
            )
        elif results and self._is_relevant_answer(resolved_question, syllabus_chunks):
            # Answer from subject knowledge base
            # Use single intelligent prompt that adapts to any question type
            prompt = f"""You are a friendly, knowledgeable tutor having a conversation with a student.

🗨️ PREVIOUS CONVERSATION (for context only - to understand flow):
{conversation_context if conversation_context else '(This is the first question in the conversation)'}

❓ STUDENT'S CURRENT QUESTION:
"{question}"

📚 RELEVANT CONTENT FROM TEXTBOOK:
{syllabus_text}

📝 YOUR TASK:
- Answer ONLY the student's current question above (in quotes)
- Use previous conversation ONLY to understand references and maintain conversational flow
- Intelligently determine the scope:
  • If question is GENERAL (e.g., "what is data structure"), provide comprehensive overview covering multiple types/aspects
  • If question is SPECIFIC (e.g., "advantages of list"), focus on that specific aspect
- Use textbook content as foundation, but expand naturally for complete understanding
- Be conversational and natural - acknowledge previous topics when relevant
- Explain clearly with examples
- Break down complex concepts in simple terms

Your answer:"""
            
            # Add confidence indicator if grounding is weak
            confidence_badge = ""
            if max_confidence < 0.55:
                confidence_badge = "⚠️ <span style='color:#d48806'><b>Low Confidence:</b> Limited relevant content found in syllabus.</span><br>"
            elif max_confidence < 0.65:
                confidence_badge = "⚠️ <span style='color:#d48806'><b>Moderate Confidence:</b> Some relevant content found.</span><br>"
            elif max_confidence >= 0.75:
                confidence_badge = "✅ <span style='color:#52c41a'><b>High Confidence:</b> Strong match found in syllabus.</span><br>"
            
            # Build source citations
            source_citations = self._format_sources(sources)
            
            # Build image references if available
            image_references = ""
            if has_images and image_results:
                image_references = self._format_image_references(image_results)
            
            system_output = (
                "<div style='background:#f6f8fa;border-radius:8px;padding:16px 20px 16px 20px;margin-bottom:8px;'>"
                f"<h4 style='color:#2d6cdf;margin-top:0'>📘 {subject} Tutor Answer</h4>"
                f"{confidence_badge}"
                f"<b>Question:</b> {question}<br>"
                f"<b>Answer:</b><br>"
                f"<div style='margin-left:1em'>{self.llm.invoke(prompt, task_type=TaskType.ANSWER_GENERATION) if self.use_multi_model else self.llm.invoke(prompt)}</div>"
                f"<div style='margin-top:8px;font-size:0.85em;color:#6a737d;'>Confidence: {max_confidence:.2f} | Sources: {len(results)} chunks"
                f"{' | 🖼️ ' + str(len(image_results)) + ' relevant diagrams' if has_images else ''}</div>"
                f"{source_citations}"
                f"{image_references}"
                "</div>"
            )
        elif self._is_academic_question(question):
            # Out-of-syllabus but academic: explicit insufficient grounding signaling
            if not results:
                # No grounding available at all
                system_output = (
                    "<div style='background:#fff4e6;border:2px solid #fa8c16;border-radius:8px;padding:16px 20px;margin-bottom:8px;'>"
                    "<h4 style='color:#d48806;margin-top:0'>⚠️ Insufficient Knowledge Grounding</h4>"
                    f"<b>Question:</b> {question}<br>"
                    f"<b>Status:</b> No relevant content found in the <b>{subject}</b> knowledge base (0 matches above threshold).<br><br>"
                    "<b>Options:</b><br>"
                    "1. This topic may not be covered in your uploaded textbooks<br>"
                    "2. Try rephrasing your question with different keywords<br>"
                    "3. Upload additional materials covering this topic<br><br>"
                    "<i style='color:#8c8c8c;'>As per paper principles: Refusing to guess is better than hallucinating.</i>"
                    "</div>"
                )
            else:
                # Weak grounding but has some matches - show recommendation
                system_output = (
                    "<div style='background:#fffbe6;border:2px solid #fa8c16;border-radius:8px;padding:16px 20px;margin-bottom:8px;'>"
                    "<h4 style='color:#d48806;margin-top:0'>⚠️ Weak Knowledge Grounding</h4>"
                    f"<b>Question:</b> {question}<br>"
                    f"<b>Note:</b> This topic has weak coverage in the <b>{subject}</b> textbook ({len(results)} weak matches).<br><br>"
                    "<b>💡 Did you select the right subject?</b><br>"
                    "• Computer Science questions? Select 'Computer Science - 12 - TN'<br>"
                    "• Commerce questions? Select 'Commerce - Higher Secondary Second Year - TN'<br>"
                    "• Physics questions? Select 'Physics - 12 - TN'<br><br>"
                    "<b>Other options:</b><br>"
                    "1. Try rephrasing your question with different keywords<br>"
                    "2. Upload more comprehensive materials on this topic<br><br>"
                    f"<i style='color:#8c8c8c;'>Retrieved {len(results)} chunks but relevance check indicates weak grounding.</i>"
                    "</div>"
                )

        else:
            # Block anything not academic
            system_output = (
                "❌ <span style='color:red'><b>Sorry, as your tutor, I cannot answer questions unrelated to the academic syllabus or class context.</b></span> "
                "If you have a question about your subject, please ask!"
            )
        self.logger.log(question, system_output)
        return system_output

    def _is_academic_question(self, question: str) -> bool:
        """
        Use LLM to determine if the question is academic/educational in nature.
        """
        prompt = f"""Is this question academic or educational in nature?

Question: "{question}"

Reply with ONLY "yes" or "no".

Academic questions include:
- Subject explanations ("what is", "explain", "define", "describe")
- Visualizations ("show me", "diagram", "illustrate", "visualize")
- Concepts, theories, processes, calculations
- Examples, comparisons, advantages/disadvantages  
- Homework help, study questions
- Follow-ups ("more details", "it", "that", "them")
- Affirmations ("yes", "sure", "interested", "tell me more")
- ANY question about school/college subjects (math, science, history, programming, commerce, etc.)

Non-academic ONLY:
- Entertainment gossip (movies, celebrities)
- Personal life questions
- Sports scores/news
- Current events unrelated to study

When in doubt, answer "yes" (be permissive).

Answer (yes/no):"""

        try:
            # Use fast model for quick validation (if multi-model enabled)
            if self.use_multi_model:
                response = self.llm.invoke(prompt, task_type=TaskType.QUICK_CHECK).strip().lower()
            else:
                response = self.llm.invoke(prompt).strip().lower()
            # Be more permissive - accept yes, y, academic, educational
            return any(word in response for word in ['yes', 'academic', 'educational', 'learning'])
        except:
            # Fallback: assume it's academic to avoid blocking legitimate questions
            return True

    def _build_conversation_context(self, chat_history: List[Any], max_turns: int = 3) -> str:
        """
        Build conversation context from recent chat history to understand follow-up questions.
        Returns a string with the last N turns of conversation.
        """
        if not chat_history:
            return ""
        
        # Get last N turns (each turn has user question and assistant answer)
        recent_turns = chat_history[-max_turns:] if len(chat_history) > max_turns else chat_history
        context_parts = []
        
        for turn in recent_turns:
            if isinstance(turn, dict):
                user_msg = turn.get('user', '')
                assistant_msg = turn.get('assistant', '')
                if user_msg:
                    context_parts.append(f"User: {user_msg}")
                if assistant_msg:
                    # Extract plain text from HTML (simple approach)
                    import re
                    plain_text = re.sub(r'<[^>]+>', '', assistant_msg)
                    # For follow-up understanding, capture enough context for comparisons
                    # Increased to 3000 chars to handle detailed multi-topic discussions
                    plain_text = plain_text[:3000] + "..." if len(plain_text) > 3000 else plain_text
                    context_parts.append(f"Assistant: {plain_text}")
        
        return "\n".join(context_parts)

    def _filter_context_by_type(self, full_context: str, question_type: QuestionType, 
                                current_subject: str, chat_history: List[Any]) -> str:
        """
        Filter conversation context based on question type to prevent context pollution.
        
        Critical for handling topic switches correctly:
        - NEW_TOPIC: Clear context (avoid confusing LLM with unrelated previous topics)
        - Subject switch: Clear context (e.g., Commerce → Biology)
        - FOLLOW_UP/CLARIFICATION: Keep context (need it for understanding)
        """
        # NEW_TOPIC: Student is starting fresh, clear context to avoid pollution
        if question_type == QuestionType.NEW_TOPIC:
            return ""
        
        # Check for subject switching (prevent cross-subject context pollution)
        if chat_history and len(chat_history) > 0:
            # Get the subject from the last turn (if stored)
            last_turn = chat_history[-1] if isinstance(chat_history, list) else None
            if isinstance(last_turn, dict):
                last_subject = last_turn.get('subject', current_subject)
                # If subject changed, this is effectively a new conversation
                if last_subject != current_subject:
                    return ""
        
        # FOLLOW_UP, CLARIFICATION, VISUAL_REQUEST: Keep full context
        # These question types REQUIRE previous context to make sense
        return full_context

    def _resolve_question_with_context(self, question: str, conversation_context: str) -> str:
        """
        Use LLM to intelligently resolve questions based on conversation context.
        Handles follow-ups, affirmations, and topic shifts intelligently.
        """
        if not conversation_context:
            return question
        
        # Quick check for simple follow-ups that need context
        simple_followups = ['more', 'explain more', 'tell me more', 'continue', 'elaborate', 
                           'further', 'what else', 'more details', 'go on', 'keep going']
        q_lower = question.lower().strip()
        
        is_simple_followup = any(q_lower == phrase or q_lower.startswith(phrase + ' ') for phrase in simple_followups)
        
        # For simple follow-ups, extract topic from last user question
        if is_simple_followup:
            lines = conversation_context.split('\n')
            for line in reversed(lines):
                if line.startswith("User:"):
                    last_question = line.replace("User:", "").strip()
                    # Return the last question topic for continuity
                    return last_question
        
        # Use LLM to understand the question in context
        prompt = f"""Analyze the student's question in the context of the conversation.

CONVERSATION HISTORY:
{conversation_context}

STUDENT'S NEW INPUT: "{question}"

TASK: Determine what the student is asking about. Reply with ONLY the interpreted question.

Rules:
1. If it's an affirmation (yes, sure, interested, ok, please, etc.), extract what they're confirming interest in from the conversation and return that as a proper question (e.g., "explain OSI model")
2. If it's a follow-up (more, explain more, further, elaborate, etc.), identify the topic from the conversation and expand it (e.g., "explain secondary market" if they said "explain more" after discussing secondary markets)
3. If it's a follow-up with pronouns (it, that, this, them), replace with the actual topic from conversation
4. If asking for diagrams/images/charts without specifying topic, extract the topic from conversation (e.g., "show diagram of primary market" if they asked about primary market before)
5. If it's a NEW topic (complete question about something different), return it as-is
6. If it references a previous topic implicitly, make it explicit

Examples:
- Conversation: "...explain OSI model to you later"
  Input: "yes interested"
  Output: explain OSI model

- Conversation: "...secondary market is where..."
  Input: "explain more" 
  Output: explain secondary market in detail

- Conversation: "...list data structures..."
  Input: "what are its advantages"
  Output: what are the advantages of list data structure

- Conversation: "...primary market is where new securities..."
  Input: "explain with relevant diagrams if any"
  Output: explain primary market with diagrams

- Conversation: "...tuples in Python are immutable..."
  Input: "give a flow chart diagram"
  Output: give a flow chart diagram of tuples in Python

- Conversation: "...stack data structure works by..."
  Input: "show me a visual representation"
  Output: show visual representation of stack data structure

- Input: "what is stack data structure"  
  Output: what is stack data structure

Return ONLY the interpreted question, nothing else:"""

        try:
            # Use reasoning model for pronoun resolution (if multi-model enabled)
            if self.use_multi_model:
                resolved = self.llm.invoke(prompt, task_type=TaskType.ANALYSIS).strip()
            else:
                resolved = self.llm.invoke(prompt).strip()
            # Fallback if LLM returns empty or too long
            if not resolved or len(resolved) > 200:
                return question
            return resolved
        except:
            # Fallback to original question if LLM fails
            return question

    def _build_smart_prompt(self, question: str, conversation_context: str,
                           syllabus_text: str, has_images: bool, image_count: int) -> str:
        """
        Build ONE intelligent prompt that handles everything:
        - Relevance checking
        - Answer generation  
        - Visual content awareness
        - Context awareness
        
        Pure LLM intelligence - no hardcoded rules.
        """
        visual_note = ""
        if has_images and image_count > 0:
            visual_note = f"\n\nNOTE: There are {image_count} relevant diagrams/images available that will be shown below your answer. Reference them naturally if they help explain the concept."
        
        prompt = f"""You are an expert educational tutor. Answer the student's question using the textbook content provided.

CONVERSATION CONTEXT:
{conversation_context if conversation_context else '(First question in this session)'}

STUDENT'S QUESTION:
{question}

TEXTBOOK CONTENT (may include exam questions, diagrams, and explanations):
{syllabus_text}

YOUR TASK:
1. First, verify the textbook content is relevant to the student's question above
2. If NOT relevant or insufficient, respond with: "I cannot find relevant information about this topic in the textbook."
3. If relevant, provide a clear, accurate answer to the STUDENT'S QUESTION ONLY (not any exam questions in the textbook)
4. Be natural and educational - explain concepts clearly
5. Use examples from the textbook when available
6. Reference diagrams naturally if images are available{visual_note}

IMPORTANT: Answer ONLY the student's question "{question}" - ignore any practice/exam questions that may appear in the textbook content.

Provide your answer:"""
        
        return prompt
    
    def _generate_dynamic_prompt(self, question: str, analysis, conversation_context: str,
                                 syllabus_sample: str, has_images: bool, image_count: int) -> str:
        """
        META-PROMPTING: Use LLM to generate optimal answering instructions based on question characteristics.
        More reliable approach: LLM generates INSTRUCTIONS, we build the prompt structure.
        """
        
        meta_prompt = f"""Analyze this student question and generate OPTIMAL INSTRUCTIONS for how an AI tutor should answer it.

QUESTION ANALYSIS:
- Type: {analysis.type.value}
- Topic: {analysis.topic}
- Question: {question}
- Has context: {bool(conversation_context)}
- Visual aids: {has_images} ({image_count} diagrams available)

TEXTBOOK CONTENT PREVIEW:
{syllabus_sample[:600]}

YOUR TASK:
Generate a numbered list of 4-6 SPECIFIC INSTRUCTIONS that will produce the BEST educational answer for this question.

INSTRUCTION GUIDELINES by question type:
- **Definition questions** ("what is"): Extract definition + purpose + characteristics + examples + practical context
- **Comparison questions** ("difference"): Create structured comparison (table/bullet points) highlighting ALL key distinctions
- **Procedural questions** ("how"): Break into sequential steps with explanations and examples
- **Complex questions**: Decompose into sub-topics, address each thoroughly

REQUIREMENTS:
- Be SPECIFIC to this question and topic (not generic advice)
- Emphasize COMPREHENSIVE extraction of ALL relevant textbook content
- Specify structure (paragraphs, lists, tables, examples)
- Warn: "Answer the STUDENT'S question - NOT textbook practice questions"
- Each instruction should be actionable and precise

EXAMPLE OUTPUT (for reference):
"1. Extract the complete definition of [TOPIC] from the textbook, including its purpose and characteristics
2. Explain how [TOPIC] operates, using the specific examples and mechanisms described in the textbook
3. Provide real-world context by discussing the practical applications mentioned in the textbook
4. Structure your answer in clear paragraphs, using terminology from the textbook
5. IMPORTANT: Answer the student's question about [TOPIC] - ignore any practice questions in the textbook content"

NOW GENERATE INSTRUCTIONS for: "{question}" (Topic: {analysis.topic}, Type: {analysis.type.value}):

INSTRUCTIONS:"""

        try:
            # Use specialized reasoning model for meta-prompting (if multi-model enabled)
            if self.use_multi_model:
                instructions = self.llm.invoke(meta_prompt, task_type=TaskType.META_REASONING).strip()
                logger.info(f"🧠 Meta-reasoning: Used Qwen2.5-14B for instruction generation")
            else:
                instructions = self.llm.invoke(meta_prompt).strip()
            
            # Build the full prompt programmatically with proper structure
            full_prompt = f"""You are an expert educational tutor. Answer the student's question using ALL relevant information from the textbook.

CONVERSATION CONTEXT:
{{{{CONTEXT}}}}

STUDENT'S QUESTION:
{{{{QUESTION}}}}

TEXTBOOK CONTENT:
{{{{TEXTBOOK_CONTENT}}}}

YOUR TASK - Follow these specific instructions:
{instructions}

Provide your complete, comprehensive answer:"""
            
            logger.info(f"🎨 Generated dynamic prompt with custom instructions ({len(instructions)} chars)")
            return full_prompt
            
        except Exception as e:
            logger.warning(f"Meta-prompt generation failed: {e}, using fallback")
            return None  # Will trigger fallback to static template
    
    def _build_final_prompt_from_template(self, dynamic_prompt_template: str,
                                          question: str, conversation_context: str,
                                          syllabus_text: str, has_images: bool, image_count: int) -> str:
        """
        Fill in the LLM-generated dynamic prompt template with actual content.
        Handles both {{PLACEHOLDER}} and {PLACEHOLDER} formats.
        """
        # Replace placeholders - handle both single and double braces
        prompt = dynamic_prompt_template
        
        # Replace QUESTION
        prompt = prompt.replace('{{{{QUESTION}}}}', question)
        prompt = prompt.replace('{{QUESTION}}', question)
        prompt = prompt.replace('{QUESTION}', question)
        
        # Replace CONTEXT
        context_text = conversation_context if conversation_context else '(First question in this session)'
        prompt = prompt.replace('{{{{CONTEXT}}}}', context_text)
        prompt = prompt.replace('{{CONTEXT}}', context_text)
        prompt = prompt.replace('{CONTEXT}', context_text)
        
        # Replace TEXTBOOK_CONTENT
        prompt = prompt.replace('{{{{TEXTBOOK_CONTENT}}}}', syllabus_text)
        prompt = prompt.replace('{{TEXTBOOK_CONTENT}}', syllabus_text)
        prompt = prompt.replace('{TEXTBOOK_CONTENT}', syllabus_text)
        
        logger.info(f"✅ Filled prompt template: {len(prompt)} chars total, {len(syllabus_text)} chars textbook content")
        
        return prompt
    
    def _is_relevant_answer(self, question: str, syllabus_chunks: list) -> bool:
        """
        CRITICAL INTELLIGENCE: Check if retrieved content can actually answer the question.
        Uses LLM to verify relevance - prevents hallucinations and wrong answers.
        
        LENIENT CHECK: We retrieve broadly (low thresholds), then verify relevance here.
        """
        if not syllabus_chunks:
            return False
        
        # OPTIMIZATION: Use fewer, shorter chunks for faster relevance check (10.87s -> ~4s)
        sample_content = "\n\n".join(syllabus_chunks[:3])[:1500]
        
        # Concise prompt for faster processing
        prompt = f"""Does this content help answer the question?

QUESTION: {question}

CONTENT: {sample_content}

Reply ONLY "yes" or "no":"""

        try:
            # Use fast model for quick binary checks (if multi-model enabled)
            if self.use_multi_model:
                response = self.llm.invoke(prompt, task_type=TaskType.QUICK_CHECK).strip().lower()
            else:
                response = self.llm.invoke(prompt, max_tokens=64).strip().lower()
            
            is_relevant = 'yes' in response
            
            if not is_relevant:
                logger.info(f"⚠️ Relevance check FAILED: Content cannot answer question")
                logger.info(f"   LLM response: {response}")
            else:
                logger.info(f"✅ Relevance check PASSED")
            
            return is_relevant
        except Exception as e:
            logger.warning(f"Relevance check error: {e}")
            return True  # Default to allowing answer if check fails
    
    def _is_relevant_answer_fast(self, question: str, syllabus_sample: str) -> bool:
        """
        ULTRA-FAST relevance check (cuts 10.87s -> ~3s).
        Uses QUICK_CHECK model with minimal prompt and sample text.
        
        Args:
            question: Question text
            syllabus_sample: Pre-sampled content (already shortened by caller)
        
        Returns:
            True if relevant, False otherwise
        """
        if not syllabus_sample or len(syllabus_sample) < 50:
            return False
        
        # Ultra-concise prompt for speed
        prompt = f"""Can this answer the question?
Q: {question[:150]}
Content: {syllabus_sample[:800]}
Reply yes/no:"""

        try:
            if self.use_multi_model:
                response = self.llm.invoke(prompt, task_type=TaskType.QUICK_CHECK).strip().lower()
            else:
                response = self.llm.invoke(prompt, max_tokens=32).strip().lower()
            
            return 'yes' in response
        except Exception as e:
            logger.warning(f"Fast relevance check error: {e}")
            return True  # Default allow if check fails

    # _check_subject_mismatch - REMOVED (Dead code)
    # Reason: File system isolation (vector_db/Subject/) architecturally guarantees
    # all retrieved content is from the correct subject. Runtime check is redundant.
    # This function contained hardcoded examples (Commerce, Computer Science) which
    # violated pure intelligence principle.
    def _check_subject_mismatch_DEPRECATED(self, question: str, expected_subject: str, sample_content: str) -> bool:
        """
        DEPRECATED: Replaced by architectural file system isolation.
        This check is redundant - vector_db/Subject/ structure guarantees subject purity.
        """
        # Function kept for reference but never called
        return False  # Always return no mismatch (architectural guarantee)

        try:
            response = self.llm.invoke(prompt).strip().lower()
            # Return True if mismatch (content doesn't match subject)
            return 'no' in response
        except:
            # On error, assume no mismatch to avoid false positives
            return False

    def _format_sources(self, sources: list) -> str:
        """Format source citations with page numbers"""
        if not sources:
            return ""
        
        # Group by source file and collect unique pages
        from collections import defaultdict
        import os
        source_pages = defaultdict(set)
        
        for meta in sources[:5]:  # Limit to top 5 sources
            if meta and 'source' in meta and 'page' in meta:
                source_file = os.path.basename(meta['source'])
                source_pages[source_file].add(meta['page'])
        
        if not source_pages:
            return ""
        
        # Build citation string
        citations = []
        for source_file, pages in source_pages.items():
            sorted_pages = sorted(list(pages))
            page_str = ", ".join(str(p) for p in sorted_pages[:3])  # Max 3 pages per source
            if len(sorted_pages) > 3:
                page_str += ", ..."
            citations.append(f"{source_file} (p. {page_str})")
        
        return (
            f"<div style='margin-top:12px;padding:10px;background:#f0f9ff;border-left:3px solid #3b82f6;font-size:0.9em;'>"
            f"📚 <b>Sources:</b> {' • '.join(citations)}"
            "</div>"
        )

    def _format_image_references(self, image_results: list) -> str:
        """Format and display actual images inline with base64 encoding"""
        if not image_results:
            return ""
        
        import base64
        from io import BytesIO
        import os
        
        html_parts = []
        html_parts.append(
            "<div style='margin-top:16px;padding:12px;background:#f0f7ff;border-left:4px solid #1890ff;border-radius:4px;'>" 
            "<h4 style='margin-top:0;color:#1890ff;'>🖼️ Relevant Diagrams from Textbook</h4>"
        )
        
        # Display up to 3 images
        for img_data, meta, score in image_results[:3]:
            try:
                # Get PIL image from img_data dict
                pil_image = img_data.get('image') if isinstance(img_data, dict) else None
                
                if pil_image:
                    # Convert PIL image to base64
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
                # Fallback to reference only if image display fails
                source_file = os.path.basename(meta.get('source', 'Unknown'))
                page = meta.get('page', '?')
                html_parts.append(
                    f"<div style='padding:4px;font-size:0.9em;'>📚 {source_file} - Page {page}</div>"
                )
        
        html_parts.append("</div>")
        return "".join(html_parts)

    def generate_practice_questions(self, topic: str, subject: str, num_questions: int = 5, 
                                   difficulty: str = "medium") -> str:
        """
        Generate practice questions for a given topic.
        
        Args:
            topic: Topic to generate questions about
            subject: Subject name
            num_questions: Number of questions to generate
            difficulty: easy, medium, or hard
        
        Returns:
            HTML-formatted questions
        """
        if not self.vector_store:
            self.load_subject(subject)
        
        result = self.question_generator.generate_from_topic(
            topic=topic,
            subject=subject,
            vector_store=self.vector_store,
            num_questions=num_questions,
            difficulty=difficulty
        )
        
        # Handle both old format (list) and new format (dict with images)
        questions = result.get('questions', []) if isinstance(result, dict) else result
        
        if not questions:
            return f"""<div style='background:#fff3cd;border:2px solid #ffc107;border-radius:8px;padding:20px;margin:20px 0;'>
                <h4 style='color:#856404;margin-top:0;'>⚠️ Insufficient Knowledge Base Content</h4>
                <p style='color:#856404;margin:8px 0;'><b>Topic:</b> {topic}</p>
                <p style='color:#856404;margin:8px 0;'><b>Subject:</b> {subject}</p>
                <p style='color:#856404;margin:8px 0;'>This topic was not found in your uploaded textbooks.</p>
                <p style='color:#856404;margin:8px 0;'><b>Suggestions:</b></p>
                <ul style='color:#856404;margin:8px 0 8px 20px;'>
                    <li>Verify the topic name matches content in your textbooks</li>
                    <li>Try a related or broader topic (e.g., "markets" instead of "recursion" for Commerce)</li>
                    <li>Upload textbooks that cover this topic</li>
                </ul>
            </div>"""
        
        return PracticeQuestionGenerator.format_for_display(result)
