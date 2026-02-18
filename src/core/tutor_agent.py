"""
tutor_agent.py: Intelligent orchestrator for handling tutoring conversations
Uses structured decision-making with LLM assistance
"""
from typing import List, Dict, Any, Optional, Tuple
from .llm import OllamaLLM
from .embeddings import embed_texts_batched
from .image_embeddings import embed_text_for_image_search
from .multimodal_vector_store import MultimodalVectorStore


class TutorOrchestrator:
    """
    Intelligent tutoring orchestrator that handles:
    - Conversation context management
    - Follow-up question resolution
    - Knowledge base search
    - Subject validation
    - Answer generation
    
    Uses structured flow with LLM-assisted decisions rather than brittle prompt engineering.
    """
    
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.llm = rag_engine.llm
        self.conversation_history = []  # Structured history
    
    def answer(self, question: str, subject: str, chat_history: List[Dict] = None) -> str:
        """
        Main entry point for answering questions.
        
        Args:
            question: User's raw question
            subject: Current subject
            chat_history: Previous conversation turns
            
        Returns:
            Formatted HTML answer
        """
        # Step 1: Update internal state
        self.conversation_history = chat_history or []
        self.rag_engine.current_subject = subject
        
        # Step 2: Resolve question (handle follow-ups, affirmations)
        resolved_question = self._resolve_question(question)
        
        # Step 3: Check if academic/appropriate
        if not self._is_academic(resolved_question):
            return self._format_non_academic_response
        """Create tools that the agent can use"""
        
        def search_knowledge_base(query: str) -> str:
            """
            Search the current subject's knowledge base for relevant information.
            Use this when you need to retrieve factual information from the textbook.
            """
            try:
                subject = self.rag_engine.current_subject
                if not subject:
                    return "Error: No subject loaded. Please specify a subject first."
                
                # Ensure correct KB is loaded
                if not self.rag_engine.vector_store or self.rag_engine.current_subject != subject:
                    self.rag_engine.load_subject(subject)
                
                # Perform search
                from .multimodal_vector_store import MultimodalVectorStore
                is_multimodal = isinstance(self.rag_engine.vector_store, MultimodalVectorStore)
                
                if is_multimodal:
                    # Check if query wants visuals
                    visual_keywords = ['show', 'visualize', 'diagram', 'illustration', 'picture', 
                                     'image', 'looks like', 'structure', 'architecture']
                    wants_visual = any(kw in query.lower() for kw in visual_keywords)
                    
                    # Multimodal search
                    q_vec_text = embed_texts_batched([query])
                    q_vec_image = embed_text_for_image_search(query)
                    k_images = 8 if wants_visual else 3
                    
                    results = self.rag_engine.vector_store.search_multimodal(
                        q_vec_text, q_vec_image, k_text=10, k_images=k_images
                    )
                    
                    text_results = results['texts']
                    image_results = results['images']
                    
                    # Format results
                    response = f"Found {len(text_results)} text chunks"
                    if image_results:
                        response += f" and {len(image_results)} relevant diagrams"
                    response += ":\n\n"
                    
                    # Add text content
                    for i, (text, score, meta) in enumerate(text_results[:5], 1):
                        page = meta.get('page', '?')
                        response += f"{i}. [Page {page}, Score: {score:.2f}] {text[:1000]}...\n\n"
                    
                    # Add image info
                    if image_results:
                        response += "\nRelevant diagrams available:\n"
                        for img_data, meta, score in image_results[:3]:
                            page = meta.get('page', '?')
                            response += f"- Diagram on page {page} (relevance: {score:.2f})\n"
                    
                    return response
                else:
                    # Text-only search
                    q_vec = embed_texts_batched([query])
                    results = self.rag_engine.vector_store.search(q_vec, k=30, threshold=0.5)
                    
                    if not results:
                        return "No relevant information found in the knowledge base."
                    
                    response = f"Found {len(results)} relevant chunks:\n\n"
                    for i, (text, score, meta) in enumerate(results[:5], 1):
                        page = meta.get('page', '?')
                        response += f"{i}. [Page {page}, Score: {score:.2f}] {text[:1000]}...\n\n"
                    
                    return response
                    
            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"
        
        def resolve_question_context(question: str, history: str) -> str:
            """
            Resolve ambiguous questions using conversation history.
            Use this for follow-ups like 'explain more', 'tell me about it', etc.
            Returns the resolved/clarified question.
            """
            if not history:
                return question
            
            prompt = f"""Given this conversation history and a new question, determine what the user is really asking about.

Conversation History:
{history}

New Question: "{question}"

Instructions:
- If it's a follow-up (more, it, that, continue, etc.), identify what topic they're referring to
- If it's an affirmation (yes, sure, interested), extract what they're confirming interest in
- If it's a complete new question, return as-is
- Return ONLY the clarified/resolved question, nothing else

Resolved question:"""
            
            try:
                resolved = self.llm.invoke(prompt).strip()
                return resolved if resolved else question
            except:
                return question
        
        def check_subject_match(question: str, content_sample: str, subject: str) -> str:
            """
            Check if retrieved content actually matches the expected subject.
            Use this to validate that search results are relevant to the subject.
            """
            subject_name = subject.split('-')[0].strip()
            
            prompt = f"""Does this content belong to the subject "{subject_name}"?

Question: {question}
Subject: {subject_name}

Sample Content:
{content_sample[:3000]}

Reply with ONLY "yes" or "no":"""
            
            try:
                response = self.llm.invoke(prompt).strip().lower()
                return "yes" if "yes" in response else "no"
            except:
                return "yes"  # Assume match on error
        
        # Return tools list
        return [
            Tool(
                name="SearchKnowledgeBase",
                func=search_knowledge_base,
                description="Search the current subject's textbook knowledge base. Input should be a clear question or topic to search for. Returns relevant text chunks and diagrams if available."
            ),
            Tool(
                name="ResolveQuestionContext",
                func=resolve_question_context,
                description="Resolve ambiguous follow-up questions using conversation history. Input format: 'question|history'. Returns the clarified question."
            ),
            Tool(
                name="CheckSubjectMatch",
                func=check_subject_match,
                description="Validate if content matches the expected subject. Input format: 'question|content|subject'. Returns 'yes' or 'no'."
            ),
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent"""
        
        template = """You are a friendly, intelligent tutor helping a student learn from their textbooks.

You have access to these tools:
{tools}

Current Subject: {subject}

Student's Question: {input}

Conversation History:
{chat_history}

Think step by step:
1. If the question is a follow-up (like "explain more", "tell me about it"), use ResolveQuestionContext first
2. Then use SearchKnowledgeBase to find relevant information
3. If needed, use CheckSubjectMatch to verify content relevance
4. Finally, provide a clear, educational answer based on the retrieved information

{agent_scratchpad}

Remember:
- Be conversational and friendly
- Acknowledge previous topics naturally
- If asking for visuals (show, diagram, etc.), mention that in your search
- Provide comprehensive answers for general questions
- Be specific for targeted questions
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "subject", "chat_history", "agent_scratchpad"],
            partial_variables={"tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])}
        )
        
        agent = create_react_agent(
            llm=self.llm.llm,  # Get underlying LangChain LLM
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def answer(self, question: str, subject: str) -> str:
        """
        Answer a question using the agent
        
        Args:
            question: User's question
            subject: Current subject
            
        Returns:
            Agent's answer
        """
        try:
            # Set current subject
            self.rag_engine.current_subject = subject
            
            # Run agent
            response = self.agent.invoke({
                "input": question,
                "subject": subject
            })
            
            return response.get("output", "I apologize, but I couldn't generate an answer.")
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def reset_conversation(self):
        """Clear conversation memory"""
        self.memory.clear()
