"""
agent_orchestrator.py: Intelligent agent system for tutoring
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """Types of questions the system can handle"""
    NEW_TOPIC = "new_topic"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    AFFIRMATION = "affirmation"
    VISUAL_REQUEST = "visual_request"
    OFF_TOPIC = "off_topic"

@dataclass
class AnalyzedQuestion:
    """Result of question analysis"""
    original: str
    expanded: str
    type: QuestionType
    topic: str
    confidence: float

class AgentOrchestrator:
    """
    Orchestrates all AI agents for intelligent tutoring.
    Manages: question analysis, query rewriting, retrieval, reasoning, and response generation.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_question(self, question: str, conversation_context: str, subject: str) -> AnalyzedQuestion:
        """
        Step 1: Analyze the user's question using pure LLM intelligence
        No examples, no hardcoding - clearer task decomposition
        """
        prompt = f"""TASK: Analyze and expand a student's question for knowledge retrieval.

SUBJECT: {subject}

CONVERSATION HISTORY:
{conversation_context if conversation_context else "(This is the first question - no prior context)"}

STUDENT'S CURRENT QUESTION: "{question}"

=== ANALYSIS STEPS ===

STEP 1 - QUESTION TYPE:
Determine if this question:
- References previous topics in the conversation history → ANSWER: follow_up
- Is a brand new topic unrelated to previous conversation → ANSWER: new_topic
- Is off-topic/non-academic → ANSWER: off_topic

STEP 2 - TOPIC IDENTIFICATION:
What are the main topics/concepts being discussed?
- For follow-ups: Extract topic names from recent "User:" questions above
- For new questions: Use topic mentioned in current question
- For comparisons ("their", "both", "differences"): List ALL topics being compared

STEP 4 - QUESTION EXPANSION (CRITICAL FOR RETRIEVAL):
Rewrite to be completely self-contained by replacing ALL vague words:
- "it" → the actual thing being referenced
- "them"/"their"/"both" → the actual entities/topics
- "more" → the specific topic to explain more about
- "these"/"those"/"that" → concrete references

HOW TO EXPAND:
1. Scan the question for vague words (it, them, their, more, both, these, that, those)
2. Look at last 2-3 "User:" questions to find what they refer to
3. Replace vague words with actual topic names
4. Result must make sense without any prior context

CRITICAL: If question is "explain more" and last User question was about "secondary market", then expanded version MUST be "explain secondary market in more detail" NOT just "explain more".

=== OUTPUT FORMAT ===
TYPE: <new_topic OR follow_up OR off_topic>
TOPIC: <topic name(s)>
EXPANDED: <fully expanded question with NO vague references>"""

        try:
            response = self.llm.invoke(prompt).strip()
            logger.info(f"LLM Analysis - Question: '{question}' | Response: {response}")
            
            # Parse response
            type_line = [l for l in response.split('\n') if l.startswith('TYPE:')]
            topic_line = [l for l in response.split('\n') if l.startswith('TOPIC:')]
            expanded_line = [l for l in response.split('\n') if l.startswith('EXPANDED:')]
            
            q_type = QuestionType.NEW_TOPIC
            if type_line:
                type_str = type_line[0].split(':', 1)[1].strip().lower()
                if 'follow' in type_str:
                    q_type = QuestionType.FOLLOW_UP
                elif 'clarif' in type_str:
                    q_type = QuestionType.CLARIFICATION
                elif 'affirm' in type_str:
                    q_type = QuestionType.AFFIRMATION
                elif 'off' in type_str:
                    q_type = QuestionType.OFF_TOPIC
            
            topic = topic_line[0].split(':', 1)[1].strip() if topic_line else question
            expanded = expanded_line[0].split(':', 1)[1].strip() if expanded_line else question
            
            logger.info(f"Analysis Result - Type: {q_type}, Topic: '{topic}', Expanded: '{expanded}'")
            
            return AnalyzedQuestion(
                original=question,
                expanded=expanded,
                type=q_type,
                topic=topic,
                confidence=0.8
            )
            
        except Exception as e:
            # Fallback: Use simpler LLM call for expansion
            logger.warning(f"Main analysis failed: {e}, using fallback")
            try:
                # Expand question
                fallback_prompt = f"""Conversation:
{conversation_context if conversation_context else "(no context)"}

Question: {question}

Task: Expand this question to be self-contained by replacing pronouns (it, them, their, more, etc.) with actual topic names from the conversation.

Expanded question:"""
                expanded = self.llm.invoke(fallback_prompt).strip()
            except:
                expanded = question
            
            return AnalyzedQuestion(
                original=question,
                expanded=expanded,
                type=QuestionType.NEW_TOPIC,
                topic=question,
                confidence=0.5
            )
    
    def is_academic(self, question: str) -> bool:
        """Check if question is academic using pure LLM intelligence"""
        prompt = f"""Is this an academic/educational question that a student would ask about learning materials?

Question: "{question}"

Answer 'yes' if it's an educational question, 'no' if it's clearly entertainment/casual/off-topic.
Reply ONLY 'yes' or 'no':"""
        
        try:
            response = self.llm.invoke(prompt).strip().lower()
            return 'yes' in response or 'academic' in response
        except:
            return True  # Default to academic to avoid blocking valid questions
    
    # check_subject_match removed - redundant due to architectural file system isolation
    # Vector stores are completely isolated: vector_db/Commerce/, vector_db/Biology/, etc.
    # load_subject() ensures only one subject's FAISS index is loaded at a time
    # Therefore, cross-subject contamination is architecturally impossible
    
    def generate_chain_of_thought(self, question: str, context: str) -> str:
        """
        Generate reasoning steps for complex questions
        """
        if len(question.split()) < 5:
            return ""  # Simple questions don't need CoT
        
        prompt = f"""Break down how to answer this question step-by-step.

Question: {question}
Available Context: {context[:5000]}...

Provide 3-5 reasoning steps:
1.
2.
3.

Steps:"""

        try:
            cot = self.llm.invoke(prompt).strip()
            return cot if len(cot) < 500 else ""
        except:
            return ""
    
    def synthesize_answer(self, question: str, conversation_context: str, 
                         syllabus_text: str, chain_of_thought: str,
                         is_general: bool) -> str:
        """
        Generate the final answer using all available information.
        Pure intelligence - no hardcoded examples or templates.
        """
        base_prompt = f"""You are an expert tutor. Your task is to answer the student's question accurately using ONLY the textbook content provided below.

CONTEXT FROM PREVIOUS CONVERSATION:
{conversation_context if conversation_context else '(This is the first question in this conversation)'}

STUDENT'S QUESTION:
{question}

RELEVANT TEXTBOOK CONTENT:
{syllabus_text}

INSTRUCTIONS:
- Answer the question using ONLY information from the textbook content above
- Be accurate and precise - do not add information not present in the textbook
- If the question asks for a definition, structure, process, or explanation, provide it clearly
- Use examples from the textbook when available
- Connect to previous conversation context only if directly relevant"""

        if is_general:
            base_prompt += """
- This is a general/overview question - provide comprehensive coverage
- Cover multiple aspects, types, or categories if mentioned in the textbook
- Give a well-rounded explanation"""
        else:
            base_prompt += """
- Focus your answer on what the question specifically asks for
- Be detailed and thorough in explaining the concept"""
        
        if chain_of_thought:
            base_prompt += f"""

REASONING GUIDANCE:
{chain_of_thought}"""
        
        base_prompt += """

Now provide your answer based strictly on the textbook content:
"""
        
        return base_prompt
