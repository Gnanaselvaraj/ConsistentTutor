"""
retrieval_agent.py: Intelligent agent for retrieval strategy decisions
Pure LLM-based decision making - NO hardcoding
"""
import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalStrategy:
    """Strategy for retrieving content"""
    search_depth: str  # "shallow", "medium", "deep"
    multimodal_priority: str  # "text_only", "balanced", "visual_heavy"
    context_window: int  # Number of chunks to retrieve
    requires_comparison: bool  # Whether comparing multiple concepts
    focus_areas: list  # Specific aspects to focus retrieval on


class RetrievalStrategyAgent:
    """
    Intelligent agent that determines optimal retrieval strategy
    based on question analysis AND student's learning history.
    Pure LLM intelligence - no hardcoding, memory-aware decisions.
    """
    
    def __init__(self, llm, student_profile=None):
        self.llm = llm
        self.student_profile = student_profile  # Rolling long-term memory
    
    def determine_strategy(self, question: str, analyzed_question, subject: str) -> RetrievalStrategy:
        """
        Use LLM intelligence + student history to determine best retrieval strategy.
        Memory-aware: adapts based on student's learning patterns.
        No hardcoded rules - let LLM decide everything based on context.
        """
        # Build student context from long-term memory
        student_context = ""
        if self.student_profile:
            recent_topics = list(self.student_profile.topics_studied.keys())[-5:]
            weak_areas = list(self.student_profile.weak_areas)
            style = self.student_profile.preferred_explanation_style
            
            student_context = f"""
STUDENT LEARNING PROFILE (Session Memory):
- Recent topics studied: {', '.join(recent_topics) if recent_topics else 'First session'}
- Weak areas needing more context: {', '.join(weak_areas) if weak_areas else 'None identified yet'}
- Preferred style: {style} explanations
- Total questions this session: {len(self.student_profile.questions_asked)}
"""
        
        prompt = f"""You are an adaptive retrieval strategy expert for educational content.

SUBJECT: {subject}
ORIGINAL QUESTION: {question}
EXPANDED QUESTION: {analyzed_question.expanded}
QUESTION TYPE: {analyzed_question.type.value}
{student_context}

DECISIONS TO MAKE:

1. SEARCH DEPTH:
   - "shallow" = High-level overview questions (definitions, "what is")
   - "medium" = StaPure conceptual/definitions (no visuals needed)
   - "balanced" = Text primary, images supportive (most questions)
   - "visual_heavy" = Diagrams/charts/images CRITICAL (structures, processes, comparisons)
   - If student prefers "visual": Favor "visual_heavy" or "balanced"
2. MULTIMODAL PRIORITY:
   - "text_only" = No visual content needed
   - "balanced" = Text primary, images supportive
   - "visual_heavy" = Diagrams/images are critical to understanding

3. CONTEXT WINDOW (number of text chunks to retrieve):
   - Small questions (definitions): Choose ONE number between 30-40
   - Medium questions (explanations): Choose ONE number between 50-70
   - Large questions (comparisons, analysis): Choose ONE number between 80-100
   - IMPORTANT: Output ONLY a single integer number, not a range

4. REQUIRES COMPARISON:
   - Does this question ask to compare/contrast multiple concepts? yes/no

5. FOCUS AREAS:
   - List 2-4 specific aspects/subtopics to prioritize in retrieval
   - Examples: ["definition", "examples"], ["advantages", "disadvantages", "comparison"]

OUTPUT FORMAT (be precise):
SEARCH_DEPTH: <shallow | medium | deep>
MULTIMODAL_PRIORITY: <text_only | balanced | visual_heavy>
CONTEXT_WINDOW: <number>
REQUIRES_COMPARISON: <yes | no>
FOCUS_AREAS: <comma-separated list>"""

        try:
            response = self.llm.invoke(prompt).strip()
            logger.info(f"Retrieval Strategy LLM Response: {response}")
            
            # Parse LLM response
            lines = {line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip() 
                    for line in response.split('\n') if ':' in line}
            
            search_depth = lines.get('SEARCH_DEPTH', 'medium').lower()
            multimodal = lines.get('MULTIMODAL_PRIORITY', 'balanced').lower()
            
            # Parse context window - extract first number from string (handles "50" or "50-70 chunks")
            context_str = lines.get('CONTEXT_WINDOW', '50')
            import re
            context_match = re.search(r'(\d+)', context_str)
            context_window = int(context_match.group(1)) if context_match else 50
            
            requires_comp = 'yes' in lines.get('REQUIRES_COMPARISON', 'no').lower()
            focus_raw = lines.get('FOCUS_AREAS', 'general understanding')
            focus_areas = [f.strip() for f in focus_raw.split(',')]
            
            strategy = RetrievalStrategy(
                search_depth=search_depth,
                multimodal_priority=multimodal,
                context_window=context_window,
                requires_comparison=requires_comp,
                focus_areas=focus_areas
            )
            
            logger.info(f"Retrieval Strategy: {strategy}")
            return strategy
            
        except Exception as e:
            logger.warning(f"Strategy determination failed: {e}, using defaults")
            # Intelligent defaults
            return RetrievalStrategy(
                search_depth="medium",
                multimodal_priority="balanced",  # Always balanced - let similarity decide
                context_window=50,
                requires_comparison=False,
                focus_areas=["explanation", "examples"]
            )
    
    def adjust_thresholds(self, strategy: RetrievalStrategy) -> Dict[str, float]:
        """
        Use LLM intelligence to determine optimal similarity thresholds.
        Different strategies need different strictness levels.
        """
        prompt = f"""Given this retrieval strategy, determine optimal similarity thresholds.

STRATEGY:
- Search Depth: {strategy.search_depth}
- Multimodal Priority: {strategy.multimodal_priority}
- Context Window: {strategy.context_window}
- Requires Comparison: {strategy.requires_comparison}

Your task: Set similarity thresholds (0.0 to 1.0) for:
1. TEXT_THRESHOLD - How similar text must be to the query
2. IMAGE_THRESHOLD - How similar images must be to the query

GUIDELINES:
- Shallow searches can be stricter (higher threshold = more precise)
- Deep searches should be lenient (lower threshold = broader recall)
- Visual-heavy needs lower image threshold to find more diagrams
- Text-only can ignore image threshold
- **CRITICAL: Default to LENIENT thresholds** - We have LLM relevance verification (_is_relevant_answer)
- **Philosophy: Threshold filters noise, LLM filters irrelevance** - Be generous in retrieval
- Educational content uses varied terminology, so lower thresholds catch semantic matches

**Recommended ranges (be lenient - LLM will verify relevance):**
- Text: **0.20-0.30** (0.20-0.25 for broad/standard, 0.28-0.30 for focused only)
- Images: **0.15-0.25** (diagrams have low similarity, be very generous)

OUTPUT FORMAT (provide exact decimal numbers):
TEXT_THRESHOLD: 0.XX
IMAGE_THRESHOLD: 0.XX
IMAGE_COUNT: XX"""

        try:
            response = self.llm.invoke(prompt).strip()
            logger.info(f"Threshold LLM Response: {response}")
            
            lines = {line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip() 
                    for line in response.split('\n') if ':' in line}
            
            # Parse with robust extraction
            import re
            text_str = lines.get('TEXT_THRESHOLD', '0.25')
            text_match = re.search(r'(\d+\.?\d*)', text_str)
            text_thresh = float(text_match.group(1)) if text_match else 0.25
            
            # Safety caps: Lenient to maximize recall (LLM verifies relevance anyway)
            text_thresh = min(text_thresh, 0.35)  # Cap at 0.35 max (lower than before)
            text_thresh = max(text_thresh, 0.20)  # Floor at 0.20 min
            
            image_str = lines.get('IMAGE_THRESHOLD', '0.20')
            image_match = re.search(r'(\d+\.?\d*)', image_str)
            image_thresh = float(image_match.group(1)) if image_match else 0.20
            
            # Safety caps for images: Very lenient for diagrams
            image_thresh = min(image_thresh, 0.30)  # Cap at 0.30 max (lower than before)
            image_thresh = max(image_thresh, 0.15)  # Floor at 0.15 min
            
            count_str = lines.get('IMAGE_COUNT', '10')
            count_match = re.search(r'(\d+)', count_str)
            image_count = int(count_match.group(1)) if count_match else 10
            
            return {
                'text': text_thresh,
                'image': image_thresh
            }
            
        except Exception as e:
            logger.warning(f"Threshold adjustment failed: {e}, using lenient defaults")
            # Lenient defaults: Threshold filters noise, LLM verifies relevance
            base_text = 0.25  # Lenient - let LLM decide final relevance
            base_image = 0.18  # Very lenient for diagrams
            
            # Adjust based on strategy  
            if strategy.search_depth == "deep":
                base_text -= 0.05  # Even more lenient for deep searches
                base_image -= 0.03
            elif strategy.search_depth == "shallow":
                base_text += 0.05  # Slightly stricter for shallow
            
            if strategy.multimodal_priority == "visual_heavy":
                base_image -= 0.05  # More lenient for visuals
            
            return {
                'text_threshold': base_text,
                'image_threshold': base_image,
                'k_text': strategy.context_window,
                'k_images': 15 if strategy.multimodal_priority == "visual_heavy" else 10
            }
