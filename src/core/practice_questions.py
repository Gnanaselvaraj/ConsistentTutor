"""
practice_questions.py: Generate practice questions from content
"""
from typing import List, Dict, Any
import json

class PracticeQuestionGenerator:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_from_content(self, content: str, topic: str = "", num_questions: int = 5, 
                             difficulty: str = "medium") -> List[Dict[str, Any]]:
        """
        Generate practice questions from textbook content using intelligent question count.
        
        Args:
            content: Text content from which to generate questions
            topic: Optional topic name
            num_questions: Target number (system may generate more or fewer based on content)
            difficulty: easy, medium, or hard
        
        Returns:
            List of dicts with: {question, options, correct_answer, explanation, type}
        """
        
        # Define difficulty-specific instructions
        difficulty_guidelines = {
            'easy': 'Focus on basic recall, definitions, and simple identification. Questions should be straightforward.',
            'medium': 'Test understanding and application of concepts. Questions should require some analysis.',
            'hard': 'Test deep understanding, analysis, synthesis, and critical thinking. Questions should be challenging.'
        }
        
        difficulty_guide = difficulty_guidelines.get(difficulty, difficulty_guidelines['medium'])
        
        prompt = f"""You are an educational assessment expert. Analyze this content and generate practice questions.

Topic: {topic if topic else 'General'}
Difficulty Level: {difficulty.upper()}
Difficulty Guidelines: {difficulty_guide}

Content:
{content[:3000]}

CRITICAL TASK: 
1. Read the content carefully and identify ALL distinct concepts, definitions, processes, or facts that can be tested
2. Generate ONE unique question for EACH distinct testable concept
3. If the content is narrow (only 1-2 concepts), generate 2-4 questions
4. If the content is broad (many concepts), generate 5-10 questions
5. NEVER create duplicate or repetitive questions - each question MUST test a DIFFERENT aspect

Target: Around {num_questions} questions, but ADJUST based on actual content richness

IMPORTANT: Do NOT use HTML tags, special symbols, or formatting codes in your response. Use plain text only.

Generate questions in this JSON format (ONLY JSON, no extra text):
[
  {{
    "question": "Clear, specific question testing ONE concept",
    "type": "multiple_choice",
    "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
    "correct_answer": "A) First option",
    "explanation": "2-3 sentences explaining why this is correct"
  }}
]

QUALITY RULES:
1. Each question MUST test a DIFFERENT concept or aspect
2. NO repetitive questions (e.g., multiple questions asking "what opportunities exist")
3. Diverse question types: definitions, comparisons, applications, examples
4. Adjust question count naturally - don't force duplicates to hit a number
5. Every explanation must be 2-3 sentences minimum
6. Options must be 4 choices (A, B, C, D) for multiple choice

JSON array only:"""

        try:
            response = self.llm.invoke(prompt).strip()
            
            # Clean up response - remove markdown and extra text
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Try to find JSON array in response
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                response = json_match.group(0)
            
            questions = json.loads(response)
            
            if not isinstance(questions, list):
                raise ValueError("Response is not a list")
            
            # Validate structure and remove duplicates
            valid_questions = []
            seen_questions = set()
            
            for q in questions:
                if isinstance(q, dict) and 'question' in q and 'correct_answer' in q:
                    question_text = q.get('question', '').strip().lower()
                    
                    # Skip if this question is too similar to one we've already added
                    is_duplicate = False
                    for seen in seen_questions:
                        # Simple similarity check: if 70%+ of words match, it's a duplicate
                        q_words = set(question_text.split())
                        seen_words = set(seen.split())
                        if len(q_words) > 0 and len(seen_words) > 0:
                            overlap = len(q_words & seen_words)
                            similarity = overlap / max(len(q_words), len(seen_words))
                            if similarity > 0.7:
                                is_duplicate = True
                                break
                    
                    if is_duplicate:
                        continue
                    
                    seen_questions.add(question_text)
                    
                    explanation = q.get('explanation', '').strip()
                    
                    # If explanation is empty or too short, generate one
                    if not explanation or len(explanation) < 20:
                        explanation = self._generate_explanation(
                            q.get('question', ''),
                            q.get('correct_answer', ''),
                            content[:2000]
                        )
                    
                    valid_questions.append({
                        'question': q.get('question', ''),
                        'type': q.get('type', 'multiple_choice'),
                        'options': q.get('options', []),
                        'correct_answer': q.get('correct_answer', ''),
                        'explanation': explanation,
                        'difficulty': difficulty,
                        'topic': topic
                    })
            
            # Don't force additional questions if we have quality unique ones
            return valid_questions
        
        except (json.JSONDecodeError, ValueError, Exception) as e:
            # Fallback: generate simple questions
            return self._generate_simple_questions(content, topic, num_questions, difficulty)
    
    def _generate_explanation(self, question: str, answer: str, content: str) -> str:
        """Generate explanation for a question-answer pair using LLM"""
        prompt = f"""Provide a clear, educational explanation for why this answer is correct.

Question: {question}
Correct Answer: {answer}

Reference Content:
{content}

Write 2-3 sentences explaining why this is correct:"""

        try:
            explanation = self.llm.invoke(prompt).strip()
            return explanation if explanation else "This answer is correct based on the content."
        except:
            return "This answer is correct based on the content."
    
    def _generate_simple_questions(self, content: str, topic: str, num_questions: int, difficulty: str = 'medium') -> List[Dict[str, Any]]:
        """Fallback method to generate questions one by one"""
        questions = []
        
        # Generate questions individually
        for i in range(num_questions):
            difficulty_prompt = {
                'easy': 'Generate a simple recall question',
                'medium': 'Generate a moderate understanding question',
                'hard': 'Generate a challenging analytical question'
            }.get(difficulty, 'Generate a question')
            
            prompt = f"""{difficulty_prompt} from this content about {topic}:

{content[:2000]}

Provide:
1. Question text
2. Four multiple choice options (A, B, C, D)
3. Correct answer
4. 2 sentence explanation

Format as:
QUESTION: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [correct option]
EXPLANATION: [explanation]"""

            try:
                response = self.llm.invoke(prompt).strip()
                
                # Parse response
                question_text = ''
                options = []
                correct_answer = ''
                explanation = ''
                
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('QUESTION:'):
                        question_text = line.replace('QUESTION:', '').strip()
                    elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                        options.append(line)
                    elif line.startswith('ANSWER:'):
                        correct_answer = line.replace('ANSWER:', '').strip()
                    elif line.startswith('EXPLANATION:'):
                        explanation = line.replace('EXPLANATION:', '').strip()
                
                if question_text and correct_answer:
                    questions.append({
                        'question': question_text,
                        'type': 'multiple_choice',
                        'options': options if options else [],
                        'correct_answer': correct_answer,
                        'explanation': explanation if explanation else 'Based on the content.',
                        'difficulty': difficulty,
                        'topic': topic
                    })
            except:
                continue
        
        return questions
    
    def generate_from_topic(self, topic: str, subject: str, vector_store, 
                           num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Generate questions for a specific topic by retrieving relevant content.
        NOW RETURNS: Dict with 'questions' and 'images' lists for multimodal support
        
        Args:
            topic: Topic name (e.g., "recursion", "binary trees")
            subject: Subject name
            vector_store: VectorStore instance
            num_questions: Number of questions
            difficulty: Difficulty level
        
        Returns:
            Dict with 'questions' list and 'images' list
        """
        from .embeddings import embed_texts_batched
        
        # Retrieve content related to topic
        q_vec = embed_texts_batched([f"Explain {topic} in {subject}"])
        
        # Get more chunks for question generation
        image_results = []
        if hasattr(vector_store, 'search_text'):
            # Multimodal store - also get images!
            results = vector_store.search_text(q_vec, k=50, threshold=0.5)
            
            # Also search for relevant diagrams using CLIP
            try:
                from .image_embeddings import embed_text_for_image_search
                q_vec_img = embed_text_for_image_search([f"diagram illustration of {topic}"])
                image_results = vector_store.search_images(q_vec_img, k=3, threshold=0.5)
            except Exception as e:
                # CLIP not available or no images
                image_results = []
        else:
            # Regular store - text only
            results = vector_store.search(q_vec, k=50, threshold=0.5)
        
        # KB Grounding Check: Verify sufficient relevant content exists
        if not results or len(results) < 3:
            return []  # Not enough content found
        
        # Check if content is actually relevant to the topic
        sample_content = " ".join([r[0] for r in results[:3]]).lower()
        topic_lower = topic.lower()
        
        # Simple relevance check: topic keywords should appear in content
        topic_words = topic_lower.split()
        matches = sum(1 for word in topic_words if len(word) > 3 and word in sample_content)
        
        if matches == 0 and len(results) < 5:
            return []  # Topic not found in content
        
        # Combine content from top results
        content = "\\n\\n".join([r[0] for r in results[:10]])
        
        questions = self.generate_from_content(content, topic=topic, 
                                         num_questions=num_questions, 
                                         difficulty=difficulty)
        
        return {
            'questions': questions,
            'images': image_results
        }
    
    @staticmethod
    def format_for_display(data: Any) -> str:
        """Format questions as clean, user-friendly HTML with optional diagrams"""
        # Handle both old format (list) and new format (dict with questions + images)
        if isinstance(data, dict):
            questions = data.get('questions', [])
            images = data.get('images', [])
        else:
            questions = data if isinstance(data, list) else []
            images = []
        
        if not questions:
            return "<p style='color:#999;'>❌ No questions generated. Try a different topic.</p>"
        
        import html
        
        html_output = "<div style='max-width:900px;margin:20px auto;'>"
        
        # Display reference diagrams if available
        if images:
            import base64
            from io import BytesIO
            import os
            
            html_output += """<div style='margin-bottom:24px;padding:16px;background:#f0f7ff;border-left:4px solid #1890ff;border-radius:8px;'>
                <h4 style='margin-top:0;color:#1890ff;'>🖼️ Reference Diagrams</h4>
                <p style='color:#666;font-size:0.9em;margin-bottom:12px;'>These diagrams from your textbook may help you answer the questions below:</p>
            """
            
            for img_data, meta, score in images[:3]:  # Show up to 3 images
                try:
                    pil_image = img_data.get('image') if isinstance(img_data, dict) else None
                    
                    if pil_image:
                        buffered = BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        source_file = os.path.basename(meta.get('source', 'Unknown'))
                        page = meta.get('page', '?')
                        confidence = int(score * 100)
                        
                        html_output += f"""<div style='margin:12px 0;padding:12px;background:white;border:1px solid #d9d9d9;border-radius:8px;'>
                            <img src='data:image/png;base64,{img_str}' style='max-width:100%;height:auto;border-radius:6px;box-shadow:0 2px 4px rgba(0,0,0,0.1);'/>
                            <div style='margin-top:8px;font-size:0.85em;color:#666;'>
                                📚 <b>{source_file}</b> - Page {page} | Relevance: {confidence}%
                            </div>
                        </div>"""
                except:
                    pass
            
            html_output += "</div>"
        
        for i, q in enumerate(questions, 1):
            qtype = q.get('type', 'short_answer')
            # Escape LLM-generated content to prevent HTML injection
            question_text = html.escape(q.get('question', ''))
            options = q.get('options', [])
            correct_answer = html.escape(q.get('correct_answer', 'Not provided'))
            explanation = html.escape(q.get('explanation', 'No explanation provided.'))
            difficulty = q.get('difficulty', 'medium')
            
            # Difficulty badge color
            badge_colors = {
                'easy': '#4CAF50',
                'medium': '#FF9800',
                'hard': '#F44336'
            }
            badge_color = badge_colors.get(difficulty, '#999')
            
            html_output += f"""
<div style='background:white;border:1px solid #e0e0e0;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,0.08);'>
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;'>
        <h3 style='margin:0;color:#1976d2;font-size:1.1em;'>Question {i}</h3>
        <span style='background:{badge_color};color:white;padding:4px 12px;border-radius:12px;font-size:0.85em;font-weight:500;'>{difficulty.upper()}</span>
    </div>
    <div style='font-size:1.1em;color:#333;line-height:1.6;margin:16px 0;'>{question_text}</div>
"""
            
            # Render options for multiple choice
            if qtype == 'multiple_choice' and options:
                html_output += "<div style='margin:20px 0;'>"
                for opt in options:
                    # Escape option text too
                    opt_text = html.escape(opt if isinstance(opt, str) else str(opt))
                    html_output += f"""
        <div style='background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;padding:12px 16px;margin:8px 0;'>
            <span style='color:#495057;font-size:1em;'>{opt_text}</span>
        </div>"""
                html_output += "</div>"
            
            # Answer and explanation toggle
            html_output += f"""
    <details style='margin-top:16px;cursor:pointer;'>
        <summary style='color:#4CAF50;font-weight:600;font-size:1em;padding:8px 0;user-select:none;'>
            👁️ Show Answer & Explanation
        </summary>
        <div style='margin-top:16px;padding:16px;background:#e8f5e9;border-left:4px solid #4CAF50;border-radius:4px;'>
            <p style='margin:0 0 12px 0;'><strong style='color:#2e7d32;'>✓ Answer:</strong> <span style='color:#1b5e20;font-weight:500;'>{correct_answer}</span></p>
            <p style='margin:0;'><strong style='color:#2e7d32;'>📖 Explanation:</strong> <span style='color:#333;line-height:1.6;'>{explanation}</span></p>
        </div>
    </details>
</div>
"""
        
        html_output += "</div>"
        return html_output
