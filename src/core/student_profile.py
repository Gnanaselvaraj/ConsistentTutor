"""
student_profile.py: Track student learning progress and preferences
"""
import json
import os
from typing import Dict, List, Set
from datetime import datetime
from collections import defaultdict

class StudentProfile:
    """
    Tracks individual student's learning journey:
    - Topics studied
    - Knowledge gaps
    - Conversation style  
    - Learning pace
    - Question patterns
    """
    
    def __init__(self, profile_dir: str = "student_profiles"):
        self.profile_dir = profile_dir
        os.makedirs(profile_dir, exist_ok=True)
        self.profile_file = os.path.join(profile_dir, "default_student.json")
        
        # Profile data
        self.topics_studied: Dict[str, int] = {}  # topic -> count
        self.questions_asked: List[Dict] = []
        self.weak_areas: Set[str] = set()
        self.strong_areas: Set[str] = set()
        self.preferred_explanation_style: str = "detailed"  # detailed | concise | visual
        self.total_sessions: int = 0
        self.last_session: str = ""
        
        self._load_profile()
    
    def _load_profile(self):
        """Load existing profile"""
        if os.path.exists(self.profile_file):
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                    self.topics_studied = data.get('topics_studied', {})
                    self.questions_asked = data.get('questions_asked', [])
                    self.weak_areas = set(data.get('weak_areas', []))
                    self.strong_areas = set(data.get('strong_areas', []))
                    self.preferred_explanation_style = data.get('preferred_explanation_style', 'detailed')
                    self.total_sessions = data.get('total_sessions', 0)
                    self.last_session = data.get('last_session', '')
            except:
                pass
    
    def save_profile(self):
        """Save profile to disk"""
        try:
            data = {
                'topics_studied': self.topics_studied,
                'questions_asked': self.questions_asked[-100:],  # Keep last 100
                'weak_areas': list(self.weak_areas),
                'strong_areas': list(self.strong_areas),
                'preferred_explanation_style': self.preferred_explanation_style,
                'total_sessions': self.total_sessions,
                'last_session': self.last_session
            }
            with open(self.profile_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save profile: {e}")
    
    def log_question(self, question: str, topic: str, subject: str, 
                    confidence: float, answered: bool):
        """Log a question asked by the student"""
        self.questions_asked.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'topic': topic,
            'subject': subject,
            'confidence': confidence,
            'answered': answered
        })
        
        # Update topics studied
        topic_key = f"{subject}::{topic}"
        self.topics_studied[topic_key] = self.topics_studied.get(topic_key, 0) + 1
        
        # Identify weak areas (low confidence or repeated questions)
        if confidence < 0.6 or self.topics_studied[topic_key] > 3:
            self.weak_areas.add(topic_key)
        elif confidence > 0.8 and self.topics_studied[topic_key] == 1:
            self.strong_areas.add(topic_key)
        
        # Auto-save every 5 questions
        if len(self.questions_asked) % 5 == 0:
            self.save_profile()
    
    def start_session(self):
        """Mark start of a learning session"""
        self.total_sessions += 1
        self.last_session = datetime.now().isoformat()
        self.save_profile()
    
    def get_study_recommendations(self) -> List[str]:
        """Get recommended topics to review"""
        recommendations = []
        
        # Suggest weak areas first
        if self.weak_areas:
            recommendations.append(f"📚 Review these topics: {', '.join(list(self.weak_areas)[:3])}")
        
        # Suggest related topics
        if self.topics_studied:
            most_studied = sorted(self.topics_studied.items(), key=lambda x: x[1], reverse=True)[:3]
            recommendations.append(f"🔄 Continue learning: {', '.join([t[0] for t in most_studied])}")
        
        return recommendations
    
    def get_stats(self) -> Dict:
        """Get profile statistics"""
        return {
            'total_sessions': self.total_sessions,
            'topics_studied': len(self.topics_studied),
            'questions_asked': len(self.questions_asked),
            'weak_areas': len(self.weak_areas),
            'strong_areas': len(self.strong_areas),
            'last_session': self.last_session
        }
    
    def adapt_response_level(self, base_level: str = "intermediate") -> str:
        """Adapt explanation complexity based on student's history"""
        # If student asks many followup questions, they prefer detailed explanations
        recent_questions = self.questions_asked[-10:]
        if len(recent_questions) > 5:
            followup_count = sum(1 for q in recent_questions 
                               if len(q['question'].split()) < 5)
            if followup_count > 3:
                return "detailed"
        
        return self.preferred_explanation_style
