"""
Multi-Model LLM Router - Use different models for different tasks
Maintains pure intelligence philosophy while optimizing performance
"""
from typing import Literal
from enum import Enum

class TaskType(Enum):
    """Different types of LLM tasks with different requirements"""
    META_REASONING = "meta_reasoning"      # Complex: meta-prompting, strategy
    ANSWER_GENERATION = "answer_generation" # Quality: educational answers
    QUICK_CHECK = "quick_check"            # Fast: relevance, validation
    ANALYSIS = "analysis"                  # Reasoning: question analysis

class MultiModelLLM:
    """
    Intelligent LLM router that selects the best model for each task.
    Maintains pure intelligence - no hardcoding, just smarter model selection.
    """
    
    def __init__(self, 
                 reasoning_model: str = "qwen2.5:14b",
                 generation_model: str = "llama3.1:8b", 
                 fast_model: str = "llama3:latest",
                 temperature: float = 0.2):
        """
        Initialize multi-model system.
        
        Args:
            reasoning_model: Best model for complex reasoning (meta-prompting, strategy)
            generation_model: Best model for content generation (answers)
            fast_model: Fast model for simple checks (relevance, validation)
            temperature: Default temperature for generation
        """
        from .llm import OllamaLLM
        
        self.models = {
            TaskType.META_REASONING: OllamaLLM(model=reasoning_model, temperature=temperature),
            TaskType.ANSWER_GENERATION: OllamaLLM(model=generation_model, temperature=temperature),
            TaskType.QUICK_CHECK: OllamaLLM(model=fast_model, temperature=0.1),  # Lower temp for consistency
            TaskType.ANALYSIS: OllamaLLM(model=reasoning_model, temperature=temperature)
        }
        
        self.task_stats = {task: {"calls": 0, "total_time": 0.0} for task in TaskType}
        
    def invoke(self, prompt: str, task_type: TaskType = TaskType.ANSWER_GENERATION) -> str:
        """
        Invoke appropriate model for the task.
        
        Args:
            prompt: Input prompt
            task_type: Type of task to perform
            
        Returns:
            Model response
        """
        import time
        start = time.time()
        
        model = self.models[task_type]
        response = model.invoke(prompt)
        
        # Track stats
        elapsed = time.time() - start
        self.task_stats[task_type]["calls"] += 1
        self.task_stats[task_type]["total_time"] += elapsed
        
        return response
    
    def stream(self, prompt: str, task_type: TaskType = TaskType.ANSWER_GENERATION):
        """
        Stream response from appropriate model.
        
        Args:
            prompt: Input prompt
            task_type: Type of task to perform
            
        Yields:
            Response chunks
        """
        model = self.models[task_type]
        yield from model.stream(prompt)
    
    def get_stats(self) -> dict:
        """Get usage statistics for each task type"""
        stats = {}
        for task, data in self.task_stats.items():
            if data["calls"] > 0:
                avg_time = data["total_time"] / data["calls"]
                stats[task.value] = {
                    "calls": data["calls"],
                    "total_time": f"{data['total_time']:.2f}s",
                    "avg_time": f"{avg_time:.2f}s"
                }
        return stats


# Example usage in RAG engine:
"""
# In rag_engine.py __init__:
from .multi_model_llm import MultiModelLLM, TaskType

self.llm = MultiModelLLM(
    reasoning_model="qwen2.5:14b",      # Best for meta-prompting
    generation_model="llama3.1:8b",    # Good for answers
    fast_model="llama3:latest"          # Fast for checks
)

# In _generate_dynamic_prompt:
instructions = self.llm.invoke(meta_prompt, task_type=TaskType.META_REASONING)

# In _is_relevant_answer:
response = self.llm.invoke(prompt, task_type=TaskType.QUICK_CHECK)

# In analyze_question:
response = self.llm.invoke(prompt, task_type=TaskType.ANALYSIS)

# In answer generation:
for chunk in self.llm.stream(prompt, task_type=TaskType.ANSWER_GENERATION):
    yield chunk
"""
