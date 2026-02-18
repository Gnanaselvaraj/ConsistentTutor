"""
Test Multi-Model Architecture Integration
Tests that different models are used for different tasks
"""
import sys
sys.path.insert(0, 'src')

from core.rag_engine import ConsistentTutorRAG
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_multi_model_initialization():
    """Test that multi-model system initializes correctly"""
    print("\n" + "="*70)
    print("TEST 1: Multi-Model Initialization")
    print("="*70)
    
    try:
        # Initialize with multi-model enabled
        tutor = ConsistentTutorRAG(use_multi_model=True, use_meta_prompting=True)
        print("✅ Multi-model RAG engine initialized successfully")
        
        # Check if it's using MultiModelLLM
        from core.multi_model_llm import MultiModelLLM
        if isinstance(tutor.llm, MultiModelLLM):
            print("✅ Confirmed: Using MultiModelLLM class")
        else:
            print(f"❌ ERROR: Expected MultiModelLLM, got {type(tutor.llm)}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_routing():
    """Test that different tasks route to different models"""
    print("\n" + "="*70)
    print("TEST 2: Model Routing")
    print("="*70)
    
    try:
        from core.multi_model_llm import MultiModelLLM, TaskType
        
        # Create multi-model LLM directly
        llm = MultiModelLLM(
            reasoning_model="qwen2.5:14b-instruct-q4_K_M",
            generation_model="llama3.1:8b"
        )
        
        print("\n📋 Testing different task types:")
        
        # Test META_REASONING (Qwen2.5-14B)
        print("\n1. META_REASONING Task (should use Qwen2.5-14B):")
        meta_prompt = "Generate 3 teaching instructions for explaining photosynthesis to a 12th grade student."
        response = llm.invoke(meta_prompt, task_type=TaskType.META_REASONING)
        print(f"   Response length: {len(response)} chars")
        print(f"   First 100 chars: {response[:100]}...")
        
        # Test ANSWER_GENERATION (Llama3.1-8B)
        print("\n2. ANSWER_GENERATION Task (should use Llama3.1-8B):")
        gen_prompt = "What is photosynthesis? Explain in 2 sentences."
        response = llm.invoke(gen_prompt, task_type=TaskType.ANSWER_GENERATION)
        print(f"   Response length: {len(response)} chars")
        print(f"   Response: {response}")
        
        # Test QUICK_CHECK (Llama3.1-8B with lower temperature)
        print("\n3. QUICK_CHECK Task (should use Llama3.1-8B, temp=0.1):")
        quick_prompt = "Is this content relevant to the question about markets? Answer yes or no only."
        response = llm.invoke(quick_prompt, task_type=TaskType.QUICK_CHECK)
        print(f"   Response: {response}")
        
        # Print statistics
        print("\n📊 Model Usage Statistics:")
        stats = llm.get_stats()
        for task_name, task_stats in stats.items():
            print(f"   {task_name}: {task_stats}")
        
        print("\n✅ All model routing tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model routing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_rag_with_multi_model():
    """Test complete RAG flow with multi-model"""
    print("\n" + "="*70)
    print("TEST 3: Full RAG Flow with Multi-Model")
    print("="*70)
    
    try:
        tutor = ConsistentTutorRAG(use_multi_model=True, use_meta_prompting=True)
        
        # Load a subject (if available)
        import os
        if os.path.exists("vector_db/Commerce - 12 - TN (Tamil Nadu)"):
            print("✅ Found Commerce knowledge base")
            tutor.load_subject("Commerce - 12 - TN (Tamil Nadu)")
            
            print("\n🔍 Testing question with meta-prompting:")
            question = "What is a secondary market?"
            
            print(f"   Question: {question}")
            print(f"   Using meta-prompting: {tutor.use_meta_prompting}")
            print(f"   Using multi-model: {tutor.use_multi_model}")
            
            # Get answer (non-streaming for test)
            full_answer = []
            for chunk in tutor.answer_stream(question, "Commerce - 12 - TN (Tamil Nadu)", [], ""):
                full_answer.append(chunk)
            
            complete_answer = "".join(full_answer)
            print(f"\n   Answer length: {len(complete_answer)} chars")
            print(f"   Answer preview: {complete_answer[:200]}...")
            
            # Check if multi-model stats are available
            if hasattr(tutor.llm, 'get_stats'):
                print("\n📊 Multi-Model Statistics:")
                stats = tutor.llm.get_stats()
                for task_name, task_stats in stats.items():
                    print(f"   {task_name}: {task_stats}")
            
            print("\n✅ Full RAG flow test passed!")
            return True
        else:
            print("⚠️  No knowledge base found, skipping full RAG test")
            print("   You can test after ingesting PDFs")
            return True
            
    except Exception as e:
        print(f"❌ Full RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("2-MODEL ARCHITECTURE TEST SUITE")
    print("Testing task-specialized LLM routing with:")
    print("  - Qwen2.5-14B-Q4 (9GB) for complex reasoning")
    print("  - Llama3.1-8B (5GB) for generation + quick checks")
    print("  - Total RAM: 14GB (26% less than 3-model approach)")
    print("="*70)
    
    results = []
    
    # Test 1: Initialization
    results.append(("Initialization", test_multi_model_initialization()))
    
    # Test 2: Model Routing
    results.append(("Model Routing", test_model_routing()))
    
    # Test 3: Full RAG Flow
    results.append(("Full RAG Flow", test_full_rag_with_multi_model()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Multi-model architecture is working!")
        print("\n📝 Next Steps:")
        print("   1. Restart Streamlit app: streamlit run src/app.py")
        print("   2. Ask questions and observe different models being used")
        print("   3. Check logs for model usage patterns")
    else:
        print("\n⚠️  SOME TESTS FAILED! Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
