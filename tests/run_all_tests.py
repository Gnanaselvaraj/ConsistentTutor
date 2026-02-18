"""
run_all_tests.py: Run all test suites for ConsistentTutor
Phase 1 & 2: Multi-Model Architecture Tests
"""
import sys
import os

# Add parent and src directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def run_all_tests():
    """Run all available test suites"""
    
    print("=" * 80)
    print("ConsistentTutor Phase 1 & 2 Test Suite")
    print("Multi-Model Architecture + Lenient Thresholds + Meta-Prompting")
    print("=" * 80)
    print()
    
    # Import and run test_multi_model
    print("Running Multi-Model Architecture Tests...")
    print("=" * 80)
    try:
        import test_multi_model
        test_multi_model.test_multi_model_initialization()
        print("\n" + "="*70)
        test_multi_model.test_model_routing()
        print("\n" + "="*70)
        test_multi_model.test_full_rag_with_multi_model()
        print("\n✅ All Multi-Model Tests PASSED")
        return True
    except Exception as e:
        print(f"\n❌ Multi-Model Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
