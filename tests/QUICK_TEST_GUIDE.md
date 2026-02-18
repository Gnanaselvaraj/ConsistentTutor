# Quick Test Reference

## Run All Tests (2 minutes)
```powershell
C:/Users/selva/Documents/ConsistentTutor/venv/Scripts/python.exe tests/run_all_tests.py
```

## Run Individual Suites (< 1 second each, except multimodal)

| Test Suite | Command | Time | Tests |
|------------|---------|------|-------|
| Requirements | `python tests/test_tutor_requirements.py` | ~38s | 6 |
| Citations | `python tests/test_source_citations.py` | <1s | 5 |
| Multimodal | `python tests/test_multimodal.py` | ~68s | 9 |
| Questions | `python tests/test_practice_questions.py` | <1s | 7 |
| KB Mgmt | `python tests/test_kb_management.py` | <1s | 8 |
| Integration | `python tests/test_integration.py` | ~5s | 8 |

## Total: 43 Tests ✅

## What's Tested

✅ RAG retrieval with real embeddings  
✅ Source citations with page numbers  
✅ Multimodal CLIP (text + images)  
✅ Practice question generation  
✅ KB management (CRUD operations)  
✅ Hallucination mitigation (refusal to guess)  
✅ Memory & context retention  
✅ Edge cases & error handling  

## Quick Verification

```bash
# Test embeddings working
python -c "from src.core.embeddings import embed_texts_batched; print('✓ Embeddings OK')"

# Test KB can be loaded
python -c "from src.core.rag_engine import ConsistentTutorRAG; t = ConsistentTutorRAG(); t.load_subject('Commerce'); print('✓ KB Load OK')"

# Test LLM connection
python -c "from src.core.llm import OllamaLLM; llm = OllamaLLM(); print('✓ LLM OK')"
```

## Expected Output
```
================================================================================
ConsistentTutor Test Suite
================================================================================

test_tutor_requirements      ✓ PASS     Tests:   6
test_source_citations        ✓ PASS     Tests:   5
test_multimodal              ✓ PASS     Tests:   9
test_practice_questions      ✓ PASS     Tests:   7
test_kb_management           ✓ PASS     Tests:   8
test_integration             ✓ PASS     Tests:   8

TOTAL                                   Tests:  43  Failures: 0

Success Rate: 100.0%
🎉 ALL TESTS PASSED! 🎉
```
