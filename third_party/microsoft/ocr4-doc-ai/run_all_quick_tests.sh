#!/bin/bash

echo "============================================================"
echo "OCR-4 Quick Regression Test Suite"
echo "============================================================"

echo ""
echo "1. Running Confidence Score Tests..."
python quick_confidence_test.py

CONFIDENCE_RESULT=$?

echo ""
echo "2. Running Sparse Table Tests..."
python quick_sparse_test.py

SPARSE_RESULT=$?

echo ""
echo "3. Running Multilingual Tests..."
python quick_multilingual_test.py

MULTI_RESULT=$?

echo ""
echo "============================================================"
echo "Test Summary:"
echo "============================================================"

if [ $CONFIDENCE_RESULT -eq 0 ]; then
    echo "✅ Confidence Score Tests: PASSED"
else
    echo "❌ Confidence Score Tests: FAILED"
fi

if [ $SPARSE_RESULT -eq 0 ]; then
    echo "✅ Sparse Table Tests: PASSED"
else
    echo "❌ Sparse Table Tests: FAILED"
fi

if [ $MULTI_RESULT -eq 0 ]; then
    echo "✅ Multilingual Tests: PASSED"
else
    echo "❌ Multilingual Tests: FAILED"
fi

echo "============================================================"

# Exit with error if any test failed
if [ $CONFIDENCE_RESULT -ne 0 ] || [ $SPARSE_RESULT -ne 0 ] || [ $MULTI_RESULT -ne 0 ]; then
    exit 1
fi
