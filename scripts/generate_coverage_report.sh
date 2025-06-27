#!/bin/bash

# Script to generate coverage report locally
# Usage: ./scripts/generate_coverage_report.sh

# Don't exit on errors - we want to generate reports even if coverage fails
set +e

echo "🧪 Running tests with coverage..."

# Run tests with coverage (continue even if coverage requirement fails)
python -m pytest tests/ --cov=src/feafea --cov-report=xml --cov-report=html --cov-report=term-missing || true

echo ""
echo "📊 Generating coverage report..."

# Generate coverage report
echo "## 📊 Test Coverage Report" > coverage_report.md
echo "" >> coverage_report.md

# Get coverage percentage (extract the percentage value from the TOTAL line)
COVERAGE=$(coverage report | grep TOTAL | grep -o '[0-9]\+\.[0-9]\+%' | sed 's/%//')

# Add summary with emoji based on coverage
if python3 -c "import sys; sys.exit(0 if float('$COVERAGE') >= 100 else 1)"; then
    echo "### ✅ Coverage Status: **PASSED** 🎉" >> coverage_report.md
    echo "**Current Coverage:** ${COVERAGE}% (Required: 100%)" >> coverage_report.md
else
    echo "### ❌ Coverage Status: **FAILED** 🚨" >> coverage_report.md
    echo "**Current Coverage:** ${COVERAGE}% (Required: 100%)" >> coverage_report.md
fi

echo "" >> coverage_report.md
echo "### 📋 Detailed Coverage Report" >> coverage_report.md
echo "" >> coverage_report.md
echo '```' >> coverage_report.md
coverage report --show-missing >> coverage_report.md
echo '```' >> coverage_report.md

# Add missing coverage section if needed
if ! python3 -c "import sys; sys.exit(0 if float('$COVERAGE') >= 100 else 1)"; then
    echo "" >> coverage_report.md
    echo "### 🔍 Lines Missing Coverage" >> coverage_report.md
    echo "" >> coverage_report.md
    echo "The following lines need to be covered by tests:" >> coverage_report.md
    echo '```' >> coverage_report.md
    coverage report --show-missing | grep -E "^src/" | grep -v "100%" >> coverage_report.md 2>/dev/null || echo "No specific missing lines found" >> coverage_report.md
    echo '```' >> coverage_report.md
    echo "" >> coverage_report.md
    echo "💡 **Tip:** Open \`htmlcov/index.html\` in your browser to see detailed coverage highlighting." >> coverage_report.md
fi

echo "" >> coverage_report.md
echo "---" >> coverage_report.md
echo "*Generated locally on $(date)*" >> coverage_report.md

echo ""
echo "✅ Coverage report generated:"
echo "   📄 Markdown: coverage_report.md"
echo "   🌐 HTML: htmlcov/index.html"
echo ""

# Display summary
echo "📊 Coverage Summary:"
coverage report

echo ""
if python3 -c "import sys; sys.exit(0 if float('$COVERAGE') >= 100 else 1)"; then
    echo "✅ Coverage requirement met: ${COVERAGE}% 🎉"
    exit 0
else
    echo "❌ Coverage requirement not met: ${COVERAGE}% (required: 100%)"
    echo ""
    echo "💡 To see detailed coverage:"
    echo "   - Open htmlcov/index.html in your browser"
    echo "   - Or run: coverage report --show-missing"
    exit 1
fi
