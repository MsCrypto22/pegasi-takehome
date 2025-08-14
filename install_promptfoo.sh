#!/bin/bash

# Installation script for promptfoo CLI dependency
# This script installs the promptfoo CLI tool required for AI security testing

echo "🔧 Installing promptfoo CLI for AI security testing..."
echo "=================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first:"
    echo "   Visit: https://nodejs.org/"
    echo "   Or use: brew install node (on macOS)"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Node.js and npm are available"

# Install promptfoo CLI globally
echo "📦 Installing promptfoo CLI..."
npm install -g promptfoo

# Verify installation
if command -v promptfoo &> /dev/null; then
    echo "✅ promptfoo CLI installed successfully!"
    echo "📋 Version: $(promptfoo --version)"
    echo ""
    echo "🎉 Setup complete! You can now run the AI security tests."
    echo ""
    echo "Next steps:"
    echo "1. Set up your API keys in environment variables:"
    echo "   export OPENAI_API_KEY='your-key-here'"
    echo "   export ANTHROPIC_API_KEY='your-key-here'"
    echo "   export GOOGLE_API_KEY='your-key-here'"
    echo "2. Run the tests: python test_promptfoo_wrapper.py"
else
    echo "❌ Failed to install promptfoo CLI"
    exit 1
fi 