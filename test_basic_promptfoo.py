#!/usr/bin/env python3
"""
Basic test script to verify promptfoo CLI installation and basic functionality.
This test doesn't require API keys and just checks that the CLI is available.
"""

import subprocess
import sys
from pathlib import Path

def test_promptfoo_installation():
    """Test that promptfoo CLI is installed and accessible."""
    print("🔧 Testing promptfoo CLI installation...")
    
    try:
        # Test basic command
        result = subprocess.run(
            ["promptfoo", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ promptfoo CLI is installed: {version}")
            return True
        else:
            print(f"❌ promptfoo CLI command failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ promptfoo CLI not found. Please install it with: npm install -g promptfoo")
        return False
    except subprocess.TimeoutExpired:
        print("❌ promptfoo CLI command timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing promptfoo CLI: {e}")
        return False

def test_promptfoo_help():
    """Test that promptfoo help command works."""
    print("\n📖 Testing promptfoo help command...")
    
    try:
        result = subprocess.run(
            ["promptfoo", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ promptfoo help command works")
            return True
        else:
            print(f"❌ promptfoo help command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing promptfoo help: {e}")
        return False

def test_config_file_exists():
    """Test that the configuration file exists."""
    print("\n📁 Testing configuration file...")
    
    config_path = Path("configs/promptfooconfig.yaml")
    if config_path.exists():
        print(f"✅ Configuration file exists: {config_path}")
        return True
    else:
        print(f"❌ Configuration file not found: {config_path}")
        return False

def main():
    """Run all basic tests."""
    print("🚀 Basic Promptfoo CLI Test Suite")
    print("=" * 50)
    
    tests = [
        test_promptfoo_installation,
        test_promptfoo_help,
        test_config_file_exists
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        print("\n✅ promptfoo CLI is properly installed and ready to use")
        print("\nNext steps:")
        print("1. Set up API keys in environment variables:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("   export GOOGLE_API_KEY='your-key-here'")
        print("2. Run the full test suite: python test_promptfoo_wrapper.py")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTo install promptfoo CLI:")
        print("npm install -g promptfoo")
        print("Or run: ./install_promptfoo.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 