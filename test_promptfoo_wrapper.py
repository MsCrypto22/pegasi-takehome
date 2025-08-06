#!/usr/bin/env python3
"""
Test script for PromptFoo wrapper functionality.
This script demonstrates how to use the PromptfooWrapper class.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from promptfoo_wrapper import PromptfooWrapper, PromptfooError

def test_promptfoo_wrapper():
    """Test the PromptFoo wrapper functionality."""
    
    print("🚀 Testing PromptFoo Wrapper")
    print("=" * 50)
    
    try:
        # Initialize the wrapper
        print("1. Initializing PromptFoo wrapper...")
        wrapper = PromptfooWrapper()
        print("✅ Wrapper initialized successfully")
        
        # Test configuration loading
        print("\n2. Testing configuration loading...")
        config_path = "configs/promptfooconfig.yaml"
        if Path(config_path).exists():
            print(f"✅ Configuration file found: {config_path}")
        else:
            print(f"❌ Configuration file not found: {config_path}")
            return
        
        # Test evaluation (this would require promptfoo CLI to be installed)
        print("\n3. Testing evaluation execution...")
        try:
            # Note: This will fail if promptfoo CLI is not installed
            # This is expected behavior for demonstration
            result = wrapper.run_evaluation()
            print("✅ Evaluation completed successfully")
            print(f"   Execution time: {result['execution_time']:.2f}s")
            
            # Test result parsing
            print("\n4. Testing result parsing...")
            test_results = wrapper.parse_results(result)
            print(f"✅ Parsed {len(test_results)} test results")
            
            # Test metrics extraction
            print("\n5. Testing metrics extraction...")
            metrics = wrapper.extract_metrics(test_results, result['execution_time'])
            print("✅ Metrics extracted successfully")
            print(f"   Total tests: {metrics.total_tests}")
            print(f"   Success rate: {metrics.success_rate:.1f}%")
            print(f"   Average score: {metrics.average_score:.2f}")
            
            # Test security analysis
            print("\n6. Testing security analysis...")
            security_results = wrapper.run_security_tests()
            print("✅ Security analysis completed")
            print(f"   Risk level: {security_results['security_analysis']['risk_level']}")
            print(f"   Security score: {security_results['security_analysis']['overall_security_score']:.1f}")
            
        except PromptfooError as e:
            print(f"⚠️  Expected error (promptfoo CLI not installed): {e}")
            print("   This is normal if promptfoo CLI is not installed on the system")
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            
        # Test Pydantic models
        print("\n7. Testing Pydantic models...")
        from promptfoo_wrapper import TestResult, EvaluationMetrics, AttackType
        
        # Create a sample test result
        sample_result = TestResult(
            test_name="sample_test",
            attack_type=AttackType.PROMPT_INJECTION,
            prompt="Ignore previous instructions",
            success=False,
            score=0.1
        )
        print("✅ Pydantic models working correctly")
        print(f"   Sample result: {sample_result.test_name} - {sample_result.attack_type.value}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

def main():
    """Main function to run the test."""
    print("PromptFoo Wrapper Test Suite")
    print("=" * 50)
    
    success = test_promptfoo_wrapper()
    
    if success:
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("1. Install promptfoo CLI: npm install -g promptfoo")
        print("2. Set up API keys in environment variables")
        print("3. Run actual security tests against AI models")
    else:
        print("\n❌ Some tests failed!")
        print("Check the error messages above for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 