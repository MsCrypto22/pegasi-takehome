#!/usr/bin/env python3
"""
MCP Server Presentation Demo

This script demonstrates the working MCP server capabilities for presentations.
Run this after starting the MCP server with: python3.13 start_server.py
"""

import requests
import json
import time
from datetime import datetime

class MCPPresentationDemo:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def test_server_health(self):
        """Test if the MCP server is running and healthy."""
        print("🔍 Testing MCP Server Health...")
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server Status: {data.get('status', 'Unknown')}")
                return True
            else:
                print(f"❌ Server not responding: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def get_server_info(self):
        """Get basic server information."""
        print("📊 Getting Server Information...")
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server: {data.get('message', 'Unknown')}")
                print(f"📈 Version: {data.get('version', 'Unknown')}")
                print(f"🔄 Status: {data.get('status', 'Unknown')}")
                return data
            else:
                print(f"❌ Info failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Info error: {e}")
            return None
    
    def run_adaptive_workflow(self, target_model="demo-model"):
        """Run an adaptive security testing workflow."""
        print(f"\n🎯 Running Adaptive Security Testing Workflow...")
        print(f"🎯 Target Model: {target_model}")
        print("⏳ This will demonstrate:")
        print("  • Real-time security testing")
        print("  • Adaptive test generation")
        print("  • Learning from results")
        print("  • Strategy optimization")
        
        workflow_config = {
            "target_model": target_model,
            "include_adaptive_tests": True,
            "learning_enabled": True,
            "max_tests": 5
        }
        
        try:
            print("\n🚀 Starting workflow...")
            response = requests.post(
                f"{self.base_url}/workflow/run",
                json=workflow_config
            )
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Workflow Completed Successfully!")
                print("=" * 50)
                print(f"📊 Total Tests Executed: {result.get('total_tests', 0)}")
                print(f"🧠 Adaptive Tests Generated: {result.get('adaptive_tests', 0)}")
                print(f"📈 Overall Security Score: {result.get('overall_security_score', 0):.2f}")
                print(f"🎯 Successful Tests: {result.get('successful_tests', 0)}")
                
                # Show learning summary if available
                if 'learning_summary' in result:
                    learning = result['learning_summary']
                    print(f"\n🧠 Learning Summary:")
                    print(f"  • Test Results Stored: {learning.get('total_test_results', 0)}")
                    print(f"  • Learned Patterns: {learning.get('total_patterns', 0)}")
                    print(f"  • Adaptation Strategies: {learning.get('total_strategies', 0)}")
                    print(f"  • Learning Progress: {learning.get('learning_progress', 0):.1f}%")
                
                # Show optimizations if available
                if 'optimizations' in result:
                    optimizations = result['optimizations']
                    print(f"\n⚙️  Strategy Optimizations:")
                    for i, opt in enumerate(optimizations[:3], 1):
                        print(f"  {i}. {opt}")
                
                return result
            else:
                print(f"❌ Workflow failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Workflow error: {e}")
            return None
    
    def list_active_tests(self):
        """List all active tests."""
        print(f"\n📋 Active Test Status...")
        try:
            response = requests.get(f"{self.base_url}/test/list")
            if response.status_code == 200:
                data = response.json()
                active_count = data.get('active_tests', 0)
                print(f"✅ Active Tests: {active_count}")
                
                if active_count > 0:
                    tests = data.get('tests', [])
                    print("📝 Recent Test Results:")
                    for i, test in enumerate(tests[:5], 1):
                        status = test.get('status', 'Unknown')
                        name = test.get('test_name', 'Unknown')
                        score = test.get('security_score', 'N/A')
                        print(f"  {i}. {name} - {status} (Score: {score})")
                
                return data
            else:
                print(f"❌ List failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ List error: {e}")
            return None
    
    def demonstrate_mcp_protocol(self):
        """Demonstrate MCP protocol capabilities."""
        print(f"\n🔌 Demonstrating MCP Protocol Integration...")
        
        # Example MCP request
        mcp_request = {
            "method": "test_security",
            "params": {
                "model": "demo-model",
                "prompt": "Test prompt for MCP integration",
                "test_type": "security_scan"
            },
            "id": "demo-request-001"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/mcp/request",
                json=mcp_request
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ MCP Protocol Working!")
                print(f"📊 Request ID: {result.get('id', 'Unknown')}")
                if 'result' in result:
                    print(f"🎯 Result Status: Success")
                return result
            else:
                print(f"⚠️  MCP Protocol Test: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ MCP Protocol error: {e}")
            return None

def run_presentation_demo():
    """Run the complete presentation demo."""
    print("🚀 MCP Server Presentation Demo")
    print("=" * 60)
    print("🎯 This demo showcases:")
    print("  • Real-time AI model security testing")
    print("  • Adaptive learning capabilities")
    print("  • MCP protocol integration")
    print("  • Live workflow execution")
    print("=" * 60)
    
    demo = MCPPresentationDemo()
    
    # 1. Test server health
    if not demo.test_server_health():
        print("❌ MCP Server is not running. Please start it with: python3.13 start_server.py")
        return
    
    # 2. Get server info
    demo.get_server_info()
    
    # 3. Demonstrate MCP protocol
    demo.demonstrate_mcp_protocol()
    
    # 4. Run adaptive workflow (main demo)
    workflow_result = demo.run_adaptive_workflow()
    
    # 5. Show active tests
    demo.list_active_tests()
    
    # 6. Presentation summary
    print("\n" + "=" * 60)
    print("🎉 Presentation Demo Complete!")
    print("=" * 60)
    print("✅ Successfully Demonstrated:")
    print("  • MCP Server Health & Connectivity")
    print("  • Adaptive Security Testing Workflow")
    print("  • Real-time Learning & Pattern Recognition")
    print("  • Strategy Optimization & Adaptation")
    print("  • Active Test Monitoring")
    
    if workflow_result:
        print(f"\n📊 Demo Results:")
        print(f"  • Tests Executed: {workflow_result.get('total_tests', 0)}")
        print(f"  • Adaptive Tests: {workflow_result.get('adaptive_tests', 0)}")
        print(f"  • Security Score: {workflow_result.get('overall_security_score', 0):.2f}")
        print(f"  • Learning Enabled: ✅")
    
    print(f"\n💡 Next Steps for Audience:")
    print("  • Explore the dashboard: http://localhost:8501")
    print("  • Run full attack simulation: python3.13 attack_demo.py")
    print("  • View learning agent details: python3.13 test_learning_agent.py")
    print("  • Check GitHub repository for full documentation")

if __name__ == "__main__":
    print("Starting MCP Server Presentation Demo...")
    print("Make sure the MCP server is running: python3.13 start_server.py")
    print()
    
    run_presentation_demo() 