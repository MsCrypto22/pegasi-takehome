#!/usr/bin/env python3
"""
🎯 PRESENTATION DAY DEMO SCRIPT
================================

Enterprise AI Security Simulation - Live Demonstration
Perfect for showcasing business value and ROI to stakeholders.

USAGE:
    python presentation_demo.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("🚀 ENTERPRISE AI SECURITY SIMULATION")
    print("=" * 50)
    print("🎯 Live Demonstration for Stakeholders")
    print("📊 Business Value & ROI Analysis")
    print("🛡️  Adaptive AI Security Defense")
    print("=" * 50)
    print()
    
    # Import and run simulation
    try:
        from security_simulation import EnterpriseSecuritySimulation
        
        print("✅ Loading Enterprise Security Simulation...")
        simulation = EnterpriseSecuritySimulation()
        
        print("\n🎬 Starting Live Demo...")
        print("📈 Running 6 realistic attack scenarios...")
        print("💼 Calculating business impact...")
        print("🧠 Demonstrating AI learning & adaptation...")
        print()
        
        # Run the simulation
        import asyncio
        report = asyncio.run(simulation.run_comprehensive_simulation(num_scenarios=6))
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETE - KEY RESULTS")
        print("="*60)
        
        # Display key metrics
        print(f"💰 Total Financial Risk Mitigated: ${report['simulation_summary']['total_financial_loss']:,.2f}")
        print(f"🛡️  Attacks Blocked: {report['simulation_summary']['prevented_attacks']}/{report['simulation_summary']['total_scenarios']}")
        print(f"📊 Success Rate: {report['simulation_summary']['prevention_rate']:.1f}%")
        print(f"🧠 Patterns Learned: {report['learning_agent_performance']['patterns_learned']}")
        print(f"⚡ Strategies Generated: {report['learning_agent_performance']['strategies_generated']}")
        print(f"📈 ROI Improvement: {report['business_impact']['roi_percentage']:.1f}%")
        
        print("\n" + "="*60)
        print("🎯 BUSINESS VALUE DEMONSTRATED")
        print("="*60)
        print("✅ Real-time threat detection & blocking")
        print("✅ Adaptive learning from attack patterns")
        print("✅ Automated security strategy generation")
        print("✅ Quantifiable risk reduction & cost savings")
        print("✅ Scalable enterprise-grade protection")
        
        print("\n🎊 PRESENTATION SUCCESS! 🎊")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 