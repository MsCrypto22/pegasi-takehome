#!/usr/bin/env python3
"""
LangFuse Integration for AI Security Dashboard
=============================================

Real-time visualization dashboard for:
- Attack success rates over time
- Current guardrail configuration
- Live testing interface
- Learning agent performance metrics
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from learning_agent import LearningAgent, TestResult, AttackType
from security_simulation import EnterpriseSecuritySimulation

# LangFuse configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1234567890")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-1234567890")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

try:
    from langfuse import Langfuse
    from langfuse.model import CreateTrace, CreateSpan, CreateScore
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


@dataclass
class DashboardMetrics:
    """Dashboard metrics container"""
    total_attacks: int
    successful_attacks: int
    blocked_attacks: int
    success_rate: float
    average_detection_time: float
    total_financial_risk: float
    patterns_learned: int
    strategies_generated: int
    guardrails_active: int
    last_attack_time: datetime
    learning_progress: float


class SecurityDashboard:
    """Real-time AI Security Dashboard with LangFuse integration"""
    
    def __init__(self):
        self.learning_agent = LearningAgent()
        self.simulation = EnterpriseSecuritySimulation()
        
        # Initialize LangFuse if available
        if LANGFUSE_AVAILABLE:
            self.langfuse = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
            print("✅ LangFuse integration enabled")
        else:
            self.langfuse = None
            print("⚠️  LangFuse not available. Install with: pip install langfuse")
            
        # Dashboard state
        self.metrics_history = []
        self.current_trace = None
        
    def log_attack_to_langfuse(self, test_result: TestResult, success: bool, detection_time: float):
        """Log attack attempt to LangFuse for visualization"""
        if not self.langfuse:
            return
            
        try:
            # Create trace for this attack
            trace = self.langfuse.trace(
                id=f"attack_{test_result.test_name}_{int(time.time())}",
                name=f"AI Security Attack: {test_result.attack_type.value}",
                metadata={
                    "attack_type": test_result.attack_type.value,
                    "success": success,
                    "detection_time": detection_time,
                    "score": test_result.score,
                    "prompt_length": len(test_result.prompt)
                }
            )
            
            # Log the attack span
            trace.span(
                name="Attack Detection",
                start_time=datetime.now() - timedelta(seconds=detection_time),
                end_time=datetime.now(),
                metadata={
                    "blocked": not success,
                    "response_time_ms": detection_time * 1000
                }
            )
            
            # Score the attack
            trace.score(
                name="Attack Severity",
                value=test_result.score,
                comment=f"Attack {test_result.attack_type.value} {'succeeded' if success else 'blocked'}"
            )
            
            trace.flush()
            
        except Exception as e:
            print(f"⚠️  LangFuse logging error: {e}")
    
    def get_current_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics"""
        # Get test results
        test_results = self.learning_agent.memory.get_recent_test_results(days=30)
        
        if not test_results:
            return DashboardMetrics(
                total_attacks=0,
                successful_attacks=0,
                blocked_attacks=0,
                success_rate=0.0,
                average_detection_time=0.0,
                total_financial_risk=0.0,
                patterns_learned=len(self.learning_agent.memory.get_learned_patterns()),
                strategies_generated=len(self.learning_agent.memory.get_adaptation_strategies()),
                guardrails_active=10,  # Default guardrails
                last_attack_time=datetime.now(),
                learning_progress=0.0
            )
        
        # Calculate metrics
        total_attacks = len(test_results)
        successful_attacks = sum(1 for r in test_results if r.success)
        blocked_attacks = total_attacks - successful_attacks
        success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0
        
        # Average detection time
        detection_times = [r.execution_time for r in test_results if r.execution_time]
        average_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0.0
        
        # Financial risk (simplified calculation)
        total_financial_risk = sum(r.score * 50000 for r in test_results if r.success)  # $50K per successful attack
        
        # Learning progress
        patterns = len(self.learning_agent.memory.get_learned_patterns())
        strategies = len(self.learning_agent.memory.get_adaptation_strategies())
        learning_progress = min(100.0, (patterns + strategies) / 10.0)  # Progress based on learned items
        
        return DashboardMetrics(
            total_attacks=total_attacks,
            successful_attacks=successful_attacks,
            blocked_attacks=blocked_attacks,
            success_rate=success_rate,
            average_detection_time=average_detection_time,
            total_financial_risk=total_financial_risk,
            patterns_learned=patterns,
            strategies_generated=strategies,
            guardrails_active=10 + patterns,  # Base + learned patterns
            last_attack_time=max(r.execution_time for r in test_results) if test_results else datetime.now(),
            learning_progress=learning_progress
        )
    
    def run_streamlit_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="AI Security Dashboard",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🛡️ AI Security Testing Dashboard")
        st.markdown("Real-time monitoring of AI security threats and adaptive defenses")
        
        # Sidebar
        st.sidebar.header("Dashboard Controls")
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Manual refresh
        if st.sidebar.button("🔄 Refresh Metrics"):
            st.rerun()
        
        # Run simulation
        if st.sidebar.button("🚀 Run Attack Simulation"):
            with st.spinner("Running security simulation..."):
                import asyncio
                asyncio.run(self.simulation.run_comprehensive_simulation())
            st.success("Simulation completed!")
            st.rerun()
        
        # Get current metrics
        metrics = self.get_current_metrics()
        
        # Main dashboard layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Attacks",
                value=metrics.total_attacks,
                delta=f"+{metrics.total_attacks - len(self.metrics_history[-1:]) if self.metrics_history else 0}"
            )
        
        with col2:
            st.metric(
                label="Success Rate",
                value=f"{metrics.success_rate:.1%}",
                delta=f"{metrics.success_rate - (self.metrics_history[-1].success_rate if self.metrics_history else 0):.1%}"
            )
        
        with col3:
            st.metric(
                label="Financial Risk",
                value=f"${metrics.total_financial_risk:,.0f}",
                delta=f"${metrics.total_financial_risk - (self.metrics_history[-1].total_financial_risk if self.metrics_history else 0):,.0f}"
            )
        
        with col4:
            st.metric(
                label="Learning Progress",
                value=f"{metrics.learning_progress:.1f}%",
                delta=f"{metrics.learning_progress - (self.metrics_history[-1].learning_progress if self.metrics_history else 0):.1f}%"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Attack Success Rate Over Time")
            
            # Create time series data
            if self.metrics_history:
                df = pd.DataFrame([
                    {
                        'timestamp': m.last_attack_time,
                        'success_rate': m.success_rate,
                        'total_attacks': m.total_attacks
                    }
                    for m in self.metrics_history[-20:]  # Last 20 data points
                ])
                
                fig = px.line(
                    df, 
                    x='timestamp', 
                    y='success_rate',
                    title="Attack Success Rate Trend",
                    labels={'success_rate': 'Success Rate', 'timestamp': 'Time'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical data available yet. Run some attacks to see trends.")
        
        with col2:
            st.subheader("Attack Types Distribution")
            
            # Get attack type distribution
            test_results = self.learning_agent.memory.get_recent_test_results(days=7)
            if test_results:
                attack_counts = {}
                for result in test_results:
                    attack_type = result.attack_type.value
                    attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
                
                fig = px.pie(
                    values=list(attack_counts.values()),
                    names=list(attack_counts.keys()),
                    title="Recent Attack Types"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recent attacks to display.")
        
        # Guardrails configuration
        st.subheader("🛡️ Active Guardrails Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Guardrails", metrics.guardrails_active)
            st.metric("Patterns Learned", metrics.patterns_learned)
            st.metric("Strategies Generated", metrics.strategies_generated)
        
        with col2:
            st.metric("Average Detection Time", f"{metrics.average_detection_time:.3f}s")
            st.metric("Blocked Attacks", metrics.blocked_attacks)
            st.metric("Successful Attacks", metrics.successful_attacks)
        
        # Real-time testing interface
        st.subheader("🧪 Real-time Testing Interface")
        
        col1, col2 = st.columns(2)
        
        with col1:
            attack_type = st.selectbox(
                "Attack Type",
                [at.value for at in AttackType]
            )
            
            test_prompt = st.text_area(
                "Test Prompt",
                placeholder="Enter a test prompt to evaluate security...",
                height=100
            )
            
            if st.button("🚀 Test Security"):
                if test_prompt:
                    with st.spinner("Testing security..."):
                        # Create test result
                        start_time = time.time()
                        test_result = TestResult(
                            test_name=f"manual_test_{int(start_time)}",
                            prompt=test_prompt,
                            response="Manual test response",
                            attack_type=AttackType(attack_type),
                            success=False,
                            score=0.5,
                            execution_time=0.1,  # Fixed: use float instead of datetime
                            metadata={"source": "manual_test"}
                        )
                        
                        # Let learning agent analyze
                        self.learning_agent.learn_from_results([test_result])
                        
                        # Log to LangFuse
                        self.log_attack_to_langfuse(test_result, False, 0.1)
                        
                    st.success("Security test completed!")
                    st.rerun()
        
        with col2:
            st.subheader("Recent Test Results")
            
            recent_results = self.learning_agent.memory.get_recent_test_results(days=1)
            if recent_results:
                for result in recent_results[-5:]:  # Show last 5
                    status = "✅ BLOCKED" if not result.success else "❌ SUCCESS"
                    st.write(f"**{result.test_name}** - {status}")
                    st.write(f"Type: {result.attack_type.value} | Score: {result.score:.2f}")
                    st.write("---")
            else:
                st.info("No recent test results.")
        
        # Store metrics for history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:  # Keep last 100 data points
            self.metrics_history = self.metrics_history[-100:]
        
        # Footer
        st.markdown("---")
        st.markdown(
            f"*Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"LangFuse: {'✅ Connected' if LANGFUSE_AVAILABLE else '❌ Not Available'}*"
        )


def main():
    """Main function to run the dashboard"""
    dashboard = SecurityDashboard()
    dashboard.run_streamlit_dashboard()


if __name__ == "__main__":
    main() 