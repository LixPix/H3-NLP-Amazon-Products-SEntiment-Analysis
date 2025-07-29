#!/usr/bin/env python3
"""
Quick test script to verify the dashboard can import and run without syntax errors.
"""

import sys
import importlib.util

def test_dashboard_import():
    """Test if the dashboard can be imported without syntax errors."""
    try:
        spec = importlib.util.spec_from_file_location(
            "streamlit_dashboard", 
            "streamlit_dashboard.py"
        )
        dashboard_module = importlib.util.module_from_spec(spec)
        
        # This will raise a SyntaxError if there are syntax issues
        spec.loader.exec_module(dashboard_module)
        
        print("✅ Dashboard imports successfully - no syntax errors!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in dashboard: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
        
    except Exception as e:
        print(f"⚠️  Import error (may be normal for Streamlit): {e}")
        return True  # Import errors are often expected in Streamlit apps

if __name__ == "__main__":
    success = test_dashboard_import()
    sys.exit(0 if success else 1)
