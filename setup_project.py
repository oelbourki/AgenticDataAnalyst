#!/usr/bin/env python3
"""
Setup script for the root project.
Installs codibox package and sets up Docker container.
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install project dependencies including codibox from pip."""
    print("="*70)
    print("Installing project dependencies...")
    print("="*70)
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print(f"❌ Error: requirements.txt not found at {requirements_file}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully (including codibox)")
            return True
        else:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_docker_container():
    """Set up Docker container using codibox."""
    print("\n" + "="*70)
    print("Setting up Docker container...")
    print("="*70)
    
    try:
        from codibox import CodeExecutor
        import os
        
        # Check if Docker is available
        docker_available = os.path.exists("/var/run/docker.sock") or os.path.exists("/.dockerenv")
        
        if docker_available:
            executor = CodeExecutor(backend="docker", container_name="sandbox")
            status = executor.get_container_status()
        else:
            print("ℹ️  Docker not available. Using host backend (no container setup needed).")
            return True
        
            if status["running"]:
                print("✅ Container 'sandbox' is already running")
                return True
            
            if status["exists"]:
                print("Container exists but is stopped. Starting it...")
                if executor.start_container():
                    print("✅ Container started successfully")
                    return True
                else:
                    print("❌ Failed to start container")
                    return False
            else:
                print("Container not found. Setting it up (this may take a few minutes)...")
                if executor.setup_container():
                    print("✅ Container set up successfully")
                    return True
                else:
                    print("❌ Failed to set up container")
                    return False
                
    except ImportError:
        print("❌ codibox package not installed. Run install_codibox() first.")
        return False
    except Exception as e:
        print(f"❌ Error setting up container: {e}")
        return False

def main():
    """Main setup function."""
    print("\n" + "="*70)
    print("Project Setup")
    print("="*70)
    print()
    
    # Step 1: Install dependencies (including codibox)
    if not install_dependencies():
        print("\n❌ Setup failed at package installation step")
        sys.exit(1)
    
    # Step 2: Setup Docker container
    if not setup_docker_container():
        print("\n⚠️  Setup completed but Docker container setup failed")
        print("You can set it up later by running:")
        print("  python -c 'from codibox import CodeExecutor; CodeExecutor(backend=\"docker\").setup_container()'")
        sys.exit(0)
    
    print("\n" + "="*70)
    print("✅ Setup completed successfully!")
    print("="*70)
    print("\nYou can now:")
    print("  - Run Streamlit app: streamlit run streamlit_app.py")
    print("  - Use codibox in your code: from codibox import CodeExecutor")
    print("  - Check container status: python scripts/check_docker.py")
    print()

if __name__ == "__main__":
    main()
