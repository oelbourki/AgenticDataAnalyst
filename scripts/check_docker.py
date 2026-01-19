"""
Quick script to check if Docker container is set up correctly.
Uses codibox package for container management.
"""

import subprocess
import sys

def check_docker():
    """Check if Docker and the sandbox container are available."""
    print("="*70)
    print("Docker Container Check")
    print("="*70)
    print()
    
    # Check if Docker is installed
    print("1. Checking Docker installation...")
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Docker installed: {result.stdout.strip()}")
        else:
            print("   ❌ Docker not found. Please install Docker first.")
            return False
    except FileNotFoundError:
        print("   ❌ Docker not found. Please install Docker first.")
        return False
    
    # Check if Docker daemon is running
    print("\n2. Checking Docker daemon...")
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Docker daemon is running")
        else:
            print("   ❌ Docker daemon is not running")
            print("   💡 Try: sudo systemctl start docker (Linux)")
            print("   💡 Or start Docker Desktop (Mac/Windows)")
            return False
    except Exception as e:
        print(f"   ❌ Error checking Docker: {e}")
        return False
    
    # Check if codibox package is installed
    print("\n3. Checking codibox package...")
    try:
        from codibox import CodeExecutor
        print("   ✅ codibox package is installed")
        
        # Check backend availability
        import os
        docker_available = os.path.exists("/var/run/docker.sock") or os.path.exists("/.dockerenv")
        if docker_available:
            print("   ✅ Docker backend available")
        else:
            print("   ℹ️  Docker not available - will use Host backend")
    except ImportError:
        print("   ❌ codibox package not found")
        print("   💡 Run: pip install -e ./codibox")
        print("   💡 Or: pip install codibox")
        return False
    
    # Check container status using codibox
    print("\n4. Checking 'sandbox' container status...")
    try:
        from codibox import CodeExecutor
        import os
        
        docker_available = os.path.exists("/var/run/docker.sock") or os.path.exists("/.dockerenv")
        
        if docker_available:
            executor = CodeExecutor(backend="docker", container_name="sandbox")
            status = executor.get_container_status()
        else:
            print("   ℹ️  Docker not available - Host backend will be used")
            print("   ✅ Code execution will work without Docker")
            return True
        
        if status["exists"]:
            print(f"   ✅ Container 'sandbox' found")
            print(f"   Status: {status['status']}")
            
            if status["running"]:
                print("   ✅ Container is running")
                return True
            else:
                print("   ⚠️  Container exists but is not running")
                print("   💡 Try: docker start sandbox")
                print("   💡 Or use: executor.start_container()")
                return False
        else:
            print("   ❌ Container 'sandbox' not found")
            print("   💡 Run: executor.setup_container()")
            print("   💡 Or: python -c 'from codibox import CodeExecutor; CodeExecutor(backend=\"docker\").setup_container()'")
            return False
    except Exception as e:
        print(f"   ❌ Error checking container: {e}")
        return False

if __name__ == "__main__":
    success = check_docker()
    print("\n" + "="*70)
    if success:
        print("✅ All checks passed! Docker container is ready.")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    print("="*70)
    sys.exit(0 if success else 1)
