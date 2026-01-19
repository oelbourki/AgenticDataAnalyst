import subprocess
import os

def setup_docker_environment():
    """Set up a secure Docker environment for code execution."""
    
    # Create a Dockerfile
    dockerfile_content = """
    FROM python:3.10-slim
    
    # Create a non-root user
    RUN useradd -m sandboxuser
    
    # Install required packages
    COPY requirements.txt /tmp/
    RUN pip install --no-cache-dir -r /tmp/requirements.txt
    
    # Set up working directory
    WORKDIR /home/sandboxuser
    
    # Switch to non-root user
    USER sandboxuser
    
    # Container will run indefinitely until stopped
    CMD ["sleep", "infinity"]
    """
    
    # Create requirements.txt with necessary packages
    requirements_content = """
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    statsmodels
    jupyter
    nbconvert
    jupytext
    ipynb-py-convert
    xlsxwriter
    """
    
    # Use root docker folder (don't create duplicate)
    docker_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker")
    os.makedirs(docker_path, exist_ok=True)
    
    # Write Dockerfile and requirements.txt to root docker folder
    with open(os.path.join(docker_path, "Dockerfile"), "w") as f:
        f.write(dockerfile_content)
    
    with open(os.path.join(docker_path, "requirements.txt"), "w") as f:
        f.write(requirements_content)
    
    # Check if container already exists
    check_result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=sandbox", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    
    if "sandbox" in check_result.stdout:
        print("Container 'sandbox' already exists. Removing it...")
        subprocess.run(["docker", "rm", "-f", "sandbox"], check=False)
    
    # Build the Docker image
    print("Building Docker image...")
    # Use root docker folder
    docker_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker")
    build_result = subprocess.run(
        ["docker", "build", "-t", "python_sandbox:latest", docker_path],
        capture_output=True,
        text=True
    )
    
    if build_result.returncode != 0:
        print(f"Error building Docker image: {build_result.stderr}")
        return False
    
    print("✅ Docker image built successfully")
    
    # Run the container with security restrictions
    print("Starting Docker container...")
    run_result = subprocess.run([
        "docker", "run", "-d",
        "--name", "sandbox",
        "--network", "none",  # No network access
        "--cap-drop", "all",  # Drop all capabilities
        "--pids-limit", "124",  # Limit number of processes
        "--tmpfs", "/tmp:rw,size=124M",  # Limit temp directory size
        "python_sandbox:latest"
    ], capture_output=True, text=True)
    
    if run_result.returncode != 0:
        print(f"Error starting container: {run_result.stderr}")
        return False
    
    # Verify container is running
    verify_result = subprocess.run(
        ["docker", "ps", "--filter", "name=sandbox", "--format", "{{.Status}}"],
        capture_output=True,
        text=True
    )
    
    if "Up" in verify_result.stdout:
        print("✅ Docker container 'sandbox' is now running.")
        print(f"   Status: {verify_result.stdout.strip()}")
        return True
    else:
        print("⚠️  Container created but may not be running. Check with: docker ps -a")
        return False

if __name__ == "__main__":
    setup_docker_environment() 