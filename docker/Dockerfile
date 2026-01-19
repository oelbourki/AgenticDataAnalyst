
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
    