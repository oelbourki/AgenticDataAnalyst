"""
Backend status and execution service.
"""

import os
from typing import Dict, Any, Optional

from ..config import USE_DOCKER


class BackendService:
    """Service for managing execution backend status."""
    
    def __init__(self):
        self._codibox_available: Optional[bool] = None
        self._workflow_available: Optional[bool] = None
        self._docker_available: Optional[bool] = None
        self._backend_status: Optional[Dict[str, Any]] = None
    
    def _check_codibox(self) -> bool:
        """Check if codibox is available."""
        if self._codibox_available is None:
            try:
                from codibox import CodeExecutor
                self._codibox_available = True
            except ImportError:
                self._codibox_available = False
        return self._codibox_available
    
    def _check_workflow(self) -> bool:
        """Check if workflow is available."""
        if self._workflow_available is None:
            try:
                from agent_coder.simple_workflow import process_query, simple_coder
                from agent_coder.utils import analyze_csv, generate_detailed_dataframe_description
                self._workflow_available = True
            except Exception:
                self._workflow_available = False
        return self._workflow_available
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is None:
            if USE_DOCKER and self._check_codibox():
                self._docker_available = (
                    os.path.exists("/var/run/docker.sock") or 
                    os.path.exists("/.dockerenv")
                )
            else:
                self._docker_available = False
        return self._docker_available
    
    def check_backend_status(self) -> Dict[str, Any]:
        """Check execution backend status (Host by default, Docker if requested)."""
        if self._backend_status is not None:
            return self._backend_status
        
        if not self._check_codibox():
            self._backend_status = {
                "available": False,
                "backend": None,
                "ready": False,
                "status": "codibox_not_available"
            }
            return self._backend_status
        
        try:
            # Host backend is default - always ready (no container needed)
            if not USE_DOCKER:
                self._backend_status = {
                    "available": True,
                    "backend": "host",
                    "ready": True,
                    "status": "ready"
                }
                return self._backend_status
            
            # Docker backend (only if explicitly requested)
            if self._check_docker():
                try:
                    from codibox import CodeExecutor
                    executor = CodeExecutor(backend="docker", container_name="sandbox")
                    container_running = executor.check_container()
                    self._backend_status = {
                        "available": True,
                        "backend": "docker",
                        "ready": container_running,
                        "status": "running" if container_running else "not_running"
                    }
                except Exception as e:
                    # Fall back to host if Docker check fails
                    self._backend_status = {
                        "available": True,
                        "backend": "host",
                        "ready": True,
                        "status": f"ready (docker check failed: {str(e)})"
                    }
            else:
                # Docker requested but not available - fall back to host
                self._backend_status = {
                    "available": True,
                    "backend": "host",
                    "ready": True,
                    "status": "ready (docker not available, using host)"
                }
        except Exception as e:
            # On error, fall back to host backend
            self._backend_status = {
                "available": True,
                "backend": "host",
                "ready": True,
                "status": f"ready (fallback from error: {str(e)})"
            }
        
        return self._backend_status
    
    @property
    def is_execution_ready(self) -> bool:
        """Check if execution backend is ready."""
        status = self.check_backend_status()
        return status.get("ready", False) and self._check_workflow()
    
    @property
    def codibox_available(self) -> bool:
        """Check if codibox is available."""
        return self._check_codibox()
    
    @property
    def workflow_available(self) -> bool:
        """Check if workflow is available."""
        return self._check_workflow()
    
    @property
    def docker_available(self) -> bool:
        """Check if Docker is available."""
        return self._check_docker()
