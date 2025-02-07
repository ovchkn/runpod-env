# JupyterHub Configuration for ML Environment
import os

# Basic Configuration
c = get_config()  # noqa
c.JupyterHub.ip = '0.0.0.0'
c.JupyterHub.port = 8888
c.JupyterHub.allow_root = True

# User Configuration
c.LocalAuthenticator.create_system_users = True
c.Authenticator.admin_users = {'root'}

# Spawner Configuration
c.Spawner.default_url = '/lab'  # Use JupyterLab by default
c.Spawner.env = {
    'JUPYTER_ENABLE_LAB': 'yes',
    'PATH': '/opt/conda/envs/mlops/bin:/opt/conda/bin:' + os.environ['PATH'],
    'CONDA_DEFAULT_ENV': 'mlops',
    'MLFLOW_TRACKING_URI': 'http://localhost:5000',
    'PYTHONPATH': '/workspace'
}

# Directory Configuration
c.Spawner.notebook_dir = '/workspace/notebooks'
c.JupyterHub.data_files_path = '/workspace/services/jupyterhub/data'
c.JupyterHub.runtime_dir = '/workspace/services/jupyterhub/runtime'

# Server Configuration
c.ConfigurableHTTPProxy.command = ['configurable-http-proxy']
c.JupyterHub.cleanup_servers = True
c.JupyterHub.cleanup_proxy = True

# SSL/Security Configuration
c.JupyterHub.ssl_key = ''
c.JupyterHub.ssl_cert = ''
c.JupyterHub.cookie_secret_file = '/workspace/services/jupyterhub/jupyterhub_cookie_secret'

# Resource Limits
c.Spawner.cpu_limit = None
c.Spawner.mem_limit = None
c.Spawner.environment = {
    'NVIDIA_VISIBLE_DEVICES': 'all'
}

# Extensions and services
c.JupyterHub.load_roles = [
    {
        "name": "user",
        "scopes": ["access:services", "self"]
    }
]

# Add JupyterLab Git extension
c.Spawner.args = [
    '--NotebookApp.token=""',
    '--NotebookApp.password=""',
    '--NotebookApp.allow_origin="*"',
    '--NotebookApp.base_url=/"'
]

# Template paths
c.JupyterHub.template_paths = ['/workspace/services/jupyterhub/templates']

# Services
c.JupyterHub.services = [
    {
        'name': 'idle-culler',
        'command': [
            sys.executable,
            '-m', 'jupyterhub_idle_culler',
            '--timeout=3600'
        ],
        'admin': True
    }
]

# Debug logging
c.JupyterHub.log_level = 'INFO'
c.Spawner.debug = True

# Shutdown on logout
c.JupyterHub.shutdown_on_logout = False

# Custom logo/branding
c.JupyterHub.logo_file = ''