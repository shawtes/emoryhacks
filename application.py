"""
Elastic Beanstalk entry point - redirects to FastAPI app
"""
import sys
from pathlib import Path

# Add emoryhacks to path
sys.path.insert(0, str(Path(__file__).parent / "emoryhacks"))

from emoryhacks.api.main import app

# This is the application object that Elastic Beanstalk expects
application = app




