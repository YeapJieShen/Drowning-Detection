import sys
import platform
import socket
import psutil
from datetime import datetime
from IPython.display import display, Markdown


def display_system_info(check_gpu: bool = False, markdown: bool = False) -> str:
    # TODO : Write method description

    # Get system details
    python_version = sys.version.split(' ')[0]
    os_name = platform.system()
    os_version = platform.version()
    architecture = platform.architecture()[0]
    hostname = socket.gethostname()
    processor = platform.processor()

    # Get total RAM size
    ram_size = round(psutil.virtual_memory().total /
                     (1024 ** 3), 2)  # Convert bytes to GB

    # Check for GPU (CUDA support)
    if check_gpu:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_status = f'Available ({gpu_name})'
        else:
            gpu_status = 'Not Available'

    # Get timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    info = (
f"""
**Last Updated**: {timestamp}

**Python Version**: {python_version}  
**OS**: {os_name} {os_version}  
**Architecture**: {architecture}  
**Hostname**: {hostname}  
**Processor**: {processor}  
**RAM Size**: {ram_size} GB  
{f"**GPU**: {gpu_status}" if check_gpu else ""}  
"""
    )

    if markdown:
        display(Markdown(info))
    else:
        print(info)
