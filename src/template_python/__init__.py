"Placeholder module info"
__version__ = "0.0.1"

from .utils.logger import log

log.info('Template Module Initialized')

def debug(status=False):
    """Import this in a new module and enable debug to use debug
    example:
    ```python
    from template_python import debug
    debug(True)
    ```
    """
    if status:
        log.setLevel(10) #debug
    else:
        log.setLevel(20) #info

    log.debug(f'Debug Mode: {status}')
    
debug(status=False)