from hilti_profis.Config import MasterModule

class Template(MasterModule):
    def __init__(self, basefile: dict) -> None:
        headerpath = ['Template']
        submodules_list = [
            
        ]
        super().__init__(basefile, headerpath, submodules_list)