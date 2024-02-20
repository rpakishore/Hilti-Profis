from hilti_profis.Config import MasterModule
import uuid

class Combos(MasterModule):
    def __init__(self, basefile: dict) -> None:
        headerpath = ['ProjectDesignConcreteEntity','Loads', 'LoadCombinations']
        submodules_list = []
        super().__init__(basefile, headerpath, submodules_list)
        
    def add(self, Fx: float=0, Fy: float=0, Fz: float=0, 
            Mx:float=0, My:float=0, Mz:float=0, LoadType='Seismic', Comment: str=""):
        new_load = {
            'Id': str(uuid.uuid4()),
            'ForceZ': str(Fz),
            'ForceX': str(Fx),
            'ForceY': str(Fy),
            'MomentZ': str(Mz),
            'MomentX': str(Mx),
            'MomentY': str(My),
            'ForceZVariable': '0',
            'ForceXVariable': '0',
            'ForceYVariable': '0',
            'MomentZVariable': '0',
            'MomentXVariable': '0',
            'MomentYVariable': '0',
            'LoadType': LoadType,
            'Comment': Comment,
            'IsWizardGenerated': 'false',
            'LoadCharacteristic': {'@i:nil': 'true'}}
        
        existing = self.data.get('LoadCombinationEntity')

        if isinstance(existing, list):
            existing.append(new_load)
            self.data['LoadCombinationEntity'] = existing
        elif isinstance(existing, dict):
            self.data['LoadCombinationEntity'] = [existing, new_load]
        elif existing is None:
            self.data['LoadCombinationEntity'] = new_load
        else:
            raise Exception(f'Method {self.__class__.__name__}.{self.__name__} not implemented.')