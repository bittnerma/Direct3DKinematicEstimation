from dataclasses import dataclass

@dataclass
class MarkerTransform:
    # original marker name in the dataset
    name: str
    # new marker name compatible to the opensim model
    transform: str

class MarkerSetTransform:

    def __init__(
            self,
            transforms: [MarkerTransform],
    ):
        self.transforms=transforms
        self.transformsDict={}
        self.update_dict()

    def update_dict(self):
        self.transformsDict={}
        for transform in self.transforms:
            self.transformsDict[transform.name]=transform.transform

    def copy(self):
        copyClass=MarkerSetTransform(self.transforms.copy())
        copyClass.update_dict()
        return copyClass