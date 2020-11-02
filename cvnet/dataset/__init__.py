from pathlib import Path

from cvnet.utils.registry_utils import import_all_modules

FILE_ROOT = Path(__file__).parent
# automatically import any Python files in the dataset/ directory
import_all_modules(FILE_ROOT, "cvnet.dataset")

from cvnet.dataset.pascalvoc_dataset import PascalVOCDataset
from cvnet.dataset.cityscapes_dataset import CityscapesDataset


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": PascalVOCDataset,
        # "camvid": camvidLoader,
        # "ade20k": ADE20KLoader,
        # "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": CityscapesDataset,
        # "nyuv2": NYUv2Loader,
        # "sunrgbd": SUNRGBDLoader,
        # "vistas": mapillaryVistasLoader,
    }[name]
