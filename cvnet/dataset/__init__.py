from cvnet.utils.registry_utils import import_all_modules
from pathlib import Path
FILE_ROOT = Path(__file__).parent
# automatically import any Python files in the dataset/ directory
import_all_modules(FILE_ROOT, "cvnet.dataset")