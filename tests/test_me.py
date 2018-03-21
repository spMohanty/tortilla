from nose.tools import assert_raises
import numpy as np
from scripts.data_preparation.utils import *

class TestClass:

    def test_valid_input_folder(self):
        with assert_raises(SystemExit) as cm:
            get_classes_from_input_folder('tests/data/plant_diseases/c_5/File_1', non_interactive_mode=True)
        assert cm.exception.args[0] == 'Invalid Input Folder Path'

    def test_subfolders(self):
        with assert_raises(SystemExit) as cm:
            get_classes_from_input_folder('tests/data/no-valid-image-folder/c_6',non_interactive_mode=True)
        assert cm.exception.args[0] == 'No Valid Subfolders'

    def test_valid_subfolders(self):
        with assert_raises(SystemExit) as cm:
            get_classes_from_input_folder('tests/data/no-valid-image-folder',non_interactive_mode=True)
        assert cm.exception.args[0] == 'No Valid Images'

    def test_classes_creation(self):
        classes = get_classes_from_input_folder('tests/data/plant_diseases', non_interactive_mode=True)
        assert len(classes) == 5
        assert set(classes) == set(np.array(['c_0', 'c_1', 'c_2', 'c_3', 'c_5']))

    def test_output_folder_creation(self):
        test_classes = np.array(['A', 'B', 'C'])
        output_folder_path_validation('tests/test1', test_classes, non_interactive_mode=True)
        assert os.path.exists('tests/test1') == True
        assert set(os.listdir('tests/test1/images')) == set(test_classes)
        shutil.rmtree('tests/test1')

    def test_output_folder_deletion(self):
        test_classes_2 = np.array(['D', 'E', 'F'])
        os.mkdir('tests/test2')
        assert os.path.exists('tests/test2') == True
        with assert_raises(SystemExit) as cm:
            output_folder_path_validation('tests/test2', test_classes_2, non_interactive_mode=True)
        assert cm.exception.args[0] == 'No deletion of Output Folder'
        shutil.rmtree('tests/test2')
