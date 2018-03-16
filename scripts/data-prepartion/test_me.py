from nose.tools import assert_raises
from utils import *
import numpy as np



class TestClass:

    def test_valid_input_folder(self):
        with assert_raises(SystemExit) as cm:
            get_classes_from_input_folder('tests/data/plant_diseases/c_5/File_1', non_interactive_mode=True)
        assert cm.exception.args[0] == 'Invalid Input Folder Path'

    def test_subfolders(self):
        with assert_raises(SystemExit) as cm:
            get_classes_from_input_folder('tests/data/plant/c_6',non_interactive_mode=True)
        assert cm.exception.args[0] == 'No Valid Subfolders'

    def test_valid_subfolders(self):
        with assert_raises(SystemExit) as cm:
            get_classes_from_input_folder('tests/data/plant',non_interactive_mode=True)
        assert cm.exception.args[0] == 'No Valid Images'

    def test_classes_creation(self):
        classes = get_classes_from_input_folder('tests/data/plant_diseases', non_interactive_mode=True)
        assert len(classes) == 5
        assert set(classes) == set(np.array(['c_0', 'c_1', 'c_2', 'c_3', 'c_5']))

    def test_output_folder_creation(self):
        test_classes = np.array(['A', 'B', 'C'])
        output_folder_path_validation('datasets/test1', test_classes, non_interactive_mode=True)
        assert os.path.exists('datasets/test1') == True
        assert set(os.listdir('datasets/test1/images')) == set(test_classes)
        shutil.rmtree('datasets/test1')

    def test_output_folder_deletion(self):
        test_classes_2 = np.array(['D', 'E', 'F'])
        os.mkdir('datasets/test2')
        assert os.path.exists('datasets/test2') == True
        with assert_raises(SystemExit) as cm:
            output_folder_path_validation('datasets/test2', test_classes_2, non_interactive_mode=True)
        assert cm.exception.args[0] == 'No deletion of Output Folder'
        shutil.rmtree('datasets/test2')
