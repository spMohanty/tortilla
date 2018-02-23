import sys
import os
import shutil
import glob

def quick_compute_class_frequency_from_folder(folder_path, classes):
    _classes = get_classes_from_input_folder(folder_path)
    assert _classes == classes

    _class_frequency = {}

    for _class in classes:
        _class_frequency[_class] = len(os.listdir(
                            os.path.join(folder_path, _class)
                        ))

    return _class_frequency

def output_folder_path_validation(output_folder_path, classes):
    """
        Validation of the Output folder path

        If output_folder_path exists, check if the user wants to overwrite
        else exit
    """
    if os.path.exists(output_folder_path):
        response = query_yes_no(
                    "Output Folder seems to exist, do you want to overwrite ?",
                    default='no')

        if response:
            shutil.rmtree(output_folder_path)
        else:
            print("Exiting, because output folder path exists \
                  and cannot be deleted.")
            exit(0)

    """
    Create output_folder_path for images root folder
    and corresponding class subfolders
    """
    os.mkdir(output_folder_path)
    os.mkdir(
        os.path.join(
            output_folder_path,
            "images"
        )
    )
    for _class in classes:
        os.mkdir(os.path.join(
            output_folder_path,
            "images",
            _class
        ))

def get_classes_from_input_folder(input_folder_path):
    dot_files = glob.glob(input_folder_path+"/.*")
    dot_files = [x.replace(input_folder_path+"/","") for x in dot_files]
    classes = set(os.listdir(input_folder_path)) - set(dot_files)
    final_classes = []
    for _class in classes:
        if os.path.isdir(os.path.join(input_folder_path, _class)):
            final_classes.append(_class)
    return final_classes

def input_folder_path_validation(input_folder_path):
    """
        Validation of the Input folder path
    """
    classes = get_classes_from_input_folder(input_folder_path)
    for _class in classes:
        assert(os.path.isdir(os.path.join(input_folder_path, _class)))

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
