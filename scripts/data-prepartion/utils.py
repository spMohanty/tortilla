import sys
import os
import shutil
import glob
from PIL import Image

def files_in_a_directory(folder_path):
    files = glob.glob(folder_path+"/*")
    final_files = []
    for _file in files:
        if os.path.isfile(_file):
            final_files.append(_file)
    return final_files

def quick_compute_class_frequency_from_folder(folder_path, classes):
    _classes = get_classes_from_input_folder(folder_path)
    assert _classes == classes

    _class_frequency = {}

    for _class in classes:
        _class_frequency[_class] = len(files_in_a_directory(
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
    dot_files = [x.replace(input_folder_path,"") for x in dot_files]
    folders_list = set(os.listdir(input_folder_path)) - set(dot_files)
    print(folders_list)
    final_classes = []
    for _folder in folders_list:
        if os.path.isdir(os.path.join(input_folder_path, _folder)): #Only keep folders
            files_list = set(os.listdir(os.path.join(input_folder_path, _folder)))
            if files_list: #Discard empty folders
                print("Full folder: {} ".format(_folder))
                print(list(files_list)[:10])
                stop = False;
                print(len(files_list))
                while not stop:
                    for _idx, _file in enumerate(files_list):
                        print("Processing {}/{} :: {}".format(str(_idx), str(len(files_list)), _file))
                        try:
                            print("Try")
                            im=Image.open(os.path.join(input_folder_path, _folder,_file))
                            print("Yes")
                            stop = True;
                            final_classes.append(_folder)
                            print("Class added.")
                            break
                        except:
                            print("Except")
                            if _idx == len(files_list)-1:
                                print("Warning: the folder {} does not contain valid images.".format(_folder))
                                response = query_yes_no(
                                            "Do you want to continue ?",
                                            default='no')

                                if not response:
                                    print("Exiting.")
                                    exit(0)
                                stop = True;
                            else:
                                continue
                #if one_valid_image:
                    #break
                #for _file in sub_folders_list:
                # need editing for valid image check
                # either while true try Image or check for .jpg...
                # final_classes.append(_folder)
    if not final_classes:
        print("Exiting because none of the subfolders contains valid images.")
        exit(0)
    print(final_classes)
    return final_classes

def input_folder_path_validation(input_folder_path):
    """
        Validation of the Input folder path

        If the folder is not a directory, then the program exits.
    """
    if os.path.isdir(input_folder_path):
        classes = get_classes_from_input_folder(input_folder_path)
        #I don't know if this is really useful because already done in get_classes
        for _class in classes:
            assert(os.path.isdir(os.path.join(input_folder_path, _class)))
    else:
        print("Exiting because Input Folder Path is not a directory, \
                please provide a folder containing sub-folders of images")
        exit(0)

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
