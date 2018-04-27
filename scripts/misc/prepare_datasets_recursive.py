import os
import uuid
import argparse
import sys
import shutil

def all_valid_files(rootDir, is_valid):
    all_files = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            filePath = os.path.abspath(os.path.join(dirName, fname))
            if is_valid(filePath):
                all_files.append(filePath)
    return all_files

def is_valid(filePath):
    """
        Validation function
    """
    ext = [".jpg", ".JPG", ".jpeg", ".png"]
    if filePath.endswith(tuple(ext)):
        return True
    else:
        return False


def sanitise_class_name(className):
    className = className[len(SOURCE_IMAGES):]
    if className == "":
        className = "ROOT"

    if className[0] == "/":
        className = className[1:]

    className = className.replace("/","::").replace(" ", "_")
    return className

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Converts a taxonomy tree into multiple datasets \
                with folder-subfolder format (one subfolder per class)")

    parser.add_argument('--input-folder-path', action='store', dest='input_folder_path', required=True, help='Path to input folder containing images')
    parser.add_argument('--output-folder-path', action='store', dest='output_folder_path', required=True, help='Path to output folder to write images')

    args = parser.parse_args()

    SOURCE_IMAGES = args.input_folder_path
    OUTPUT_FOLDER = args.output_folder_path

    if os.path.exists(OUTPUT_FOLDER):
        response = query_yes_no(
                    "Output Folder seems to exist, do you want to overwrite ?",
                    default='no')
        if response:
            shutil.rmtree(OUTPUT_FOLDER)
        else:
            print("Exiting, because output folder path exists and cannot be deleted.")
            exit('No deletion of Output Folder')

    os.mkdir(OUTPUT_FOLDER)

    for dirName, subdirList, fileList in os.walk(SOURCE_IMAGES):
        if len(subdirList) == 0:
            """
            THis is a leaf node, no need to create a dataset for this
            """
        elif len(subdirList) == 1:
            """
            This is not a leaf node, but still has just 1 sub-class,
            and we need atleast two subclasses to create a dataset
            """
        else:
            """
            Case when this directory has more than one sub-classes
            """
            CLASSES = []
            FILES = []
            DATASET_NAME = sanitise_class_name(dirName)
            DATASET_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, DATASET_NAME)

            for subDir in subdirList:
                subDirPath = os.path.join(dirName, subDir)
                className = os.path.join(dirName, subDir)
                className = sanitise_class_name(className)

                all_files = all_valid_files(subDirPath, is_valid)
                CLASSES.append(className)
                FILES.append(all_files)

            print("dataset_name: {}, classes: {}, number of images per class: {}".format(DATASET_NAME, CLASSES, [len(x) for x in FILES]))
            """
            TODO


            - Create symlinks for all the files in the corresponding subdirectories : Done
                - Use the original filename to create the symlink :: Done
            - Write a bash script which iterates over all the datasets in OUTPUT_FOLDER and then use `prepare_data.py` to create tortilla datasets :Done
            - Use a min-image of 100
            """
            os.mkdir(os.path.join(OUTPUT_FOLDER, DATASET_NAME))
            for _idx, _class in enumerate(CLASSES):
                _files = FILES[_idx]
                classRoot = os.path.join(OUTPUT_FOLDER, DATASET_NAME, _class)
                try:
                    os.mkdir(classRoot)
                except:
                    pass
                    #TODO: Throw some error or handle them well

                for _file in _files:
                    fileName = str(uuid.uuid4())[:4]+"___"+os.path.basename(_file)
                    target_path = os.path.join(classRoot, fileName)
                    os.symlink(_file, target_path)
