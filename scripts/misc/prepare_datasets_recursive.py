import os
import uuid

SOURCE_IMAGES = "/mount/SDE/instagram/myfoodrepo-images/myfoodrepo-images/images"
OUTPUT_FOLDER = "output" #Should create tortilla compatible dataset folders for all possible datasets

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
    if filePath.endswith(".jpg"):
        return True
    else:
        return False


def sanitise_class_name(className):
    className = className.replace(SOURCE_IMAGES, "")
    if className == "":
        className = "ROOT"

    if className[0] == "/":
        className = className[1:]

    className = className.replace("/","::").replace(" ", "_")
    return className

def main(SOURCE_IMAGES):
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

            print(DATASET_NAME, CLASSES, [len(x) for x in FILES])
            """
            TODO


            - Create symlinks for all the files in the corresponding subdirectories : Done
                - Use the original filename to create the symlink :: Done
            - Write a bash script which iterates over all the datasets in OUTPUT_FOLDER and then use `prepare_data.py` to create tortilla datasets
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
                    os.symlink(fileName, target_path)

main(SOURCE_IMAGES)
