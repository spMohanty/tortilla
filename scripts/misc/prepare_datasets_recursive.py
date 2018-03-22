import os

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

        for subDir in subdirList:
            subDirPath = os.path.join(dirName, subDir)
            className = os.path.join(dirName, subDir)
            className = sanitise_class_name(className)

            all_files = all_valid_files(subDirPath, is_valid)
            CLASSES.append(className)
            FILES.append(all_files)

        print(DATASET_NAME, CLASSES, [len(x) for x in FILES])
