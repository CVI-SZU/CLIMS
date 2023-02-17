from .voc import VOC, VOCAug, VOC_test
from .cocostuff import CocoStuff10k, CocoStuff164k


def get_dataset(name):
    return {
        "cocostuff10k": CocoStuff10k,
        "cocostuff164k": CocoStuff164k,
        "voc": VOC,
        "vocaug": VOCAug,
        "voc_test": VOC_test,  # for test set
    }[name]
