from utils.logger import Log

logger = Log(__name__).getlog()

def dataset_path(dataset_name):
    if dataset_name == "dataset_base":
        dataset_trian_path = "./data/mk_3k_long_0322_new_train.json"
        dataset_dev_path = "./data/mk_3k_long_0322_new_dev.json"
        dataset_test_path = "./data/mk_3k_long_0322_new_dev.json"
    elif dataset_name == "other":
        pass
    else:
        logger.info(f"dataset name is wrong")
        assert False

    return dataset_trian_path, dataset_dev_path, dataset_test_path