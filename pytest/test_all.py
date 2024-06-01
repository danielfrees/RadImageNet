"""
Test pre-commit to make sure the model pipeline is functional. Should be able to train,
load, eval, etc. for one epoch.
"""


def test():
    # no tests for now
    # TODO: run a basic test which basically runs (main.py is in the parent dir,
    # this test_all script is in pytest/test_all)
    # python main.py --data_dir breast --database RadImageNet --backbone_model_name
    # ResNet50 --clf NonLinear  --structure freezeall --verbose --dropout_prob 0.5
    # --fc_hidden_size_ratio 0.5 --num_filters 8 --kernel_size 2 --epoch 1
    # --batch_size 512  --lr_decay_method cosine --amp
    # should just make sure this completes without raising errors

    # waiting to implement this until expt is complete
    return


if __name__ == "__main__":
    test()
