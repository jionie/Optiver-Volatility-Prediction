from dataset.dataset import get_train_val_loader, get_test_loader
from config import Config


def main():

    config = Config(
        0,
        model_type="bert",
        seed=42,
        batch_size=8,
        accumulation_steps=4
    )

    print("loading train val loader")
    train_loader, val_loader = get_train_val_loader(config)

    print("testing train loader")
    for idx, (cate_x, cont_x, mask, target) in enumerate(train_loader):

        print("category tensor shape {}".format(cate_x.shape))
        print("continuous tensor shape {}".format(cont_x.shape))
        print("mask tensor shape {}".format(mask.shape))
        print("target tensor shape {}".format(target.shape))

        break

    print("testing val loader")
    for idx, (cate_x, cont_x, mask, target) in enumerate(val_loader):

        print("category data tensor shape {}".format(cate_x.shape))
        print("continuous data tensor shape {}".format(cont_x.shape))
        print("mask tensor shape {}".format(mask.shape))
        print("target data tensor shape {}".format(target.shape))

        break

    print("loading test loader")
    test_loader = get_test_loader(config)

    for idx, (cate_x, cont_x, mask, target) in enumerate(test_loader):

        print("category data tensor shape {}".format(cate_x.shape))
        print("continuous data tensor shape {}".format(cont_x.shape))
        print("mask tensor shape {}".format(mask.shape))
        print("target data tensor {}".format(target))

        break

    return


if __name__ == "__main__":
    main()
