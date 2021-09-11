import torch

from dataset.dataset import get_train_val_loader
from models.transformer import TransfomerModel, LSTMATTNModel
from config import Config


def main():

    config = Config(
        0,
        model_type="bert",
        seed=42,
        batch_size=8,
        accumulation_steps=4
    )

    # detect gpu
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # load data
    print("loading train val loader")
    train_loader, _ = get_train_val_loader(config)

    # mode testing
    print("testing model bert")
    model_bert = TransfomerModel(config).to(device)

    for idx, (cate_x, cont_x, mask, target) in enumerate(train_loader):

        print("category tensor shape {}".format(cate_x.shape))
        print("continuous tensor shape {}".format(cont_x.shape))
        print("mask tensor shape {}".format(mask.shape))
        print("target tensor shape {}".format(target.shape))

        cate_x = cate_x.to(device).long()
        cont_x = cont_x.to(device).float()
        mask = mask.to(device).float()

        pred = model_bert(cate_x, cont_x, mask)

        print("prediction shape {}".format(pred.shape))
        break

    # mode testing
    print("testing model lstm")
    model_lstm = LSTMATTNModel(config).to(device)

    for idx, (cate_x, cont_x, mask, target) in enumerate(train_loader):
        print("category tensor shape {}".format(cate_x.shape))
        print("continuous tensor shape {}".format(cont_x.shape))
        print("mask tensor shape {}".format(mask.shape))
        print("target tensor shape {}".format(target.shape))

        cate_x = cate_x.to(device).long()
        cont_x = cont_x.to(device).float()
        mask = mask.to(device).float()

        pred = model_lstm(cate_x, cont_x, mask)

        print("prediction shape {}".format(pred.shape))
        break

    return


if __name__ == "__main__":
    main()
