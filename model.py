import math
from os.path import join, exists
from os import makedirs
from torch import nn, optim, save, load
from torch.optim.lr_scheduler import LambdaLR


class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(2, 8)
            self.fc2 = nn.Linear(8, 8)
            self.fc3 = nn.Linear(8, 1)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            
            return x


def save_checkpoint(model, optimizer, checkpoint_dir, epoch, f1_score):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_epoch{:09d}_f1-{}.pth".format(epoch, str(f1_score)))
    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir, exist_ok=True)
    optimizer_state = optimizer.state_dict()
    save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    checkpoint = load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    return model


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform(m.weight, a=-0.1, b=0.1)


def get_model(device):
    net = Net().to(device)
    #net.apply(init_weights)
    loss = nn.BCELoss()
    sgd = optim.SGD(net.parameters(), lr=0.1)
    return net, loss, sgd


def create_loss_fn(device):
    criterion = nn.BCELoss()
    return criterion.to(device)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
