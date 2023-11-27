import os
import json
import matplotlib.pyplot as plt

json_file_dir = './tasks/'
train_sample_num = 48000
valid_sample_num = 2000

def get_json_file_path(json_file_dir):
    json_file_path_list = []
    for root, dirs, files in os.walk(json_file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                json_file_path_list.append(os.path.join(root, file))
    return json_file_path_list


def plot_loss_acc(json_file_path):
    with open(json_file_path, 'r') as f:
        log = json.load(f)

    train_loss = log['train_loss']
    train_acc = log['train_acc']

    valid_loss = log['valid_loss']
    valid_acc = log['valid_acc']

    train_loss = [float(i) / train_sample_num for i in train_loss]
    valid_loss = [float(i) / valid_sample_num for i in valid_loss]

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(json_file_path[:-5] + '.png')
    plt.show()


for file in get_json_file_path(json_file_dir):
    try:
        plot_loss_acc(file)
    except Exception as e:
        print(file, e)
        continue
