import torch

COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": (0,255,0) #green
             },
    "index": {"ids": [0, 5, 6, 7, 8], "color": (0,255,255) #cyan
             },
    "middle": {"ids": [0, 9, 10, 11, 12], "color": (0,0,255) #blue
              },
    "ring": {"ids": [0, 13, 14, 15, 16], "color": (255,0,255) #pink
            },
    "little": {"ids": [0, 17, 18, 19, 20], "color": (255,0,0) #red
              },
}

config = {
    "data_dir": "Dataset/Freihand",
    "n_sample": 4,
    "epochs": 40,
    "n_joints": 21,
    "batch_size": 128,
    "learning_rate": 0.001,
    "decay": 0.0001,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}