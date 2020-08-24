import torch
# from models.base import get_resnest

pth = './weights/efficientnet-b0/lightning_logs/version_0/checkpoints/epoch=28.ckpt'
#pth = './weights/efficientnet-b0test_epoch=28-val_loss=0.002397.ckpt'

state_dict = torch.load(pth)
# print(state_dict.keys())
# model = get_resnest()
# weights = state_dict['state_dict'].copy()
# for k in state_dict['state_dict'].keys():
#     nk = k[4:]
#     weights[nk] = weights.pop(k)

# model.load_state_dict(weights)
print(f"best score: {state_dict['checkpoint_callback_best_model_score']}")
print(1)