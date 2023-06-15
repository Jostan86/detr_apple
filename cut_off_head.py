import torch

# # Get pretrained weights
# checkpoint = torch.hub.load_state_dict_from_url(
#             url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
#             map_location='cpu',
#             check_hash=True)
checkpoint = torch.load('./weights/detr-r101-dc5-a2e86def.pth', map_location='cpu')

# Remove class weights
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]

# Save
torch.save(checkpoint,
           'weights/detr-r101-dc5_no-class-head.pth')