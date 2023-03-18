import torch
import models_vit

# parameters
device = 'cuda'
model_name = 'vit_base_patch16'

# AffectNet model
# The head includes 7 dims of expression classes and 2 dims of Valence+Arousal.
# e.g.
'''
outputs = model(inputs)
expression = outputs[..., :7]
valence = outputs[..., 7]
arousal = outputs[..., 8]
'''
num_classes = 7 + 2
ckpt_path = '/project/mbw/mae/models.tmp/AffectNet/9001/model-20.pth'

# ABAW5.EXPR model
# num_classes = 8
# ckpt_path = './ABAW5.EXPR/12303/model-10.pth'

# create model
model = getattr(models_vit, model_name)(
    global_pool=True,
    num_classes=num_classes,
    drop_path_rate=0.1,
    img_size=224,
)

# load pre-trained weights
print(f"Load pre-trained checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')
checkpoint_model = checkpoint['model']
model.load_state_dict(checkpoint_model, strict=False)
model.to(device)

inputs_dummpy = torch.ones((8,3,224,224)).cuda()
model.eval()
out = model(inputs_dummpy, ret_feature=False)
print(out)