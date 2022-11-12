import torch
import torchvision
from modeling_t5 import VLT5


def build_model():
    cnn = getattr(torchvision.models, 'resnet101')(pretrained=True)
    layers = [cnn.conv1,
              cnn.bn1,
              cnn.relu,
              cnn.maxpool]
    for i in range(4):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    model.cuda()
    model.eval()
    return model


class VLT5Geo(VLT5):
    def __init__(self, config):
        super().__init__(config)
        self.resnet = build_model()

    def train_step(self, batch):
        device = next(self.parameters()).device
        image = batch['image_list'].to(device)

        with torch.no_grad():
            vis_feats = self.resnet(image)

        N, C, H, W = vis_feats.shape
        vis_feats = vis_feats.reshape(N, C, -1).permute(0, 2, 1)

        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        vis_attention_mask = batch['vis_attention_mask'].to(device)

        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            labels=lm_labels,
            reduce_loss=True,
            return_dict=True
        )

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        image = batch['image_list'].to(device)

        with torch.no_grad():
            vis_feats = self.resnet(image)

        N, C, H, W = vis_feats.shape
        vis_feats = vis_feats.reshape(N, C, -1).permute(0, 2, 1)

        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        vis_attention_mask = batch['vis_attention_mask'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result
