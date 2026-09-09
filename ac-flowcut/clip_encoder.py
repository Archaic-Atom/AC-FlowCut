import torch
import torch.nn as nn

class CLIPVisionTower_FlowCut(nn.Module):

    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            image_features = []
            keep_indices = []
            token_importance = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                keep_index = self.vision_tower.vision_model.encoder.last_keep_indices
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
                keep_indices.append(keep_index)
                token_importance.append(
                    getattr(self.vision_tower.vision_model.encoder, "last_token_importance", None)
                )
            return image_features, keep_indices, token_importance
        else:
            outputs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            keep_indices = self.vision_tower.vision_model.encoder.last_keep_indices
            hidden_states = self.feature_select(outputs)
            token_importance = getattr(
                self.vision_tower.vision_model.encoder, "last_token_importance", None
            )
            # keep_indices = outputs.all_keep_indices

        return hidden_states.to(images.dtype), keep_indices, token_importance
