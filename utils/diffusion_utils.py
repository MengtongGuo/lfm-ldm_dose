import pickle
import glob
import os
import torch


def load_latents(latent_path):
    r"""
    Simple utility to save latents to speed up ldm training
    :param latent_path:
    :return:
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps


def drop_text_condition(text_embed, im, empty_text_embed, text_drop_prob):
    if text_drop_prob > 0:
        text_drop_mask = torch.zeros((im.shape[0]), device=im.device).float().uniform_(0,
                                                                                       1) < text_drop_prob
        assert empty_text_embed is not None, ("Text Conditioning required as well as"
                                        " text dropping but empty text representation not created")
        text_embed[text_drop_mask, :, :] = empty_text_embed[0]
    return text_embed


def drop_image_condition(image_condition, im, im_drop_prob):
    if im_drop_prob > 0:
        im_drop_mask = torch.zeros((im.shape[0], 1, 1, 1), device=im.device).float().uniform_(0,
                                                                                        1) > im_drop_prob
        return image_condition * im_drop_mask
    else:
        return image_condition


def drop_class_condition(class_condition, class_drop_prob, im):
    if class_drop_prob > 0:
        class_drop_mask = torch.zeros((im.shape[0], 1), device=class_condition.device).float().uniform_(0,
                                                                                           1) > class_drop_prob
        return class_condition * class_drop_mask
    else:
        return class_condition


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def clip_gradients(model, max_norm):
    """
    Clip gradients of the model parameters to prevent exploding gradients.
    Args:
        model: The model whose gradients will be clipped.
        max_norm: Maximum allowable gradient norm.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
