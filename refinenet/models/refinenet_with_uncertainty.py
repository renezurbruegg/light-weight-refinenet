from refinenet.models.resnet import rf_lw50, rf_lw101, rf_lw152
from refinenet.models.uncertainty_utils import UncertaintyModel


def get_uncertainty_net(size, num_classes, n_components=None, n_feature_for_uncertainty=128,
                        feature_layer="mflow_conv_g4_pool", imagenet=False, pretrained=True, covariance_type="tied",
                        reg_covar=1e-6, **kwargs):
  model = None
  if size == 50:
    model = rf_lw50(num_classes, imagenet=imagenet, pretrained=pretrained, **kwargs)
  elif size == 101:
    model = rf_lw101(num_classes, imagenet=imagenet, pretrained=pretrained, **kwargs)
  elif size == 152:
    model = rf_lw152(num_classes, imagenet=imagenet, pretrained=pretrained, **kwargs)

  if model is None:
    raise RuntimeError("Unsupported model size {}".format(size))

  return UncertaintyModel(model, feature_layer, n_feature_for_uncertainty=n_feature_for_uncertainty,
                          n_components=num_classes if n_components is None else n_components,
                          covariance_type=covariance_type, reg_covar=reg_covar)
