from monai.networks.nets import SwinUNETR


def create_model(config: dict):

    if config['arch'] == 'swin_unetr':
        model = SwinUNETR(
            img_size=(config['M_rc_size'], config['M_rc_size'], config['M_rc_size']),
            in_channels=config['M_in_channels'],
            out_channels= config['M_out_channels'],
            feature_size=config['M_feature_size'],
            use_checkpoint=config['M_use_checkpoint']
        )

    return model

if __name__=='__main__':
    config = {
        'arch': 'swin_unetr',
        'M_rc_size': 96,
        'M_in_channels': 2,
        'M_out_channels': 2,
        'M_feature_size': 48,
        'M_use_checkpoint': False
    }

    print(type(create_model(config)))
