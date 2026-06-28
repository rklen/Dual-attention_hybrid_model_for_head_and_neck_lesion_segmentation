from monai.networks.nets import SwinUNETR

SwinUNETR = SwinUNETR(
    in_channels=1,     
    out_channels=1,
    feature_size=24,
    depths=(2,2,2,2),
    num_heads=(3,6,12,24),
    spatial_dims=2,
 
)