import os

class GlobalConfigs:
    """ base architecture configurations """
    seq_len = 1  # input timesteps
    lidar_seq_len = 1
    # Conv Encoder
    img_vert_anchors = 5
    img_horz_anchors = 20 + 2
    lidar_vert_anchors = 8
    lidar_horz_anchors = 8

    img_anchors = img_vert_anchors * img_horz_anchors
    lidar_anchors = lidar_vert_anchors * lidar_horz_anchors

    detailed_losses = ['loss_wp', 'loss_bev', 'loss_depth', 'loss_semantic', 'loss_center_heatmap', 'loss_wh',
                       'loss_offset', 'loss_yaw_class', 'loss_yaw_res', 'loss_velocity', 'loss_brake']
    detailed_losses_weights = [1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]

    perception_output_features = 512  # Number of features outputted by the perception branch.用了
    bev_features_chanels = 64  # Number of channels for the BEV feature pyramid
    bev_upsample_factor = 2

    deconv_channel_num_1 = 128  # Number of channels at the first deconvolution layer
    deconv_channel_num_2 = 64  # Number of channels at the second deconvolution layer
    deconv_channel_num_3 = 32  # Number of channels at the third deconvolution layer

    deconv_scale_factor_1 = 8  # Scale factor, of how much the grid size will be interpolated after the first layer
    deconv_scale_factor_2 = 4  # Scale factor, of how much the grid size will be interpolated after the second layer

    gps_buffer_max_len = 100  # Number of past gps measurements that we track.
    carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
    carla_fps = 20  # Simulator Frames per second
    iou_treshold_nms = 0.2  # Iou threshold used for Non Maximum suppression on the Bounding Box predictions for the ensembles
    steer_damping = 0.5  # Damping factor by which the steering will be multiplied when braking
    route_planner_min_distance = 7.5
    route_planner_max_distance = 50.0
    action_repeat = 2  # Number of times we repeat the networks action. It's 2 because the LiDAR operates at half the frame rate of the simulation
    stuck_threshold = 1100 / action_repeat  # Number of frames after which the creep controller starts triggering. Divided by
    creep_duration = 30 / action_repeat  # Number of frames we will creep forward

    # GPT Encoder
    n_embd = 512
    block_exp = 4
    #n_layer = 1     #8  4
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0 # Mean of the normal distribution with which the linear layers in the GPT are initialized
    gpt_linear_layer_init_std  = 0.02 # Std  of the normal distribution with which the linear layers in the GPT are initialized
    gpt_layer_norm_init_weight = 1.0 # Initial weight of the layer norms in the gpt.


#training:
    lr=0.01
    weight_decay=0.0000001
    bs=16
    scheduler= 'steplr'
    gamma= 0.1
    step_size=15
    rebalancing_fake=0.3
    rebalancing_real=1
    frames_per_video=30 # Equidistant frames

#model:
    image_size=224
    num_classes=1