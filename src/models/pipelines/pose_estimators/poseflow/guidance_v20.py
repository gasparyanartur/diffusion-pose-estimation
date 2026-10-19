import torch


# CrowdPose shoulders, elbows, wrists, hips, knees, ankles -> Human3.6M.
COMMON_JOINTS = (11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3)
GFPOSE_SCALE = 30.


def project_pose3d(poses, *, joints=14):
    if poses.shape[-2:] != (17, 3) or not 1 <= joints <= 14:
        raise ValueError('V20 requires 17 teacher joints and 1-14 CrowdPose joints')
    return (poses[..., list(COMMON_JOINTS[:joints]), :2]/GFPOSE_SCALE+1)/2


def projected_pose_loss(poses, targets, labeled):
    points = project_pose3d(poses, joints=targets.shape[-2])
    common = points.shape[-2]
    return ((points-targets[..., :common, :]).abs()
            *labeled[..., :common, None]).sum()/max(poses.shape[0], 1)


def gfpose_marginal(times):
    if not torch.isfinite(times).all() or ((times <= 0) | (times > 1)).any():
        raise ValueError('GFPose times must be finite and in (0,1]')
    log_mean = -.25*times.square()*19.9-.5*times*.1
    return torch.exp(log_mean), 1-torch.exp(2*log_mean)


def score_distillation_loss(poses, teacher, times, noise):
    if poses.shape != noise.shape or times.shape != poses.shape[:1]:
        raise ValueError('Teacher poses, noise and times must agree')
    alpha, sigma = gfpose_marginal(times)
    with torch.no_grad(), torch.autocast(device_type=poses.device.type, enabled=False):
        noisy = alpha[:, None, None]*poses.detach().float()+sigma[:, None, None]*noise.float()
        score = teacher.score(noisy, times.float())
        force = -sigma[:, None, None]*score-noise.float()
        if not torch.isfinite(force).all():
            raise ValueError('Nonfinite GFPose guidance')
    # w_KL=sigma/alpha: the detached noise residual is the pose-space gradient.
    loss = (poses.float()*force).mean()
    metrics = {'prior/force_rms': force.square().mean().sqrt(),
               'prior/score_rms': score.square().mean().sqrt(),
               'prior/time_mean': times.mean()}
    return loss, metrics
