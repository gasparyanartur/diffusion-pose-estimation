import numpy as np

def joints_to_heatmap(joints, size=64, sigma=2):

    num_joints = joints.shape[0]

    heatmaps = np.zeros((num_joints, size, size))

    for j in range(num_joints):

        x, y = joints[j][:2]

        for i in range(size):
            for k in range(size):

                heatmaps[j,i,k] = np.exp(
                    -((i-y)**2 + (k-x)**2)/(2*sigma*sigma)
                )

    return heatmaps