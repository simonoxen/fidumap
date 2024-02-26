import torch

class AffineAligner:
    def __init__(self):
        pass

    def alignKeypoints(self, p1, p2):
        """
        Kabsch algorithm
        p1, p2: points, shape [n_batch, n_points, dim]
        returns: transform, shape [n_batch, dim+1, dim+1]
        """
        dim = p1.shape[2]

        # Calculate centroids
        p1_c = torch.mean(p1, 1, keepdim=True)
        p2_c = torch.mean(p2, 1, keepdim=True)

        # Subtract centroids
        q1 = p1 - p1_c
        q2 = p2 - p2_c

        # Calculate covariance matrix
        H = torch.bmm(q1.transpose(1, 2), q2)

        # Calculate singular value decomposition (SVD)
        U, _, V_t = torch.linalg.svd(H) # the SVD of linalg gives you Vt
        V = V_t.transpose(1, 2)
        U_t = U.transpose(1, 2)

        # ensure right-handedness
        M = torch.eye(dim, device=U.device).reshape((1, dim, dim)).repeat(p1.shape[0], 1, 1)
        M[:,-1,-1] = torch.sign(torch.linalg.det(torch.bmm(V, U))).squeeze()

        R = torch.bmm(V, torch.bmm(M, U_t))
        T = p2_c.transpose(1, 2) - torch.bmm(R, p1_c.transpose(1, 2))

        output = torch.eye(dim+1, device=U.device).reshape((1, dim+1, dim+1)).repeat(p1.shape[0], 1, 1)
        output[:, :dim, :dim] = R.squeeze()
        output[:, :dim, dim] = T.squeeze()
        return output