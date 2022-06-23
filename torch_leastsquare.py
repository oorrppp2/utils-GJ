import torch


class LeastSquares:
    def __init__(self):
        pass
    
    def lstq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        # Assuming A to be full column rank
        cols = A.shape[1]
        print (torch.matrix_rank(A))
        if cols == torch.matrix_rank(A):
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
            Y_dash = A.permute(1, 0) @ Y
            x = self.lstq(A_dash, Y_dash)
        return x

PA = [[-14.2, 17, -1], [1, 1, 1], [2.3, 4.1, 9.8], [1,2,3]]
PB = [[1.3, 1.3, -10], [12.1, -17.2, 1.1], [19.2, 31.8, 3.5], [4,5,6]]


PA = torch.tensor(PA)
PB = torch.tensor(PB)
print(PA.shape)

Si = PB - PA
ni = torch.sqrt(torch.sum(torch.pow(Si, 2), axis=1)).view(-1,1)
ni = Si / ni.repeat(1,3)

# ni = torch.sum(torch.pow(Si, 2), axis=1)
# print(ni)
nx = ni[:,0]
ny = ni[:,1]
nz = ni[:,2]
SXX = torch.sum(torch.pow(nx, 2)-1)
SYY = torch.sum(torch.pow(ny, 2)-1)
SZZ = torch.sum(torch.pow(nz, 2)-1)
SXY = torch.sum(nx*ny)
SXZ = torch.sum(nx*nz)
SYZ = torch.sum(ny*nz)
S = torch.Tensor([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])
CX = torch.sum(PA[:,0]*(torch.pow(nx, 2)-1) + PA[:,1]*(nx*ny) + PA[:,2]*(nx*nz))
CY = torch.sum(PA[:,0]*(nx*ny) + PA[:,1]*(torch.pow(ny, 2)-1) + PA[:,2]*(ny*nz))
CZ = torch.sum(PA[:,0]*(nx*nz) + PA[:,1]*(ny*nz) + PA[:,2]*(torch.pow(nz, 2)-1))
C = torch.Tensor([[CX], [CY], [CZ]])
P_intersect, _ = torch.lstsq(C, S)

# # print(C_)
# # print(S[:,0])
# C = torch.Tensor((1,3))
# C[0] = S[:,0] / C_
# print(C[0])
# # C[0]
# P_intersect = S/C
print(P_intersect.T)


A = torch.tensor(S)
B = torch.tensor(C)
# A = torch.tensor([[1., 1, 1],
#                       [2, 3, 3],
#                       [3, 5, 5],
#                       [4, 2, 2],
#                       [5, 4.0, 4]])
# B = torch.tensor([[-10.],
#                       [ 12.0],
#                       [ 14],
#                       [ 16],
#                       [ 18]])
A.requires_grad = True
ls = LeastSquares()
x = ls.lstq(A, B, 0.010)
print (x)
# l = torch.mean(x)
# print(l)
# l.backward()
