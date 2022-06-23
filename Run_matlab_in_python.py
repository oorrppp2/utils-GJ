# from oct2py import Oct2Py
# oc = Oct2Py()


# # script = "function y = myScript(x)\n" \
# #          "    y = x-5" \
# #          "end"

# script = "function [P_intersect, distances] = lineIntersect3D(PA,PB) \n"\
# "% Find intersection point of lines in 3D space, in the least squares sense. \n"\
# "% PA :          Nx3-matrix containing starting point of N lines \n"\
# "% PB :          Nx3-matrix containing end point of N lines \n"\
# "% P_Intersect : Best intersection point of the N lines, in least squares sense. \n"\
# "% distances   : Distances from intersection point to the input lines \n"\
# "% Anders Eikenes, 2012 \n"\
# " \n"\
# "Si = PB - PA; %N lines described as vectors \n"\
# "ni = Si ./ (sqrt(sum(Si.^2,2))*ones(1,3)); %Normalize vectors \n"\
# "nx = ni(:,1); ny = ni(:,2); nz = ni(:,3); \n"\
# "SXX = sum(nx.^2-1); \n"\
# "SYY = sum(ny.^2-1); \n"\
# "SZZ = sum(nz.^2-1); \n"\
# "SXY = sum(nx.*ny); \n"\
# "SXZ = sum(nx.*nz); \n"\
# "SYZ = sum(ny.*nz); \n"\
# "S = [SXX SXY SXZ;SXY SYY SYZ;SXZ SYZ SZZ]; \n"\
# "CX  = sum(PA(:,1).*(nx.^2-1) + PA(:,2).*(nx.*ny)  + PA(:,3).*(nx.*nz)); \n"\
# "CY  = sum(PA(:,1).*(nx.*ny)  + PA(:,2).*(ny.^2-1) + PA(:,3).*(ny.*nz)); \n"\
# "CZ  = sum(PA(:,1).*(nx.*nz)  + PA(:,2).*(ny.*nz)  + PA(:,3).*(nz.^2-1)); \n"\
# "C   = [CX;CY;CZ]; \n"\
# "P_intersect = (S\C)'; \n"\
# "if nargout>1 \n"\
# "    N = size(PA,1); \n"\
# "    distances=zeros(N,1); \n"\
# "    for i=1:N %This is faster: \n"\
# "        ui=(P_intersect-PA(i,:))*Si(i,:)'/(Si(i,:)*Si(i,:)'); \n"\
# "        distances(i)=norm(P_intersect-PA(i,:)-ui*Si(i,:)); \n"\
# "    end \n"\
# "    %for i=1:N %http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html: \n"\
# "    %    distances(i) = norm(cross(P_intersect-PA(i,:),P_intersect-PB(i,:))) / norm(Si(i,:)); \n"\
# "    %end \n"\
# "end \n"\
# "end "
# "end "

# with open("lineIntersect3D.m","w+") as f:
#     f.write(script)

PA = [[-14.2, 17, -1], [1, 1, 1], [2.3, 4.1, 9.8], [1,2,3]]
PB = [[1.3, 1.3, -10], [12.1, -17.2, 1.1], [19.2, 31.8, 3.5], [4,5,6]]
# PA = [[-14.2, 17, -1], [1, 1, 1]]
# PB = [[1.3, 1.3, -10], [12.1, -17.2, 1.1]]

# print(oc.lineIntersect3D(PA, PB))


"""
    Closest point of lines in 3D space
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

PA = torch.FloatTensor(PA)
PB = torch.FloatTensor(PB)

Si = PB - PA
ni = torch.sqrt(torch.sum(torch.pow(Si, 2), axis=1)).view(-1,1)
print(ni.shape)
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
print(C.shape)
print(S.shape)
P_intersect, _ = torch.lstsq(C, S)

# # print(C_)
# # print(S[:,0])
# C = torch.Tensor((1,3))
# C[0] = S[:,0] / C_
# print(C[0])
# # C[0]
# P_intersect = S/C
print(P_intersect.T)