import os
import numpy as np
import math
from common import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from DataLoader import *
import time
# decirption: 运行非结构化的有限体积法求解Euler 方程
# autor: JLX
# date: 2024-6-6
class Airfoil:
    def __init__(self,data_loader:SU2Loader, far_cond, restart=False, restart_path='./flow.npz', restart_iter=None, ouput_path='./output'):
        print('airfoil init...')
        # geometry
        self.Node = data_loader.Nodes      # 存储网格节点坐标
        self.Element = data_loader.Element  # 控制体
        self.Eedge = data_loader.Edge       # 边
        #
        self.NNode = len(self.Node)
        self.NElem = len(self.Element)      # 控制体数量
        self.NFace = len(self.Eedge)        # 边数量
        # flow
        if restart:                    # 之前流场条件、信息继续算
            data = np.load(restart_path)
            self.free = data['free']        # 远场边界条件
            self.U = data['U']              # 流场信息
            if restart_iter == None:
                self.iter = data['iter']        # 步数信息
            else:
                self.iter = restart_iter
        else:                               # 新的流场条件
            self.iter = 0
            self.far_cond = far_cond
            p, r, Ma, alpha = far_cond
            c = math.sqrt(gamma*p/r)
            u = Ma*c*np.cos(np.radians(alpha))
            v = Ma*c*np.sin(np.radians(alpha))
            e = p/(r*(gamma-1.0)) + 0.5*(u**2 + v**2)
            print('rho: ', r, 'Vel:', math.sqrt(u ** 2 + v ** 2))
            self.free = np.array([1.0, u/math.sqrt(u**2+v**2), v/math.sqrt(u**2+v**2), r*e/(u**2+v**2)]) # 自由流
            # self.free = np.array([r, r*u, r*v, r*e])
            p = 101325
            R = 287.87
            T = 273.15
            alpha = 1.25
            Ma = 0.8
            rho = p/(R*T)
            c = math.sqrt(gamma*p/rho)
            u = Ma*c*np.cos(np.radians(alpha))
            v = Ma*c*np.sin(np.radians(alpha))
            print(rho, u, v, p)
            self.free = np.array([rho, rho*u, rho*v, rho*(p/((gamma-1.0)*rho) + 0.5*(u**2+v**2))], dtype=np.float64)
            #
            print(self.free)
            self.U = np.tile(self.free, (self.NElem, 1))
        # resid
        self.res = np.zeros((self.NElem, 4))
        # timestep
        self.tstep = np.zeros((self.NElem, 1), dtype=np.float64)
        # adjoint
        self.Adjoint = False
        # area
        self.area = np.zeros((self.NElem, 1), dtype=np.float64)
        self.metrics()
        #
        self.output_path = ouput_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not restart:
            self.output(self.output_path + '/0',restart_path=self.output_path+'/'+'flow')
    def metrics(self):
        """计算网格大小"""
        def tri_area(p1, p2, p3):
            v1 = p2 - p1
            v2 = p3 - p1
            return 0.5*np.abs(np.cross(v1, v2))
        for i in range(self.NElem):
            nodes = []
            for j in range(len(self.Element[i])):
                nodes.append(self.Node[self.Element[i].data[j]])
            nodes = np.array(nodes)
            center = np.sum(nodes, axis=0)/len(nodes)
            area = 0.0
            for j in range(len(nodes)):
                k1 = j
                k2 = 0 if j ==(len(nodes) - 1) else j + 1
                area += tri_area(center, nodes[k1], nodes[k2])
            self.area[i] = area
    def TimeStep(self):
        """计算局部时间步长"""
        for i in range(self.NElem):
            X = np.array([self.Node[idx] for idx in self.Element[i].data])
            U = self.U[i]
            self.tstep[i] = self._time_cell(X, U, self.area[i])

    def _time_cell(self, X, U, area):
        # X: 单元节点
        # U: 单元守恒变量
        Lambda = .0
        r = U[0]
        u = U[1]/ U[0]
        v =  U[2]/ U[0]
        p = (gamma - 1.0) * (U[3] - 0.5 * r* (u** 2 + v ** 2))
        c = math.sqrt(gamma * p /r)
        for i in range(X.shape[0]):
            dx = (X[i + 1][0] - X[i][0]) if (i != X.shape[0] - 1) else (X[0][0] - X[-1][0])
            dy = (X[i + 1][1] - X[i][1]) if (i != X.shape[0] - 1) else (X[0][1] - X[-1][1])
            Lambda += abs(u * dy - v * dx ) + c * math.sqrt(dx ** 2 + dy ** 2)
        dt = CFL*area/Lambda
        return dt
    def index(self, idx=[]):
        """记录索引"""
        self.idx = idx
    def _vanleer_flux(self, X, UL, UR):
        """van leer 通量分裂"""
        # X: 边节点坐标
        # UL: 左状态
        # UR: 右状态
        # 几何变量
        dx = X[1, 0] - X[0, 0]
        dy = X[1, 1] - X[0, 1]
        S = math.sqrt(dx ** 2 + dy ** 2)  # 面的大小
        nx = dy / S  # Note
        ny = -dx / S
        # 左原始变量
        rL = UL[0]
        uL = UL[1] / UL[0]
        vL = UL[2] / UL[0]
        pL = (gamma - 1.0) * (UL[3] - 0.5 * rL * (uL ** 2 + vL ** 2))
        HL = UL[3] / UL[0] + pL / rL
        # 右原始变量
        rR = UR[0]
        uR = UR[1] / UR[0]
        vR = UR[2] / UR[0]
        pR = (gamma - 1.0) * (UR[3] - 0.5 * rR * (uR ** 2 + vR ** 2))
        HR = UR[3] / UR[0] + pR / rR
        #
        cL = math.sqrt(gamma*pL/rL)
        cR = math.sqrt(gamma * pR / rR)
        V_L = uL*nx + vL*ny
        V_R = uR*nx+vR*ny
        ML = V_L/cL
        MR = V_R/cR
        # print(ML, MR)
        if ML >=1:
            MLp = ML
        elif ML <= -1:
            MLp = 0
        else:
            MLp = 0.25*(ML+1)**2
        if MR >=1:
            MRn = 0
        elif MR <= -1:
            MRn = MR
        else:
            MRn = -0.25*(MR-1.0)**2
        Mn = MLp + MRn
        if Mn>=1.0 :
            F1 = rL * V_L
            F2 = rL * uL * V_L + pL * nx
            F3 = rL * vL * V_L + pL * ny
            F4 = rL * HL * V_L
        elif Mn <= -1.0 :
            F1 = rR * V_R
            F2 = rR * uR * V_R + pR * nx
            F3 = rR * vR * V_R + pR * ny
            F4 = rR * HR * V_R
        else:
            # F+
            factor_p = rL*cL*0.25*(ML+1)**2
            FL1 = factor_p
            FL2 = factor_p*(nx*(-V_L+2.0*cL)/gamma+uL)
            FL3 = factor_p * (ny * (-V_L + 2.0 * cL) / gamma + vL)
            FL4 = factor_p*0.5*(((gamma-1.0)*V_L+2*cL)**2/(gamma**2-1.0) + uL**2 + vL**2 - V_L**2)
            # F-
            factor_n = -rR * cR * 0.25 * (MR - 1) ** 2
            FR1 = factor_n
            FR2 = factor_n * (nx * (-V_R - 2.0 * cR) / gamma + uR)
            FR3 = factor_n * (ny * (-V_R - 2.0 * cR) / gamma + vR)
            FR4 = factor_n * 0.5 * (
                        ((gamma - 1.0) * V_R - 2 * cR) ** 2 / (gamma ** 2 - 1.0) + uR ** 2 + vR ** 2 - V_R ** 2)
            F1 = FL1 + FR1
            F2 = FL2 + FR2
            F3 = FL3 + FR3
            F4 = FL4 + FR4
        #
        F1 *= S
        F2 *= S
        F3 *= S
        F4 *= S
        self.res[self.idx[0], 0] += F1
        self.res[self.idx[0], 1] += F2
        self.res[self.idx[0], 2] += F3
        self.res[self.idx[0], 3] += F4
        self.res[self.idx[1], 0] -= F1
        self.res[self.idx[1], 1] -= F2
        self.res[self.idx[1], 2] -= F3
        self.res[self.idx[1], 3] -= F4


    def _roe_flux(self, X, UL, UR):
        """Roe 通量分裂"""
        # X: 边节点坐标
        # UL: 左状态
        # UR: 右状态
        if self.Adjoint == False:
            pass
        else:
            pass
        # 几何变量
        dx = X[1,0] - X[0,0]
        dy = X[1,1] - X[0, 1]
        S = math.sqrt(dx**2 + dy**2) # 面的大小
        nx = dy/S      # Note: 拓扑网格内为顺时针
        ny = -dx/S
        # 左原始变量
        rL = UL[0]
        uL = UL[1] / UL[0]
        vL = UL[2] / UL[0]
        pL = (gamma - 1.0) * (UL[3] - 0.5 * rL * (uL ** 2 + vL ** 2))
        HL = UL[3]/UL[0] + pL/rL
        # 右原始变量
        rR = UR[0]
        uR = UR[1] / UR[0]
        vR = UR[2] / UR[0]
        pR = (gamma - 1.0) * (UR[3] - 0.5*rR*(uR**2 + vR**2))
        HR = UR[3]/UR[0] + pR/rR
        #  Roe 平均
        ra = math.sqrt(rL*rR)
        ua = (uL*math.sqrt(rL) + uR*math.sqrt(rR))/(math.sqrt(rL) + math.sqrt(rR))
        va = (vL * math.sqrt(rL) + vR * math.sqrt(rR)) / (math.sqrt(rL) + math.sqrt(rR))
        Ha = (HL*math.sqrt(rL) + HR*math.sqrt(rR))/(math.sqrt(rL) + math.sqrt(rR))
        qa = math.sqrt(ua**2 + va**2)
        ca = math.sqrt((gamma-1.0)*(Ha - 0.5*qa**2))
        Va = ua*nx + va*ny
        # FL
        V_L = nx*uL + ny*vL
        FL1 = rL*V_L
        FL2 = rL*uL*V_L + pL*nx
        FL3 = rL*vL*V_L + pL*ny
        FL4 = rL*HL*V_L
        # FR
        V_R = nx*uR + ny*vR
        FR1 = rR*V_R
        FR2 = rR*uR*V_R + pR*nx
        FR3 = rR*vR*V_R + pR*ny
        FR4 = rR*HR*V_R
        # DF1
        dp = pR - pL
        dV = V_R - V_L
        factor1 = abs(Va-ca)*0.5*(dp-ra*ca*dV)/(ca*ca)
        DF1_1 = factor1
        DF1_2 = factor1*ua-ca*nx
        DF1_3 = factor1*va-ca*ny
        DF1_4 = factor1*(Ha - ca*Va)
        # DF2,3,4
        dr = rR -rL
        du = uR - uL
        dv = vR - vL
        factor2 = dr - dp/(ca*ca)
        DF24_1 = abs(Va)*factor2
        DF24_2 = abs(Va)*(factor2*ua + ra*(du-dV*nx))
        DF24_3 = abs(Va)*(factor2*va + ra*(dv-dV*ny))
        DF24_4 = abs(Va)*(factor2*0.5*qa*qa + ra*(ua*du+va*dv-Va*dV))
        # DF5
        factor3 = abs(Va + ca)*0.5*(dp+ra*ca*dV)/(ca*ca)
        DF5_1 = factor3
        DF5_2 = factor3*(ua+ca*nx)
        DF5_3 = factor3*(va+ca*ny)
        DF5_4 = factor3*(Ha + ca*Va)
        # flux
        F1 = 0.5 * (FL1 + FR1) - (DF1_1 + DF24_1 + DF5_1)
        F2 = 0.5 * (FL2 + FR2) - (DF1_2 + DF24_2 + DF5_2)
        F3 = 0.5 * (FL3 + FR3) - (DF1_3 + DF24_3 + DF5_3)
        F4 = 0.5 * (FL4 + FR4) - (DF1_4 + DF24_4 + DF5_4)
        #
        F1 *= S
        F2 *= S
        F3 *= S
        F4 *= S
        #
        self.res[self.idx[0], 0] += F1
        self.res[self.idx[0], 1] += F2
        self.res[self.idx[0], 2] += F3
        self.res[self.idx[0], 3] += F4
        self.res[self.idx[1], 0] -= F1
        self.res[self.idx[1], 1] -= F2
        self.res[self.idx[1], 2] -= F3
        self.res[self.idx[1], 3] -= F4

    def _wall_flux(self, X, U):
        r = U[0]
        u = U[1] / U[0]
        v = U[2] / U[0]
        p = (gamma - 1.0) * (U[3] - 0.5 * r * (u ** 2 + v ** 2))
        rb = r
        pb = p
        dx = X[1, 0] - X[0, 0]
        dy = X[1, 1] - X[0, 1]
        S = math.sqrt(dx ** 2 + dy ** 2)
        nx = dy / S  # Note: 拓扑网格内为顺时针
        ny = -dx / S
        V = u*nx + v*ny
        ub = u - V*nx
        vb = v - V*ny
        #
        rL = r
        uL = u
        vL = v
        pL = p
        # 右原始变量
        rR = rb
        uR = ub
        vR = vb
        pR = pb
        #
        cL = math.sqrt(gamma * pL / rL)
        cR = math.sqrt(gamma * pR / rR)
        V_L = uL * nx + vL * ny
        V_R = uR * nx + vR * ny
        ML = V_L / cL
        MR = V_R / cR
        # print(ML, MR)
        # F+
        factor_p = rL * cL * 0.25 * (ML + 1) ** 2
        FL1 = factor_p
        FL2 = factor_p * (nx * (-V_L + 2.0 * cL) / gamma + uL)
        FL3 = factor_p * (ny * (-V_L + 2.0 * cL) / gamma + vL)
        FL4 = factor_p * 0.5 * (
                        ((gamma - 1.0) * V_L + 2 * cL) ** 2 / (gamma ** 2 - 1.0) + uL ** 2 + vL ** 2 - V_L ** 2)
        # F-
        factor_n = -rR * cR * 0.25 * (MR - 1) ** 2
        FR1 = factor_n
        FR2 = factor_n * (nx * (-V_R - 2.0 * cR) / gamma + uR)
        FR3 = factor_n * (ny * (-V_R - 2.0 * cR) / gamma + vR)
        FR4 = factor_n * 0.5 * (
                    ((gamma - 1.0) * V_R - 2 * cR) ** 2 / (gamma ** 2 - 1.0) + uR ** 2 + vR ** 2 - V_R ** 2)
        F1 = FL1 + FR1
        F2 = FL2 + FR2
        F3 = FL3 + FR3
        F4 = FL4 + FR4
        #
        F1 *= S
        F2 *= S
        F3 *= S
        F4 *= S
        #
        self.res[self.idx[0], 0] += F1
        self.res[self.idx[0], 1] += F2
        self.res[self.idx[0], 2] += F3
        self.res[self.idx[0], 3] += F4

    def Residual(self):
        """计算残差"""
        self.res.fill(.0)  # 残差清零
        # 计算残差
        for i in range(self.NFace):
            X = [self.Node[self.Eedge[i].node[0]], self.Node[self.Eedge[i].node[1]]]
            X = np.array(X)
            if self.Eedge[i].type == EdgeType.INTER:  # 内边
                self.index([self.Eedge[i].own, self.Eedge[i].neb])
                U = [self.U[self.Eedge[i].own], self.U[self.Eedge[i].neb]]
                if FLUX_SCHEME == 'ROE':
                    self._roe_flux(X, U[0], U[1])
                elif FLUX_SCHEME == 'VAN_LEER':
                    self._vanleer_flux(X, U[0], U[1])
                else:
                    raise RuntimeError('not support flux scheme!')
            elif self.Eedge[i].type == EdgeType.WALL:  # 壁面
                self.index([self.Eedge[i].own])
                U = self.U[self.Eedge[i].own]
                self._wall_flux(X, U)
            elif self.Eedge[i].type == EdgeType.FAR:  # 远场
                # 所有边通量计算完后才能更新，不然会将边界单元的其他边的通量代入
                continue
            else:
                raise RuntimeError('Error boundary condition!')
    def Boundary(self):
        """处理边界"""
        # 处理远场边界
        for i in range(self.NFace):
            if self.Eedge[i].type == EdgeType.FAR:  # 远场
                ncell = self.Eedge[i].own
                self.U[ncell] = self.free
                self.res[ncell].fill(.0)
    def solve(self):
        """求解"""
        print('euler 2D solve...')
        while self.iter < ITER:
            t1 = time.time()
            U0 = self.U.copy()      #
            # 计算时间步长
            if GLOBAL_TIME_STEP == True:
                tstep = TIME_STEP
            else:
                self.TimeStep()
                tstep = np.min(self.tstep)
            for k in range(len(RK_COEF)):
                self.Residual()     # 计算残差
                self.Boundary()     # 远场边界残差清零
                self.U = U0 - RK_COEF[k]*tstep*self.res/self.area
                self.Boundary()     # 远场边界残差清零
            self.iter += 1
            if self.iter%WRT_ITER == 0:
                self.output(self.output_path+'/'+str(self.iter), restart_path=self.output_path+'/'+'flow')
            max_rho_error = np.max(np.abs(self.res[:,0]))
            max_ru_error = np.max(np.abs(self.res[:,1]))
            max_rv_error = np.max(np.abs(self.res[:, 2]))
            max_re_error = np.max(np.abs(self.res[:, 3]))
            print('iter: %d, rms_rho: %.10f, rms_ru: %.10f, rms_rv: %.10f, rms_re: %.10f, time:%.4e' %(self.iter, max_rho_error, max_ru_error, max_rv_error, max_re_error, time.time()-t1))
            print('max_rho_idx: %5d, max_ru_idx: %5d, max_rv_idx: %5d, max_re_idx: %5d' %(np.argmax(max_rho_error), np.argmax(max_ru_error), np.argmax(max_rv_error), np.argmax(max_re_error)))

    def output(self, path, restart_path = './flow.npy'):
        rho = self.U[:,0]
        u = self.U[:,1]/self.U[:,0]
        v = self.U[:, 2] / self.U[:, 0]
        p = (gamma-1.0)*(self.U[:,3] - 0.5*rho*(u**2 + v**2))
        # 每个节点进行平均计算流场值
        AreaSum = np.zeros((self.NNode, 1))
        plot_rho = np.zeros((self.NNode, 1))
        plot_u = np.zeros((self.NNode, 1))
        plot_v = np.zeros((self.NNode, 1))
        plot_p = np.zeros((self.NNode, 1))
        for e in self.Eedge:    # 遍历所有边，边上所属单元和邻接单元(如果有)加到边上的节点的值上，并记录体积
            if e.type == EdgeType.INTER:        # 内部边记录所属和邻接控制体体积
                plot_rho[e.node] += rho[e.own]*self.area[e.own] + rho[e.neb]*self.area[e.neb]
                plot_u[e.node] += u[e.own]*self.area[e.own] + u[e.neb]*self.area[e.neb]
                plot_v[e.node] += v[e.own]*self.area[e.own] + v[e.neb]*self.area[e.neb]
                plot_p[e.node] += p[e.own]*self.area[e.own] + p[e.neb]*self.area[e.neb]
                AreaSum[e.node] += self.area[e.own] + self.area[e.neb]
            else:   # 边界边记录一次所属控制体体积
                plot_rho[e.node] += rho[e.own]*self.area[e.own]
                plot_u[e.node] += u[e.own]*self.area[e.own]
                plot_v[e.node] += v[e.own]*self.area[e.own]
                plot_p[e.node] += p[e.own]*self.area[e.own]
                AreaSum[e.node] += self.area[e.own]
        plot_rho /= AreaSum
        plot_u /= AreaSum
        plot_v /= AreaSum
        plot_p /= AreaSum
        # 输出流场
        with open(path+'.plt', 'w') as file:
            file.write('TITLE="Tecplot Unstructured Grid Data"\n')
            file.write('VARIABLES="X", "Y", "Rho", "U", "V",  "P"\n')
            file.write('ZONE T="Zone"\n')
            file.write('N={}, E={}, F=FEPOINT, ET=TRIANGLE\n'.format(self.NNode, self.NElem))
            file.write('#points:\n')
            for i in range(self.NNode):
                file.write('{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\n'.format(self.Node[i][0], self.Node[i][1],
                plot_rho[i, 0], plot_u[i, 0],plot_v[i, 0],plot_p[i, 0]))
            file.write("#Element:\n")
            for i in range(self.NElem):
                file.write('{} {} {}\n'.format(self.Element[i].data[0] +1, self.Element[i].data[1]+1, self.Element[i].data[2]+1))
        # 输出续算文件
        np.savez(restart_path, free=self.free, U=self.U, iter=self.iter)# 保存远场信息和流场,迭代步数
if __name__ == '__main__':
    airfoil = Airfoil(SU2Loader('mesh_NACA0012_inv.su2'), far_cond=[10000, 1.2, 0.8, 1.25],restart = True, restart_path='../output/flow.npz',restart_iter=None, ouput_path='../output')
    airfoil.solve()