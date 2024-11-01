import numpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from enum import Enum
# decription: 加载SU2网格
# author: JLX
# date: 2024-6-6
class ElementType:
    def __init__(self):
        self.TypeToIDMap = {'Line': 3, 'Triangle': 5, 'Quadrilateral': 9, 'Tetrahedral': 10, 'Hexahedral':12}
        self.TypeNodeCountMap = {'Line': 2, 'Triangle': 3, 'Quadrilateral': 4, 'Tetrahedral': 4, 'Hexahedral':8}
        self.IDToTypeMap = {}
        self.IDToNodeCountMap = {}
        for key in self.TypeToIDMap:
            self.IDToTypeMap[self.TypeToIDMap[key]] = key
            self.IDToNodeCountMap[self.TypeToIDMap[key]] = self.TypeNodeCountMap[key]
    def getElementNodeCount(self, ID):
        return self.IDToNodeCountMap[ID]

class ElementUnit:
    def __init__(self, N, data=None):
        if data is None:
            data = []
        self.N =N
        self.data = data
    def setData(self, data):
        self.data = data
    def __len__(self):
        return self.N
class EdgeType(Enum):
    INTER = 0
    WALL = 1
    FAR = 2

class Edge:
    """
    边数据结构要提供：
        1. 边的类型：内部边，远场边，壁面边等
        2. 边的节点索引
        3. 边的所属单元：单元遍历到第一次这条边，标记为属于该单元，第二次遍历到标记为邻接单元
        4. 边的邻接单元
    """
    def __init__(self, edg_type = EdgeType.INTER, edg_nodes = None, edg_own = -1, edg_neb = -1):
        self.type = edg_type
        self.node = edg_nodes
        self.own = edg_own
        self.neb = edg_neb

    def same(self, node:list):
        if self.node == node or self.node == node[::-1]:
            return True
        return False


class SU2Loader:
    def __init__(self, su2_path):
        self.su2_path = su2_path
        self.etype = ElementType()
        #
        self.DIM = 0
        self.Element = []
        self.Nodes = []
        self.MARK = {}
        self.__read_su2()
        #
        self.Edge = []  # 边列表
        self.__topology()
    def __get_strvalue(self, string):
        pos = string.find('=')
        return string[pos + 1:]
    def __read_su2(self):
        with open(self.su2_path, 'r') as file:
            self.__read_DIM(file=file)
            self.__read_ElEM(file=file)
            self.__read_POIN(file=file)
            self.__read_MARK(file=file)
            file.close()

    def __read_DIM(self, file):
        """读取维度"""
        file.seek(0)
        keyword = 'NDIME'
        for line in file:
            if keyword in line and '%' not in line:
                self.DIM = int(self.__get_strvalue(line))
                break

    def __read_ElEM(self, file):
        """读取单元"""
        file.seek(0)
        keyword = 'NELEM'
        for line in file:
            if keyword in line and '%' not in line:
                NELEM = int(self.__get_strvalue(line))
                while NELEM > 0:
                    line = file.readline().strip()
                    data = [int(num) for num in line.split()]
                    self.Element.append(ElementUnit(self.etype.getElementNodeCount(data[0]), data[1:self.etype.getElementNodeCount(data[0])+1]))
                    NELEM -= 1
                break

    def __read_POIN(self, file):
        """读取坐标"""
        file.seek(0)
        keyword = 'NPOIN'
        for line in file:
            if keyword in line and '%' not in line:
                NPOIN = int(self.__get_strvalue(line))
                while NPOIN > 0:
                    line = file.readline().strip()
                    data = [float(num) for num in line.split()[:self.DIM]]
                    self.Nodes.append(data)
                    NPOIN -= 1
                break
    def __read_MARK(self, file):
        """读取"""
        file.seek(0)
        keyword1 = 'NMARK'
        keyword2 = 'MARKER_TAG'
        keyword3 = 'MARKER_ELEMS'
        NMARK = 0
        for line in file:
            if keyword1 in line and '%' not in line:
                NMARK = int(self.__get_strvalue(line))
                while NMARK > 0:
                    for line in file:
                        if keyword2 in line:
                            key = self.__get_strvalue(line.strip()) # 边界名称
                            line = file.readline().strip()
                            N = int(self.__get_strvalue(line))  # 边界边数量
                            points = []
                            while N> 0:
                                line = file.readline()
                                data = [int(num) for num in line.split()]
                                points.append(data[1:self.etype.getElementNodeCount(data[0]) + 1])
                                N -= 1
                            self.MARK[key.strip()] = points
                    NMARK -= 1
                break
    def __topology(self):
        """处理拓扑"""
        EdgeToIdx = {}  # 边到对应边列表索引
        edge_count = 0
        for i in range(len(self.Element)):
            for j in range(len(self.Element[i])):
                p1 = self.Element[i].data[j]
                p2 = self.Element[i].data[0 if j == (self.Element[i].N - 1) else j + 1]
                # print(self.Element[i].data)
                # print(p1," ",  p2)
                if (p1, p2) in EdgeToIdx or (p2, p1) in EdgeToIdx:  # 已经存在过，此时单元设为该边的邻接单元
                    idx = EdgeToIdx[(p1, p2)]
                    self.Edge[idx].neb = i
                else:   # 第一次遇到这条边,将当前单元标记为所属单元，并更新边节点，边的类型先设置为内部边，并记录进临时字典中
                    EdgeToIdx[(p1, p2)] = edge_count
                    EdgeToIdx[(p2, p1)] = edge_count
                    e = Edge(edg_type = EdgeType.INTER, edg_nodes = [p1, p2], edg_own = i, edg_neb = -1)
                    self.Edge.append(e)
                    edge_count += 1
        # 处理边界边类型
        for key in self.MARK:
            bnd = self.MARK[key]
            for p in bnd:
                idx = EdgeToIdx[(p[0], p[1])]
                if key == 'airfoil':
                    self.Edge[idx].type = EdgeType.WALL
                elif key == 'farfield':
                    self.Edge[idx].type = EdgeType.FAR
                else:
                    raise RuntimeError('undefined boundary!')
    def print(self):
        print('DIM:', self.DIM)
        print('NELEM:', len(self.Element))
        print('NPOIN: ', len(self.Nodes))
        for key in self.MARK:
            print("%s: %d" % (key, len(self.MARK[key])))
        # 打印所有边
        typename = {EdgeType.INTER:'INTER', EdgeType.WALL:'WALL', EdgeType.FAR:'FAR'}
        nebneg_count = 0
        wall_count = 0
        far_count = 0
        inter_count = 0
        i = -1
        for e in self.Edge:
            i += 1
            print("%4d, %s node:[%4d, %4d], owner: %5d, nebor: %5d" %(i, typename[e.type], e.node[0], e.node[1],e.own, e.neb))
            if e.neb == -1:
                nebneg_count += 1
            if e.type == EdgeType.INTER:
                inter_count += 1
            if e.type == EdgeType.WALL:
                wall_count += 1
            if e.type == EdgeType.FAR:
                far_count += 1
        print('total: %d, nebor-1: %d, inter: %d, wall: %d, far: %d' %(len(self.Edge), nebneg_count, inter_count, wall_count, far_count))


    def plot(self, plot_dict=None):
        if plot_dict == None:
            plot_dict ={
                'Node_idx': False,
                'Elem_idx':False,
                'Edg_vec': False,
                'Edg_elem':False,
                'Bnd_edge':False,
                'Airfoil':False
            }
        plt.figure()
        # 绘制网格
        element = [elem.data for elem in self.Element]
        triangle = tri.Triangulation([data[0] for data in self.Nodes], [data[1] for data in self.Nodes], element)
        plt.triplot(triangle, 'b-', linewidth=0.6)
        # 绘制节点索引
        if plot_dict['Node_idx']:
            for i in range(len(self.Nodes)):
                plt.text(self.Nodes[i][0], self.Nodes[i][1], str(i), fontsize=7)
        # 绘制单元索引
        # if plot_dict['Elem_idx']:
        #     for i in range(len(self.Element)):
        #         idx = self.Element[i].data
        #         points = np.array([self.Nodes[j] for j in idx])
        #         xy = np.sum(points, axis=0)/3.0
        #         plt.text(xy[0], xy[1], str(i), c='r', fontsize=8)
        tt = [0]
        if plot_dict['Elem_idx']:
            for i in range(len(tt)):
                idx = self.Element[tt[i]].data
                points = np.array([self.Nodes[j] for j in idx])
                xy = np.sum(points, axis=0) / 3.0
                plt.text(xy[0], xy[1], str(tt[i]), c='r', fontsize=8)
        # 绘制边向量方向
        if plot_dict['Edg_vec']:
            for i in range(20):
                idx = self.Element[i].data
                points = np.array([self.Nodes[j] for j in idx])
                for j in range(len(points)):
                    k1 = j
                    k2 = 0 if j == len(points) - 1 else j + 1  # 循环回到第一个点
                    start_point = points[k1]
                    end_point = points[k2]
                    vector = end_point - start_point  # 计算向量
                    # 绘制向量
                    plt.quiver(start_point[0], start_point[1], vector[0], vector[1], angles='xy', scale_units='xy', scale=1)
        # 绘制边界边向量
        if plot_dict['Bnd_edge']:
            for key in self.MARK:
                idx = self.MARK[key]
                p1 = []
                p2 = []
                for k in idx:
                    p1.append(self.Nodes[k[0]])
                    p2.append(self.Nodes[k[1]])
                p1 = np.array(p1)
                p2 = np.array(p2)
                plt.text(p1[0, 0], p2[0, 1], key, fontsize='15')
                for i in range(len(p1)):
                    vector = p2[i] - p1[i]
                    plt.quiver(p1[i, 0], p1[i, 1], vector[0], vector[1], angles='xy', scale_units='xy',
                               scale=1)
        # 绘制翼型边界点索引
        if plot_dict['Airfoil']:
            for key in self.MARK:
                if key != 'airfoil':
                    continue
                idx = self.MARK[key]
                p1 = []
                p2 = []
                for k in idx:
                    p1.append(self.Nodes[k[0]])
                    p2.append(self.Nodes[k[1]])
                p1 = np.array(p1)
                p2 = np.array(p2)
                index = 0
                for i in range(len(p1)):
                    index += 1
                    plt.text(p1[i, 0], p1[i, 1], str(index), c='r')
                    # index += 1
                    # plt.text(p2[i, 0], p2[i, 1], str(index), c='g')
        plt.axis('equal')
        plt.title(self.su2_path.split('/')[-1])
        plt.show()



if __name__ == '__main__':
    a = [1,2,3,4,5]
    print(a[1:4])
    print()
    etype = ElementType()
    SU2Grid = SU2Loader('mesh_NACA0012_inv.su2')
    plot_dict = {
        'Node_idx': False,
        'Elem_idx': True,
        'Edg_vec': False,
        'Edg_elem': False,
        'Bnd_edge': False,
        'Airfoil': False
    }
    SU2Grid.print()
    SU2Grid.plot(plot_dict=plot_dict)

    e1 = Edge()
    e1.node = [1, 2]
    print(e1.same([1, 2]))
    print(e1.same([2, 1]))
    print(e1.same([1, 3]))

