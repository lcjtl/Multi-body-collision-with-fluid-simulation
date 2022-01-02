import taichi as ti
import random
from math import *


@ti.data_oriented
class Edge:
    def __init__(self):
        self.normal = ti.Vector.field(2, dtype=float, shape=())
        self.w = ti.field(dtype=float, shape=())

    @ti.kernel
    def initalize(self, px: float, py: float, nx: float, ny: float):
        self.normal[None] = [nx, ny]
        self.w[None] = nx * px + ny * py

@ti.data_oriented
class SolidSystem:
    def __init__(self, n):
        self.n = n
        self.dt = 0.0015
        self.enabled = n
        self.max_contacts = 40
        self.shape = ti.field(int, shape=n)
        self.radius = ti.field(float, shape=n)
        self.shape_beg_idx = ti.field(dtype=int, shape=n)

        self.loc = ti.Vector.field(2, dtype=float, shape=n)
        self.loc_bak = ti.Vector.field(2, dtype=float, shape=n)
        self.delta_loc = ti.field(dtype=float, shape=n)
        self.rot = ti.field(dtype=float, shape=n)
        self.v = ti.Vector.field(2, dtype=float, shape=n)
        self.w = ti.field(dtype=float, shape=n)
        self.linear_decay = 0.995
        self.angular_decay = 0.95

        self.gravity = ti.Vector.field(2, dtype=float, shape=n)
        self.mass = ti.field(dtype=float, shape=n)
        self.resolution = ti.field(dtype=float, shape=n)
        self.I = ti.field(dtype=float, shape=n)

        self.rc_vert_idx = ti.field(dtype=int, shape=n)
        self.rc_dir = ti.Vector.field(2, dtype=float, shape=n)
        self.mk = ti.Vector.field(2, dtype=float, shape=(n, 36))

        self.contact_edge_num=ti.field(dtype=int,shape=n)
        self.edge_contact_point = ti.Vector.field(2, dtype=float, shape=n)
        self.edge_delta_v = ti.Vector.field(2, dtype=float, shape=n)

        self.stable = ti.field(dtype=int, shape=n)
        self.vert_num = 0
        for i in range(self.n):
            n = random.randint(3, 8)
            self.shape[i] = n
            self.radius[i] = (random.random() - 0.5) * 0.02 + 0.05
            self.vert_num += n
        self.vert_rel = ti.Vector.field(2, dtype=float, shape=self.vert_num)
        self.vert_wor = ti.Vector.field(2, dtype=float, shape=self.vert_num)
        self.edge_dir = ti.Vector.field(2, dtype=float, shape=self.vert_num)

        self.e1=Edge()
        self.e2=Edge()
        self.e3=Edge()
        self.e4=Edge()

    @ti.kernel
    def copy_loc(self, n: int):
        for i in range(n):
            self.delta_loc[i] = (self.delta_loc[i] * 7 + 3 * (self.loc_bak[i] - self.loc[i]).norm()) / 10
            self.loc_bak[i] = self.loc[i]

    @ti.kernel
    def get_world(self, i: int, b: int):
        for j in range(self.shape[i]):
            u = b + j
            c = ti.cos(self.rot[i])
            s = ti.sin(self.rot[i])
            x = self.vert_rel[u][0]
            y = self.vert_rel[u][1]
            self.vert_wor[u] = [self.loc[i][0] + c * x - s * y, self.loc[i][1] + s * x + c * y]

        for j in range(self.shape[i]):
            u = b + j
            v = b + (j + 1) % self.shape[i]
            self.edge_dir[u] = (self.vert_wor[v] - self.vert_wor[u]).normalized()

    @ti.kernel
    def make_shape(self, i: int, b: int):
        n = self.shape[i]
        angle = 2 * pi / n
        m = self.radius[i] * self.radius[i] * ti.sin(2 * pi / n)
        self.mass[i] = n * m * 100
        for j in range(n):
            u = b + j
            self.vert_rel[u] = [self.radius[i] * ti.cos(angle * j), self.radius[i] * ti.sin(angle * j)]
        self.I[i] = self.mass[i] * (2 + ti.cos(angle)) * self.radius[i] * self.radius[i] / 6

    def initialize(self):
        k = 0
        self.resolution.fill(2)
        self.delta_loc.fill(1.)
        for i in range(self.n):
            self.shape_beg_idx[i] = k
            k += self.shape[i]
        k = 0
        u = 2*pi/self.n
        for i in range(self.n):
            self.make_shape(i, k)
            self.loc[i]=[0.35*sin(u*i)+0.5,0.35*cos(u*i)+0.5]
            self.v[i]=[5*cos(u*i),-5*sin(u*i)]
            self.rot[i] = random.random() * pi
            # self.w[i] = 5
            k += self.shape[i]
        self.e1.initalize(0., 0., 0., 1.)
        self.e2.initalize(0., 0., 1., 0.)
        self.e3.initalize(1., 1., 0., -1.)
        self.e4.initalize(1., 1., -1., 0.)

    def update(self, n):
        self.copy_loc(n)
        self.compute_force(n)
        self.update_vel(n)
        self.collsion_solid(n)
        self.collision_edge(n)
        self.update_pos(n)

    @ti.kernel
    def update_vel(self, n: int):
        for i in range(n):
            self.v[i] = self.v[i] + self.dt * self.gravity[i] / self.mass[i]

    @ti.kernel
    def update_pos(self, n: int):
        for i in range(n):
            self.v[i] = self.v[i] * self.linear_decay
            self.w[i] = self.w[i] * self.angular_decay
            self.loc[i] = self.loc[i] + self.dt * self.v[i]
            self.rot[i] = self.w[i] * self.dt + self.rot[i]

    @ti.kernel
    def update_stable(self, n: int):
        for i in range(n):
            if self.delta_loc[i] < 0.005 and self.v[i].norm() < 0.01:
                self.stable[i] = 1
            else:
                self.stable[i] = 0
                if self.resolution[i] < 0.01:
                    self.resolution[i] = 0.5

    @ti.func
    def get_K(self, u, r):
        #
        ra = r - self.loc[u]

        I_inv = 1 / self.I[u]
        K = ti.Matrix([[1 / self.mass[u] + ra[1] * ra[1] * I_inv, -ra[0] * ra[1] * I_inv],
                       [-ra[0] * ra[1] * I_inv, 1 / self.mass[u] + ra[0] * ra[0] * I_inv]])
        return K

    @ti.func
    def minko_sum(self, u: int, v: int):
        m = min(self.shape[u], self.shape[v])
        cnt = 0
        oidx_i = self.rc_vert_idx[u]
        oidx_j = self.rc_vert_idx[v]
        k = 0
        while cnt != 3 and k < 20:
            if k >= m and oidx_i == self.rc_vert_idx[u]:
                cnt |= 1
            if k > m and oidx_j == self.rc_vert_idx[v]:
                cnt |= 2
            if cnt == 3:
                break
            a1 = ti.acos(self.rc_dir[u].dot(self.edge_dir[self.shape_beg_idx[u] + self.rc_vert_idx[u]]))
            a2 = ti.acos(self.rc_dir[v].dot(self.edge_dir[self.shape_beg_idx[v] + self.rc_vert_idx[v]]))
            self.mk[u, k] = self.vert_wor[self.shape_beg_idx[u] + self.rc_vert_idx[u]] - self.vert_wor[
                self.shape_beg_idx[v] + self.rc_vert_idx[v]]
            if ti.abs(a1 - a2) < 0.0001:
                # self.rc_dir[i] = self.rc_dir[j] = self.edge_dir[self.idx[i] + self.rc_vert_idx[i]]
                R = ti.Matrix([[ti.cos(a1), -ti.sin(a1)], [ti.sin(a1), ti.cos(a1)]])
                self.rc_dir[u] = R @ self.rc_dir[u]
                self.rc_dir[v] = R @ self.rc_dir[v]
                self.rc_vert_idx[u] = (self.rc_vert_idx[u] + 1) % self.shape[u]
                self.rc_vert_idx[v] = (self.rc_vert_idx[v] + 1) % self.shape[v]
            elif a1 < a2:
                # self.rc_dir[i] = self.rc_dir[j] = self.edge_dir[self.idx[i] + self.rc_vert_idx[i]]
                R = ti.Matrix([[ti.cos(a1), -ti.sin(a1)], [ti.sin(a1), ti.cos(a1)]])
                self.rc_dir[u] = R @ self.rc_dir[u]
                self.rc_dir[v] = R @ self.rc_dir[v]
                self.rc_vert_idx[u] = (self.rc_vert_idx[u] + 1) % self.shape[u]
            else:
                # self.rc_dir[i] = self.rc_dir[j] = self.edge_dir[self.idx[j] + self.rc_vert_idx[j]]
                R = ti.Matrix([[ti.cos(a2), -ti.sin(a2)], [ti.sin(a2), ti.cos(a2)]])
                self.rc_dir[u] = R @ self.rc_dir[u]
                self.rc_dir[v] = R @ self.rc_dir[v]
                self.rc_vert_idx[v] = (self.rc_vert_idx[v] + 1) % self.shape[v]
            k += 1
        depth = inf
        dir = ti.Vector([0., 0.])
        flag = 1
        for j in range(k):
            a = self.mk[u, j]
            b = self.mk[u, (j + 1) % k]
            ab = b - a
            origin = ti.Vector([0.0, 0.0])
            cross = ab.cross(origin - a)
            if cross <= 0:
                flag = 0
                break
            d = cross / ab.norm()
            if d < depth:
                depth = d
                dir[0] = -ab[1]
                dir[1] = ab[0]

        if flag:
            dir = depth * dir.normalized()
            first_u = -1
            second_u = -1
            first_proj = -inf
            second_proj = -inf

            b = self.shape_beg_idx[u]
            for j in range(self.shape[u]):
                proj = -self.vert_wor[b + j][0] * dir[0] - self.vert_wor[b + j][1] * dir[1]
                if proj > first_proj:
                    second_proj = first_proj
                    second_u = first_u
                    first_proj = proj
                    first_u = b + j
                elif proj > second_proj:
                    second_proj = proj
                    second_u = b + j
            first_v = -1
            second_v = -1
            first_proj = -inf
            second_proj = -inf
            b = self.shape_beg_idx[v]
            for j in range(self.shape[v]):
                proj = self.vert_wor[b + j][0] * dir[0] + self.vert_wor[b + j][1] * dir[1]
                if proj > first_proj:
                    second_proj = first_proj
                    second_v = first_v
                    first_proj = proj
                    first_v = b + j
                elif proj > second_proj:
                    second_proj = proj
                    second_v = b + j
            d = dir.normalized()
            x = ti.abs(d.dot((self.vert_wor[first_u] - self.vert_wor[second_u]).normalized()))
            y = ti.abs(d.dot((self.vert_wor[first_v] - self.vert_wor[second_v]).normalized()))
            contact_point = ti.Vector([0., 0.])
            if ti.abs(x - y) < 0.05:
                minx = inf
                mi = -1
                maxx = -inf
                ma = -1
                if minx > self.vert_wor[first_u][0]:
                    minx = self.vert_wor[first_u][0]
                    mi = 1
                if maxx < self.vert_wor[first_u][0]:
                    maxx = self.vert_wor[first_u][0]
                    ma = 1
                if minx > self.vert_wor[second_u][0]:
                    minx = self.vert_wor[second_u][0]
                    mi = 2
                if maxx < self.vert_wor[second_u][0]:
                    maxx = self.vert_wor[second_u][0]
                    ma = 2
                if minx > self.vert_wor[first_v][0]:
                    minx = self.vert_wor[first_v][0]
                    mi = 4
                if maxx < self.vert_wor[first_v][0]:
                    maxx = self.vert_wor[first_v][0]
                    ma = 4
                if minx > self.vert_wor[second_v][0]:
                    minx = self.vert_wor[second_v][0]
                    mi = 8
                if maxx < self.vert_wor[second_v][0]:
                    maxx = self.vert_wor[second_v][0]
                    ma = 8
                t = 15 - mi - ma
                if t == 3:
                    contact_point = (self.vert_wor[first_u] + self.vert_wor[second_u]) / 2
                elif t == 5:
                    contact_point = (self.vert_wor[first_u] + self.vert_wor[first_v]) / 2
                elif t == 6:
                    contact_point = (self.vert_wor[first_v] + self.vert_wor[second_u]) / 2
                elif t == 9:
                    contact_point = (self.vert_wor[second_v] + self.vert_wor[first_u]) / 2
                elif t == 10:
                    contact_point = (self.vert_wor[second_v] + self.vert_wor[second_u]) / 2
                else:
                    contact_point = (self.vert_wor[first_v] + self.vert_wor[second_v]) / 2
            elif x < y:
                contact_point = self.vert_wor[first_v]
            else:
                contact_point = self.vert_wor[first_u]
            self.loc[u] += dir * self.mass[v] / (self.mass[u] + self.mass[v])
            self.loc[v] -= dir * self.mass[u] / (self.mass[u] + self.mass[v])

            ra_u = contact_point - self.loc[u]
            ra_v = contact_point - self.loc[v]

            va = self.v[u] + [-ra_u[1] * self.w[u], ra_u[0] * self.w[u]] - self.v[v] - [-ra_v[1] * self.w[v],
                                                                                        ra_v[0] * self.w[v]]
            dir = dir.normalized()
            vn = va.dot(dir) * dir
            vt = va - vn

            res = self.resolution[u]
            # sz = va.norm()
            # if sz > 5:
            #     res = 0.5
            a = max(1 - (1 + res) * vn.norm() / vt.norm(), 0)
            vn_new = -res * vn
            vt_new = a * vt
            v_new = vn_new + vt_new
            dv = v_new - va
            K1 = self.get_K(u, contact_point)
            K2 = self.get_K(v, contact_point)
            impluse = (K1 + K2).inverse() @ dv
            self.apply_impluse(u, contact_point, impluse)
            self.apply_impluse(v, contact_point, -impluse)

    @ti.func
    def apply_impluse(self, u, r, impluse):
        ra = r - self.loc[u]
        self.v[u] += 1 / self.mass[u] * impluse
        self.w[u] += (ra[0] * impluse[1] - ra[1] * impluse[0]) / self.I[u]

    @ti.func
    def detect_edge(self, i, b, e):
        n = self.shape[i]
        ra = ti.Vector([0., 0.])
        pn = e.normal[None]
        k = 0
        theta = self.rot[i]
        R = ti.Matrix([[ti.cos(theta), -ti.sin(theta)], [ti.sin(theta), ti.cos(theta)]])
        self.contact_edge_num[i]=0
        for j in range(n):
            u = b + j
            if self.vert_wor[u].dot(pn) - e.w[None] < 0:
                ri = R @ self.vert_rel[u]
                vi = self.v[i] + [- ri[1] * self.w[i], ri[0] * self.w[i]]
                if vi.dot(pn) < 0:
                    ra += ri
                    k += 1
        if k != 0:
            ra /= k
            pene = (self.loc[i] + ra).dot(pn) - e.w[None]
            self.loc[i] += ti.abs(pene) * pn
            va = self.v[i] + [-ra[1] * self.w[i], ra[0] * self.w[i]]

            vn = va.dot(pn) * pn
            vt = va - vn
            a = max(1 - (1 + 0.5) * vn.norm() / vt.norm(), 0)
            vn_new = -0.5 * vn
            # self.resolution[i] *= 0.99
            vt_new = a * vt
            v_new = vn_new + vt_new
            dv = v_new - va

            self.edge_contact_point[i] += self.loc[i] + ra
            self.contact_edge_num[i] += 1
            self.edge_delta_v[i] += dv
            K = self.get_K(i, ra + self.loc[i])
            impluse = K.inverse() @ dv
            self.apply_impluse(i, ra + self.loc[i], impluse)

    @ti.kernel
    def collision_edge(self, n: int):
        k = 0
        for i in range(n):
            self.contact_edge_num[i] = 0
            self.edge_contact_point[i] = [0.0, 0.0]
            self.edge_delta_v[i] = [0.0, 0.0]
            self.detect_edge(i, k, self.e1)
            self.detect_edge(i, k, self.e2)
            self.detect_edge(i, k, self.e3)
            self.detect_edge(i, k, self.e4)

            if self.contact_edge_num[i] != 0:
                self.edge_contact_point[i] /= self.contact_edge_num[i]
                self.edge_delta_v[i] /= self.contact_edge_num[i]
            k += self.shape[i]

    @ti.func
    def get_farest_vert(self, i, dir):
        res = 0
        max_proj = -inf
        b = self.shape_beg_idx[i]
        for j in range(self.shape[i]):
            proj = self.vert_wor[b + j][0] * dir[0] + self.vert_wor[b + j][1] * dir[1]
            if proj > max_proj:
                max_proj = proj
                res = j
        return res

    @ti.func
    def make_rc(self, i, dir):
        farest_vert = self.get_farest_vert(i, [dir[1], -dir[0]])
        self.rc_vert_idx[i] = farest_vert
        self.rc_dir[i] = dir

    @ti.kernel
    def collsion_solid(self, n: int):
        for i in range(n):
            for j in range(i + 1, n):
                self.make_rc(i, [1., 0.])
                self.make_rc(j, [-1., 0.])
                self.minko_sum(i, j)

    @ti.kernel
    def compute_force(self, n: int):
        for i in range(n):
            self.gravity[i] = [0.0,0.0]
        for i in range(n):
            pi = self.loc[i]
            mi = self.mass[i]
            for j in range(n):
                if i != j:
                    diff = pi - self.loc[j]
                    r = diff.norm(1e-5)
                    self.gravity[i] += -0.8 * mi * self.mass[j] * diff / r ** 3
