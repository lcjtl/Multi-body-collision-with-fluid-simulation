import taichi as ti
import numpy as np


@ti.data_oriented
class ParticleSystem:
    def __init__(self, m, n, r, dt):
        self.radius = r
        self.diameter = r * 2
        self.real_radius = r / 600
        self.support_radius = r * 4
        self.particle_rows = m // self.diameter
        self.particle_cols = n // self.diameter
        self.unit = 1 / self.particle_cols
        self.unit_r = self.unit / 2
        self.dt = dt

        # particle
        self.stiffness = 1
        self.density_0 = 1.0
        self.viscosity = 0.00
        self.mass = np.pi * self.radius ** 2 * self.density_0

        self.particle_free_num = int(self.particle_cols * self.particle_rows)
        self.particle_boundary_num = 3 * 4 * self.particle_cols
        self.particle_total_num = self.particle_free_num + self.particle_boundary_num
        self.density = ti.field(dtype=float, shape=self.particle_total_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_total_num)
        self.particle_pos = ti.Vector.field(2, dtype=float, shape=self.particle_total_num)
        self.particle_vel = ti.Vector.field(2, dtype=float, shape=self.particle_total_num)
        self.gravity = ti.Vector.field(2, dtype=float, shape=self.particle_total_num)

        # grid
        self.grid_origin = ti.Vector.field(2, dtype=float, shape=())
        self.grid_max_particle_num = 100
        self.particle_grid_idx = ti.field(dtype=int, shape=self.particle_total_num)
        self.grid_rows = 800 // self.support_radius
        self.grid_cols = 800 // self.support_radius
        self.grid_size = self.support_radius / 600
        self.grid_num = self.grid_cols * self.grid_rows
        self.grid_particle_num = ti.field(dtype=int, shape=self.grid_num)
        self.grid_particles = ti.field(dtype=int, shape=(self.grid_num, self.grid_max_particle_num))

    def initialize(self):
        self.particle_vel.fill(0.0)
        self.grid_origin[None] = [-1 / 6, -1 / 6]
        self.init_free_particle_pos()
        self.init_boundary_particle_pos()

    @ti.kernel
    def init_free_particle_pos(self):
        for i in range(self.particle_free_num):
            x = i // self.particle_cols
            y = i % self.particle_cols
            self.particle_pos[i] = [self.unit_r + self.unit * y, self.unit_r + self.unit * x + 1.0]

    @ti.kernel
    def init_boundary_particle_pos(self):
        beg_idx = self.particle_free_num

        for i in range(self.particle_cols):
            for j in range(4):
                self.particle_pos[beg_idx + i * 4 + j] = [-self.unit_r - self.unit * j, self.unit_r + self.unit * i]
        beg_idx += self.particle_cols * 4

        for i in range(self.particle_cols):
            for j in range(4):
                self.particle_pos[beg_idx + i * 4 + j] = [1 + self.unit_r + self.unit * j, self.unit_r + self.unit * i]
        beg_idx += self.particle_cols * 4

        for i in range(self.particle_cols):
            for j in range(4):
                self.particle_pos[beg_idx + i * 4 + j] = [self.unit_r + self.unit * i, -self.unit_r - self.unit * j]

        for i in range(self.particle_free_num, self.particle_total_num):
            self.density[i] = self.density_0

    @ti.kernel
    def compute_grid_idx(self):
        for particle_idx in range(self.grid_num):
            self.grid_particle_num[particle_idx] = 0

        for particle_idx in range(self.particle_total_num):
            x = (self.particle_pos[particle_idx][0] - self.grid_origin[None][0]) // self.grid_size
            y = (self.particle_pos[particle_idx][1] - self.grid_origin[None][1]) // self.grid_size
            grid_idx = -1
            if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                grid_idx = ti.cast(y * self.grid_cols + x, int)
            self.particle_grid_idx[particle_idx] = grid_idx
            if grid_idx != -1 and self.grid_particle_num[grid_idx] < self.grid_max_particle_num:
                self.grid_particles[grid_idx, self.grid_particle_num[grid_idx]] = particle_idx
                self.grid_particle_num[grid_idx] += 1

    @ti.func
    def detect_edge(self, i):
        p = self.particle_pos[i]
        if p[0] < self.real_radius / 2:
            # self.v[i]+=[(self.real_radius/2-p[0])/self.dt*0.5,0.0]
            self.particle_vel[i][0] = -self.particle_vel[i][0] * 0.8
            self.particle_pos[i][0] = self.real_radius / 2
        if p[1] < self.real_radius / 2:
            # self.v[i]+=[0.0,(self.real_radius/2-p[1])/self.dt*0.5]
            self.particle_vel[i][1] = -self.particle_vel[i][1] * 0.8
            self.particle_pos[i][1] = self.real_radius / 2
        if p[0] > 1 - self.real_radius / 2:
            # self.v[i]+=[(1-self.real_radius/2-p[0])/self.dt*0.5,0.0]
            self.particle_vel[i][0] = -self.particle_vel[i][0] * 0.8
            self.particle_pos[i][0] = 1 - self.real_radius / 2

    @ti.kernel
    def update_pos(self):
        for i in range(self.particle_free_num):
            self.particle_vel[i] *= 0.99
            self.detect_edge(i)
            if self.particle_vel[i].norm() > 3.0:
                self.particle_vel[i] = self.particle_vel[i].normalized() * 3.0
            self.particle_pos[i] += self.particle_vel[i] * self.dt

    @ti.func
    def is_valid(self, idx):
        return 0 <= idx < self.grid_num

    @ti.kernel
    def search_neighbors(self):
        for i in range(self.particle_free_num):
            self.density[i] = 0
            c = self.particle_grid_idx[i]
            u = c + self.grid_cols
            d = c - self.grid_cols
            v = ti.Vector([u - 1, u, u + 1,
                           c - 1, c, c + 1,
                           d - 1, d, d + 1])
            for j in ti.static(range(9)):
                grid_idx = v[j]
                if self.is_valid(grid_idx):
                    for k in range(self.grid_particle_num[grid_idx]):
                        neighbor_idx = self.grid_particles[grid_idx, k]
                        self.density[i] += self.mass * self.cubic_kernel(
                            (self.particle_pos[i] - self.particle_pos[neighbor_idx]).norm()) * 3
            self.density[i] = max(self.density[i], self.density_0)

    @ti.kernel
    def compute_pressure(self):
        for i in range(self.particle_free_num):
            self.pressure[i] = self.stiffness * (self.density[i] - self.density_0)

    @ti.kernel
    def apply_force(self):
        for particle_idx in range(self.particle_free_num):
            self.particle_vel[particle_idx] += self.gravity[particle_idx] * self.dt
            c = self.particle_grid_idx[particle_idx]
            u = c + self.grid_cols
            d = c - self.grid_cols
            v = ti.Vector([u - 1, u, u + 1,
                           c - 1, c, c + 1,
                           d - 1, d, d + 1])
            dv = ti.Vector([0.0, 0.0])
            for j in ti.static(range(9)):
                neighbor_grid = v[j]
                if self.is_valid(neighbor_grid):
                    for k in range(self.grid_particle_num[neighbor_grid]):
                        neighbor_idx = self.grid_particles[neighbor_grid, k]
                        dv -= (self.pressure[particle_idx] / self.density[particle_idx] ** 2 + self.pressure[
                            neighbor_idx] /
                               self.density[neighbor_idx] ** 2) * self.cubic_kernel_derivative(
                            self.particle_pos[particle_idx] - self.particle_pos[neighbor_idx])
                        dv += self.viscosity * (self.particle_vel[particle_idx] - self.particle_vel[particle_idx]) / \
                              self.density[
                                  particle_idx] * self.cubic_kernel_laplace(
                            self.particle_pos[particle_idx] - self.particle_pos[neighbor_idx])
            self.particle_vel[particle_idx] += dv

    @ti.func
    def cubic_kernel(self, r):
        s = self.support_radius / 600 / 2
        k = 1 / (np.pi * s ** 3)
        u = r / s
        res = 0.0
        if 0 <= u <= 1:
            res = 1 - 1.5 * u ** 2 + 0.75 * u ** 3
        elif 1 <= u <= 2:
            res = 0.25 * (2 - u) ** 3
        return k * res

    @ti.func
    def cubic_kernel_derivative(self, r):
        s = self.support_radius / 600 / 2
        k = 1 / (np.pi * s ** 4)
        u = r.norm() / s
        # print(u)
        res = 0.0
        if 0 <= u <= 1:
            res = 3 * u * (-1 + 0.75 * u)
        elif 1 <= u <= 2:
            res = -0.75 * (2 - u) ** 2
        return k * res * r.normalized(1e-5)

    @ti.func
    def cubic_kernel_laplace(self, r):
        s = self.support_radius / 600 / 2
        k = 1 / (np.pi * s ** 5)
        u = r.norm() / s
        res = 0.0
        if 0 <= u <= 1:
            res = 3 * (-1 + 1.5 * u)
        elif 1 <= u <= 2:
            res = 1.5 * (2 - u)
        return k * res * r.normalized(1e-5)

    def update(self):
        self.compute_grid_idx()
        self.search_neighbors()
        self.compute_pressure()
        self.apply_force()
        self.update_pos()
        # print(self.gravity)
