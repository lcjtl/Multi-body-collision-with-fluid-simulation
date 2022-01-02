import taichi as ti
import random
from math import *
from particle import ParticleSystem
from solid import SolidSystem
import argparse
import os

@ti.data_oriented
class MainSystem:
    def __init__(self, ps, ss):
        self.ps = ps
        self.ss = ss
        self.dt = 0.0015

    @ti.kernel
    def apply_gravity_to_particle(self):
        if particle_use_solids_gravity:
            for particle_idx in range(self.ps.particle_free_num):
                self.ps.gravity[particle_idx]=[0.0,0.0]
                for shape_idx in range(n):
                    pi = self.ps.particle_pos[particle_idx]
                    mi = self.ss.mass[shape_idx]
                    diff = pi - self.ss.loc[shape_idx]
                    r = diff.norm(1e-5)
                    self.ps.gravity[particle_idx] += -0.15 * mi * diff / r ** 3
        else:
            for particle_idx in range(self.ps.particle_free_num):
                self.ps.gravity[particle_idx] = (center[None] - self.ps.particle_pos[particle_idx]).normalized(
                    1e-5) * 15

    @ti.kernel
    def collision(self, n: int):
        for particle_idx in range(self.ps.particle_free_num):
            k = 0
            for shape_idx in range(n):
                flag = 0
                for x in range(self.ss.shape[shape_idx]):
                    y = (x + 1) % self.ss.shape[shape_idx]
                    diff = self.ps.real_radius / ti.cos(pi / self.ss.shape[shape_idx]) * 0.5
                    p1 = self.ss.vert_wor[k + x]
                    p1 += diff * (p1 - self.ss.loc[shape_idx]).normalized()
                    p2 = self.ss.vert_wor[k + y]
                    p2 += diff * (p2 - self.ss.loc[shape_idx]).normalized()
                    ab = p1 - self.ss.loc[shape_idx]
                    cross = ab.cross(self.ps.particle_pos[particle_idx] - self.ss.loc[shape_idx])
                    if cross > 0:
                        ab = self.ss.loc[shape_idx] - p2
                        cross = ab.cross(self.ps.particle_pos[particle_idx] - p2)
                        if cross > 0:
                            ab = p2 - p1
                            cross = ab.cross(self.ps.particle_pos[particle_idx] - p1)
                            if cross > 0:
                                dir_in = ti.Vector([-ab[1], ab[0]])
                                dir_in = dir_in.normalized()
                                flag = 1
                                d = cross / ab.norm()
                                dir_in *= d
                                dir_out = -dir_in
                                self.ps.particle_pos[particle_idx] += dir_out
                                ra = self.ps.particle_pos[particle_idx] - self.ss.loc[shape_idx]
                                pv = self.ss.v[shape_idx] + [-ra[1] * self.ss.w[shape_idx],
                                                             ra[0] * self.ss.w[shape_idx]]
                                if pv.dot(self.ps.particle_vel[particle_idx]) <= 0.00001:
                                    impluse = (self.ps.particle_vel[particle_idx] - pv) * 0.05
                                    self.ss.apply_impluse(shape_idx, self.ps.particle_pos[
                                        particle_idx] - dir_out.normalized() * self.ps.real_radius, impluse)
                                self.ps.particle_vel[particle_idx] += dir_out / self.dt * 0.5
                if flag:
                    break
                k += self.ss.shape[shape_idx]

if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    write_to_disk = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-sg',
                        '--use_sg',
                        action='store_true')
    args, unknowns = parser.parse_known_args()
    particle_use_solids_gravity = args.use_sg

    os.makedirs('frames', exist_ok=True)
    gui = ti.GUI(background_color=0x000000, res=600)
    num = 6

    solids = SolidSystem(num)
    solids.initialize()
    enable_idx = 0
    color = []
    for _ in range(num):
        r = random.randint(20, 255)
        g = random.randint(20, 255)
        b = random.randint(20, 255)
        color.append(r * 256 * 256 + g * 256 + b)

    center = ti.Vector.field(2, dtype=float, shape=())
    center[None] = [0.5, 0.5]

    ps = ParticleSystem(80, 600, 8, 0.0015)
    ps.initialize()

    ms = MainSystem(ps, solids)
    output = 0
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            mouse = gui.get_cursor_pos()
            if gui.is_pressed(ti.GUI.LMB):
                center[None] = [mouse[0], mouse[1]]
        #         if enable_idx < solids.n:
        #             solids.loc[enable_idx] = [mouse[0], mouse[1]]
        #             enable_idx += 1
        #             solids.enabled = enable_idx
        n = solids.enabled
        for i in range(ps.particle_free_num):
            gui.circle(ps.particle_pos[i], radius=ps.radius, color=0x7FFFD4)
        solids.update(n)
        ms.apply_gravity_to_particle()
        ps.update()

        k = 0
        for u in range(n):
            solids.get_world(u, k)
            for v in range(solids.shape[u] - 2):
                gui.triangle(solids.vert_wor[k], solids.vert_wor[k + v + 1], solids.vert_wor[k + v + 2], color=color[u])
            k += solids.shape[u]
        ms.collision(n)
        if write_to_disk:
            print(output)
            gui.show(f'frames/{output:05d}.png')
        else:
            gui.show()
        output += 1
