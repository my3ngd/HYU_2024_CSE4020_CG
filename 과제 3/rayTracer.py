import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

class XMLParser:
    def __init__(self, filename: str) -> None:
        tree = ET.parse(filename)
        root = tree.getroot()
        c = root.find('camera')
        self.viewPoint = np.array(c.findtext('viewPoint').split()).astype(np.float64)
        self.viewDir = np.array(c.findtext('viewDir').split()).astype(np.float64)
        self.projNormal = np.array(c.findtext('projNormal', default=self.viewDir).split()).astype(np.float64)
        self.projDistance = np.float64(c.findtext('projDistance', default=1))
        self.viewUp = np.array(c.findtext('viewUp').split()).astype(np.float64)
        self.viewWidth = np.float64(c.findtext('viewWidth'))
        self.viewHeight = np.float64(c.findtext('viewHeight'))
        self.imgSize = np.array(root.findtext('image').split()).astype(np.int32)

        self.shader = dict()
        for s in root.findall('shader'):
            diffuseColor = np.array(s.findtext('diffuseColor').split()).astype(np.float64)
            shaderType = s.get('type')
            cur = {'color': diffuseColor, 'type': shaderType}
            if shaderType == "Phong":
                cur['specular'] = np.array(s.findtext('specularColor').split()).astype(np.float64)
                cur['exponent'] = np.float64(s.findtext('exponent'))
            self.shader[s.get('name')] = cur

        self.surfaces = []
        for s in root.findall('surface'):
            curr_surf = dict()
            s_type = s.get('type')
            curr_surf['type'] = s_type
            curr_surf['shader'] = s.find('shader').get('ref')
            if (s_type.strip().lower() == 'box'):
                curr_surf['min'] = np.array(s.findtext('minPt').split()).astype(np.float64)
                curr_surf['max'] = np.array(s.findtext('maxPt').split()).astype(np.float64)
            if (s_type.strip().lower() == 'sphere'):
                curr_surf['radius'] = np.float64(s.findtext('radius'))
                curr_surf['center'] = np.array(s.findtext('center').split()).astype(np.float64)
            self.surfaces.append(curr_surf)
        
        self.lights = []
        for l in root.findall('light'):
            curr_light = dict()
            curr_light['position'] = np.array(l.findtext('position').split()).astype(np.float64)
            curr_light['intensity'] = np.array(l.findtext('intensity').split()).astype(np.float64)
            self.lights.append(curr_light)


class Color:
    def __init__(self, R, G, B):
        self.color=np.array([R,G,B]).astype(np.float64)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma;
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)


class Camera:
    def __init__(self, viewPoint, viewDir, projNormal, projDistance, viewUp, viewWidth, viewHeight, imgSize) -> None:
        self.width = imgSize[0]   # number of piexl (width)
        self.height = imgSize[1]  # number of piexl (height)
        u_ = np.cross(viewDir, viewUp)
        v_ = np.cross(u_, viewDir)
        self.u = u_ / np.linalg.norm(u_)  # normalized
        self.v = v_ / np.linalg.norm(v_)  # normalized
        viewDir_norm = viewDir / np.linalg.norm(viewDir)
        center = viewPoint + viewDir_norm * projDistance
        self.viewPoint = viewPoint
        self.bottomLeft = center - self.u * viewWidth / 2 - self.v * viewHeight / 2
        self.bottomRight = self.bottomLeft + self.u * viewWidth
        self.topLeft = self.bottomLeft + self.v * viewHeight
        self.topRight = self.topLeft + self.u * viewWidth

    # return (view point, direction unit vector)
    def getRay(self, i, j):  # i-width, j-height direction
        pixel = self.bottomLeft +\
            (self.bottomRight - self.bottomLeft) * (i + 0.5) / self.width +\
            (self.topLeft - self.bottomLeft) * (j + 0.5) / self.height
        viewPoint = self.viewPoint
        d_ = pixel - viewPoint
        d = d_ / np.linalg.norm(d_)
        return {'p': viewPoint, 'd': d}


class Surface:
    def __init__(self) -> None:
        pass

    def intersect(self, ray):
        pass

    def normal(self, ray):
        pass


class Sphere(Surface):
    def __init__(self, center: np.ndarray, r) -> None:
        self.c = center
        self.r = r
    
    # return t: p + td on sphere (min)
    def intersect(self, ray):
        p = ray['p']
        d = ray['d']
        c = self.c
        r = self.r
        p = p - c
        pd = np.dot(p, d)
        pp = np.dot(p, p)
        if pd*pd - pp + r*r < 0:
            return -1
        res1 = -pd + np.sqrt(pd*pd - pp + r*r)
        res2 = -pd - np.sqrt(pd*pd - pp + r*r)
        if res1 < 0 and res2 < 0:
            return -1
        if res1 < 0: res1 = np.inf
        if res2 < 0: res2 = np.inf
        return min(res1, res2)
    
    # return normal unit vector at intersection point of Sphere and ray
    def normal(self, ray):
        t = self.intersect(ray)
        if t < 0: return np.zeros(3).astype(np.float64).astype(np.float64)  # not intersect
        p = ray['p']
        d = ray['d']
        inter = p + t * d
        c = self.c
        return (inter - c) / np.linalg.norm(inter - c)


class Box(Surface):
    def __init__(self, minPoint: np.ndarray, maxPoint: np.ndarray) -> None:
        self.minPoint = minPoint
        self.maxPoint = maxPoint

    def intersect(self, ray):
        p = ray['p']
        d = ray['d']
        B = np.concatenate((self.minPoint, self.maxPoint))
        t_max = np.inf
        t_min = 0
        for i in range(3):
            if d[i] != 0:
                t1 = (B[i] - p[i]) / d[i]
                t2 = (B[i+3] - p[i]) / d[i]
                t_max = min(t_max, max(t1, t2))
                t_min = max(t_min, min(t1, t2))
        if t_max < t_min:
            return -1
        return t_min
    
    def normal(self, ray):
        t = self.intersect(ray)
        if t < 0: return np.zeros(3).astype(np.float64).astype(np.float64)
        p = ray['p']
        d = ray['d']
        inter = p + t * d
        eps = 1e-5  # for fp error
        if   np.abs(inter[0] - self.minPoint[0]) < eps: return np.array([-1, 0, 0]).astype(np.float64)
        elif np.abs(inter[0] - self.maxPoint[0]) < eps: return np.array([ 1, 0, 0]).astype(np.float64)
        elif np.abs(inter[1] - self.minPoint[1]) < eps: return np.array([ 0,-1, 0]).astype(np.float64)
        elif np.abs(inter[1] - self.maxPoint[1]) < eps: return np.array([ 0, 1, 0]).astype(np.float64)
        elif np.abs(inter[2] - self.minPoint[2]) < eps: return np.array([ 0, 0,-1]).astype(np.float64)
        elif np.abs(inter[2] - self.maxPoint[2]) < eps: return np.array([ 0, 0, 1]).astype(np.float64)
        assert False

# return (idx, t) which satisfies p + td intersect surfaces[i] where smallest t > 0
def rayTrace(ray, surfaces):
    idx = -1
    t = np.inf
    for i in range(len(surfaces)):
        surface = surfaces[i]
        inter = surface.intersect(ray)
        if 0 < inter < t:
            t = inter
            idx = i
    return (idx, t)

# return diffusely reflected light (L_d)
def lambertian(surfaces, idx, ray, t, light, diffuse):
    p = ray['p']
    d = ray['d']
    n = surfaces[idx].normal(ray)
    l = light['position'] - (p + t * d)
    l /= np.linalg.norm(l)
    I = light['intensity']
    for i in range(len(surfaces)):
        if i == idx: continue
        if surfaces[i].intersect({'p': p + t * d, 'd': l}) > 0:
            return np.zeros(3).astype(np.float64)
    return diffuse * I * max(0, np.dot(n, l))  # for each color -> not dot(diffuse, I) but diffuse * I


def phong(surfaces, idx, ray, t, light, diffuse, spc, exp):
    p = ray['p']
    d = ray['d']
    n = surfaces[idx].normal(ray)
    l = light['position'] - (p + t * d)
    l /= np.linalg.norm(l)
    I = light['intensity']
    h = (l - d) / np.linalg.norm(l - d)
    for i in range(len(surfaces)):
        if i == idx: continue
        if surfaces[i].intersect({'p': p + t * d, 'd': h}) > 0:
            return np.zeros(3).astype(np.float64)
    res = np.zeros(3).astype(np.float64)
    res += I * spc * np.power(np.maximum(0, np.dot(n, h)), exp)
    res += I * diffuse * np.maximum(0, np.dot(n, l))
    return res


def main():
    if len(sys.argv) < 2: exit()
    parser = XMLParser(sys.argv[1])

    imgSize = parser.imgSize  # imgSize[0]: --, imgSize[1]: |
    channels = 3  # rgb
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)

    shader = parser.shader
    lights = parser.lights
    baseColor = Color(0, 0, 0)

    camera = Camera(
        parser.viewPoint,
        parser.viewDir,
        parser.projNormal,
        parser.projDistance,
        parser.viewUp,
        parser.viewWidth,
        parser.viewHeight,
        parser.imgSize
    )
    
    surfaces = []
    for s in parser.surfaces:
        if s['type'] == 'Sphere':
            surfaces.append(Sphere(s['center'], s['radius']))
        elif s['type'] == 'Box':
            surfaces.append(Box(s['min'], s['max']))

    for i in range(imgSize[0]):
        for j in range(imgSize[1]-1, -1, -1):
            ray = camera.getRay(i, j)
            idx, t = rayTrace(ray, surfaces)

            if idx < 0:
                img[imgSize[1]-1-j][i] = baseColor.toUINT8()  # todo
                continue
            # print(i, j, shader[parser.surfaces[idx]['shader']])
            color = shader[parser.surfaces[idx]['shader']]['color']  # is diffusionColor
            diffuseColor = np.array([color[0], color[1], color[2]]).astype(np.float64)
            shaderType = shader[parser.surfaces[idx]['shader']]['type']
            spc = None if shaderType != 'Phong' else shader[parser.surfaces[idx]['shader']]['specular']
            exp = None if shaderType != 'Phong' else shader[parser.surfaces[idx]['shader']]['exponent']

            total_I = np.zeros(3).astype(np.float64)
            if shaderType == "Lambertian":
                for k in range(len(lights)):
                    cur = lambertian(surfaces, idx, ray, t, lights[k], diffuseColor)
                    total_I = total_I + cur
            elif shaderType == "Phong":
                for k in range(len(lights)):
                    cur = phong(surfaces, idx, ray, t, lights[k], diffuseColor, spc, exp)
                    total_I = total_I + cur

            res = Color(total_I[0], total_I[1], total_I[2])
            res.gammaCorrect(2.2)
            img[imgSize[1]-1-j][i] = res.toUINT8()

    rawimg = Image.fromarray(img, 'RGB')
    rawimg.save(f'{sys.argv[1]}.png')


if __name__ == "__main__":
    main()

