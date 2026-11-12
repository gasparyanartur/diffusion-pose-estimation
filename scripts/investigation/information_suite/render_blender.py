"""Run with Blender's Python, not the training environment."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view


def material(name, color, roughness=.65):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    result.use_nodes = True
    shader = result.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    return result


def setup_scene(scale, style_seed, light_angle=0):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 384
    scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 2
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.render.image_settings.color_mode = 'RGB'
    rng = np.random.default_rng(style_seed)
    world = bpy.data.worlds.new('StudioWorld')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.32, .34, .38, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .65
    scene.world = world
    bpy.ops.object.camera_add(location=(0, 0, 6))
    camera = bpy.context.object
    camera.name = 'Camera'
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 384 / scale
    camera.data.lens = 50
    scene.camera = camera
    for location, power, size in [((-3, 3, 5), 450, 4), ((3, 1, 3), 180, 3)]:
        angle = np.deg2rad(light_angle)
        x, y, z = location
        location = (x*np.cos(angle)+z*np.sin(angle), y, -x*np.sin(angle)+z*np.cos(angle))
        bpy.ops.object.light_add(type='AREA', location=location)
        light = bpy.context.object
        light.data.energy = power * float(rng.uniform(.95, 1.05))
        light.data.shape = 'DISK'
        light.data.size = size
        light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()
    skin = np.array([.54, .32, .20]) * rng.uniform(.95, 1.05)
    shirt = rng.uniform(.10, .32, 3)
    pants = rng.uniform(.035, .10, 3)
    palette = [material('Skin', skin), material('Shirt', shirt), material('Trousers', pants),
               material('Shoes', [.025, .025, .03]), material('Hair', [.035, .025, .018])]
    return scene, palette


def add_mesh(data, palette):
    vertices, faces = data['vertices'], data['faces']
    count = len(data['joints'])
    midpoint = len(vertices)//count
    face_count = len(faces)//count
    people = []
    for person in range(count):
        local_faces = faces[person*face_count:(person+1)*face_count] - person*midpoint
        local_vertices = vertices[person*midpoint:(person+1)*midpoint]
        mesh = bpy.data.meshes.new(f'Body{person}')
        mesh.from_pydata(local_vertices.tolist(), [], local_faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new(f'Person{person}', mesh)
        bpy.context.collection.objects.link(obj)
        for item in palette:
            mesh.materials.append(item)
        labels = data['materials'][person*len(local_faces):(person+1)*len(local_faces)]
        mesh.polygons.foreach_set('material_index', labels)
        mesh.polygons.foreach_set('use_smooth', np.ones(len(local_faces), dtype=bool))
        if 'rest_vertices' in data:
            attribute = mesh.attributes.new('RestCoordinates', 'FLOAT_VECTOR', 'POINT')
            attribute.data.foreach_set('vector', data['rest_vertices'].astype(np.float32).reshape(-1))
        people.append((obj, local_vertices, local_faces))
    return people


def add_rest_texture(palette, style_seed):
    skin = palette[0]
    nodes, links = skin.node_tree.nodes, skin.node_tree.links
    shader = nodes.get('Principled BSDF')
    base = tuple(shader.inputs['Base Color'].default_value)
    attribute = nodes.new('ShaderNodeAttribute')
    attribute.attribute_name = 'RestCoordinates'
    noise = nodes.new('ShaderNodeTexNoise')
    noise.noise_dimensions = '4D'
    noise.inputs['Scale'].default_value = 24
    noise.inputs['Detail'].default_value = 2
    noise.inputs['W'].default_value = (style_seed % 1000)/13
    ramp = nodes.new('ShaderNodeMapRange')
    ramp.inputs['To Min'].default_value = .55
    ramp.inputs['To Max'].default_value = 1.1
    multiply = nodes.new('ShaderNodeMixRGB')
    multiply.blend_type = 'MULTIPLY'
    multiply.inputs[0].default_value = 1
    multiply.inputs[1].default_value = base
    links.new(attribute.outputs['Vector'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], ramp.inputs['Value'])
    links.new(ramp.outputs[0], multiply.inputs[2])
    links.new(multiply.outputs[0], shader.inputs['Base Color'])


def add_panels(record):
    rectangles = record.get('foreground_rectangles', [])
    config = record.get('foreground_panels')
    if config is None and not rectangles:
        return
    scale = record['scale']
    vertices, faces = [], []
    if config is not None:
        radius = config['half_width_pixels']
        rectangles = [dict(bounds_xy=[x-radius, y-radius, x+radius, y+radius],
                           world_z=config['world_z'])
                      for x, y in np.unique(np.asarray(config['centers_xy']), axis=0)]
    for rectangle in rectangles:
        left, top, right, bottom = rectangle['bounds_xy']
        left, right = (left-192)/scale, (right-192)/scale
        top, bottom = (128-top)/scale, (128-bottom)/scale
        index = len(vertices)
        z = rectangle['world_z']
        vertices.extend([(left,bottom,z),(right,bottom,z),
                         (right,top,z),(left,top,z)])
        faces.append((index,index+1,index+2,index+3))
    mesh = bpy.data.meshes.new('FixedForegroundPanels')
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new('FixedForegroundPanels',mesh)
    bpy.context.collection.objects.link(obj)
    panel_material = bpy.data.materials.new('UniformPanel')
    panel_material.use_nodes = True
    nodes = panel_material.node_tree.nodes
    nodes.clear()
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (.18,.20,.22,1)
    emission.inputs['Strength'].default_value = 1
    output = nodes.new('ShaderNodeOutputMaterial')
    panel_material.node_tree.links.new(emission.outputs[0],output.inputs['Surface'])
    mesh.materials.append(panel_material)


def audit_projection(scene, joints, expected_scale):
    observed = []
    expected = []
    for point in joints.reshape(-1, 3):
        uv = world_to_camera_view(scene, scene.camera, Vector(point))
        observed.append([uv.x*384, (1-uv.y)*256])
        expected.append([point[0]*expected_scale+192, -point[1]*expected_scale+128])
    return float(np.max(np.abs(np.asarray(observed)-expected)))


def inter_person_collisions(people):
    if len(people) == 1:
        return 0, []
    trees = [BVHTree.FromPolygons([Vector(v) for v in vertices], faces.tolist(), all_triangles=True)
             for _, vertices, faces in people]
    pairs = trees[0].overlap(trees[1])
    return len(pairs), pairs[:10]


def self_collisions(people, data):
    if 'arm_core' not in data or 'torso_core' not in data:
        return None
    face_count = len(data['faces'])//len(people)
    counts = []
    for person, (_, vertices, faces) in enumerate(people):
        selected = slice(person*face_count, (person+1)*face_count)
        trees = [BVHTree.FromPolygons([Vector(v) for v in vertices], faces[data[key][selected]].tolist(), all_triangles=True)
                 for key in ('arm_core', 'torso_core')]
        counts.append(len(trees[0].overlap(trees[1])))
    return counts


def wrist_surface_audit(people, data):
    audits = []
    face_count = len(data['faces'])//len(people)
    for person, (_, vertices, faces) in enumerate(people):
        tree = BVHTree.FromPolygons([Vector(v) for v in vertices], faces.tolist(), all_triangles=True)
        wrist = data['joints'][person, 20]
        hit, normal, face_index, distance = tree.ray_cast(Vector((wrist[0], wrist[1], 6.)), Vector((0, 0, -1)))
        is_arm = face_index is not None and bool(data['arm_core'][person*face_count+face_index])
        audits.append(dict(first_surface_is_forearm_or_hand=is_arm,
                           first_surface_depth=None if hit is None else float(hit.z),
                           wrist_depth=float(wrist[2])))
    return audits


def array_people(data):
    count = len(data['joints'])
    vertices = np.split(data['vertices'], count)
    faces = np.split(data['faces'], count)
    return [(None, points, triangles-person*len(points))
            for person, (points, triangles) in enumerate(zip(vertices, faces))]


def geometry_fingerprint(data, scale):
    digest = hashlib.sha256()
    for name in ('vertices', 'faces', 'joints', 'arm_core', 'torso_core'):
        digest.update(data[name].tobytes())
    return scale, digest.hexdigest()


def audit_camera(scale):
    scene = bpy.context.scene
    if not scene.camera or scene.camera.name != 'AuditCamera':
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.object.camera_add(location=(0, 0, 6), rotation=(0, 0, 0))
        scene.camera = bpy.context.object
        scene.camera.name = 'AuditCamera'
    scene.camera.data.type = 'ORTHO'
    scene.camera.data.ortho_scale = 384/scale
    scene.render.resolution_x = 384
    scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    paths = sorted(args.root.glob('family_*/scenes.json'))
    if args.selection:
        selected = {family['family_id'] for family in json.loads(args.selection.read_text())['families']}
        paths = [path for path in paths if int(path.parent.name.split('_')[1]) in selected]
    if not paths:
        raise ValueError('No generated scene manifests found')
    processed = 0
    geometry_cache = {}
    started = time.monotonic()
    for scenes_path in paths:
        scenes = json.loads(scenes_path.read_text())
        for record in scenes:
            output = scenes_path.parent/record['rgb']
            audit_path = output.with_suffix('.render.json')
            if audit_path.exists() and (output.exists() or args.audit_only) and 'wrist_surface' in json.loads(audit_path.read_text()):
                continue
            data = np.load(scenes_path.parent/record['mesh'])
            if args.audit_only:
                scene = audit_camera(record['scale'])
                people = array_people(data)
            else:
                scene, palette = setup_scene(record['scale'], record['style_seed'], record.get('light_angle', 0))
                if record.get('rest_texture', False):
                    add_rest_texture(palette, record['style_seed'])
                people = add_mesh(data, palette)
            bpy.context.view_layer.update()
            error = audit_projection(scene, data['joints'], record['scale'])
            if error > 1e-4:
                raise ValueError(f'Blender/projection mismatch {error}')
            key = geometry_fingerprint(data, record['scale'])
            cached = geometry_cache.get(key)
            collisions, examples = inter_person_collisions(people) if cached is None else (cached['inter_person_triangle_intersections'], cached['intersection_examples'])
            audit = cached or {'projection_max_error_pixels': error, 'inter_person_triangle_intersections': collisions,
                     'forearm_hand_vs_own_torso_intersections': self_collisions(people, data),
                     'intersection_examples': examples, 'renderer': bpy.app.version_string,
                     'wrist_surface': wrist_surface_audit(people, data),
                     'engine': scene.render.engine,
                     'status': 'geometry audited; RGB not rendered' if args.audit_only else 'rendered; physical/pixel audits remain separate'}
            geometry_cache[key] = audit
            if not args.audit_only:
                add_panels(record)
                scene.render.filepath = str(output)
                bpy.ops.render.render(write_still=True)
            audit_path.write_text(json.dumps(audit, indent=2)+'\n')
            processed += 1
            print('PILOT_RENDER', output, 'collisions', collisions, 'seconds', time.monotonic()-started, flush=True)
            if args.limit and processed >= args.limit:
                return


if __name__ == '__main__':
    main()
