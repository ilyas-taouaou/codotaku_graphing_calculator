use bevy::{
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
};
use bevy_inspector_egui::{
    quick::{ResourceInspectorPlugin, WorldInspectorPlugin},
    InspectorOptions,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use evalexpr::{ContextWithMutableVariables, EvalexprError, HashMapContext, Node, Value};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(WorldInspectorPlugin::default())
        .init_resource::<Configuration>()
        .register_type::<Configuration>()
        .add_plugins(ResourceInspectorPlugin::<Configuration>::default())
        .add_plugins(WireframePlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, configuration_changed)
        .run();
}

struct Graph {
    positions: Vec<Vec3>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl Graph {
    fn new(subdivisions: u32, normal: Dir3, half_size: Vec2) -> Self {
        let z_vertex_count = subdivisions + 2;
        let x_vertex_count = subdivisions + 2;
        let num_vertices = (z_vertex_count * x_vertex_count) as usize;
        let num_indices = ((z_vertex_count - 1) * (x_vertex_count - 1) * 6) as usize;

        let mut positions: Vec<Vec3> = Vec::with_capacity(num_vertices);
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(num_vertices);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(num_vertices);
        let mut indices: Vec<u32> = Vec::with_capacity(num_indices);

        let rotation = Quat::from_rotation_arc(Vec3::Y, *normal);
        let size = half_size * 2.0;

        for z in 0..z_vertex_count {
            for x in 0..x_vertex_count {
                let tx = x as f32 / (x_vertex_count - 1) as f32;
                let tz = z as f32 / (z_vertex_count - 1) as f32;
                let pos = rotation * Vec3::new((-0.5 + tx) * size.x, 0.0, (-0.5 + tz) * size.y);
                positions.push(pos);
                normals.push(normal.to_array());
                uvs.push([tx, tz]);
            }
        }

        for z in 0..z_vertex_count - 1 {
            for x in 0..x_vertex_count - 1 {
                let quad = z * x_vertex_count + x;
                indices.push(quad + x_vertex_count + 1);
                indices.push(quad + 1);
                indices.push(quad + x_vertex_count);
                indices.push(quad);
                indices.push(quad + x_vertex_count);
                indices.push(quad + 1);
            }
        }

        Graph {
            positions,
            normals,
            uvs,
            indices,
        }
    }

    fn update(
        &mut self,
        expression: &Node,
        context: &mut HashMapContext,
        delta_time: f32,
    ) -> Result<(), EvalexprError> {
        for vertex in self.positions.iter_mut() {
            context
                .set_value("x".into(), Value::from_float(vertex.x as f64))
                .unwrap();
            context
                .set_value("z".into(), Value::from_float(vertex.z as f64))
                .unwrap();
            let target = expression.eval_number_with_context(context)? as f32;
            // lerp
            let lerp_duration = 0.1;
            vertex.y = vertex.y.lerp(target, delta_time / lerp_duration);
        }
        Ok(())
    }

    fn mesh(&self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_indices(Indices::U32(self.indices.clone()))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions.clone())
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals.clone())
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs.clone())
    }
}

use bevy_inspector_egui::prelude::ReflectInspectorOptions;

#[derive(Reflect, Resource, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct Configuration {
    expressions: Vec<String>,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            expressions: vec!["0.5*cos(x*6+t)".to_string(), "0.5*sin(x*6+t)".to_string()],
        }
    }
}

fn configuration_changed(
    configuration: Res<Configuration>,
    mut graphs_resource: ResMut<Graphs>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
    time: Res<Time>,
) {
    // if configuration.is_changed() {
    let mut context = std::mem::take(&mut graphs_resource.context);

    context
        .set_value(
            "t".into(),
            Value::from_float(time.elapsed_seconds_f64() as f64),
        )
        .unwrap();

    for (expression, (graph, entity)) in configuration
        .expressions
        .iter()
        .zip(graphs_resource.graphs.iter_mut())
    {
        if let Ok(expression) = evalexpr::build_operator_tree(expression) {
            if graph
                .update(&expression, &mut context, time.delta_seconds())
                .is_ok()
            {
                let mesh = meshes.add(graph.mesh());
                commands.get_entity(entity.clone()).unwrap().insert(mesh);
            }
        }
    }

    // generate new graphs if needed
    for expression in configuration
        .expressions
        .iter()
        .skip(graphs_resource.graphs.len())
    {
        if let Ok(expression) = evalexpr::build_operator_tree(expression) {
            let mut graph = Graph::new(64, Dir3::Y, Vec2::splat(0.5));
            if graph
                .update(&expression, &mut context, time.delta_seconds())
                .is_ok()
            {
                let mesh = meshes.add(graph.mesh());
                let entity = commands.spawn((
                    PbrBundle {
                        mesh,
                        material: graphs_resource.material.clone(),
                        ..default()
                    },
                    Wireframe,
                ));
                graphs_resource.graphs.push((graph, entity.id()));
            }
        }
    }

    graphs_resource.context = context;
    // }
}

#[derive(Resource)]
struct Graphs {
    graphs: Vec<(Graph, Entity)>,
    context: HashMapContext,
    material: Handle<StandardMaterial>,
}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    let mut mat: StandardMaterial = Color::srgba(1.0, 1.0, 1.0, 0.5).into();
    mat.cull_mode = None;
    mat.alpha_mode = AlphaMode::Blend;
    mat.unlit = true;
    let white_material = materials.add(mat);

    let context = evalexpr::context_map! {
        "cos" => Function::new(|argument| {
            let number: f64 = argument.as_number()?;
            Ok(Value::Float(number.cos()))
        }),
        "sin" => Function::new(|argument| {
            let number: f64 = argument.as_number()?;
            Ok(Value::Float(number.sin()))
        }),
    }
    .unwrap();

    let _camera = commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        PanOrbitCamera::default(),
    ));

    let graphs = vec![];
    commands.insert_resource(Graphs {
        graphs,
        context,
        material: white_material,
    });
}
