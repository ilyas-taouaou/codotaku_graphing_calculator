use std::sync::Mutex;

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
        .add_systems(Update, update_graphs)
        .run();
}

struct Graph {
    positions: Vec<Vec3>,
    indices: Vec<u32>,
    context: HashMapContext,
}

impl Graph {
    fn new(subdivisions: u32, normal: Dir3, half_size: Vec2) -> Self {
        let z_vertex_count = subdivisions + 2;
        let x_vertex_count = subdivisions + 2;
        let num_vertices = (z_vertex_count * x_vertex_count) as usize;
        let num_indices = ((z_vertex_count - 1) * x_vertex_count * 2) as usize;

        let mut positions: Vec<Vec3> = Vec::with_capacity(num_vertices);
        let mut indices: Vec<u32> = Vec::with_capacity(num_indices);

        let rotation = Quat::from_rotation_arc(Vec3::Y, *normal);
        let size = half_size * 2.0;

        for z in 0..z_vertex_count {
            for x in 0..x_vertex_count {
                let tx = x as f32 / (x_vertex_count - 1) as f32;
                let tz = z as f32 / (z_vertex_count - 1) as f32;
                let pos = rotation * Vec3::new((-0.5 + tx) * size.x, 0.0, (-0.5 + tz) * size.y);
                positions.push(pos);
            }
        }

        for z in 0..z_vertex_count - 1 {
            if z % 2 == 0 {
                for x in 0..x_vertex_count {
                    indices.push(z * x_vertex_count + x);
                    indices.push((z + 1) * x_vertex_count + x);
                }
            } else {
                for x in (0..x_vertex_count).rev() {
                    indices.push((z + 1) * x_vertex_count + x);
                    indices.push(z * x_vertex_count + x);
                }
            }
        }

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

        Graph {
            positions,
            indices,
            context,
        }
    }

    fn update(&mut self, expression: &Node, delta_time: f32, t: f64) -> Result<(), EvalexprError> {
        self.context
            .set_value("t".into(), Value::from_float(t))
            .unwrap();
        for vertex in self.positions.iter_mut() {
            self.context
                .set_value("x".into(), Value::from_float(vertex.x as f64))
                .unwrap();
            self.context
                .set_value("z".into(), Value::from_float(vertex.z as f64))
                .unwrap();
            let target = expression.eval_number_with_context(&mut self.context)? as f32;
            let lerp_duration = 0.1;
            vertex.y = vertex.y.lerp(target, delta_time / lerp_duration);
        }
        Ok(())
    }

    fn mesh(&self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleStrip,
            RenderAssetUsages::default(),
        )
        .with_inserted_indices(Indices::U32(self.indices.clone()))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions.clone())
    }
}

use bevy_inspector_egui::prelude::ReflectInspectorOptions;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[derive(Reflect, Resource, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct Configuration {
    expressions: Vec<String>,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            // beautiful 3d graphs that take x, z and t (time) as input
            expressions: vec![
                "(x * 2.0) * (z * 2.0) * sin(t * 2.0)".to_string(),
                "cos(x * 2.0) * cos(z * 2.0) * sin(t * 2.0)".to_string(),
            ],
        }
    }
}

fn update_graphs(
    configuration: Res<Configuration>,
    mut graphs_resource: ResMut<Graphs>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
    time: Res<Time>,
) {
    let t = time.elapsed_seconds_f64();
    let dt = time.delta_seconds();

    let queued_meshes = Mutex::new(Vec::new());

    configuration
        .expressions
        .par_iter()
        .zip(graphs_resource.graphs.par_iter_mut())
        .for_each(|(expression, (graph, entity))| {
            if let Ok(expression) = evalexpr::build_operator_tree(expression) {
                if graph.update(&expression, dt, t).is_ok() {
                    queued_meshes.lock().unwrap().push((graph.mesh(), *entity));
                }
            }
        });

    for (mesh, entity) in queued_meshes.lock().unwrap().iter() {
        commands.entity(*entity).insert(PbrBundle {
            mesh: meshes.add(mesh.clone()),
            material: graphs_resource.material.clone(),
            ..default()
        });
    }

    for expression in configuration
        .expressions
        .iter()
        .skip(graphs_resource.graphs.len())
    {
        if let Ok(expression) = evalexpr::build_operator_tree(expression) {
            let mut graph = Graph::new(64, Dir3::Y, Vec2::splat(0.5));
            if graph.update(&expression, dt, t).is_ok() {
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

    if configuration.expressions.len() < graphs_resource.graphs.len() {
        for (_, entity) in graphs_resource
            .graphs
            .iter()
            .skip(configuration.expressions.len())
        {
            commands.entity(*entity).despawn();
        }
        graphs_resource
            .graphs
            .truncate(configuration.expressions.len());
    }
}

#[derive(Resource)]
struct Graphs {
    graphs: Vec<(Graph, Entity)>,
    material: Handle<StandardMaterial>,
}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    let mut mat: StandardMaterial = Color::WHITE.into();
    mat.cull_mode = None;
    mat.alpha_mode = AlphaMode::Blend;
    mat.unlit = true;
    let white_material = materials.add(mat);

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
        material: white_material,
    });
}
